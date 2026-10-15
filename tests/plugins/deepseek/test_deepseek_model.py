"""生产 HTTP/SSE 接缝：网络切片、完整调用、计量、拒绝与资源生命周期。"""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from cicada.core.agent import Agent
from cicada.core.cancel import CancelToken
from cicada.core.messages import AssistantMessage, ReasoningContent, UserMessage
from cicada.core.ports import ModelRequest, ReasoningDelta, StreamDone, TextDelta, ToolCallEvent, ToolSpec
from cicada.plugins.deepseek import DeepSeekConfig, DeepSeekModel
from cicada.plugins.deepseek.protocol import build_request

KEY = "test-secret-never-log"
REQUEST = ModelRequest((UserMessage("任务"),), (ToolSpec("read", "read", {"type": "object"}),))
CTX = '<cicada-verification-context version="1">\n{"blocked":false}\n</cicada-verification-context>'


@pytest.mark.parametrize("alias, folder", [("deepseek-flash", "flash"), ("deepseek-v4-pro", "pro")])
def test_full_template_matches_pinned_reference_and_ignores_unsent_metadata(alias, folder):
    import copy
    import runpy
    from importlib import resources
    from tokenizers import Tokenizer
    from cicada.core.messages import ToolCall, ToolResult, ToolResultMessage
    from cicada.plugins.deepseek.tokenizer import FlashTokenCounter, ProTokenCounter, render_body
    counter_type = FlashTokenCounter if folder == "flash" else ProTokenCounter

    root = resources.files("cicada.plugins.deepseek.tokenizer")
    reference = runpy.run_path(str(root.joinpath(folder + "/reference.py.txt")))
    tokenizer = Tokenizer.from_file(str(root.joinpath(folder + "/tokenizer.json")))
    plain = ModelRequest((UserMessage("What is 2+2?"),), (), "You are a helpful assistant.")
    call = ToolCall("c1", "lookup", '{"query":"value","count":5}')
    history = (UserMessage("Look up a value"),
               AssistantMessage("", (call,), "tool_use", reasoning=ReasoningContent("search first", True)),
               ToolResultMessage(ToolResult("c1", "lookup", "result", details={"private": "not sent"})))
    tools = (ToolSpec("lookup", "Look up a value", {"type": "object"}),)
    tool_request = ModelRequest(history, tools, "Search tools.", CTX)
    for config, request in [(DeepSeekConfig(model=alias), plain),
                            (DeepSeekConfig(model=alias, reasoning_effort="low"), tool_request),
                            (DeepSeekConfig(model=alias, reasoning_effort="max"), replace(tool_request, system_prompt="")),
                            (DeepSeekConfig(model=alias, think=False), tool_request)]:
        body = build_request(request, config)
        mapped = copy.deepcopy(body["messages"])
        if body.get("tools"):
            if mapped[0]["role"] != "system":
                mapped.insert(0, {"role": "system", "content": ""})
            mapped[0]["tools"] = body["tools"]
        expected = reference["encode_messages"](
            mapped, thinking_mode="thinking" if config.think else "chat",
            reasoning_effort=body.get("reasoning_effort"),
        )
        rendered = render_body(copy.deepcopy(body))
        assert rendered == expected
        assert tokenizer.encode(rendered, add_special_tokens=False).ids == tokenizer.encode(expected, add_special_tokens=False).ids
        assert rendered.count("<｜begin▁of▁sentence｜>") == 1
        counter = counter_type(config)
        count = counter.count_request(request)
        assert count.tokens == len(tokenizer.encode(expected, add_special_tokens=False).ids)
        assert counter.count_request(request) is count
    counter = counter_type(DeepSeekConfig(model=alias))
    first = counter.count_request(tool_request)
    changed = replace(tool_request, messages=(history[0], replace(history[1], error="not sent"),
        ToolResultMessage(replace(history[2].result, details={"different": "not sent"}))))
    assert counter.count_request(changed) is first
    assert counter.count_request(replace(tool_request, request_context=CTX + " changed")) is not first


@pytest.mark.parametrize("alias, folder", [("deepseek-flash", "flash"), ("deepseek-v4-pro", "pro")])
def test_assets_and_wire_limit_fail_safely_before_render(monkeypatch, tmp_path, alias, folder):
    import sys
    from cicada.plugins.deepseek import tokenizer as module
    from cicada.plugins.deepseek.tokenizer import FlashTokenCounter, ProTokenCounter
    counter_type = FlashTokenCounter if folder == "flash" else ProTokenCounter

    missing = counter_type(DeepSeekConfig(model=alias))
    with monkeypatch.context() as patch:
        patch.setattr(module.resources, "files", lambda package: tmp_path)
        assert missing.count_request(REQUEST).reason == "asset-missing"
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "manifest.json").write_bytes(b"corrupt")
        assert counter_type(DeepSeekConfig(model=alias)).count_request(REQUEST).reason == "asset-invalid"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "tokenizers", None)
        assert counter_type(DeepSeekConfig(model=alias)).count_request(REQUEST).reason == "dependency-unavailable"
    too_large = counter_type(DeepSeekConfig(model=alias, max_request_bytes=10))
    monkeypatch.setattr(too_large, "_load", lambda: pytest.fail("oversized body must not load/render"))
    assert too_large.count_request(REQUEST).reason == "request-too-large"
    with pytest.raises(ValueError, match="requires tools"):
        too_large.count_request(ModelRequest((UserMessage("x"),), (), request_context=CTX))


def frame(delta=None, finish=None, usage=None):
    obj = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        obj["usage"] = usage
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\r\n\r\n").encode()


def tool(index=0, id="svc-0", name="read", arguments='{"path":"x"}'):
    return {"index": index, "id": id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


DONE = b"data: [DONE]\r\n\r\n"


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, data, *, slice_size=1, fail_close=False):
        self.data = data
        self.slice_size = slice_size
        self.closed = False
        self.fail_close = fail_close

    async def __aiter__(self):
        for index in range(0, len(self.data), self.slice_size):
            yield self.data[index:index + self.slice_size]

    async def aclose(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError(KEY)


async def test_cost_reservation_boundary_and_summary_share_call_limit(tmp_path):
    from decimal import Decimal
    from cicada.costs import CostLedger, CostProfile, ModelPrice

    profile = CostProfile('https://example.invalid/prices', '2026-10-15', 'a' * 64,
                          (ModelPrice('deepseek-flash', Decimal('1'), Decimal('1')),))
    sent = []
    def handle(request):
        sent.append(request)
        return httpx.Response(200, stream=BytesStream(frame({'content': 'answer'}, finish='stop',
            usage={'prompt_tokens': 10, 'completion_tokens': 2}) + DONE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        config = DeepSeekConfig(think=False, max_tokens=2048)
        refused = CostLedger(tmp_path / 'cost', limit=Decimal('1.10'), profile=profile)
        model = DeepSeekModel(client, config, KEY, cost_control=refused)
        events = [e async for e in model.stream(REQUEST, CancelToken())]
        assert events[-1].stop_reason == 'error' and sent == [] and refused.snapshot.calls == 0
        equal = CostLedger(tmp_path / 'cost', limit=Decimal('1.11'), profile=profile)
        model = DeepSeekModel(client, config, KEY, cost_control=equal)
        events = [e async for e in model.stream(REQUEST, CancelToken())]
        assert events[-1].stop_reason == 'stop' and len(sent) == 1
        assert equal.snapshot.settled_upper_cny == '0.01' and equal.snapshot.pending_upper_cny == '0.00'
        calls = CostLedger(tmp_path / 'cost', max_calls=2)
        model = DeepSeekModel(client, config, KEY, cost_control=calls)
        for channel in (model.initial_binding.summary_model, model.prepare_model('deepseek-v4-pro').model, model):
            events = [e async for e in channel.stream(REQUEST, CancelToken())]
        assert events[-1].stop_reason == 'error' and len(sent) == 3 and calls.snapshot.calls == 2


@pytest.mark.parametrize('failure', ['usage-missing', 'length', 'close', 'over-cap'])
async def test_cost_incomplete_call_keeps_reserve_and_blocks_next_http(tmp_path, failure):
    from decimal import Decimal
    from cicada.costs import CostLedger, CostProfile, ModelPrice

    profile = CostProfile('https://example.invalid/prices', '2026-10-15', 'a' * 64,
                          (ModelPrice('deepseek-flash', Decimal('1'), Decimal('1')),))
    ledger = CostLedger(tmp_path / 'cost', limit=Decimal('10'), profile=profile)
    sent = []
    usage = None if failure == 'usage-missing' else {'prompt_tokens': 10, 'completion_tokens': 2049 if failure == 'over-cap' else 2}
    def handle(request):
        sent.append(request)
        return httpx.Response(200, stream=BytesStream(frame({'content': 'answer'},
            finish='length' if failure == 'length' else 'stop', usage=usage) + DONE,
            fail_close=failure == 'close'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(think=False, max_tokens=2048), KEY, cost_control=ledger)
        for _ in range(2):
            events = [e async for e in model.stream(REQUEST, CancelToken())]
            assert events[-1].stop_reason in ('error', 'length')
    assert len(sent) == 1 and ledger.snapshot.sealed_reason
    assert ledger.snapshot.settled_upper_cny == '0.00' and ledger.snapshot.pending_upper_cny == '1.11'


async def test_cost_ledger_write_failure_prevents_http(tmp_path, monkeypatch):
    from cicada import costs
    sent = []
    ledger = costs.CostLedger(tmp_path / 'cost', max_calls=1)
    def fail(*args):
        raise OSError('injected write failure')
    monkeypatch.setattr(costs.os, 'replace', fail)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: sent.append(request))) as client:
        model = DeepSeekModel(client, DeepSeekConfig(), KEY, cost_control=ledger)
        events = [e async for e in model.stream(REQUEST, CancelToken())]
    assert sent == [] and events[-1].stop_reason == 'error' and ledger.snapshot.sealed_reason


async def collect(data, *, config=None, status=200, fail_close=False, request=REQUEST):
    stream = BytesStream(data, fail_close=fail_close)
    requests = []
    def handle(req):
        requests.append(req)
        return httpx.Response(status, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = DeepSeekModel(client, config or DeepSeekConfig(), KEY)
        events = [e async for e in model.stream(request, CancelToken())]
    assert stream.closed or not requests
    return events, requests


async def test_unicode_crlf_keepalive_multi_index_current_usage_and_original_ids():
    data = b": keep-alive\r\n\r\n" + frame({"reasoning_content": " 原始\n思考 "})
    data += frame({"content": "答案", "tool_calls": [tool(1, "svc-1", arguments='{"path":'),
                                                     tool(arguments='{"path":')]})
    data += frame({"tool_calls": [{"index": 0, "function": {"arguments": '"甲"}'}},
                                  {"index": 1, "function": {"arguments": '"乙"}'}}]})
    data += frame(finish="tool_calls", usage={"prompt_tokens": 100, "completion_tokens": 20,
        "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 10},
        "prompt_tokens_details": {"cached_tokens": 70}}) + DONE
    events, requests = await collect(data)
    assert events[:2] == [ReasoningDelta(" 原始\n思考 "), TextDelta("答案")]
    assert events[2:4] == [ToolCallEvent("svc-0", "read", '{"path":"甲"}'),
                           ToolCallEvent("svc-1", "read", '{"path":"乙"}')]
    assert events[-1].stop_reason == "tool_use"
    assert (events[-1].metrics.input_tokens, events[-1].metrics.output_tokens) == (100, 20)
    body = json.loads(requests[0].content)
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 32768 and "num_ctx" not in body
    assert requests[0].headers["Authorization"] == "Bearer " + KEY


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 7, "completion_tokens": 8}])
async def test_old_usage_only_and_unknown_metrics(usage):
    data = frame({"reasoning_content": ""}) + frame(finish="stop")
    if usage is not None:
        data += ("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode()
    events, _ = await collect(data + DONE)
    assert events[0] == ReasoningDelta("")
    assert events[-1].stop_reason == "stop"
    assert events[-1].metrics is None if usage is None else events[-1].metrics.input_tokens == 7


async def test_cache_usage_snapshots_and_protocol_errors():
    """同帧成对、不跨帧拼接；无效后帧保留前一次有效 usage，EOF 不冒充成功。"""
    paired = {"prompt_tokens": 10, "prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3}
    for usage in (paired, {"prompt_cache_hit_tokens": 0}, {"prompt_tokens_details": {"cached_tokens": 100}},
                  {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}):
        data = frame(finish="stop", usage=paired)
        data += ("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode() + DONE
        events, _ = await collect(data)
        metrics = events[-1].metrics
        assert events[-1].stop_reason == "stop"
        assert (metrics.cache_hit_input_tokens, metrics.cache_miss_input_tokens) == (
            usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens"),
        )
        assert metrics.input_tokens == usage.get("prompt_tokens")
    for bad in (True, -1, 1.5, "7"):
        invalid = dict(paired, prompt_cache_hit_tokens=bad)
        events, _ = await collect(frame(usage=paired) + frame(finish="stop", usage=invalid) + DONE)
        assert events[-1].stop_reason == "error"
        assert events[-1].metrics.cache_hit_input_tokens == 7
    events, _ = await collect(frame(finish="stop", usage=dict(paired, prompt_cache_miss_tokens=4)) + DONE)
    assert events[-1].stop_reason == "error" and events[-1].metrics is None
    events, _ = await collect(frame(usage=paired))
    assert events[-1].stop_reason == "error" and events[-1].metrics.cache_miss_input_tokens == 3


async def test_cache_usage_survives_cancellation_once():
    token = CancelToken()
    class CancelAfterUsage(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield frame(usage={"prompt_tokens": 10, "prompt_cache_hit_tokens": 7,
                               "prompt_cache_miss_tokens": 3})
            token.cancel()
            await token.wait()
        async def aclose(self):
            pass
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, stream=CancelAfterUsage())
    )) as client:
        agent = Agent(model=DeepSeekModel(client, DeepSeekConfig(), KEY), tools={})
        from cicada.core.events import ModelCallCompleted
        events = []
        agent.subscribe(events.append)
        await agent.run("任务", cancel=token)
    calls = [event for event in events if isinstance(event, ModelCallCompleted)]
    assert len(calls) == 1 and calls[0].stop_reason == "aborted"
    assert calls[0].metrics.cache_hit_input_tokens == 7


@pytest.mark.parametrize("tail", [
    frame(finish="stop"), DONE, b"data: broken\n\n",
    frame({"tool_calls": [tool(arguments="[]")]}, "tool_calls") + DONE,
    frame({"tool_calls": [tool(), tool(1)]}, "tool_calls") + DONE,
    frame({"tool_calls": [tool(32)]}, "tool_calls") + DONE,
    frame({"tool_calls": [tool()]}, "stop") + DONE,
    frame({"tool_calls": [tool(arguments="{")]}, "tool_calls") + DONE,
    frame(finish="content_filter") + DONE,
    frame(finish="insufficient_system_resource") + DONE,
    frame(finish="aborted") + DONE,
    frame(finish="unexpected") + DONE,
    frame({"tool_calls": [tool(), {"index": 0, "id": "conflict"}]}, "tool_calls") + DONE,
])
async def test_bad_stream_preserves_partial_and_never_delivers_executable_calls(tail):
    events, _ = await collect(frame({"reasoning_content": "片段"}) + tail)
    assert events[0] == ReasoningDelta("片段")
    assert events[-1].stop_reason == "error"
    assert not any(isinstance(e, ToolCallEvent) for e in events)


async def test_length_does_not_execute_partial_tool_and_close_failure_is_error():
    data = frame({"reasoning_content": "片段", "tool_calls": [tool(arguments="{")]}, "length") + DONE
    events, _ = await collect(data)
    assert events[-1].stop_reason == "length" and not any(isinstance(e, ToolCallEvent) for e in events)
    events, _ = await collect(frame(finish="stop") + DONE, fail_close=True)
    assert events[-1].stop_reason == "error" and KEY not in repr(events)


async def test_missing_reasoning_complete_calls_are_intents_with_error_results():
    data = frame({"tool_calls": [tool()]}, "tool_calls") + DONE
    stream = BytesStream(data)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=stream))) as client:
        result = await Agent(model=DeepSeekModel(client, DeepSeekConfig(), KEY), tools={}).run("q")
    assert result.stop_reason == "error"
    assert result.messages[1].tool_calls[0].id == "svc-0"
    assert result.messages[2].result.is_error
    assert result.messages[1].reasoning is None


@pytest.mark.parametrize("status", [400, 401, 402, 422, 429, 500, 503])
async def test_http_errors_are_bounded_secret_free_and_no_retry(status):
    events, requests = await collect(KEY.encode() * 100, status=status)
    assert len(requests) == 1
    assert events == [StreamDone("error", f"deepseek HTTP {status}")]
    assert KEY not in repr(events)


async def test_connect_exception_secret_is_never_repeated():
    def handle(req):
        raise httpx.ConnectError(KEY, request=req)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        events = [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, CancelToken())]
    assert events[-1].stop_reason == "error" and KEY not in repr(events)


@pytest.mark.parametrize("message", [
    AssistantMessage("", stop_reason="error"), AssistantMessage("", stop_reason="aborted"),
    AssistantMessage("", stop_reason="length"),
    AssistantMessage("answer", reasoning=ReasoningContent("partial", False)),
])
async def test_incompatible_history_rejected_before_send(message):
    events, requests = await collect(b"", request=replace(REQUEST, messages=(UserMessage("q"), message)))
    assert events[-1].stop_reason == "error" and requests == []


async def test_request_limit_refuses_zero_send_and_nonthinking_omits_effort_and_reasoning():
    events, requests = await collect(b"", config=DeepSeekConfig(max_request_bytes=10))
    assert events[-1].stop_reason == "error" and requests == []
    request = replace(REQUEST, tools=(), messages=(UserMessage("q"),
        AssistantMessage("answer", reasoning=ReasoningContent("partial", False))))
    events, requests = await collect(frame(finish="stop") + DONE,
                                    config=DeepSeekConfig(think=False), request=request)
    body = json.loads(requests[0].content)
    assert "tools" not in body and "reasoning_effort" not in body
    assert "reasoning_content" not in body["messages"][1]
    assert body["thinking"] == {"type": "disabled"}
    assert events[-1].stop_reason == "stop"


async def test_nonthinking_rejects_unexpected_reasoning():
    events, _ = await collect(frame({"reasoning_content": "unexpected"}, "stop") + DONE,
                              config=DeepSeekConfig(think=False))
    assert events[-1].stop_reason == "error"


async def test_request_context_maps_to_trailing_user_and_requires_tools():
    """缓存 02 (契约 C3): 非空 context 恰好追加最后一条 user; 无工具本地拒绝; 空值不加消息."""
    context = '<cicada-verification-context version="1">\n{"blocked":false}\n</cicada-verification-context>'
    with_context = replace(REQUEST, request_context=context)
    events, requests = await collect(frame(finish="stop") + DONE, request=with_context)
    body = json.loads(requests[0].content)
    assert body["messages"] == [
        {"role": "user", "content": "任务"},
        {"role": "user", "content": context},
    ], "尾部是最后一条, 恰好追加一条 user"
    assert events[-1].stop_reason == "stop"

    events, requests = await collect(b"", request=replace(with_context, tools=()))
    assert events[-1].stop_reason == "error" and requests == [], "无工具且非空本地拒绝"

    events, requests = await collect(frame(finish="stop") + DONE, request=REQUEST)
    body = json.loads(requests[0].content)
    assert body["messages"] == [{"role": "user", "content": "任务"}], "空 context 不追加消息"


@pytest.mark.parametrize("phase", ["send", "read", "task"])
async def test_cancellation_interrupts_blocked_io_and_closes_response(phase, tmp_path):
    from decimal import Decimal
    from cicada.costs import CostLedger, CostProfile, ModelPrice
    profile = CostProfile('https://example.invalid/prices', '2026-10-15', 'a' * 64,
                          (ModelPrice('deepseek-flash', Decimal('1'), Decimal('1')),))
    ledger = CostLedger(tmp_path / 'cost', limit=Decimal('10'), profile=profile)
    entered = asyncio.Event()
    class Blocked(BytesStream):
        async def __aiter__(self):
            yield frame({"reasoning_content": "partial"})
            entered.set()
            await asyncio.Event().wait()
    stream = Blocked(b"")
    async def handle(req):
        if phase == "send":
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        token = CancelToken()
        events = []
        async def run():
            async for event in DeepSeekModel(client, DeepSeekConfig(), KEY, cost_control=ledger).stream(REQUEST, token):
                events.append(event)
        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 2)
        if phase == "task":
            task.cancel()
        else:
            token.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert events[-1].stop_reason == "aborted"
    assert phase == "send" or stream.closed
    assert ledger.snapshot.sealed_reason and ledger.snapshot.pending_upper_cny == '1.14'
    assert ledger.snapshot.settled_upper_cny == '0.00'


async def test_total_deadline_interrupts_send(monkeypatch):
    import cicada.plugins.deepseek.model as module
    monkeypatch.setattr(module, "CALL_DEADLINE_S", 0.01)
    async def handle(req):
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        events = [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, CancelToken())]
    assert events == [StreamDone("error", "deepseek call timed out")]


@pytest.mark.parametrize("kwargs", [{"max_tokens": True}, {"num_ctx": 1048577},
    {"max_tokens": 393217}, {"max_tokens": 32768, "num_ctx": 10000},
    {"model": "deepseek-chat"}, {"reasoning_effort": "none"}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DeepSeekConfig(**kwargs)


@pytest.mark.parametrize("data", [b"data: " + b"x" * (256 * 1024),
    frame({"tool_calls": [tool(arguments="x" * (64 * 1024 + 1))]}),
    b": " + b"x" * (2 * 1024 * 1024 + 1)], ids=["frame", "arguments", "response"])
async def test_response_and_argument_limits_are_errors(data):
    # 使用大网络切片，避免为限额检查逐字节创建数百万异步任务。
    stream = BytesStream(frame({"reasoning_content": "partial"}) + data, slice_size=32768)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
        events = [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, CancelToken())]
    assert events[0] == ReasoningDelta("partial")
    assert events[-1].stop_reason == "error" and stream.closed


async def test_cancel_when_send_and_token_ready_together_closes_acquired_response():
    stream = BytesStream(frame(finish="stop") + DONE)
    token = CancelToken()
    async def handle(req):
        token.cancel()
        return httpx.Response(200, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        events = [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, token)]
    assert events == [StreamDone("aborted")] and stream.closed


async def test_task_cancel_during_response_close_waits_and_never_publishes_tools():
    entered, release = asyncio.Event(), asyncio.Event()
    class Closing(BytesStream):
        async def aclose(self):
            entered.set()
            await release.wait()
            self.closed = True
    stream = Closing(frame({"reasoning_content": "ok", "tool_calls": [tool()]}, "tool_calls") + DONE,
                     slice_size=4096)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
        events = []
        async def run():
            async for event in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, CancelToken()):
                events.append(event)
        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not any(isinstance(e, ToolCallEvent) for e in events)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert stream.closed and events[-1].stop_reason == "aborted"


async def test_send_cancel_race_with_slow_close_is_bounded(monkeypatch):
    import cicada.plugins.deepseek.model as module
    monkeypatch.setattr(module, "CALL_DEADLINE_S", 0.01)
    close_entered = asyncio.Event()
    class Closing(BytesStream):
        async def aclose(self):
            close_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True
    stream = Closing(b"")
    token = CancelToken()
    async def handle(req):
        token.cancel()
        return httpx.Response(200, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        async def run():
            return [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, token)]
        events = await asyncio.wait_for(run(), 1)
    assert close_entered.is_set() and stream.closed and events[-1].stop_reason == "aborted"


@pytest.mark.parametrize("separate", [False, True])
async def test_data_after_done_is_rejected_regardless_of_network_slice(separate):
    head = frame({"reasoning_content": "ok", "tool_calls": [tool()]}, "tool_calls") + DONE
    tail = b"data: broken\n\n"
    class Trailing(BytesStream):
        async def __aiter__(self):
            if separate:
                yield head
                yield tail
            else:
                yield head + tail
    stream = Trailing(b"")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
        events = [e async for e in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, CancelToken())]
    assert events[-1].stop_reason == "error"
    assert not any(isinstance(e, ToolCallEvent) for e in events) and stream.closed


async def test_send_observation_tracks_shape_changes_and_preserved_prefix():
    """缓存 03 (契约 C7): 首发观察、相同/延长/替换/尾部与工具顺序变化及保留前导数."""
    requests = []
    def handle(req):
        requests.append(req)
        return httpx.Response(200, stream=BytesStream(frame(finish="stop") + DONE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(), KEY)

        async def send(request):
            events = [e async for e in model.stream(request, CancelToken())]
            assert events[-1].stop_reason == "stop"

        def facts():
            observation = model.cache_observation
            return (observation.sequence, observation.changes, observation.preserved_messages)

        await send(REQUEST)
        assert facts() == (1, ("first_request",), None)
        await send(REQUEST)
        assert facts() == (2, ("unchanged",), 1)
        await send(replace(REQUEST, request_context=CTX))
        assert facts() == (3, ("request_context_changed",), 1), "稳定历史全部保留, 只标尾部变化"
        grown = replace(REQUEST, request_context=CTX,
                        messages=(UserMessage("任务"), AssistantMessage("答案")))
        await send(grown)
        assert facts() == (4, ("messages_extended",), 1)
        rewritten = replace(grown, messages=(UserMessage("任务"), AssistantMessage("改写")))
        await send(rewritten)
        assert facts() == (5, ("messages_replaced",), 1)
        diverged = replace(rewritten, messages=(UserMessage("其他"), AssistantMessage("改写")))
        await send(diverged)
        assert facts() == (6, ("messages_replaced",), 0), "无共同前导时如实为 0"
        write = ToolSpec("write", "write", {"type": "object"})
        rebased = replace(diverged, system_prompt="规则", tools=(REQUEST.tools[0], write))
        await send(rebased)
        assert set(model.cache_observation.changes) == {"system_changed", "tools_changed"}
        assert (model.cache_observation.sequence, model.cache_observation.preserved_messages) == (7, 2)
        swapped = replace(rebased, tools=(write, REQUEST.tools[0]))
        await send(swapped)
        assert facts() == (8, ("tools_changed",), 2), "顺序集合相同不能掩盖工具序列变化"


def test_tools_schema_canonical_copy_and_mutation_immunity():
    """缓存 03 (契约 C3): 复制 schema 并确定性编码对象键, 数组顺序保持, 外部修改不回灌."""
    parameters = {"required": ["b", "a"], "type": "object",
                  "properties": {"b": {"type": "string"}, "a": {"type": "integer"}}}
    body = build_request(
        ModelRequest((UserMessage("任务"),), (ToolSpec("read", "read", parameters),)),
        DeepSeekConfig())
    prepared = body["tools"][0]["function"]["parameters"]
    assert list(prepared) == ["properties", "required", "type"], "对象键递归排序"
    assert list(prepared["properties"]) == ["a", "b"]
    assert prepared["required"] == ["b", "a"], "数组顺序保持语义"
    serialized = json.dumps(body, ensure_ascii=False)
    parameters["properties"]["a"] = {"type": "mutated"}
    parameters["extra"] = 1
    assert json.dumps(body, ensure_ascii=False) == serialized, "外部 dict 修改不影响已准备 body"
    reordered = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                 "required": ["b", "a"]}
    again = build_request(
        ModelRequest((UserMessage("任务"),), (ToolSpec("read", "read", reordered),)),
        DeepSeekConfig())
    assert json.dumps(again, ensure_ascii=False) == serialized, "相同输入不同键序生成相同 body"


async def test_observation_requires_actual_send_and_survives_send_failure():
    """缓存 03 (契约 C7): 本地拒绝与字节守卫零 send 不更新; 发送后失败仍保留; 新实例从空起."""
    sent = []
    def fail(req):
        sent.append(req)
        return httpx.Response(500)
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(), KEY)
        events = [e async for e in model.stream(
            replace(REQUEST, request_context="x", tools=()), CancelToken())]
        assert events[-1].stop_reason == "error" and sent == [] and model.cache_observation is None
        guarded = DeepSeekModel(client, DeepSeekConfig(max_request_bytes=10), KEY)
        events = [e async for e in guarded.stream(REQUEST, CancelToken())]
        assert events[-1].stop_reason == "error" and sent == [] and guarded.cache_observation is None
        for sequence, changes in ((1, ("first_request",)), (2, ("unchanged",))):
            events = [e async for e in model.stream(REQUEST, CancelToken())]
            assert events[-1].stop_reason == "error" and len(sent) == sequence
            assert (model.cache_observation.sequence, model.cache_observation.changes) == (
                sequence, changes), "发送后失败仍保留观察并可比较"
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        assert DeepSeekModel(client, DeepSeekConfig(), KEY).cache_observation is None, "新实例不继承基线"


async def test_observation_failure_clears_baseline_keeps_send_result(monkeypatch):
    """缓存 03 (契约 C7): 诊断异常不改变发送终结, 清空基线后下次为 first_request 且序列不回退."""
    import cicada.plugins.deepseek.model as module
    original = module._request_shape
    calls = {"count": 0}
    def flaky(request, identity, body):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("observation boom")
        return original(request, identity, body)
    monkeypatch.setattr(module, "_request_shape", flaky)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=BytesStream(frame(finish="stop") + DONE)))) as client:
        model = DeepSeekModel(client, DeepSeekConfig(), KEY)
        for send_number in range(1, 4):
            events = [e async for e in model.stream(REQUEST, CancelToken())]
            assert events[-1].stop_reason == "stop", "诊断异常不改变发送终结"
            observation = model.cache_observation
            if send_number == 2:
                assert observation is None, "诊断失败显示不可得"
            elif send_number == 1:
                assert (observation.sequence, observation.changes) == (1, ("first_request",))
            else:
                assert (observation.sequence, observation.changes,
                        observation.preserved_messages) == (3, ("first_request",), None), "不假称紧邻前次"
