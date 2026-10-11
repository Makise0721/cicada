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

KEY = "test-secret-never-log"
REQUEST = ModelRequest((UserMessage("任务"),), (ToolSpec("read", "read", {"type": "object"}),))


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
async def test_cancellation_interrupts_blocked_io_and_closes_response(phase):
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
            async for event in DeepSeekModel(client, DeepSeekConfig(), KEY).stream(REQUEST, token):
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
