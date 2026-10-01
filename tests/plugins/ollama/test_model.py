import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import ToolResult, ToolResultMessage, UserMessage
from cicada.core.ports import ModelMetrics, ModelRequest, StreamDone, ToolCallEvent, TextDelta, ToolSpec
from cicada.plugins.ollama.model import OllamaModel
from cicada.plugins.ollama.protocol import OllamaConfig, build_request

CONFIG = OllamaConfig(base_url="http://ollama.test")
RECORDINGS = Path(__file__).parent / "recordings"

TEXT_STREAM_METRICS = ModelMetrics(
    input_tokens=9, output_tokens=12, provider_duration_s=4320753500 / 1e9
)


def recording(name: str) -> bytes:
    return (RECORDINGS / f"{name}.ndjson").read_bytes()


def ndjson(*entries) -> bytes:
    """拼接 NDJSON 字节流; bytes 条目原样嵌入 (坏行注入用), 其余 json 序列化."""
    lines = [entry if isinstance(entry, bytes) else json.dumps(entry).encode("utf-8") for entry in entries]
    return b"\n".join(lines) + b"\n"


def ndjson_response(body: bytes, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=body, headers={"content-type": "application/x-ndjson"})


def ok_done() -> bytes:
    return ndjson({"message": {"role": "assistant", "content": "hi"}, "done": True, "done_reason": "stop"})


class CapturingTransport(httpx.MockTransport):
    """记录实际 POST 次数的 MockTransport; 无命中回 200 合法终结流."""

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            return ndjson_response(ok_done())

        super().__init__(handler)


def capped_model(limit: int, transport: CapturingTransport) -> OllamaModel:
    config = replace(CONFIG, max_request_bytes=limit)
    return OllamaModel(httpx.AsyncClient(transport=transport), config)


def wire_size(model: OllamaModel, request: ModelRequest) -> int:
    """用适配器自身的 serializer 测同一请求的 wire 字节数 (含 JSON 转义)."""
    payload = build_request(request, model.config)
    return len(model.client.build_request("POST", "http://ollama.test/api/chat", json=payload).content)


def size_boundary_request(target: int) -> tuple[ModelRequest, int]:
    """构造 wire 字节恰好等于 target 的请求 (正文长度取差值, 与 model 上限无关)."""
    tools = (ToolSpec("t", "d" * 64, {"type": "object", "properties": {"a": {"type": "string"}}}),)
    probe = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(lambda request: ndjson_response(ok_done()))), CONFIG)
    padding = target - wire_size(probe, ModelRequest((UserMessage(text=""),), tools, system_prompt="sys"))
    assert padding > 0, "target 太小: 固定开销已超出目标字节数"
    request = ModelRequest((UserMessage(text="q" * padding),), tools, system_prompt="sys")
    return request, wire_size(probe, request)


def make_model(response, seen: list) -> OllamaModel:
    """response 为 httpx.Response (回放) 或 Exception (注入发送失败)."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaModel(client, CONFIG)


async def collect(model: OllamaModel, cancel=None, request=None) -> list:
    cancel = cancel or CancelToken()
    request = request or ModelRequest((UserMessage(text="hi"),), ())
    gen = model.stream(request, cancel)
    events = [event async for event in gen]
    await gen.aclose()
    await model.client.aclose()
    return events


async def test_plain_text_stream_replays_recording():
    seen: list[httpx.Request] = []
    model = make_model(ndjson_response(recording("text-stream")), seen)
    events = await collect(model)
    assert events == [
        TextDelta("The capital"),
        TextDelta(" of France is Paris."),
        StreamDone("stop", metrics=TEXT_STREAM_METRICS),
    ]
    assert seen[0].url == httpx.URL("http://ollama.test/api/chat")
    payload = json.loads(seen[0].content)
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload


async def test_thinking_deltas_are_discarded():
    model = make_model(ndjson_response(recording("thinking-stream")), [])
    events = await collect(model)
    assert events == [
        TextDelta("The answer is 42."),
        StreamDone(
            "stop",
            metrics=ModelMetrics(
                input_tokens=12, output_tokens=20, provider_duration_s=5210753500 / 1e9
            ),
        ),
    ]


async def test_single_tool_call_yields_tool_use_despite_done_reason_stop():
    # 附录 A.1 形态: 工具轮 done_reason 为 "stop", stop_reason 推导不依赖它
    model = make_model(ndjson_response(recording("tool-call")), [])
    events = await collect(model)
    assert events == [
        ToolCallEvent("call_jiv93d1a", "get_weather", json.dumps({"city": "Paris"})),
        StreamDone(
            "tool_use",
            metrics=ModelMetrics(
                input_tokens=294, output_tokens=26, provider_duration_s=4320753500 / 1e9
            ),
        ),
    ]


async def test_mixed_content_and_tool_call():
    model = make_model(ndjson_response(recording("mixed-content-toolcall")), [])
    events = await collect(model)
    assert events == [
        TextDelta("Let me check the weather."),
        ToolCallEvent("call_abc123", "get_weather", json.dumps({"city": "Rome"})),
        StreamDone(
            "tool_use",
            metrics=ModelMetrics(
                input_tokens=310, output_tokens=30, provider_duration_s=4320753500 / 1e9
            ),
        ),
    ]


async def test_done_reason_length_maps_to_length():
    model = make_model(ndjson_response(recording("length-cutoff")), [])
    events = await collect(model)
    assert events == [
        TextDelta("once upon a time there was a little"),
        TextDelta(" cicada that sang all summer long and"),
        StreamDone(
            "length",
            metrics=ModelMetrics(
                input_tokens=15, output_tokens=128, provider_duration_s=4320753500 / 1e9
            ),
        ),
    ]


async def test_tool_calls_take_precedence_over_length():
    body = ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "t", "arguments": {}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length"},
    )
    model = make_model(ndjson_response(body), [])
    events = await collect(model)
    assert events[-1] == StreamDone("tool_use")


async def test_http_404_carries_body_error_field():
    body = b'{"error":"model \\"qwen3.5:9b\\" not found. For more information, check https://ollama.com"}'
    model = make_model(ndjson_response(body, status_code=404), [])
    events = await collect(model)
    assert len(events) == 1
    assert events[-1].stop_reason == "error"
    assert "404" in events[-1].error
    assert "not found" in events[-1].error


async def test_http_500_carries_body_error_field():
    model = make_model(ndjson_response(b'{"error":"registry unavailable"}', status_code=500), [])
    events = await collect(model)
    assert len(events) == 1
    assert events[-1].stop_reason == "error"
    assert "500" in events[-1].error
    assert "registry unavailable" in events[-1].error


async def test_bad_ndjson_line_emits_prior_text_then_error():
    body = ndjson(
        {"message": {"role": "assistant", "content": "a"}, "done": False},
        b"not-json",
    )
    model = make_model(ndjson_response(body), [])
    events = await collect(model)
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[0].text == "a"
    assert events[-1].stop_reason == "error"
    assert "protocol error" in events[-1].error


async def test_stream_without_done_line_reports_error():
    body = b'{"message":{"role":"assistant","content":"a"},"done":false}\n'
    model = make_model(ndjson_response(body), [])
    events = await collect(model)
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[0].text == "a"
    assert events[-1].stop_reason == "error"
    assert "without a done line" in events[-1].error


async def test_partial_final_line_without_done_reports_error():
    body = b'{"message":{"role":"assistant","content":"a"},"done":false}\n{"message":{"role":"ass'
    model = make_model(ndjson_response(body), [])
    events = await collect(model)
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[0].text == "a"
    assert events[-1].stop_reason == "error"
    # 流末残留半行: 作为一行解析, 以坏行协议错误终结 (同样不会裸抛)
    assert "protocol error" in events[-1].error


async def test_connection_failure_reports_error():
    model = make_model(httpx.ConnectError("connection refused"), [])
    events = await collect(model)
    assert len(events) == 1
    assert events[-1].stop_reason == "error"
    assert "connection refused" in events[-1].error


async def test_read_timeout_reports_error():
    model = make_model(httpx.ReadTimeout("timed out"), [])
    events = await collect(model)
    assert len(events) == 1
    assert events[-1].stop_reason == "error"
    assert "timed out" in events[-1].error


async def test_stream_broken_midway_reports_error():
    line = b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n'

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield line
            raise httpx.ReadError("connection reset")

    model = make_model(
        httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=BrokenStream()), []
    )
    events = await collect(model)
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[0].text == "Hello"
    assert events[-1].stop_reason == "error"
    assert "connection reset" in events[-1].error


ERROR_SHAPES = {
    "http_500": lambda: ndjson_response(b'{"error":"boom"}', 500),
    "http_404": lambda: ndjson_response(b'{"error":"model not found"}', 404),
    "bad_line": lambda: ndjson_response(b'{"message":{"content":"a"},"done":false}\nnot-json\n'),
    "missing_done": lambda: ndjson_response(b'{"message":{"content":"a"},"done":false}\n'),
    "protocol_bad_tool_call": lambda: ndjson_response(ndjson(
        {"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]}, "done": False},
    )),
    "connect_error": lambda: httpx.ConnectError("connection refused"),
    "read_timeout": lambda: httpx.ReadTimeout("timed out"),
}


@pytest.mark.parametrize("shape", sorted(ERROR_SHAPES))
async def test_all_error_shapes_yield_exactly_one_error_done(shape):
    """不裸抛契约: 全部错误形态只产出恰好一个 StreamDone("error"), 无异常外泄."""
    seen: list[httpx.Request] = []
    model = make_model(ERROR_SHAPES[shape](), seen)
    events = await collect(model)
    dones = [event for event in events if isinstance(event, StreamDone)]
    assert len(dones) == 1
    assert dones[0].stop_reason == "error"
    assert dones[0].error
    assert events[-1] is dones[0]


async def test_cancelled_before_entry_sends_no_request():
    seen: list[httpx.Request] = []
    model = make_model(ndjson_response(recording("text-stream")), seen)
    cancel = CancelToken()
    cancel.cancel()
    events = await collect(model, cancel=cancel)
    assert events == [StreamDone("aborted")]
    assert seen == []


async def test_cancel_during_stream_closes_response_and_leaves_no_tasks():
    gate = asyncio.Event()
    closed: list[bool] = []
    line1 = b'{"model":"m","message":{"role":"assistant","content":"Hello"},"done":false}\n'
    line2 = b'{"model":"m","message":{"role":"assistant","content":" world"},"done":false}\n'

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield line1
            await gate.wait()
            yield line2

        async def aclose(self):
            closed.append(True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=GatedStream())

    model = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), CONFIG)
    cancel = CancelToken()
    gen = model.stream(ModelRequest((UserMessage(text="hi"),), ()), cancel)
    first = await gen.__anext__()
    assert first == TextDelta("Hello")
    cancel.cancel()
    second = await gen.__anext__()
    assert second == StreamDone("aborted")
    assert closed == [True]  # 响应在产出 aborted 前已关闭, 释放连接
    await gen.aclose()
    await model.client.aclose()
    assert [task for task in asyncio.all_tasks() if task is not asyncio.current_task()] == []


class BoomCloseStream(httpx.AsyncByteStream):
    """aclose 抛错的流 (审查 R1 探针): 连接拆卸已在故障中, 关闭再出错."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self) -> None:
        raise httpx.HTTPError("close boom")


def boom_close_model(body: bytes) -> OllamaModel:
    response = httpx.Response(
        200, headers={"content-type": "application/x-ndjson"}, stream=BoomCloseStream(body)
    )
    return make_model(response, [])


async def test_close_failure_on_done_path_does_not_bare_throw():
    # R1①: 正常 done 路径上 response.aclose() 抛错不得顶掉 StreamDone("stop")
    body = ndjson({"message": {"role": "assistant", "content": "hi"}, "done": True, "done_reason": "stop"})
    events = await collect(boom_close_model(body))
    assert events == [TextDelta("hi"), StreamDone("stop")]


async def test_close_failure_on_cancel_path_still_yields_aborted():
    # R1②: 流中取消路径上 response.aclose() 抛错不得顶掉 StreamDone("aborted")
    gate = asyncio.Event()
    line1 = b'{"model":"m","message":{"role":"assistant","content":"Hello"},"done":false}\n'

    class GatedBoomCloseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield line1
            await gate.wait()
            yield b'{"model":"m","message":{"role":"assistant","content":" world"},"done":false}\n'

        async def aclose(self) -> None:
            raise httpx.HTTPError("close boom")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=GatedBoomCloseStream())

    model = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), CONFIG)
    cancel = CancelToken()
    gen = model.stream(ModelRequest((UserMessage(text="hi"),), ()), cancel)
    assert await gen.__anext__() == TextDelta("Hello")
    cancel.cancel()
    assert await gen.__anext__() == StreamDone("aborted")
    await gen.aclose()
    await model.client.aclose()


async def test_close_failure_on_error_path_does_not_replace_error_done():
    # R1③: 坏行错误路径上 response.aclose() 抛错不得顶替 StreamDone("error")
    body = ndjson({"message": {"role": "assistant", "content": "a"}, "done": False}, b"not-json")
    events = await collect(boom_close_model(body))
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[-1].stop_reason == "error"
    assert "protocol error" in events[-1].error


async def test_midstream_connection_reset_is_normalized_to_error_done():
    # R2: 读取路径的 OSError (Windows 连接重置 errno 10054) 不得裸抛, 归一 StreamDone("error")
    line = b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n'

    class ResetStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield line
            raise OSError(10054, "Connection reset by peer")

    model = make_model(
        httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=ResetStream()), []
    )
    events = await collect(model)
    assert [type(event).__name__ for event in events] == ["TextDelta", "StreamDone"]
    assert events[0].text == "Hello"
    assert events[-1].stop_reason == "error"
    assert "stream failed" in events[-1].error


async def test_http_error_body_read_failure_is_normalized_to_error_done():
    # R2: 非 2xx 响应体 aread() 的 OSError 同样归一 StreamDone("error"), 不裸抛

    class BrokenBodyStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise OSError(10054, "Connection reset by peer")
            yield  # 使 __aiter__ 成为异步生成器, 首个迭代即抛错

    model = make_model(
        httpx.Response(500, headers={"content-type": "application/x-ndjson"}, stream=BrokenBodyStream()), []
    )
    events = await collect(model)
    assert len(events) == 1
    assert events[-1].stop_reason == "error"
    assert "500" in events[-1].error


async def test_done_line_without_metrics_fields_yields_none_metrics():
    body = ndjson({"message": {"role": "assistant", "content": "hi"}, "done": True, "done_reason": "stop"})
    events = await collect(make_model(ndjson_response(body), []))
    assert events == [TextDelta("hi"), StreamDone("stop")]


async def test_partial_terminal_metrics_are_forwarded():
    body = ndjson(
        {"message": {"role": "assistant", "content": "hi"}, "done": True, "eval_count": 3}
    )
    events = await collect(make_model(ndjson_response(body), []))
    assert events == [
        TextDelta("hi"),
        StreamDone("stop", metrics=ModelMetrics(output_tokens=3)),
    ]


class SlowCloseStream(httpx.AsyncByteStream):
    """aclose 挂起直到被取消: 探针关闭响应的 await 被 Task 取消的分支."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self) -> None:
        await asyncio.sleep(3600)


async def test_done_metrics_survive_task_cancellation_during_close():
    # P3: 合法 done 行已解析保存计量后, 关闭响应的 await 被 Task 取消 →
    # 产出一次 StreamDone("aborted", metrics=已知计量), 不裸抛丢计量
    body = ndjson(
        {
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "total_duration": 1_000_000_000,
        }
    )
    response = httpx.Response(
        200, headers={"content-type": "application/x-ndjson"}, stream=SlowCloseStream(body)
    )
    model = make_model(response, [])
    gen = model.stream(ModelRequest((UserMessage(text="hi"),), ()), CancelToken())
    assert await gen.__anext__() == TextDelta("hi")
    pending = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.2)  # 让关闭响应的 await 挂上
    pending.cancel()
    done = await pending
    assert done == StreamDone(
        "aborted", metrics=ModelMetrics(input_tokens=7, output_tokens=3, provider_duration_s=1.0)
    )
    await gen.aclose()
    await model.client.aclose()


async def test_cancel_winning_race_over_done_line_keeps_metrics_unknown():
    # 取消抢先于终结行消费: 未消费的行不计入 metrics, aborted + metrics=None
    gate = asyncio.Event()
    done_line = (
        b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",'
        b'"prompt_eval_count":7,"eval_count":3,"total_duration":1000000000}\n'
    )

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n'
            await gate.wait()
            yield done_line

        async def aclose(self):
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=GatedStream())

    model = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), CONFIG)
    cancel = CancelToken()
    gen = model.stream(ModelRequest((UserMessage(text="hi"),), ()), cancel)
    assert await gen.__anext__() == TextDelta("Hello")
    cancel.cancel()
    assert await gen.__anext__() == StreamDone("aborted")
    await gen.aclose()
    await model.client.aclose()


async def test_oversized_request_sends_no_post_and_reports_actual_bytes_and_limit():
    transport = CapturingTransport()
    model = capped_model(200, transport)
    request = ModelRequest((UserMessage(text="x" * 500),), (), system_prompt="sys")
    expected = wire_size(model, request)
    assert expected > 200
    events = await collect(model, request=request)
    assert transport.seen == []  # 未发 POST
    assert len(events) == 1
    assert events[0].stop_reason == "error"
    assert f"{expected} bytes" in events[0].error
    assert "200 bytes" in events[0].error


async def test_request_exactly_at_limit_is_sent_and_payload_matches_wire_bytes():
    transport = CapturingTransport()
    model = capped_model(4096, transport)
    base = "中文" * 300
    padding = 4096 - wire_size(model, ModelRequest((UserMessage(text=base),), (), system_prompt="规则"))
    sized = ModelRequest((UserMessage(text=base + "x" * padding),), (), system_prompt="规则")
    assert wire_size(model, sized) == 4096
    events = await collect(model, request=sized)
    assert [event.stop_reason for event in events if isinstance(event, StreamDone)] == ["stop"]
    assert transport.seen != []
    assert len(transport.seen[0].content) == 4096
    payload = json.loads(transport.seen[0].content)
    assert payload["messages"][0] == {"role": "system", "content": "规则"}
    assert "中文" in payload["messages"][1]["content"]


async def test_one_byte_over_limit_is_not_sent():
    transport = CapturingTransport()
    model = capped_model(3000, transport)
    request, size = size_boundary_request(3000)
    assert size == 3000
    events = await collect(model, request=request)
    assert [event.stop_reason for event in events if isinstance(event, StreamDone)] == ["stop"]
    assert transport.seen != []

    transport2 = CapturingTransport()
    model2 = capped_model(2999, transport2)
    assert wire_size(model2, request) == 3000
    events2 = await collect(model2, request=request)
    assert transport2.seen == []
    assert [event.stop_reason for event in events2 if isinstance(event, StreamDone)] == ["error"]


async def test_json_escaped_tool_result_counts_escaped_bytes():
    # 转义膨胀: 同样字符数的正文, 含需要转义的字符时 wire 字节更大
    transport = CapturingTransport()
    model = capped_model(1024, transport)
    plain = ModelRequest((ToolResultMessage(result=ToolResult("c1", "check", "a" * 460)),), ())
    escaped = ModelRequest((ToolResultMessage(result=ToolResult("c1", "check", "\\" * 460)),), ())
    assert wire_size(model, escaped) > wire_size(model, plain)
    assert wire_size(model, escaped) > 1024
    events = await collect(model, request=escaped)
    assert transport.seen == []
    dones = [event for event in events if isinstance(event, StreamDone)]
    assert dones[0].stop_reason == "error"
    assert f"{wire_size(model, escaped)} bytes" in dones[0].error


async def test_tools_count_toward_request_limit():
    transport = CapturingTransport()
    tool = ToolSpec(
        "glob",
        "Find files",
        {"type": "object", "properties": {"pattern": {"type": "string", "description": "p" * 400}}},
    )
    model = capped_model(300, transport)
    request = ModelRequest((UserMessage(text="hi"),), (tool,), system_prompt="s")
    events = await collect(model, request=request)
    assert transport.seen == []
    assert [event.stop_reason for event in events if isinstance(event, StreamDone)] == ["error"]

    # 同一请求放宽上限后正常发出: 超限来自 tools 体积, 不是固定行为
    relaxed = capped_model(4000, CapturingTransport())
    assert wire_size(relaxed, request) < 4000
    assert build_request(request, relaxed.config)["tools"][0]["function"]["name"] == "glob"


async def test_max_request_bytes_none_keeps_legacy_path_without_cap():
    transport = CapturingTransport()
    model = OllamaModel(httpx.AsyncClient(transport=transport), CONFIG)
    assert CONFIG.max_request_bytes is None
    request = ModelRequest((UserMessage(text="y" * 20000),), ())
    events = await collect(model, request=request)
    assert [event.stop_reason for event in events if isinstance(event, StreamDone)] == ["stop"]
    assert len(transport.seen[0].content) > 20000


async def test_oversized_request_leaves_no_pending_tasks():
    transport = CapturingTransport()
    model = capped_model(100, transport)
    events = await collect(model, request=ModelRequest((UserMessage(text="z" * 400),), ()))
    assert [event.stop_reason for event in events if isinstance(event, StreamDone)] == ["error"]
    assert [task for task in asyncio.all_tasks() if task is not asyncio.current_task()] == []


async def test_cancelled_before_oversized_check_yields_aborted_without_post():
    # 取消优先于体量判定: 进入时已取消 -> aborted, 不检查也不发送
    transport = CapturingTransport()
    model = capped_model(100, transport)
    cancel = CancelToken()
    cancel.cancel()
    events = await collect(
        model, cancel=cancel, request=ModelRequest((UserMessage(text="z" * 400),), ())
    )
    assert events == [StreamDone("aborted")]
    assert transport.seen == []
