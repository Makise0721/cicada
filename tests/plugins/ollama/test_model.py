import asyncio
import json
from pathlib import Path

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelRequest, StreamDone, ToolCallEvent, TextDelta
from cicada.plugins.ollama.model import OllamaModel
from cicada.plugins.ollama.protocol import OllamaConfig

CONFIG = OllamaConfig(base_url="http://ollama.test")
RECORDINGS = Path(__file__).parent / "recordings"


def recording(name: str) -> bytes:
    return (RECORDINGS / f"{name}.ndjson").read_bytes()


def ndjson(*entries) -> bytes:
    """拼接 NDJSON 字节流; bytes 条目原样嵌入 (坏行注入用), 其余 json 序列化."""
    lines = [entry if isinstance(entry, bytes) else json.dumps(entry).encode("utf-8") for entry in entries]
    return b"\n".join(lines) + b"\n"


def ndjson_response(body: bytes, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=body, headers={"content-type": "application/x-ndjson"})


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
        StreamDone("stop"),
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
    assert events == [TextDelta("The answer is 42."), StreamDone("stop")]


async def test_single_tool_call_yields_tool_use_despite_done_reason_stop():
    # 附录 A.1 形态: 工具轮 done_reason 为 "stop", stop_reason 推导不依赖它
    model = make_model(ndjson_response(recording("tool-call")), [])
    events = await collect(model)
    assert events == [
        ToolCallEvent("call_jiv93d1a", "get_weather", json.dumps({"city": "Paris"})),
        StreamDone("tool_use"),
    ]


async def test_mixed_content_and_tool_call():
    model = make_model(ndjson_response(recording("mixed-content-toolcall")), [])
    events = await collect(model)
    assert events == [
        TextDelta("Let me check the weather."),
        ToolCallEvent("call_abc123", "get_weather", json.dumps({"city": "Rome"})),
        StreamDone("tool_use"),
    ]


async def test_done_reason_length_maps_to_length():
    model = make_model(ndjson_response(recording("length-cutoff")), [])
    events = await collect(model)
    assert events == [
        TextDelta("once upon a time there was a little"),
        TextDelta(" cicada that sang all summer long and"),
        StreamDone("length"),
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
