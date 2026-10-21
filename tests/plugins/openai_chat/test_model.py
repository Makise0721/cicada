"""兼容协议的映射与关键失败行为；资源生命周期使用共用的已验接缝。"""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import (
    AssistantMessage, ReasoningContent, ToolCall, ToolResult, ToolResultMessage, UserMessage,
)
from cicada.core.ports import ModelRequest, ToolCallEvent, ToolSpec
from cicada.models import ConfigurationRequest
from cicada.plugins.deepseek import DeepSeekConfig, DeepSeekModel
from cicada.plugins.openai_chat import ChatConfig, ChatModel
from cicada.plugins.openai_chat.protocol import build_request
from cicada.user_config import SecretValue

KEY = "synthetic-chat-key"
REQUEST = ModelRequest((UserMessage("任务"),), (ToolSpec("read", "read", {"type": "object"}),))
DONE = b"data: [DONE]\n\n"


def frame(delta=None, finish=None, usage=None):
    payload = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        for start in range(0, len(self.data), 3):
            yield self.data[start:start + 3]

    async def aclose(self):
        self.closed = True


def config(**changes):
    return replace(ChatConfig("work", "coder", "org/model", "https://gateway.example/v1",
                              32768, 4096), **changes)


def test_projection_omits_private_reasoning_and_preserves_tools_context_and_limits():
    call = ToolCall("original-call", "read", '{"path":"x"}')
    history = (UserMessage("read"), AssistantMessage("", (call,), "tool_use",
               reasoning=ReasoningContent("private thinking", True)),
               ToolResultMessage(ToolResult(call.id, call.name, "result")))
    request = ModelRequest(history, REQUEST.tools, "system", "program context")
    body = build_request(request, config(stream_usage=False, output_token_field="max_completion_tokens"))
    assert body["model"] == "org/model" and body["max_completion_tokens"] == 4096
    assert "max_tokens" not in body and "stream_options" not in body
    assert "thinking" not in body and "reasoning_effort" not in body
    assert "reasoning_content" not in body["messages"][2]
    assert body["messages"][2]["tool_calls"][0]["id"] == call.id
    assert body["messages"][3]["tool_call_id"] == call.id
    assert body["messages"][-1] == {"role": "user", "content": "program context"}
    assert history[1].reasoning.text == "private thinking"


async def test_endpoint_authentication_usage_and_summary_follow_configured_instance():
    captured = []
    streams = []
    def handle(request):
        captured.append(request)
        stream = BytesStream(frame({"content": "答案"}, "stop",
                                   {"prompt_tokens": 20, "completion_tokens": 2}) + DONE)
        streams.append(stream)
        return httpx.Response(200, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        binding = ChatModel(client, config(), KEY).initial_binding
        for model in (binding.model, binding.summary_model):
            events = [event async for event in model.stream(REQUEST, CancelToken())]
            assert events[-1].stop_reason == "stop"
            assert events[-1].metrics.input_tokens == 20
            assert events[-1].metrics.cache_hit_input_tokens is None
        assert binding.identity.provider == binding.summary_identity.provider == "work"
        assert binding.identity.model == "org/model"
        assert dict(binding.summary_identity.options)["max_tokens"] == 2048
    assert all(stream.closed for stream in streams)
    assert all(str(request.url) == "https://gateway.example/v1/chat/completions" for request in captured)
    assert all(request.headers["authorization"] == "Bearer " + KEY for request in captured)
    assert [json.loads(request.content)["max_tokens"] for request in captured] == [4096, 2048]


@pytest.mark.parametrize("tail", ["missing-done", "length", "bad-arguments", "private-reasoning"])
async def test_unfinished_or_unsupported_stream_never_releases_tool_calls(tail):
    args = "{" if tail == "bad-arguments" else '{"path":"x"}'
    data = frame({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                  "function": {"name": "read", "arguments": args}}]})
    if tail == "private-reasoning":
        data = frame({"reasoning_content": ""}) + data
    data += frame(finish="length" if tail == "length" else "tool_calls")
    if tail != "missing-done":
        data += DONE
    stream = BytesStream(data)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream))) as client:
        events = [event async for event in ChatModel(client, config(), KEY).stream(REQUEST, CancelToken())]
    assert events[-1].stop_reason in ("length", "error")
    assert not any(isinstance(event, ToolCallEvent) for event in events)
    assert stream.closed and KEY not in repr(events)


async def test_loopback_no_auth_and_explicit_key_replacement():
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, stream=BytesStream(frame({"content": "ok"}, "stop") + DONE))
    local = config(base_url="http://127.0.0.1:1234/compat/", max_tokens=1000)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ChatModel(client, local, None)
        assert dict(model.initial_binding.summary_identity.options)["max_tokens"] == 1000
        await _consume(model)
        binding = model.prepare_configuration(ConfigurationRequest(
            local.identity, "replace", SecretValue(KEY)))
        await _consume(binding.model)
        with pytest.raises(ValueError, match="loopback"):
            ChatModel(client, config(), None)
    assert "authorization" not in captured[0].headers
    assert captured[1].headers["authorization"] == "Bearer " + KEY
    assert str(captured[0].url) == "http://127.0.0.1:1234/compat/chat/completions"


async def _consume(model):
    return [event async for event in model.stream(REQUEST, CancelToken())]


async def test_generic_deadline_and_http_error_are_bounded_and_secret_free(monkeypatch):
    from cicada.plugins.openai_chat import model as module
    monkeypatch.setattr(module, "CALL_DEADLINE_S", 0.01)
    sent = []
    async def blocked(request):
        sent.append(request)
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(blocked)) as client:
        events = await _consume(ChatModel(client, config(), KEY))
    assert events[-1].stop_reason == "error" and "timed out" in events[-1].error
    assert len(sent) == 1 and KEY not in repr(events)


async def test_deepseek_identity_override_survives_summary_and_reconfiguration():
    ordinary = DeepSeekConfig()
    identity = replace(ordinary.identity, provider="work-deepseek", options=(
        *ordinary.identity.options, ("protocol", "deepseek-chat"),
        ("model_key", "reasoner"), ("profile_version", 1)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        model = DeepSeekModel(client, ordinary, KEY, identity_override=identity)
        binding = model.prepare_reasoning_effort("low")
        assert binding.identity.provider == binding.summary_identity.provider == "work-deepseek"
        assert dict(binding.identity.options)["model_key"] == "reasoner"
        assert dict(binding.identity.options)["reasoning_effort"] == "low"
        assert binding.summary_identity.think is False
        assert dict(binding.summary_identity.options)["max_tokens"] == 2048
        updated = binding.model.prepare_configuration(ConfigurationRequest(binding.identity, "replace", SecretValue("new-key")))
        assert updated.identity == binding.identity
        assert updated.model._api_key == "new-key"
        with pytest.raises(ValueError, match="does not match"):
            DeepSeekModel(client, ordinary, KEY, identity_override=replace(identity, model="deepseek-v4-pro"))


async def test_repeated_task_cancellation_still_closes_late_acquired_response():
    import time
    from cicada.plugins.chat_stream import wait_operation

    started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stream = BytesStream(b'')
    response = httpx.Response(200, stream=stream)

    async def acquire():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            return response

    owner = asyncio.create_task(wait_operation(acquire(), CancelToken(), time.monotonic() + 5))
    await started.wait()
    owner.cancel()
    await cancelling.wait()
    owner.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert stream.closed
