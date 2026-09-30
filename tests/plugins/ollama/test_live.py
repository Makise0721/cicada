"""live 冒烟: 真实 Ollama + 默认模型, 纳入默认 pytest.

Gate: GET /api/version (2s) 可达且 /api/tags 含 CONFIG.model, 否则 skip.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelRequest, StreamDone, ToolCallEvent, TextDelta, ToolSpec
from cicada.plugins.ollama import OllamaConfig, OllamaModel

pytestmark = pytest.mark.live

CONFIG = OllamaConfig()
FIRST_EVENT_WAIT = 15.0
CANCEL_BUDGET = 30.0


async def _gate_reason() -> str | None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            version = await client.get(f"{CONFIG.base_url}/api/version")
            version.raise_for_status()
            tags = await client.get(f"{CONFIG.base_url}/api/tags")
            tags.raise_for_status()
    except httpx.HTTPError as exc:
        return f"ollama unreachable at {CONFIG.base_url}: {exc}"
    names = {entry.get("name") for entry in tags.json().get("models", [])}
    if CONFIG.model not in names:
        return f"model {CONFIG.model!r} not present in /api/tags"
    return None


@pytest.fixture(autouse=True)
async def _require_live_ollama():
    reason = await _gate_reason()
    if reason:
        pytest.skip(reason)


def _weather_spec() -> ToolSpec:
    return ToolSpec(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )


async def test_live_plain_text():
    async with httpx.AsyncClient() as client:
        model = OllamaModel(client, CONFIG)
        request = ModelRequest((UserMessage(text="Reply with the single word OK."),), ())
        gen = model.stream(request, CancelToken())
        events = [event async for event in gen]
        await gen.aclose()
    assert any(isinstance(event, TextDelta) for event in events)
    done = events[-1]
    assert isinstance(done, StreamDone) and done.stop_reason == "stop"
    # P3 M2 起 terminal 携带真实计量
    assert done.metrics is not None and done.metrics.input_tokens is not None


async def test_live_tool_call():
    request = ModelRequest(
        (UserMessage(text="What is the weather in Paris? Use the get_weather tool."),),
        (_weather_spec(),),
    )
    async with httpx.AsyncClient() as client:
        model = OllamaModel(client, CONFIG)
        gen = model.stream(request, CancelToken())
        events = [event async for event in gen]
        await gen.aclose()
    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    assert calls, events
    assert calls[0].id
    assert calls[0].name == "get_weather"
    json.loads(calls[0].arguments_json)  # arguments JSON 可解析
    done = events[-1]
    assert isinstance(done, StreamDone) and done.stop_reason == "tool_use"
    assert done.metrics is not None and done.metrics.output_tokens is not None


async def test_live_cancel_during_long_generation():
    config = replace(CONFIG, think=True)
    cancel = CancelToken()
    seen: list = []
    start = time.monotonic()

    async def consume() -> list:
        async with httpx.AsyncClient() as client:
            model = OllamaModel(client, config)
            request = ModelRequest(
                (UserMessage(text="Write a very long story about a cicada (at least 2000 words)."),),
                (),
            )
            gen = model.stream(request, cancel)
            async for event in gen:
                seen.append(event)
            await gen.aclose()
            return seen

    task = asyncio.create_task(consume())
    # 等流进入生成中 (think 开启, 首个 TextDelta 晚于思考); 超时也照常取消 (取消命中挂起的行读取)
    deadline = time.monotonic() + FIRST_EVENT_WAIT
    while not seen and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    cancel.cancel()
    events = await task
    elapsed = time.monotonic() - start
    assert events, "no events before cancellation"
    done = events[-1]
    assert isinstance(done, StreamDone) and done.stop_reason == "aborted"
    assert elapsed < CANCEL_BUDGET
