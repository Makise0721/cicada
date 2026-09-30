import asyncio

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, TextDelta, ToolContext
from cicada.plugins.fake_model import FakeModel
from cicada.plugins.fake_tools import EchoTool, SlowTool


async def collect(model, cancel):
    return [event async for event in model.stream(ModelRequest(messages=(), tools=()), cancel)]


async def test_fake_model_records_requests_and_replays_script():
    model = FakeModel([[TextDelta("a"), StreamDone("stop")]])
    cancel = CancelToken()
    assert await collect(model, cancel) == [TextDelta("a"), StreamDone("stop")]
    assert await collect(model, cancel) == [StreamDone("stop")]  # 剧本用尽回退 stop
    assert len(model.requests) == 2


async def test_fake_model_raises_scripted_exception():
    model = FakeModel([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        await collect(model, CancelToken())


async def test_fake_model_aborts_when_cancelled():
    model = FakeModel([[TextDelta("a"), TextDelta("b"), StreamDone("stop")]])
    cancel = CancelToken()
    cancel.cancel()
    assert await collect(model, cancel) == [StreamDone("aborted")]


async def test_echo_tool_records_invocations():
    echo = EchoTool()
    result = await echo.execute({"text": "hi"}, ToolContext(call_id="c1", cancel=CancelToken()))
    assert result.call_id == "c1"
    assert result.content == "hi"
    assert echo.invocations == [{"text": "hi"}]


async def test_slow_tool_timeout_and_cancel():
    slow = SlowTool()
    result = await slow.execute({"timeout": 0.01}, ToolContext(call_id="c1", cancel=CancelToken()))
    assert result.content == "completed without cancel"

    cancel = CancelToken()
    task = asyncio.create_task(slow.execute({}, ToolContext(call_id="c2", cancel=cancel)))
    await slow.started.wait()
    cancel.cancel()
    result2 = await task
    assert result2.is_error
    assert result2.call_id == "c2"
