import asyncio
import dataclasses

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.events import ModelCallCompleted
from cicada.core.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from cicada.core.ports import ModelMetrics, ModelRequest, StreamDone
from cicada.core.session import SessionState


def test_messages_are_frozen_dataclasses():
    msg = AssistantMessage(text="hi", tool_calls=(ToolCall("c1", "echo", "{}"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.text = "changed"
    assert msg.role == "assistant"
    assert msg.tool_calls[0].id == "c1"


def test_tool_result_pairs_by_call_id():
    result = ToolResult(call_id="c1", name="echo", content="ok")
    msg = ToolResultMessage(result=result)
    assert msg.role == "tool"
    assert msg.result.call_id == "c1"
    assert msg.result.is_error is False


def test_session_snapshot_is_immutable_copy():
    session = SessionState()
    session.append(UserMessage(text="a"))
    snapshot = session.messages
    session.append(UserMessage(text="b"))
    assert len(snapshot) == 1
    assert isinstance(snapshot, tuple)


async def test_cancel_token():
    token = CancelToken()
    assert not token.cancelled
    waiter = asyncio.create_task(token.wait())
    await asyncio.sleep(0)
    assert not waiter.done()
    token.cancel()
    await waiter
    assert token.cancelled
    with pytest.raises(asyncio.CancelledError):
        token.throw_if_cancelled()


def test_model_metrics_defaults_to_unknown():
    metrics = ModelMetrics()
    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.provider_duration_s is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.input_tokens = 1


def test_model_request_old_construction_still_valid():
    request = ModelRequest(messages=(UserMessage(text="hi"),), tools=())
    assert request.system_prompt == ""
    explicit = ModelRequest(messages=(), tools=(), system_prompt="be terse")
    assert explicit.system_prompt == "be terse"


def test_stream_done_old_constructions_still_valid():
    assert StreamDone("stop").metrics is None
    failed = StreamDone("error", "boom")
    assert failed.error == "boom"
    assert failed.metrics is None


def test_stream_done_accepts_metrics():
    metrics = ModelMetrics(input_tokens=10, output_tokens=5, provider_duration_s=0.5)
    done = StreamDone("stop", metrics=metrics)
    assert done.metrics == metrics


def test_model_call_completed_is_frozen_dataclass():
    metrics = ModelMetrics(input_tokens=1)
    event = ModelCallCompleted(
        run_id="run-1", turn_index=2, stop_reason="stop", metrics=metrics, elapsed_s=0.25
    )
    assert event.run_id == "run-1"
    assert event.turn_index == 2
    assert event.metrics == metrics
    assert event.elapsed_s == 0.25
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.elapsed_s = 1.0
