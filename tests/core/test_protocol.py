import asyncio
import dataclasses

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
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
