"""审批档最小回归: broker 决定语义、取消/关闭收敛、包装器未批准不执行."""

from __future__ import annotations

import asyncio

import pytest

from cicada.approval import DENIED_CONTENT, ApprovalBroker, ApprovalTool
from cicada.core.cancel import CancelToken
from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec


class RecordingTool:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.spec = ToolSpec(name="powershell", description="pwsh", parameters={
            "type": "object", "properties": {}, "additionalProperties": False,
        })

    async def execute(self, arguments, ctx):
        self.calls.append(arguments)
        return ToolResult(call_id=ctx.call_id, name="powershell", content="ok",
                          details={"exit_code": 0, "output_complete": True})


async def _start(broker: ApprovalBroker, seen: list):
    broker.enable(seen.append)
    return asyncio.create_task(broker.request("powershell", "echo hi", CancelToken()))


async def test_without_ui_denies_without_waiting():
    broker = ApprovalBroker()
    assert await broker.request("powershell", "echo hi", CancelToken()) == "deny"


async def test_allow_once_resolves_pending():
    seen: list = []
    broker = ApprovalBroker()
    task = await _start(broker, seen)
    await asyncio.sleep(0)
    assert seen and broker.resolve(seen[0].request_id, "allow_once")
    assert await task == "allow_once"


async def test_allow_session_skips_next_request():
    seen: list = []
    broker = ApprovalBroker()
    task = await _start(broker, seen)
    await asyncio.sleep(0)
    broker.resolve(seen[0].request_id, "allow_session")
    assert await task == "allow_session"
    assert await broker.request("powershell", "echo again", CancelToken()) == "allow_session"


async def test_wrong_request_id_or_bad_decision_changes_nothing():
    seen: list = []
    broker = ApprovalBroker()
    task = await _start(broker, seen)
    await asyncio.sleep(0)
    assert not broker.resolve("other", "allow_once")
    assert not broker.resolve(seen[0].request_id, "maybe")
    assert broker.resolve(seen[0].request_id, "deny")
    assert await task == "deny"


async def test_cancel_token_denies():
    seen: list = []
    broker = ApprovalBroker()
    broker.enable(seen.append)
    token = CancelToken()
    task = asyncio.create_task(broker.request("powershell", "echo hi", token))
    await asyncio.sleep(0)
    token.cancel()
    assert await task == "deny"


async def test_close_denies_pending():
    seen: list = []
    broker = ApprovalBroker()
    task = await _start(broker, seen)
    await asyncio.sleep(0)
    broker.close()
    assert await task == "deny"
    assert await broker.request("powershell", "echo hi", CancelToken()) == "deny"


async def test_tool_wrapper_denies_before_running_inner():
    inner = RecordingTool()
    tool = ApprovalTool(inner, ApprovalBroker())
    ctx = ToolContext(call_id="c1", cancel=CancelToken())
    result = await tool.execute({"command": "echo hi"}, ctx)
    assert result.is_error and result.content == DENIED_CONTENT
    assert inner.calls == []
    assert tool.spec.name == "powershell"
