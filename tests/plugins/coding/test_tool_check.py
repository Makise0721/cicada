"""check 工具的聚焦验证 (04 小步2: 固定命令执行、结构化回执身份、有界 content).

经内核真实分发路径 (arguments JSON -> jsonschema -> Tool.execute) 与 bootstrap+FakeModel
验证; `details.receipt_id` 是 K/应用层关联历史回执的键, 模型 call_id 不作键。
"""

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import ToolResultMessage
from cicada.core.ports import StreamDone, ToolCallEvent, ToolContext
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.process import PowerShellRunner
from cicada.plugins.coding.snapshot import Snapshotter
from cicada.plugins.coding.tool_check import CheckTool, check_plugin
from cicada.plugins.coding.verification import (
    CONTENT_MAX_BYTES,
    Verifier,
    verification_plugin,
)
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.coding.workspace import Workspace

PY = Path(".venv/Scripts/python.exe").resolve().as_posix()


def git(root, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "import sample\n\n\ndef test_value():\n    assert sample.VALUE == 1\n",
        encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "baseline")
    return root


async def service_for(repo, *checks):
    workspace = Workspace.create(repo)
    snapshotter = Snapshotter(workspace, GitInventory(workspace))
    plan = VerificationPlan("vrun-tool", workspace.root, tuple(checks))
    verifier = Verifier(workspace, snapshotter, PowerShellRunner(), plan)
    capture = await verifier.initialize(CancelToken())
    assert capture.available, (capture.failure_kind, capture.error)
    return CheckTool(verifier)


def passing(check_id="check-1") -> CheckDefinition:
    return CheckDefinition(check_id, f"& '{PY}' -m pytest test_sample.py -q")


async def call(tool, arguments, cancel=None):
    return await tool.execute(arguments, ToolContext(call_id="call-1", cancel=cancel or CancelToken()))


async def test_run_returns_structured_receipt_identity_and_bounded_content(repo):
    tool = await service_for(repo, passing())
    result = await call(tool, {"action": "run", "check_id": "check-1"})
    assert result.name == "check" and result.is_error is False
    assert len(result.content.encode("utf-8")) <= CONTENT_MAX_BYTES
    details = result.details
    assert details["receipt_id"].startswith("chk-")
    assert details["check_id"] == "check-1" and details["verification_status"] == "passed"
    assert details["freshness"] == "current"
    assert details["snapshot_ref"] == details["snapshot_after"] == details["snapshot_before"]
    assert details["baseline_snapshot_ref"] == details["snapshot_ref"]
    assert details["command"] == passing().command
    assert details["command_sha256"] == hashlib.sha256(passing().command.encode()).hexdigest()
    assert details["process_uncertain"] is False and details["blocking_reasons"] == []
    assert details["output_artifact_path"] and details["output_artifact_sha256"]
    assert f"receipt_id={details['receipt_id']}" in result.content
    assert "freshness" in tool.spec.description


async def test_status_reports_not_run_then_current_with_history(repo):
    tool = await service_for(repo, passing())
    fresh = await call(tool, {"action": "status"})
    assert fresh.is_error is False
    assert fresh.details["checks"] == [{
        "check_id": "check-1", "status": "not_run", "freshness": "unknown",
        "receipt_id": None, "failure_kind": None}]
    assert fresh.details["receipt_ids"] == []
    assert "not_run" in fresh.content

    first = await call(tool, {"action": "run", "check_id": "check-1"})
    after = await call(tool, {"action": "status"})
    assert after.details["checks"][0]["status"] == "passed"
    assert after.details["checks"][0]["freshness"] == "current"
    assert after.details["receipt_ids"] == [first.details["receipt_id"]]
    assert after.details["scope_id"] and after.details["snapshot_ref"]
    assert len(after.content.encode("utf-8")) <= CONTENT_MAX_BYTES


async def test_status_goes_stale_after_the_code_changes(repo):
    tool = await service_for(repo, passing())
    receipt = await call(tool, {"action": "run", "check_id": "check-1"})
    (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    status = await call(tool, {"action": "status"})
    assert status.details["checks"][0]["status"] == "passed"
    assert status.details["checks"][0]["freshness"] == "stale"
    assert status.details["checks"][0]["receipt_id"] == receipt.details["receipt_id"]
    assert status.details["snapshot_ref"] != receipt.details["snapshot_ref"]
    assert any("stale" in reason for reason in status.details["blocking_reasons"])


async def test_latest_attempt_replaces_state_and_keeps_history_in_receipt_ids(repo):
    tool = await service_for(repo, passing())
    first = await call(tool, {"action": "run", "check_id": "check-1"})
    (repo / "sample.py").write_text("VALUE = 5\n", encoding="utf-8")
    second = await call(tool, {"action": "run", "check_id": "check-1"})
    assert first.details["verification_status"] == "passed"
    assert second.details["verification_status"] == "failed"
    status = await call(tool, {"action": "status"})
    assert status.details["checks"][0]["status"] == "failed"
    assert status.details["receipt_ids"] == [
        first.details["receipt_id"], second.details["receipt_id"]]
    assert "receipt history" in status.content


async def test_unknown_check_id_is_a_tool_error_without_a_receipt(repo):
    tool = await service_for(repo, passing())
    result = await call(tool, {"action": "run", "check_id": "check-9"})
    assert result.is_error is True and "unknown check_id" in result.content
    assert result.details is None
    status = await call(tool, {"action": "status"})
    assert status.details["receipt_ids"] == []


async def test_model_cannot_run_its_own_command_through_the_check_tool(repo):
    tool = await service_for(repo, passing())
    for arguments in (
        {"action": "run", "check_id": "check-1", "command": f"& '{PY}' -c \"print('owned')\""},
        {"action": "run", "check_id": "check-1", "timeout": 1},
        {"action": "run", "check_id": "check-1", "cwd": "C:/"},
        {"action": "status", "check_id": "check-1"},
    ):
        import jsonschema

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(arguments, tool.spec.parameters)
    status = await call(tool, {"action": "status"})
    assert status.details["receipt_ids"] == []


async def test_blocked_receipt_still_carries_receipt_id_and_is_not_a_tool_error(repo):
    tool = await service_for(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=1.0))
    result = await call(tool, {"action": "run", "check_id": "check-1"})
    assert result.is_error is False
    assert result.details["verification_status"] == "blocked"
    assert result.details["failure_kind"] == "timed_out"
    assert result.details["timed_out"] is True
    assert result.details["process_uncertain"] is True
    status = await call(tool, {"action": "status"})
    assert status.details["process_uncertain"] is True
    assert "process_uncertain" in status.content


async def test_cancellation_is_recorded_as_a_blocked_receipt_not_a_failure(repo):
    tool = await service_for(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=30.0))
    token = CancelToken()
    pending = asyncio.create_task(call(tool, {"action": "run", "check_id": "check-1"}, token))
    await asyncio.sleep(0.8)
    token.cancel()
    result = await asyncio.wait_for(pending, timeout=15)
    assert result.is_error is False
    assert result.details["verification_status"] == "blocked"
    assert result.details["failure_kind"] == "cancelled"
    assert result.details["cancelled"] is True and result.details["exit_code"] is None


async def test_receipt_id_survives_the_kernel_tool_result_message(repo):
    """K/应用层按程序 receipt_id 关联: 它必须出现在原始 ToolResult.details 里."""
    tool = await service_for(repo, passing())
    result = await call(tool, {"action": "run", "check_id": "check-1"})
    message = ToolResultMessage(result=result)
    assert isinstance(message.result.details, dict)
    assert message.result.details["receipt_id"] == result.details["receipt_id"]
    assert message.result.details["verification_status"] == "passed"


async def test_k_projection_marks_the_receipt_current_and_stale_by_program_ledger(repo):
    """跨层: K 的 RunPolicy 用 receipt_id 在程序账本里查 freshness, 不解析输出文字."""
    from cicada.core.ports import ModelRequest
    from cicada.plugins.fake_model import FakeModel
    from cicada.run_policy import RunPolicy

    workspace = Workspace.create(repo)
    snapshotter = Snapshotter(workspace, GitInventory(workspace))
    plan = VerificationPlan("vrun-k", workspace.root, (passing(),))
    verifier = Verifier(workspace, snapshotter, PowerShellRunner(), plan)
    assert (await verifier.initialize(CancelToken())).available
    tool = CheckTool(verifier)
    result = await call(tool, {"action": "run", "check_id": "check-1"})

    inner = FakeModel([[StreamDone("stop")], [StreamDone("stop")]])
    policy = RunPolicy(inner, verifier)
    request = ModelRequest(messages=(ToolResultMessage(result=result),), tools=())
    for _ in range(2):
        async for _event in policy.stream(request, CancelToken()):
            pass
    assert "freshness=current" in inner.requests[0].messages[0].result.content
    assert "stale=false" in inner.requests[0].messages[0].result.content
    assert "<cicada-verification-state>" in inner.requests[0].system_prompt
    assert "check check-1: status=passed freshness=current" in inner.requests[0].system_prompt

    (repo / "sample.py").write_text("VALUE = 11\n", encoding="utf-8")
    async for _event in policy.stream(request, CancelToken()):
        pass
    stale_projection = inner.requests[2].messages[0].result.content
    assert "freshness=stale" in stale_projection and "stale=true" in stale_projection
    assert "freshness=stale" in inner.requests[2].system_prompt


async def test_check_tool_runs_through_bootstrap_agent_loop_without_model_text_evidence(repo):
    from cicada.boot import bootstrap
    from cicada.plugins.coding.inventory import inventory_plugin
    from cicada.plugins.coding.process import process_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    workspace = Workspace.create(repo)
    plan = VerificationPlan("vrun-loop", workspace.root, (passing(),))
    model = FakeModel([
        [ToolCallEvent("m1", "check", json.dumps({"action": "run", "check_id": "check-1"})),
         StreamDone("tool_use")],
        [StreamDone("stop")],
    ])
    app = await bootstrap(
        [
            workspace_plugin(repo),
            inventory_plugin(),
            process_plugin(),
            verification_plugin(plan),
            check_plugin(),
            fake_model_plugin(model),
        ],
        tool_capabilities=("tool.check",),
    )
    service = app.runtime.capability("coding.verification")
    assert (await service.initialize(CancelToken())).available
    result = await app.agent.run("run the required check")
    assert result.stop_reason == "stop"
    messages = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert len(messages) == 1
    outcome = messages[0].result
    assert outcome.is_error is False
    assert outcome.details["check_id"] == "check-1"
    assert outcome.details["verification_status"] == "passed"
    assert outcome.details["receipt_id"] in {
        receipt.receipt_id for receipt in service.receipts}
    view = await service.refresh(CancelToken())
    assert view.process_uncertain is False
    assert [state.status for state in view.checks] == ["passed"]
    await app.aclose()


async def test_tool_exception_is_normalised_by_the_kernel_so_policy_can_latch(repo):
    """保守路径: 工具异常时内核给 details=None; K 必须据此锁存 process_uncertain."""
    from cicada.core.agent import Agent
    from cicada.core.ports import ModelRequest
    from cicada.plugins.coding.verification_contracts import (
        CheckState,
        VerificationView,
    )
    from cicada.plugins.fake_model import FakeModel
    from cicada.run_policy import RunPolicy

    real_schema = CheckTool(service=None).spec

    class Exploding:
        spec = real_schema

        async def execute(self, arguments, ctx):
            raise RuntimeError("simulated failure after a process was started")

    agent = Agent(
        model=FakeModel([[
            ToolCallEvent("m1", "check", json.dumps(
                {"action": "run", "check_id": "check-1"})),
            StreamDone("tool_use")],
            [StreamDone("stop")]]),
        tools={"check": Exploding()},
    )
    result = await agent.run("trigger the failure")
    outcome = [m for m in result.messages if isinstance(m, ToolResultMessage)][0].result
    assert outcome.is_error is True and outcome.details is None
    assert "tool raised" in outcome.content

    class FakeService:
        def __init__(self):
            self.reasons = []

        def mark_process_uncertain(self, reason):
            self.reasons.append(reason)

        async def refresh(self, cancel):
            return VerificationView(
                "vrun-x", None, None, (CheckState("check-1"),), (), True,
                tuple(self.reasons))

    service = FakeService()
    policy = RunPolicy(FakeModel([[StreamDone("stop")]]), service)
    async for _ in policy.stream(
        ModelRequest(messages=result.messages, tools=()), CancelToken()
    ):
        pass
    assert service.reasons and "no structured details" in service.reasons[0]
