"""验证服务的聚焦验证 (04 小步2: 固定命令回执、最新尝试、基线/最终证据、取消锁存).

用真实 Git/Windows/PowerShell fixture 与 bootstrap+FakeModel 覆盖公开行为:
断言 `VerificationService` 的回执/视图/证据, 不用 helper 替代真实命令执行。
"""

import asyncio
import hashlib
import subprocess
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.process import PowerShellRunner
from cicada.plugins.coding.snapshot import Snapshotter
from cicada.plugins.coding.verification import (
    CONTENT_MAX_BYTES,
    DIFF_MAX_BYTES,
    Verifier,
    VerificationError,
    generate_verification_run_id,
    verification_plugin,
)
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.coding.workspace import Workspace

# check 命令在 fixture 根 (= 进程 cwd) 执行, 因此解释器必须用绝对路径。
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


def build(repo, *checks, snapshot_limits=None, **kwargs):
    workspace = Workspace.create(repo)
    snapshotter = Snapshotter(workspace, GitInventory(workspace), **(snapshot_limits or {}))
    plan = VerificationPlan(
        verification_run_id=kwargs.pop("run_id", "vrun-test"),
        root=workspace.root,
        checks=tuple(checks),
    )
    verifier = Verifier(workspace, snapshotter, PowerShellRunner(), plan, **kwargs)
    return workspace, verifier


def passing(command=f"& '{PY}' -m pytest test_sample.py -q") -> CheckDefinition:
    return CheckDefinition("check-1", command)


async def initialized(repo, *checks, **kwargs):
    _, verifier = build(repo, *checks, **kwargs)
    capture = await verifier.initialize(CancelToken())
    assert capture.available, (capture.failure_kind, capture.error)
    return verifier


async def test_fixed_command_passes_and_receipt_carries_real_facts(repo):
    verifier = await initialized(repo, passing())
    receipt = await verifier.run_check("check-1", CancelToken())
    assert receipt.verification_status == "passed"
    assert receipt.failure_kind is None
    assert receipt.execution_status == "exited" and receipt.exit_code == 0
    assert receipt.command == passing().command  # 启动者原样命令
    assert receipt.cwd == verifier.plan.root
    assert receipt.output_complete is True and receipt.timed_out is False
    assert receipt.snapshot_before == receipt.snapshot_after == verifier.baseline_snapshot_ref
    assert receipt.elapsed_s > 0 and receipt.receipt_id.startswith("chk-")
    assert receipt.verification_run_id == "vrun-test"
    assert receipt.output_artifact_path is not None
    assert receipt.output_artifact_sha256 == hashlib.sha256(
        receipt.output_artifact_path.read_bytes()).hexdigest()
    assert receipt.output_artifact_sha256 is not None and receipt.error is None
    view = await verifier.refresh(CancelToken())
    assert [state.status for state in view.checks] == ["passed"]
    assert [state.freshness for state in view.checks] == ["current"]
    assert view.snapshot_ref == verifier.baseline_snapshot_ref
    assert view.process_uncertain is False and view.blocking_reasons == ()


async def test_unknown_check_id_is_rejected_and_never_runs_model_text(repo):
    verifier = await initialized(repo, passing())
    with pytest.raises(VerificationError, match="unknown check_id"):
        await verifier.run_check("check-9", CancelToken())
    assert verifier.receipts == ()


async def test_nonzero_exit_is_failed_and_records_real_exit_code(repo):
    verifier = await initialized(repo, CheckDefinition("check-1", "exit 3"))
    receipt = await verifier.run_check("check-1", CancelToken())
    assert receipt.verification_status == "failed"
    assert receipt.failure_kind == "nonzero_exit"
    assert receipt.execution_status == "exited" and receipt.exit_code == 3
    assert receipt.output_complete is True
    assert verifier.process_uncertain is False  # 终结完整: 不是进程不确定
    view = await verifier.refresh(CancelToken())
    assert [r for r in view.blocking_reasons if "check-1" in r]


async def test_check_that_changes_its_input_is_blocked(repo):
    verifier = await initialized(
        repo, CheckDefinition("check-1", "Set-Content -NoNewline side-effect.txt changed"))
    receipt = await verifier.run_check("check-1", CancelToken())
    assert receipt.verification_status == "blocked"
    assert receipt.failure_kind == "snapshot_changed"
    assert receipt.exit_code == 0  # 命令本身正常退出, 但不能作为通过证据
    assert receipt.snapshot_before != receipt.snapshot_after
    # 进程正常终结: 不是 process_uncertain, 但依然 blocked。
    assert verifier.process_uncertain is False


async def test_timeout_latches_process_uncertain_and_a_later_pass_cannot_clear_it(repo):
    verifier = await initialized(
        repo,
        CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=1.0),
        CheckDefinition("check-2", f"& '{PY}' -m pytest test_sample.py -q"),
    )
    blocked = await verifier.run_check("check-1", CancelToken())
    assert blocked.verification_status == "blocked"
    assert blocked.failure_kind == "timed_out"
    assert blocked.timed_out is True and blocked.output_complete is False
    assert verifier.process_uncertain is True
    after = await verifier.run_check("check-2", CancelToken())
    assert after.verification_status == "passed"
    view = await verifier.refresh(CancelToken())
    assert view.process_uncertain is True
    assert view.checks[1].status == "passed"
    assert any("process_uncertain" in reason for reason in view.blocking_reasons)


async def test_cancelled_check_is_blocked_latches_and_propagates_cancel(repo):
    verifier = await initialized(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=30.0))
    token = CancelToken()
    task = asyncio.create_task(verifier.run_check("check-1", token))
    await asyncio.sleep(0.6)
    token.cancel()
    receipt = await asyncio.wait_for(task, timeout=15)
    assert receipt.verification_status == "blocked"
    assert receipt.failure_kind == "cancelled" and receipt.cancelled is True
    assert verifier.process_uncertain is True

    other = await initialized(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=30.0))
    pending = asyncio.create_task(other.run_check("check-1", CancelToken()))
    await asyncio.sleep(0.6)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=15)


async def test_latest_attempt_replaces_an_earlier_pass_in_state_but_history_remains(repo):
    verifier = await initialized(repo, passing())
    first = await verifier.run_check("check-1", CancelToken())
    assert first.verification_status == "passed"
    (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = await verifier.run_check("check-1", CancelToken())
    assert second.verification_status == "failed"
    view = await verifier.refresh(CancelToken())
    state = view.checks[0]
    assert state.status == "failed"
    assert state.receipt is not None and state.receipt.receipt_id == second.receipt_id
    assert len(view.receipts) == 2
    assert [r.receipt_id for r in view.receipts] == [first.receipt_id, second.receipt_id]
    assert state.freshness == "current"  # 最新尝试与当前快照一致


async def test_code_change_after_a_pass_makes_it_stale(repo):
    verifier = await initialized(repo, passing())
    await verifier.run_check("check-1", CancelToken())
    (repo / "sample.py").write_text("VALUE = 42\n", encoding="utf-8")
    view = await verifier.refresh(CancelToken())
    assert view.checks[0].status == "passed"
    assert view.checks[0].freshness == "stale"
    assert view.snapshot_ref != verifier.baseline_snapshot_ref
    assert any("stale" in reason for reason in view.blocking_reasons)


async def test_initialize_failure_has_no_baseline_and_leaves_checks_not_run(repo):
    (repo / "huge.bin").write_bytes(b"x" * 2048)
    _, verifier = build(repo, passing(), snapshot_limits={"max_file_bytes": 1024})
    capture = await verifier.initialize(CancelToken())
    assert not capture.available and capture.failure_kind == "file_limit"
    assert verifier.baseline_snapshot_ref is None
    view = await verifier.refresh(CancelToken())
    assert [state.status for state in view.checks] == ["not_run"]
    assert view.checks[0].freshness == "unknown"
    assert view.snapshot_ref is None
    assert any("unavailable" in reason for reason in view.blocking_reasons)
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is False and evidence.failure_kind == "no_baseline"


async def test_finalize_reports_added_modified_deleted_with_hashes(repo):
    verifier = await initialized(repo, passing())
    (repo / "sample.py").write_text("VALUE = 7\n", encoding="utf-8")
    (repo / "brand-new.txt").write_text("fresh\n", encoding="utf-8")
    (repo / "test_sample.py").unlink()
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    assert evidence.baseline_snapshot_ref == verifier.baseline_snapshot_ref
    assert evidence.snapshot_ref != evidence.baseline_snapshot_ref
    by_path = {change.relative_path: change for change in evidence.changes}
    assert set(by_path) == {"sample.py", "brand-new.txt", "test_sample.py"}
    assert by_path["sample.py"].kind == "modified"
    assert by_path["sample.py"].before_sha256 and by_path["sample.py"].after_sha256
    assert by_path["brand-new.txt"].kind == "added"
    assert by_path["brand-new.txt"].before_sha256 is None
    assert by_path["test_sample.py"].kind == "deleted"
    assert by_path["test_sample.py"].after_sha256 is None
    body = evidence.artifact_path.read_text(encoding="utf-8")
    assert "=== added brand-new.txt" in body
    assert "=== deleted test_sample.py" in body
    assert "-VALUE = 1" in body and "+VALUE = 7" in body
    assert evidence.artifact_sha256 == hashlib.sha256(
        evidence.artifact_path.read_bytes()).hexdigest()


async def test_corrupted_baseline_artifact_blocks_evidence(repo):
    verifier = await initialized(repo, passing())
    (repo / "sample.py").write_text("VALUE = 9\n", encoding="utf-8")
    baseline_path = verifier._baseline.path
    baseline_path.write_text('{"schema_version": 1, "snapshot_ref": "tampered"}\n',
                             encoding="utf-8")
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is False
    assert evidence.failure_kind == "baseline_corrupt"
    assert "no longer match" in evidence.error


async def test_binary_and_encoding_changes_are_reported_without_fake_text_diff(repo):
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    (repo / "windows.txt").write_bytes(b"line one\r\nline two\r\n")
    (repo / "bom.txt").write_bytes(b"plain\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "fixtures")
    verifier = await initialized(repo, passing())
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe binary")
    (repo / "windows.txt").write_bytes(b"line one\nline two\n")
    (repo / "bom.txt").write_bytes(b"\xef\xbb\xbfplain\n")
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    by_path = {change.relative_path: change for change in evidence.changes}
    assert by_path["blob.bin"].binary is True
    assert by_path["windows.txt"].newline_changed is True
    assert by_path["bom.txt"].bom_changed is True
    body = evidence.artifact_path.read_text(encoding="utf-8")
    assert "textual diff not produced (binary content)" in body
    assert "newline_changed" in body and "bom_changed" in body


async def test_dirty_start_is_the_baseline_and_is_not_reset(repo):
    # 启动时已有未提交改动: 基线用真实内容, 不自动重置用户工作; 之后的改动才是变更。
    # write_text 在 Windows 落 CRLF, 因此断言以磁盘真实 bytes 为准。
    (repo / "sample.py").write_text("VALUE = 99\n", encoding="utf-8")
    (repo / "staged-then-dirty.txt").write_text("dirty\n", encoding="utf-8")
    dirty_bytes = (repo / "sample.py").read_bytes()
    verifier = await initialized(repo, passing())
    assert verifier._baseline.files["sample.py"].data == dirty_bytes

    (repo / "sample.py").write_text("VALUE = 100\n", encoding="utf-8")
    final_bytes = (repo / "sample.py").read_bytes()
    assert final_bytes != dirty_bytes
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    assert [change.relative_path for change in evidence.changes] == ["sample.py"]
    change = evidence.changes[0]
    assert change.kind == "modified"
    assert change.before_sha256 == hashlib.sha256(dirty_bytes).hexdigest()
    assert change.after_sha256 == hashlib.sha256(final_bytes).hexdigest()
    assert change.before_sha256 != change.after_sha256
    # 启动时未跟踪但已存在的脏文件不因基线副本被改动或删除。
    assert (repo / "staged-then-dirty.txt").read_text(encoding="utf-8") == "dirty\n"


async def test_identical_state_produces_identical_snapshot_and_evidence_refs(repo):
    verifier = await initialized(repo, passing())
    (repo / "sample.py").write_text("VALUE = 3\n", encoding="utf-8")
    first = await verifier.finalize(CancelToken())
    second = await verifier.finalize(CancelToken())
    assert first.complete is True and second.complete is True
    assert first.snapshot_ref == second.snapshot_ref
    assert first.changes == second.changes
    assert first.baseline_snapshot_ref == second.baseline_snapshot_ref


async def test_no_changes_yields_complete_evidence_with_empty_change_list(repo):
    verifier = await initialized(repo, passing())
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True
    assert evidence.changes == ()
    assert evidence.snapshot_ref == evidence.baseline_snapshot_ref
    assert evidence.artifact_path is not None and evidence.artifact_sha256


async def test_snapshot_failure_at_finalize_yields_incomplete_evidence(repo):
    verifier = await initialized(repo, passing())
    # 运行中出现的超大文件让最终快照不可用: 必须明确失败, 不能给出半份证据。
    (repo / "huge.txt").write_text("a" * (16 * 1024 * 1024 + 1024), encoding="utf-8")
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is False
    assert evidence.failure_kind == "snapshot_unavailable"
    assert evidence.snapshot_ref is None and evidence.artifact_path is None


async def test_oversized_complete_diff_makes_evidence_incomplete(repo):
    verifier = await initialized(repo, passing())
    # 两个 8.5 MiB 新增文件各自合法, 但完整 unified diff 超过 16 MiB 工件上限:
    # 不能把有界预览说成完整 diff。
    chunk = "a" * (8 * 1024 * 1024 + 512 * 1024)
    for name in ("big-1.txt", "big-2.txt"):
        (repo / name).write_text(chunk, encoding="utf-8")
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is False
    assert evidence.failure_kind == "evidence_incomplete"
    assert "exceed" in evidence.error


async def test_mark_process_uncertain_is_monotonic_and_blocks_delivery(repo):
    verifier = await initialized(repo, passing())
    verifier.mark_process_uncertain("powershell result has no structured details")
    verifier.mark_process_uncertain("another reason")
    await verifier.run_check("check-1", CancelToken())
    view = await verifier.refresh(CancelToken())
    assert view.process_uncertain is True
    assert view.blocking_reasons[0].startswith("process_uncertain:")
    assert "another reason" in view.blocking_reasons[0]
    assert view.checks[0].status == "passed"


async def test_receipt_content_is_bounded_and_keeps_control_and_artifact_facts(repo):
    noisy = (
        f"& '{PY}' -c \"print('x' * 60000); print('done')\""
    )
    verifier = await initialized(repo, CheckDefinition("check-1", noisy))
    receipt = await verifier.run_check("check-1", CancelToken())
    content = verifier.render_receipt_content(receipt)
    assert len(content.encode("utf-8")) <= CONTENT_MAX_BYTES
    assert f"receipt_id={receipt.receipt_id}" in content
    assert "status=passed" in content
    assert f"command_sha256={hashlib.sha256(receipt.command.encode()).hexdigest()}" in content
    assert receipt.output_artifact_path is not None
    assert str(receipt.output_artifact_path) in content
    assert receipt.output_artifact_sha256 in content
    assert receipt.output_truncated is True  # 60 KiB 单行超 tail 预算


async def test_long_command_keeps_full_text_in_receipt_and_bounded_preview(repo):
    long_command = "Write-Output ok # " + "z" * 3000
    verifier = await initialized(repo, CheckDefinition("check-1", long_command))
    receipt = await verifier.run_check("check-1", CancelToken())
    assert receipt.command == long_command  # 完整 command 保留在回执
    content = verifier.render_receipt_content(receipt)
    assert len(content.encode("utf-8")) <= CONTENT_MAX_BYTES
    preview = content.split("command_preview=")[1].split("]")[0]
    assert len(preview.encode("utf-8")) <= 1024
    assert "z" * 100 in preview
    details = verifier.receipt_details(receipt, "current")
    assert details["command"] == long_command
    assert details["command_sha256"] == hashlib.sha256(long_command.encode()).hexdigest()


async def test_generated_run_ids_are_unique_and_not_the_kernel_run_id():
    first = generate_verification_run_id()
    second = generate_verification_run_id()
    assert first != second and first.startswith("vrun-") and second.startswith("vrun-")


async def test_verification_plugin_runs_a_real_check_through_bootstrap(repo):
    from cicada.boot import bootstrap
    from cicada.core.ports import StreamDone
    from cicada.plugins.coding.inventory import inventory_plugin
    from cicada.plugins.coding.process import process_plugin
    from cicada.plugins.coding.tool_check import check_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    workspace = Workspace.create(repo)
    plan = VerificationPlan("vrun-boot", workspace.root, (passing(),))
    app = await bootstrap(
        [
            workspace_plugin(repo),
            inventory_plugin(),
            process_plugin(),
            verification_plugin(plan),
            check_plugin(),
            fake_model_plugin(FakeModel([[StreamDone("stop")]])),
        ],
        tool_capabilities=("tool.check",),
    )
    service = app.runtime.capability("coding.verification")
    capture = await service.initialize(CancelToken())
    assert capture.available
    receipt = await service.run_check("check-1", CancelToken())
    assert receipt.verification_status == "passed"
    view = await service.refresh(CancelToken())
    assert [state.status for state in view.checks] == ["passed"]
    assert "check" in {tool.spec.name for tool in (app.agent._tools).values()}
    assert (await app.agent.run("run the required check")).stop_reason == "stop"
    await app.aclose()
