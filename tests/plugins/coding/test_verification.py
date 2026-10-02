"""验证服务的聚焦验证 (04 小步2: 固定命令回执、最新尝试、基线/最终证据、取消锁存).

用真实 Git/Windows/PowerShell fixture 与 bootstrap+FakeModel 覆盖公开行为:
断言 `VerificationService` 的回执/视图/证据, 不用 helper 替代真实命令执行。
"""

import asyncio
import hashlib
import json
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


def build(repo, *checks, snapshot_limits=None, runner=None, **kwargs):
    workspace = Workspace.create(repo)
    snapshotter = Snapshotter(workspace, GitInventory(workspace), **(snapshot_limits or {}))
    plan = VerificationPlan(
        verification_run_id=kwargs.pop("run_id", "vrun-test"),
        root=workspace.root,
        checks=tuple(checks),
    )
    verifier = Verifier(workspace, snapshotter, runner or PowerShellRunner(), plan, **kwargs)
    return workspace, verifier


class SignallingRunner:
    """真实 runner 的入口同步: 用例等 runner 已进入再取消, 不依赖固定 sleep 假设."""

    def __init__(self, inner):
        self.inner = inner
        self.entered = asyncio.Event()

    async def run(self, **kwargs):
        self.entered.set()
        return await self.inner.run(**kwargs)


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
    runner = SignallingRunner(PowerShellRunner())
    verifier = await initialized(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=30.0),
        runner=runner)
    token = CancelToken()
    task = asyncio.create_task(verifier.run_check("check-1", token))
    # 等 runner 已进入再取消: 不再假设固定 sleep 后真实进程一定已经在跑。
    await asyncio.wait_for(runner.entered.wait(), timeout=15)
    token.cancel()
    receipt = await asyncio.wait_for(task, timeout=15)
    assert receipt.verification_status == "blocked"
    assert receipt.failure_kind == "cancelled" and receipt.cancelled is True
    assert verifier.process_uncertain is True
    view = await verifier.refresh(CancelToken())
    assert view.checks[0].status == "blocked"
    assert any("process_uncertain" in reason for reason in view.blocking_reasons)


async def test_task_cancel_after_the_runner_started_latches_and_keeps_the_attempt(repo):
    runner = SignallingRunner(PowerShellRunner())
    verifier = await initialized(
        repo, CheckDefinition("check-1", "Start-Sleep -Seconds 20", timeout_s=30.0),
        runner=runner)
    pending = asyncio.create_task(verifier.run_check("check-1", CancelToken()))
    await asyncio.wait_for(runner.entered.wait(), timeout=15)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=15)
    # 服务先锁存并保留最近一次取消尝试, 然后才传播原取消。
    assert verifier.process_uncertain is True
    assert len(verifier.receipts) == 1
    latest = verifier.receipts[-1]
    assert latest.verification_status == "blocked"
    assert latest.failure_kind == "cancelled" and latest.cancelled is True
    assert latest.execution_status == "cancelled"
    assert latest.exit_code is None and latest.snapshot_after is None
    assert latest.snapshot_before == verifier.baseline_snapshot_ref
    view = await verifier.refresh(CancelToken())
    assert view.checks[0].status == "blocked"
    assert view.checks[0].receipt.receipt_id == latest.receipt_id


async def test_cancel_before_the_runner_starts_is_not_a_recorded_attempt(repo):
    class GatedSnapshotter(Snapshotter):
        """定位 seam: initialize 的快照正常, run_check 的前置快照停住等取消。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captures = 0
            self.entered = asyncio.Event()

        async def capture(self, cancel):
            self.captures += 1
            if self.captures > 1:
                self.entered.set()
                await asyncio.Event().wait()
            return await super().capture(cancel)

    runner = SignallingRunner(PowerShellRunner())
    workspace = Workspace.create(repo)
    snapshotter = GatedSnapshotter(workspace, GitInventory(workspace))
    plan = VerificationPlan("vrun-pre-cancel", workspace.root, (passing(),))
    verifier = Verifier(workspace, snapshotter, runner, plan)
    assert (await verifier.initialize(CancelToken())).available

    pending = asyncio.create_task(verifier.run_check("check-1", CancelToken()))
    await asyncio.wait_for(snapshotter.entered.wait(), timeout=15)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=15)
    # 执行前取消: 进程从未进入, 不能记成一次尝试, 也不锁存不确定状态。
    assert verifier.receipts == ()
    assert verifier.process_uncertain is False
    assert not runner.entered.is_set()


async def test_runner_exception_after_start_replaces_a_pass_and_latches(repo):
    class FailingAfterFirstRunner:
        def __init__(self):
            self.calls = 0
            self.inner = PowerShellRunner()

        async def run(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return await self.inner.run(**kwargs)
            raise RuntimeError("simulated runner crash")

    verifier = await initialized(repo, passing(), runner=FailingAfterFirstRunner())
    first = await verifier.run_check("check-1", CancelToken())
    assert first.verification_status == "passed"

    with pytest.raises(RuntimeError, match="simulated runner crash"):
        await verifier.run_check("check-1", CancelToken())
    assert verifier.process_uncertain is True
    latest = verifier.receipts[-1]
    assert latest.verification_status == "blocked"
    assert latest.failure_kind == "runner_error" and latest.execution_status == "error"
    assert latest.output_complete is False and latest.snapshot_after is None
    view = await verifier.refresh(CancelToken())
    # 旧 pass 不能留作最新候选。
    assert view.checks[0].status == "blocked"
    assert view.checks[0].receipt.receipt_id == latest.receipt_id
    assert any("process_uncertain" in reason for reason in view.blocking_reasons)
    assert [receipt.receipt_id for receipt in view.receipts] == [
        first.receipt_id, latest.receipt_id]


async def test_known_launch_failure_returns_a_blocked_receipt_without_latching(repo, monkeypatch):
    """真实缺失 exe: 进程创建边界确认未启动, 记 blocked/launch_failed 且不锁存."""
    from cicada.plugins.coding import process as process_module

    verifier = await initialized(repo, CheckDefinition("check-1", "exit 0"))
    missing = repo / "missing-pwsh.exe"  # 项目内不存在: 真实缺失 exe
    assert not missing.exists()
    monkeypatch.setattr(process_module, "resolve_pwsh", lambda: str(missing))

    receipt = await verifier.run_check("check-1", CancelToken())
    assert receipt.verification_status == "blocked"
    assert receipt.execution_status == "launch_failed"
    assert receipt.failure_kind == "launch_failed"
    assert receipt.exit_code is None
    assert receipt.timed_out is False and receipt.cancelled is False
    assert receipt.output_complete is False
    assert receipt.output_artifact_path is None
    assert receipt.receipt_id.startswith("chk-")
    assert "could not be started" in receipt.error
    # 已知未启动不是"进程终结未知": 不新锁存, 但仍是阻断交付的最近尝试。
    assert verifier.process_uncertain is False
    assert len(verifier.receipts) == 1
    view = await verifier.refresh(CancelToken())
    state = view.checks[0]
    assert state.status == "blocked"
    assert state.receipt.receipt_id == receipt.receipt_id
    assert state.freshness == "current"
    assert view.process_uncertain is False
    assert any("launch_failed" in reason for reason in view.blocking_reasons)
    assert not any("process_uncertain" in reason for reason in view.blocking_reasons)


async def test_launch_failure_replaces_a_pass_and_a_real_recheck_recovers_current(repo, monkeypatch):
    from cicada.plugins.coding import process as process_module

    real_resolve = process_module.resolve_pwsh
    verifier = await initialized(repo, passing())
    first = await verifier.run_check("check-1", CancelToken())
    assert first.verification_status == "passed"

    monkeypatch.setattr(process_module, "resolve_pwsh", lambda: str(repo / "missing-pwsh.exe"))
    failed = await verifier.run_check("check-1", CancelToken())
    assert failed.failure_kind == "launch_failed"
    view = await verifier.refresh(CancelToken())
    # 旧 pass 不能留作最新候选; 启动失败仍阻断交付, 但没有进程不确定。
    assert view.checks[0].status == "blocked"
    assert view.checks[0].receipt.receipt_id == failed.receipt_id
    assert view.process_uncertain is False
    assert any("launch_failed" in reason for reason in view.blocking_reasons)

    # 恢复合法 runner 后正常重检必须恢复 current, 不被上一轮启动失败永久卡住。
    monkeypatch.setattr(process_module, "resolve_pwsh", real_resolve)
    recovered = await verifier.run_check("check-1", CancelToken())
    assert recovered.verification_status == "passed"
    final = await verifier.refresh(CancelToken())
    assert final.checks[0].freshness == "current"
    assert final.process_uncertain is False
    assert not [reason for reason in final.blocking_reasons if "check-1" in reason]
    assert [receipt.receipt_id for receipt in final.receipts] == [
        first.receipt_id, failed.receipt_id, recovered.receipt_id]


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


async def test_new_path_makes_old_pass_stale_and_rechecking_current_range_recovers_it(repo):
    verifier = await initialized(repo, passing())
    first = await verifier.run_check("check-1", CancelToken())
    assert first.verification_status == "passed"
    baseline_ref = verifier.baseline_snapshot_ref
    baseline_bytes = verifier._baseline.files["sample.py"].data

    # 新增 nonignored 文件: 累计范围扩大, 旧 pass 对当前代码不再成立。
    (repo / "brand-new.txt").write_text("new\n", encoding="utf-8")
    stale = await verifier.refresh(CancelToken())
    assert stale.checks[0].receipt.receipt_id == first.receipt_id
    assert stale.checks[0].freshness == "stale"
    assert stale.scope_id != verifier._baseline.scope_id  # 范围身份已扩大
    assert stale.snapshot_ref != baseline_ref

    # 对扩大后的**同一当前范围**重新检查通过: 必须恢复 current, 不用初始范围硬比较。
    second = await verifier.run_check("check-1", CancelToken())
    assert second.verification_status == "passed"
    current = await verifier.refresh(CancelToken())
    assert current.checks[0].receipt.receipt_id == second.receipt_id
    assert current.checks[0].freshness == "current"
    assert not [reason for reason in current.blocking_reasons if "check-1" in reason]
    assert current.snapshot_ref == second.snapshot_after

    # 基线仍然是启动时真实 bytes; 新路径在基线里 missing, 因此算 added 而不是改写起点。
    assert verifier._baseline.files["sample.py"].data == baseline_bytes
    assert "brand-new.txt" not in verifier._baseline.files
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    by_path = {change.relative_path: change for change in evidence.changes}
    assert by_path["brand-new.txt"].kind == "added"
    assert by_path["brand-new.txt"].before_sha256 is None
    assert by_path["brand-new.txt"].after_sha256


async def test_every_check_is_compared_against_the_same_final_snapshot(repo):
    verifier = await initialized(
        repo, CheckDefinition("check-1", "exit 0"), CheckDefinition("check-2", "exit 0"))
    await verifier.run_check("check-1", CancelToken())
    (repo / "added.txt").write_text("new\n", encoding="utf-8")
    await verifier.run_check("check-2", CancelToken())
    grown = await verifier.refresh(CancelToken())
    # 只有在新范围上跑过的回执才是 current; 旧回执必须 stale。
    assert [state.freshness for state in grown.checks] == ["stale", "current"]

    await verifier.run_check("check-1", CancelToken())
    view = await verifier.refresh(CancelToken())
    assert [state.freshness for state in view.checks] == ["current", "current"]
    assert {state.receipt.snapshot_after for state in view.checks} == {view.snapshot_ref}
    assert view.snapshot_ref not in (None, verifier.baseline_snapshot_ref)
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    assert evidence.snapshot_ref == view.snapshot_ref
    assert [change.relative_path for change in evidence.changes] == ["added.txt"]


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


async def test_tampered_change_artifact_keeps_the_generation_hash_and_blocks(repo):
    verifier = await initialized(repo, passing())
    (repo / "sample.py").write_text("VALUE = 9\n", encoding="utf-8")
    original = verifier._write_diff
    generated: dict[str, object] = {}

    def tampering(*args):
        path, digest = original(*args)
        generated["path"], generated["sha256"] = path, digest
        path.write_bytes(b"forged artifact, no actual diff\n")
        return path, digest

    verifier._write_diff = tampering
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is False
    assert evidence.failure_kind == "artifact_corrupt"
    # 生成时身份保留: 不把篡改后的 hash 重认领成完整证据。
    assert evidence.artifact_sha256 is None
    assert str(generated["sha256"]) in evidence.error
    assert evidence.artifact_path == generated["path"]
    assert [change.relative_path for change in evidence.changes] == ["sample.py"]

    # 故障只属于这一次生成: 下一轮重新生成并通过重验才算完整证据。
    verifier._write_diff = original
    again = await verifier.finalize(CancelToken())
    assert again.complete is True, again.error
    assert again.artifact_sha256 == hashlib.sha256(
        again.artifact_path.read_bytes()).hexdigest()


async def test_invalid_utf8_without_nul_is_binary_and_never_a_fake_text_diff(repo):
    verifier = await initialized(repo, passing())
    (repo / "sample.py").write_bytes(b"\xff\xfe hello\n")
    evidence = await verifier.finalize(CancelToken())
    assert evidence.complete is True, evidence.error
    change = next(c for c in evidence.changes if c.relative_path == "sample.py")
    assert change.kind == "modified" and change.binary is True
    assert change.before_sha256 and change.after_sha256 != change.before_sha256
    raw = evidence.artifact_path.read_bytes()
    # 工件本身是合法 UTF-8, 不含原非法字节, 也不含 surrogate 伪文本。
    assert b"\xff\xfe" not in raw
    body = raw.decode("utf-8")
    assert "textual diff not produced (binary content)" in body
    assert "hello" not in body


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
    # 真实 stdout 预览在预算内进入 content (尾部 'done' 可见), 控制字段没有被挤掉。
    assert "[output preview]" in content and "done" in content
    assert "metadata_truncated" not in content


async def test_receipt_content_shows_real_output_preview_freshness_and_current_scope(repo):
    # 诊断文字只出现在真实 stdout (char 码拼出 DIAG), 命令本身不含它。
    command = "Write-Output ([string][char]68+[char]73+[char]65+[char]71); exit 3"
    verifier = await initialized(repo, CheckDefinition("check-1", command))
    receipt = await verifier.run_check("check-1", CancelToken())
    view = await verifier.refresh(CancelToken())
    state = view.checks[0]
    content = verifier.render_receipt_content(receipt, state.freshness, view)
    assert len(content.encode("utf-8")) <= CONTENT_MAX_BYTES
    assert "DIAG" not in command and "DIAG" in content  # 只能来自真实输出预览
    assert "[output preview]" in content
    # freshness 与当前范围身份同样在模型可见 content 里。
    assert f"freshness={state.freshness}" in content
    assert f"scope_id={view.scope_id}" in content
    assert f"snapshot_ref={view.snapshot_ref}" in content
    assert "process_uncertain=false" in content
    # 非零诊断: 失败原因与退出码可见, 模型不必先读工件再定位。
    assert "failure_kind=nonzero_exit" in content and "exit_code=3" in content
    # 命令/路径是 JSON 转义的有界预览, 且完整命令仍在回执与 details。
    command_line = next(line for line in content.splitlines() if "command_preview=" in line)
    assert command_line.endswith("]")
    preview = json.loads(command_line.split("command_preview=", 1)[1][:-1])
    assert isinstance(preview, str) and command.startswith(preview.rstrip("."))


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

    # 转义膨胀 (大量引号) 也不能靠 JSON 编码把控制字段挤出去。
    quoted_command = 'Write-Output ok # ' + '"' * 3000
    other = await initialized(repo, CheckDefinition("check-1", quoted_command))
    quoted_receipt = await other.run_check("check-1", CancelToken())
    quoted_content = other.render_receipt_content(quoted_receipt, "current")
    assert len(quoted_content.encode("utf-8")) <= CONTENT_MAX_BYTES
    assert f"receipt_id={quoted_receipt.receipt_id}" in quoted_content
    assert "status=passed" in quoted_content and "freshness=current" in quoted_content
    quoted_preview = quoted_content.split("command_preview=")[1].split("]")[0]
    assert len(quoted_preview.encode("utf-8")) <= 1024
    assert json.loads(quoted_preview).startswith("Write-Output ok #")


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
