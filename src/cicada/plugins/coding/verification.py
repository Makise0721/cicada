"""P4 §4–6 验证服务: 指定检查回执、基线工件、最终变更证据.

服务只执行启动者声明的固定命令 (模型只能选 check_id), 保存真实执行事实,
按程序生成的唯一 `receipt_id` 关联历史回执, 并用**已捕获且 hash 校验过的**
字节生成基线副本与最终 diff。

控制状态只在进程内: 工件是证据, 不能恢复权威; `mark_process_uncertain` 单调锁存,
后续 PASS 不能清除。最终交付判定属于应用层, 本服务不控制内核会话。
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.process import PowerShellLaunchError, PowerShellRunner
from cicada.plugins.coding.snapshot import SNAPSHOT_CAPABILITY, Snapshotter
from cicada.plugins.coding.verification_contracts import (
    VERIFICATION_CAPABILITY,
    CheckDefinition,
    CheckReceipt,
    CheckState,
    ChangeEvidence,
    FileChange,
    Freshness,
    Snapshot,
    SnapshotCapture,
    SnapshotEntry,
    VerificationPlan,
    VerificationStatus,
    VerificationView,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginContext, PluginDefinition

CONTENT_MAX_BYTES = 16 * 1024  # check 工具 content 的 UTF-8 总预算
CONTROL_MAX_BYTES = 4 * 1024  # 头部控制字段 (状态/范围/回执身份) 预算
COMMAND_PREVIEW_MAX_BYTES = 1024  # command_preview JSON 编码后 ≤1024 bytes
CWD_PREVIEW_MAX_BYTES = 1024  # cwd/path preview JSON 编码后 ≤1024 bytes
OUTPUT_PREVIEW_MAX_BYTES = 8 * 1024  # 模型可见 stdout/stderr 预览上限 (总预算 16 KiB)
DIFF_MAX_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 256 * 1024
_REASON_MAX_BYTES = 512


class VerificationError(RuntimeError):
    """服务用法的编程错误 (未知 check_id/非法计划); 可预期执行失败给回执, 不抛异常."""


def generate_verification_run_id() -> str:
    """程序生成的本次验证身份; 与内核 RunResult.run_id 分别记录, 不假设相同."""
    return f"vrun-{uuid.uuid4().hex}"


def _clip(text: str, limit: int) -> str:
    """按 UTF-8 字节截断; 只在字符边界切开, 不产生半个多字节字符."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    if limit <= 3:
        return data[:limit].decode("utf-8", "ignore")
    return data[: limit - 3].decode("utf-8", "ignore") + "..."


def _one_line(text: str, limit: int) -> str:
    return _clip(" ".join(text.split()), limit)


def _json_preview(text: str, limit: int) -> str:
    """JSON 编码后的有界预览 (含引号): 控制字段里的命令/路径都用它, 结果始终是合法 JSON.

    编码与转义膨胀都计入 limit, 所以长命令或大量引号不会挤掉 receipt/status/freshness。
    """
    budget = max(limit - 2, 1)  # 预留两个引号
    raw = _clip(_one_line(text, limit), budget)
    while raw and len(json.dumps(raw, ensure_ascii=False).encode("utf-8")) > limit:
        budget = max(budget // 2, 1)  # 极端转义膨胀: 减半保证收敛
        raw = _clip(_one_line(text, limit), budget)
    return json.dumps(raw, ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _BaselineFile:
    """基线副本: 启动时真实 bytes 及其 hash/长度."""

    relative_path: str
    sha256: str
    size_bytes: int
    data: bytes


@dataclass(frozen=True)
class _Baseline:
    snapshot_ref: str
    scope_id: str
    path: Path
    artifact_sha256: str
    files: dict[str, _BaselineFile]


class Verifier:
    """VerificationService 实现: 快照 + 固定命令 + 回执账本 + 证据工件."""

    def __init__(
        self,
        workspace: Workspace,
        snapshotter: Snapshotter,
        runner: PowerShellRunner,
        plan: VerificationPlan,
        *,
        content_max_bytes: int = CONTENT_MAX_BYTES,
        control_max_bytes: int = CONTROL_MAX_BYTES,
        artifact_factory: Callable[[str], Path] | None = None,
    ) -> None:
        if not isinstance(plan, VerificationPlan):
            raise TypeError("plan must be a VerificationPlan")
        for name, value, ceiling in (
            ("content_max_bytes", content_max_bytes, CONTENT_MAX_BYTES),
            ("control_max_bytes", control_max_bytes, CONTROL_MAX_BYTES),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if control_max_bytes >= content_max_bytes:
            raise ValueError("control_max_bytes must leave room for the output preview")
        self.workspace = workspace
        self.snapshotter = snapshotter
        self.runner = runner
        self._plan = plan
        self.content_max_bytes = content_max_bytes
        self.control_max_bytes = control_max_bytes
        self._artifact_factory = artifact_factory
        self._baseline: _Baseline | None = None
        self._baseline_capture: SnapshotCapture | None = None
        self._receipts: list[CheckReceipt] = []
        self._uncertain_reasons: list[str] = []
        # 模型可见的输出预览按程序 receipt_id 内部保存: 冻结 CheckReceipt 不新增字段,
        # 完整输出仍在工件里 (工件才是证据, 预览只用于模型判断)。
        self._previews: dict[str, str] = {}

    @property
    def plan(self) -> VerificationPlan:
        return self._plan

    @property
    def receipts(self) -> tuple[CheckReceipt, ...]:
        return tuple(self._receipts)

    @property
    def baseline_capture(self) -> SnapshotCapture | None:
        return self._baseline_capture

    @property
    def baseline_snapshot_ref(self) -> str | None:
        """已建立的基线快照身份; 仅在 initialize() 未建立可信基线时为 None."""
        return self._baseline.snapshot_ref if self._baseline is not None else None

    @property
    def _current_snapshot_ref(self) -> str | None:
        """最近一次已知的代码快照: 有基线工件时以基线为准, 否则用捕获结果.

        用于在 initialize() 的基线工件失败路径上仍能报出真实快照身份;
        它不代表该快照可用于交付判定 (那时 view 仍会给出 blocking reason)。
        """
        if self._baseline is not None:
            return self._baseline.snapshot_ref
        if self._baseline_capture is not None and self._baseline_capture.available:
            return self._baseline_capture.snapshot.snapshot_ref
        return None

    def baseline_failure(self) -> str | None:
        """本 run 是否缺少可信基线: 返回 blocking reason, 没有故障时为 None."""
        if self._baseline is not None:
            return None
        if self._baseline_capture is None:
            return "the verification run was not initialized; no baseline snapshot exists"
        if not self._baseline_capture.available:
            return (
                f"the baseline snapshot is unavailable ({self._baseline_capture.failure_kind}): "
                f"{self._baseline_capture.error}")
        return (
            "the baseline bytes could not be preserved for this run "
            f"({self._baseline_capture.failure_kind}): {self._baseline_capture.error}")

    @property
    def process_uncertain(self) -> bool:
        return bool(self._uncertain_reasons)

    # --- VerificationService ---

    async def initialize(self, cancel: CancelToken) -> SnapshotCapture:
        capture = await self.snapshotter.capture(cancel)
        self._baseline_capture = capture
        if not capture.available:
            return capture
        try:
            baseline = self._write_baseline(capture.snapshot, cancel)
        except (OSError, ValueError) as exc:
            self._baseline = None
            return SnapshotCapture(
                snapshot=None, failure_kind="baseline_artifact",
                error=f"cannot preserve the baseline bytes: {exc}")
        self._baseline = baseline
        return capture

    async def run_check(self, check_id: str, cancel: CancelToken) -> CheckReceipt:
        definition = next((c for c in self._plan.checks if c.check_id == check_id), None)
        if definition is None:
            raise VerificationError(f"unknown check_id {check_id!r}")
        uncertain_before = self.process_uncertain
        started = time.monotonic()
        before = await self.snapshotter.capture(cancel)
        if not before.available:
            return self._blocked(
                definition, started, None, None, "snapshot_unavailable",
                f"pre-run snapshot unavailable ({before.failure_kind}): {before.error}",
                uncertain_before)
        before_ref = before.snapshot.snapshot_ref
        result = None
        try:
            result = await self.runner.run(
                command=definition.command,
                cwd=self._plan.root,
                timeout=definition.timeout_s,
                cancel=cancel,
                output_dir=self.workspace.output_dir,
            )
            bounded = result.output
            artifact_path, artifact_sha256, artifact_error = self._artifact_facts(bounded)
            # 取消后不能再要求协作取消的快照: 用不可用结果记录"没有后快照"这一事实。
            after = None if cancel.cancelled else await self.snapshotter.capture(cancel)
        except PowerShellLaunchError as exc:
            # 进程创建边界已确认目标未启动: 记录 blocked/launch_failed 最近尝试并返回回执。
            # 没有已启动进程的未知终结, 因此不锁存 process_uncertain (已有锁存仍保持);
            # 命令未运行也不能当作 PASS, 仍阻断交付, 之后正常重检可替代它。
            return self._launch_failed(
                definition, started, before_ref, await self._post_snapshot_ref(cancel),
                exc, uncertain_before)
        except asyncio.CancelledError:
            # runner 已经进入: 先单调锁存并留下最近的取消尝试, 再传播原取消。
            self._record_interrupted(
                definition, started, before_ref, result, "cancelled",
                "the check was cancelled after the runner started; "
                "termination is not established")
            raise
        except Exception as exc:
            if result is None:
                kind = "runner_error"
                message = f"the check runner raised {type(exc).__name__}: {exc}"
            else:
                kind = "snapshot_unavailable"
                message = (
                    "the post-run snapshot failed after the check returned "
                    f"({type(exc).__name__}): {exc}")
            self._record_interrupted(definition, started, before_ref, result, kind, message)
            raise
        failure_kind, error = self._classify(result, before_ref, after, artifact_error)
        receipt = self._receipt(
            definition=definition,
            elapsed_s=time.monotonic() - started,
            execution_status=self._execution_status(result),
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            output_complete=result.output_complete,
            snapshot_before=before_ref,
            snapshot_after=(
                after.snapshot.snapshot_ref if after is not None and after.available else None),
            verification_status=(
                "passed" if failure_kind is None
                else "failed" if failure_kind == "nonzero_exit" else "blocked"),
            failure_kind=failure_kind,
            output_truncated=bounded.truncated,
            artifact_truncated=bounded.artifact_truncated,
            output_artifact_path=artifact_path,
            output_artifact_sha256=artifact_sha256,
            artifact_error=artifact_error,
            error=error,
        )
        self._record(receipt, uncertain_before)
        self._remember_output(receipt, bounded.text)
        return receipt

    async def refresh(self, cancel: CancelToken) -> VerificationView:
        """捕获当前快照并重新计算 freshness; 取不到时给 unknown view, 不抛到模型循环.

        取消之后无法再建立当前快照, 因此返回 snapshot 不可用的 view: 这是 unknown 事实,
        不是异常; 取消/超时的协作传播仍由 run_check 负责。
        """
        if cancel.cancelled:
            return self._view(
                SnapshotCapture(
                    snapshot=None, failure_kind="cancelled",
                    error="the run was cancelled; the current snapshot cannot be captured"))
        return self._view(await self.snapshotter.capture(cancel))

    def mark_process_uncertain(self, reason: str) -> None:
        self._latch(reason)

    async def finalize(self, cancel: CancelToken) -> ChangeEvidence:
        baseline = self._baseline
        if baseline is None:
            return ChangeEvidence(
                baseline_snapshot_ref=None, snapshot_ref=None, scope_id=None, changes=(),
                artifact_path=None, artifact_sha256=None, complete=False,
                failure_kind="no_baseline",
                error="initialize() did not establish a usable baseline snapshot",
            )
        final = await self.snapshotter.capture(cancel)
        if not final.available:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref, snapshot_ref=None,
                scope_id=baseline.scope_id, changes=(), artifact_path=None,
                artifact_sha256=None, complete=False, failure_kind="snapshot_unavailable",
                error=f"final snapshot unavailable ({final.failure_kind}): {final.error}",
            )
        try:
            verified = self._verify_baseline_artifact(baseline)
        except (OSError, ValueError) as exc:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=(), artifact_path=baseline.path, artifact_sha256=None, complete=False,
                failure_kind="baseline_corrupt",
                error=f"the preserved baseline bytes no longer match: {exc}",
            )
        try:
            final_bytes = self._load_final_bytes(final.snapshot, verified, cancel)
            changes = _changes(verified.files, final.snapshot, final_bytes)
            artifact_path, artifact_sha256 = self._write_diff(
                verified, final.snapshot, changes, final_bytes)
        except (OSError, ValueError) as exc:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=(), artifact_path=None, artifact_sha256=None, complete=False,
                failure_kind="evidence_incomplete",
                error=f"cannot produce complete change evidence: {exc}",
            )
        # 工件生成后再次刷新快照, 并重验工件 hash: 只有两者都成立才是完整证据。
        if cancel.cancelled:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=changes, artifact_path=artifact_path, artifact_sha256=None,
                complete=False, failure_kind="snapshot_unavailable",
                error="the run was cancelled before the change artifact could be re-verified",
            )
        recheck = await self.snapshotter.capture(cancel)
        if not recheck.available:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=changes, artifact_path=artifact_path, artifact_sha256=None,
                complete=False, failure_kind="snapshot_unavailable",
                error=(
                    "the code changed while the change artifact was being produced "
                    f"({recheck.failure_kind}): {recheck.error}"),
            )
        if recheck.snapshot.snapshot_ref != final.snapshot.snapshot_ref:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=recheck.snapshot.snapshot_ref, scope_id=recheck.snapshot.scope_id,
                changes=changes, artifact_path=artifact_path, artifact_sha256=None,
                complete=False, failure_kind="snapshot_changed",
                error="the code changed while the change artifact was being produced",
            )
        try:
            observed_sha256 = _sha256_file(artifact_path)
        except OSError as exc:
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=changes, artifact_path=artifact_path, artifact_sha256=None,
                complete=False, failure_kind="artifact_corrupt",
                error=f"the change artifact could not be re-read: {exc}",
            )
        if observed_sha256 != artifact_sha256:
            # 保留生成时身份: 重读不一致说明工件被改写, 不能重认领新 hash 当作完整证据。
            return ChangeEvidence(
                baseline_snapshot_ref=baseline.snapshot_ref,
                snapshot_ref=final.snapshot.snapshot_ref, scope_id=final.snapshot.scope_id,
                changes=changes, artifact_path=artifact_path, artifact_sha256=None,
                complete=False, failure_kind="artifact_corrupt",
                error=(
                    "the change artifact no longer matches the hash recorded when it was "
                    f"written: recorded {artifact_sha256}, observed {observed_sha256}"),
            )
        return ChangeEvidence(
            baseline_snapshot_ref=baseline.snapshot_ref,
            snapshot_ref=final.snapshot.snapshot_ref,
            scope_id=final.snapshot.scope_id,
            changes=changes,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            complete=True,
        )

    # --- check 工具的投影: 服务提供 content 拼装材料, 不自造第二份接口 ---

    def render_receipt_content(
        self,
        receipt: CheckReceipt,
        freshness: Freshness = "unknown",
        view: VerificationView | None = None,
    ) -> str:
        """回执的有界 content: 控制/状态字段优先, 真实输出预览与工件事实用剩余预算.

        content 必须能独立回答"这次检查发生了什么": 原始执行事实、freshness 与当前
        范围身份、真实 stdout/stderr 预览、工件指针。完整输出仍在工件里。
        """
        body = self._control_block(self._control_lines(receipt, freshness, view))
        footer = self._output_facts(receipt)
        remaining = self.content_max_bytes - len(body.encode("utf-8")) - 1
        if footer:
            remaining -= len(footer.encode("utf-8")) + 1
        preview = self._output_preview(receipt, remaining)
        parts = [part for part in (body, preview, footer) if part]
        return _clip("\n".join(parts), self.content_max_bytes)

    def render_status_content(
        self, view: VerificationView, freshness_reason: str | None = None
    ) -> str:
        control = [f"[check status; verification_run_id={view.verification_run_id}]"]
        if view.snapshot_ref is None:
            control.append("[snapshot=unavailable freshness=unknown]")
        else:
            control.append(f"[scope_id={view.scope_id} snapshot_ref={view.snapshot_ref}]")
        control.extend(self._state_line(state) for state in view.checks)
        control.append(f"[process_uncertain={str(view.process_uncertain).lower()}]")
        if freshness_reason:
            control.append(f"[note={_one_line(freshness_reason, _REASON_MAX_BYTES)}]")
        control.append("[blocking_reasons]")
        control.extend(
            f"- {_one_line(reason, _REASON_MAX_BYTES)}" for reason in view.blocking_reasons)
        body = self._control_block(control)
        remaining = self.content_max_bytes - len(body.encode("utf-8")) - 1
        history = self._history_block(view.receipts, remaining)
        if history:
            body = f"{body}\n{history}"
        return _clip(body, self.content_max_bytes)

    def receipt_details(self, receipt: CheckReceipt, freshness: Freshness) -> dict:
        """ToolResult.details: 程序唯一 receipt_id 供 K/应用层关联, 完整事实在此."""
        return {
            "receipt_id": receipt.receipt_id,
            "verification_run_id": receipt.verification_run_id,
            "check_id": receipt.check_id,
            "command": receipt.command,
            "command_sha256": _sha256_text(receipt.command),
            "cwd": str(receipt.cwd),
            "cwd_sha256": _sha256_text(str(receipt.cwd)),
            "timeout_s": receipt.timeout_s,
            "elapsed_s": receipt.elapsed_s,
            "execution_status": receipt.execution_status,
            "exit_code": receipt.exit_code,
            "timed_out": receipt.timed_out,
            "cancelled": receipt.cancelled,
            "output_complete": receipt.output_complete,
            "snapshot_before": receipt.snapshot_before,
            "snapshot_after": receipt.snapshot_after,
            "verification_status": receipt.verification_status,
            "freshness": freshness,
            "failure_kind": receipt.failure_kind,
            "error": receipt.error,
            "output_truncated": receipt.output_truncated,
            "artifact_truncated": receipt.artifact_truncated,
            "artifact_error": receipt.artifact_error,
            "output_artifact_path": (
                str(receipt.output_artifact_path) if receipt.output_artifact_path else None),
            "output_artifact_sha256": receipt.output_artifact_sha256,
            "schema_version": receipt.schema_version,
        }

    # --- 回执构造与分类 ---

    def _receipt(self, **fields) -> CheckReceipt:
        definition: CheckDefinition = fields.pop("definition")
        return CheckReceipt(
            verification_run_id=self._plan.verification_run_id,
            receipt_id=f"chk-{uuid.uuid4().hex}",
            check_id=definition.check_id,
            command=definition.command,
            cwd=self._plan.root,
            timeout_s=definition.timeout_s,
            **fields,
        )

    def _artifact_facts(self, bounded) -> tuple[Path | None, str | None, str | None]:
        """工件 hash 在写入后重算; 有输出却读不到工件时明确失败原因.

        无输出的命令不必创建工件 (空输出加退出 0 可以按命令约定通过),
        此时工件缺失不是"假装完整输出可读"。
        """
        if bounded.artifact_error is not None:
            return bounded.full_output_path, None, bounded.artifact_error
        if bounded.full_output_path is None:
            if bounded.total_bytes == 0:
                return None, None, None
            return None, None, (
                f"the check produced {bounded.total_bytes} bytes of output "
                "but no artifact was preserved")
        try:
            return (bounded.full_output_path, _sha256_file(bounded.full_output_path), None)
        except OSError as exc:
            return None, None, f"cannot read back the output artifact: {exc}"

    def _classify(
        self, result, before_ref: str, after: SnapshotCapture | None, artifact_error: str | None
    ) -> tuple[str | None, str | None]:
        """执行事实优先于证据事实: 取消/超时/未终结/快照改变先于工件与退出码判定."""
        if result.cancelled:
            return "cancelled", "the check was cancelled; the result is not a verdict on the code"
        if result.timed_out:
            return "timed_out", (
                "the check did not reach a bounded end within its deadline; "
                "termination is not established")
        if result.exit_code is None:
            return "launch_failed", "the check command did not report an exit status"
        if not result.output_complete:
            return "output_incomplete", (
                "the output pipe did not reach EOF; termination is not established")
        if after is None:
            return "snapshot_unavailable", (
                "the check was cancelled before the post-run snapshot could be captured")
        if not after.available:
            return "snapshot_unavailable", (
                f"post-run snapshot unavailable ({after.failure_kind}): {after.error}")
        if after.snapshot.snapshot_ref != before_ref:
            return "snapshot_changed", (
                "the check changed the code it was verifying; before/after snapshots differ")
        if result.exit_code != 0:
            return "nonzero_exit", f"the check exited with code {result.exit_code}"
        if artifact_error is not None:
            return "artifact_error", artifact_error
        return None, None

    @staticmethod
    def _execution_status(result) -> str:
        if result.cancelled:
            return "cancelled"
        if result.timed_out:
            return "timed_out"
        if result.exit_code is None:
            return "launch_failed"
        return "exited"

    def _blocked(
        self,
        definition: CheckDefinition,
        started: float,
        before_ref: str | None,
        after_ref: str | None,
        failure_kind: str,
        error: str,
        uncertain_before: bool,
    ) -> CheckReceipt:
        receipt = self._receipt(
            definition=definition,
            elapsed_s=time.monotonic() - started,
            execution_status="error",
            exit_code=None,
            timed_out=False,
            cancelled=False,
            output_complete=False,
            snapshot_before=before_ref,
            snapshot_after=after_ref,
            verification_status="blocked",
            failure_kind=failure_kind,
            output_truncated=False,
            artifact_truncated=False,
            output_artifact_path=None,
            output_artifact_sha256=None,
            artifact_error=None,
            error=error,
        )
        self._record(receipt, uncertain_before)
        # 快照不可用说明命令可能在未知状态下执行过: 与超时/取消同级锁存。
        self._latch(error)
        return receipt

    def _launch_failed(
        self,
        definition: CheckDefinition,
        started: float,
        before_ref: str,
        after_ref: str | None,
        exc: BaseException,
        uncertain_before: bool,
    ) -> CheckReceipt:
        """已知未启动: 明确 blocked/launch_failed 回执, 不锁存进程不确定状态.

        进程从未创建, 没有终结未知可言; 同 run 早先的锁存 (如超时) 仍单调保持。
        命令未运行不是 PASS, 回执继续阻断交付, 之后正常重检可替代它。
        """
        receipt = self._receipt(
            definition=definition,
            elapsed_s=time.monotonic() - started,
            execution_status="launch_failed",
            exit_code=None,
            timed_out=False,
            cancelled=False,
            output_complete=False,
            snapshot_before=before_ref,
            snapshot_after=after_ref,
            verification_status="blocked",
            failure_kind="launch_failed",
            output_truncated=False,
            artifact_truncated=False,
            output_artifact_path=None,
            output_artifact_sha256=None,
            artifact_error=None,
            error=f"the check command could not be started: {exc}",
        )
        self._record(receipt, uncertain_before)
        return receipt

    async def _post_snapshot_ref(self, cancel: CancelToken) -> str | None:
        """已知未启动时的后快照: 只用于记录"这次尝试没有改变代码", 取不到就不声称."""
        if cancel.cancelled:
            return None
        capture = await self.snapshotter.capture(cancel)
        return capture.snapshot.snapshot_ref if capture.available else None

    def _record_interrupted(
        self,
        definition: CheckDefinition,
        started: float,
        before_ref: str,
        result,
        failure_kind: str,
        error: str,
    ) -> CheckReceipt:
        """runner 已进入后的取消/异常: 先锁存并留下最近的失败尝试, 再让调用方传播异常.

        执行事实优先取 runner 已返回的结果 (例如后快照失败时进程事实已知); 没有结果时
        明确记录"进程没有可靠终结事实"。旧 pass 因此不再是最新候选。
        """
        bounded = result.output if result is not None else None
        if bounded is None:
            artifact_path, artifact_sha256, artifact_error = None, None, None
        else:
            artifact_path, artifact_sha256, artifact_error = self._artifact_facts(bounded)
        receipt = self._receipt(
            definition=definition,
            elapsed_s=time.monotonic() - started,
            execution_status=(
                self._execution_status(result) if result is not None
                else "cancelled" if failure_kind == "cancelled" else "error"),
            exit_code=result.exit_code if result is not None else None,
            timed_out=bool(result.timed_out) if result is not None else False,
            cancelled=(failure_kind == "cancelled") or (
                bool(result.cancelled) if result is not None else False),
            output_complete=bool(result.output_complete) if result is not None else False,
            snapshot_before=before_ref,
            snapshot_after=None,
            verification_status="blocked",
            failure_kind=failure_kind,
            output_truncated=bool(bounded.truncated) if bounded is not None else False,
            artifact_truncated=(
                bool(bounded.artifact_truncated) if bounded is not None else False),
            output_artifact_path=artifact_path,
            output_artifact_sha256=artifact_sha256,
            artifact_error=artifact_error,
            error=error,
        )
        self._receipts.append(receipt)
        self._latch(error)
        self._remember_output(receipt, bounded.text if bounded is not None else None)
        return receipt

    def _remember_output(self, receipt: CheckReceipt, text: str | None) -> None:
        """内部保存模型可见的输出预览; 完整输出仍在工件, 冻结回执字段不变."""
        if text:
            self._previews[receipt.receipt_id] = _clip(text, OUTPUT_PREVIEW_MAX_BYTES)

    def _record(self, receipt: CheckReceipt, uncertain_before: bool) -> None:
        self._receipts.append(receipt)
        if uncertain_before:
            return
        if receipt.timed_out or receipt.cancelled:
            self._latch(
                f"check {receipt.check_id} ended without proven termination "
                f"({receipt.execution_status})")
        elif receipt.failure_kind == "output_incomplete":
            self._latch(f"check {receipt.check_id} did not reach output EOF")

    def _latch(self, reason: str) -> None:
        if reason not in self._uncertain_reasons:
            self._uncertain_reasons.append(reason)

    # --- 视图 ---

    def _view(self, capture: SnapshotCapture) -> VerificationView:
        if capture.available:
            snapshot_ref = capture.snapshot.snapshot_ref
            scope_id = capture.snapshot.scope_id
        else:
            # 捕获失败: 报出最近已知快照身份但没有 current 回执; 这是 unknown, 不是过时事实。
            snapshot_ref = self._current_snapshot_ref
            scope_id = None
        reasons: list[str] = []
        states: list[CheckState] = []
        for definition in self._plan.checks:
            receipt = next(
                (r for r in reversed(self._receipts) if r.check_id == definition.check_id), None)
            freshness = self._freshness(receipt, snapshot_ref, capture.available)
            states.append(CheckState(definition.check_id, receipt, freshness))
            if receipt is None:
                reasons.append(
                    f"check {definition.check_id} has not been run in this verification run")
            elif freshness != "current":
                reasons.append(
                    f"check {definition.check_id} is {freshness}: its receipt does not match "
                    f"the current snapshot")
            elif receipt.verification_status != "passed":
                reasons.append(
                    f"check {definition.check_id} is {receipt.verification_status}"
                    + (f" ({receipt.failure_kind})" if receipt.failure_kind else ""))
        if snapshot_ref is None:
            reasons.append(
                f"the current code snapshot is unavailable ({capture.failure_kind}): "
                f"{capture.error}")
        if self._uncertain_reasons:
            reasons.insert(0, "process_uncertain: " + "; ".join(self._uncertain_reasons))
        baseline_error = self.baseline_failure()
        if baseline_error is not None:
            reasons.insert(1 if self._uncertain_reasons else 0, baseline_error)
        return VerificationView(
            verification_run_id=self._plan.verification_run_id,
            snapshot_ref=snapshot_ref,
            scope_id=scope_id,
            checks=tuple(states),
            receipts=tuple(self._receipts),
            process_uncertain=self.process_uncertain,
            blocking_reasons=tuple(reasons),
        )

    def _freshness(
        self,
        receipt: CheckReceipt | None,
        snapshot_ref: str | None,
        captured: bool,
    ) -> Freshness:
        """回执对当前代码是否成立: 只比较回执自身的快照身份与当前快照.

        当前快照身份已经绑定范围政策与累计范围, 因此不能拿初始 baseline 范围硬比较:
        范围扩大后对同一当前范围重新检查通过必须能恢复 current。历史回执在内容或
        范围变化后自然失效, 因为它的 `snapshot_after` 不再等于当前 `snapshot_ref`。
        """
        if receipt is None or snapshot_ref is None or not captured:
            # 无法确认当前快照时只能是 unknown: 不能把上一次已知快照当成现状。
            return "unknown"
        if receipt.snapshot_before is None or receipt.snapshot_after != snapshot_ref:
            return "stale"
        return "current"

    # --- 基线工件 ---

    def _artifact_path(self, label: str) -> Path:
        if self._artifact_factory is not None:
            return self._artifact_factory(label)
        return self.workspace.new_output_file(label)

    def _write_baseline(self, snapshot: Snapshot, cancel: CancelToken) -> _Baseline:
        files: dict[str, _BaselineFile] = {}
        for entry in snapshot.entries:
            cancel.throw_if_cancelled()
            if not entry.exists:
                continue
            data = _read_verified(self.workspace.root, entry, self.snapshotter.max_file_bytes)
            files[entry.relative_path] = _BaselineFile(
                entry.relative_path, entry.sha256, len(data), data)
        path = self._artifact_path("baseline")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "schema_version": snapshot.schema_version,
                "snapshot_ref": snapshot.snapshot_ref,
                "scope_id": snapshot.scope_id,
                "policy_id": snapshot.policy_id,
                "exclusions": list(snapshot.exclusions),
            }, ensure_ascii=False, sort_keys=True) + "\n")
            for key in sorted(files, key=lambda p: (p.casefold(), p)):
                record = files[key]
                handle.write(json.dumps({
                    "path": record.relative_path,
                    "exists": True,
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "data_b64": base64.b64encode(record.data).decode("ascii"),
                }, ensure_ascii=False, sort_keys=True) + "\n")
        return _Baseline(snapshot.snapshot_ref, snapshot.scope_id, path,
                         _sha256_file(path), files)

    def _verify_baseline_artifact(self, baseline: _Baseline) -> _Baseline:
        """重新读回工件: 工件 hash 与逐文件内容都必须仍匹配初始值."""
        if _sha256_file(baseline.path) != baseline.artifact_sha256:
            raise ValueError("the baseline artifact hash no longer matches")
        header: dict | None = None
        seen: set[str] = set()
        with open(baseline.path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                record = json.loads(line)
                if index == 0:
                    header = record
                    continue
                key = record["path"]
                original = baseline.files.get(key)
                if original is None:
                    raise ValueError(f"baseline artifact has an unknown entry {key!r}")
                data = base64.b64decode(record["data_b64"])
                if (_sha256_bytes(data) != record["sha256"]
                        or _sha256_bytes(data) != original.sha256
                        or len(data) != record["size_bytes"]):
                    raise ValueError(f"baseline content hash mismatch for {key!r}")
                seen.add(key)
        if header is None or header.get("snapshot_ref") != baseline.snapshot_ref:
            raise ValueError("the baseline artifact does not describe the same snapshot")
        if seen != set(baseline.files):
            raise ValueError("the baseline artifact is missing captured files")
        return baseline

    def _load_final_bytes(
        self, final: Snapshot, baseline: _Baseline, cancel: CancelToken
    ) -> dict[str, bytes]:
        """从最终快照重新装载字节并按已捕获 hash 校验; 不采信未校验的文件内容."""
        candidates = {
            entry.relative_path for entry in final.entries
            if baseline.files.get(entry.relative_path) is None
            or baseline.files[entry.relative_path].sha256 != entry.sha256
        }
        loaded: dict[str, bytes] = {}
        for relative_path in sorted(candidates, key=lambda p: (p.casefold(), p)):
            cancel.throw_if_cancelled()
            loaded[relative_path] = _load_entry_bytes(final, relative_path)
        return loaded

    def _write_diff(
        self,
        baseline: _Baseline,
        final: Snapshot,
        changes: tuple[FileChange, ...],
        final_bytes: dict[str, bytes],
    ) -> tuple[Path, str]:
        path = self._artifact_path("changes")
        written = 0
        # 严格 UTF-8 写出: 文本 diff 只由可解码内容组成, 不会把原非法字节写成 surrogate。
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            header = (
                "# cicada change evidence\n"
                f"# baseline_snapshot_ref: {baseline.snapshot_ref}\n"
                f"# snapshot_ref: {final.snapshot_ref}\n"
                f"# scope_id: {final.scope_id}\n"
                f"# changed_files: {len(changes)}\n"
            )
            handle.write(header)
            written += len(header.encode("utf-8"))
            for change in changes:
                block = _diff_block(change, baseline.files, final_bytes)
                size = len(block.encode("utf-8"))
                if written + size > DIFF_MAX_BYTES:
                    raise ValueError(
                        f"the complete change artifact would exceed {DIFF_MAX_BYTES} bytes")
                handle.write(block)
                written += size
        return path, _sha256_file(path)

    # --- content 拼装 ---

    def _control_lines(
        self,
        receipt: CheckReceipt,
        freshness: Freshness = "unknown",
        view: VerificationView | None = None,
    ) -> list[str]:
        # 顺序即优先级: 状态/receipt/终止事实/freshness 与范围身份先占控制预算,
        # 放不下时最后被 metadata_truncated 截掉的是较长的 path preview 与 blocking 文本。
        lines = [
            f"[check {receipt.check_id} status={receipt.verification_status} "
            f"freshness={freshness}]",
            f"[receipt_id={receipt.receipt_id} "
            f"verification_run_id={receipt.verification_run_id}]",
            f"[execution_status={receipt.execution_status} exit_code={receipt.exit_code} "
            f"timed_out={str(receipt.timed_out).lower()} "
            f"cancelled={str(receipt.cancelled).lower()} "
            f"output_complete={str(receipt.output_complete).lower()} "
            f"elapsed_s={receipt.elapsed_s:.3f} timeout_s={receipt.timeout_s}]",
        ]
        if view is not None:
            lines.append(
                f"[scope_id={view.scope_id} snapshot_ref={view.snapshot_ref}]")
            lines.append(f"[process_uncertain={str(view.process_uncertain).lower()}]")
            lines.extend(
                f"[blocking: {_one_line(reason, _REASON_MAX_BYTES)}]"
                for reason in view.blocking_reasons)
        lines.extend([
            f"[command_sha256={_sha256_text(receipt.command)} "
            f"command_preview={_json_preview(receipt.command, COMMAND_PREVIEW_MAX_BYTES)}]",
            f"[cwd_sha256={_sha256_text(str(receipt.cwd))} "
            f"cwd={_json_preview(str(receipt.cwd), CWD_PREVIEW_MAX_BYTES)}]",
            f"[snapshot_before={receipt.snapshot_before} "
            f"snapshot_after={receipt.snapshot_after}]",
        ])
        if receipt.failure_kind:
            lines.append(f"[failure_kind={receipt.failure_kind}]")
        return lines

    def _control_block(self, lines: list[str]) -> str:
        """控制字段优先; 放不下时留下明确的有界说明, 不生成半截字段."""
        suffix = "[metadata_truncated=true control fields exceeded their budget]"
        out = ""
        for line in lines:
            if len(out.encode("utf-8")) + len(line.encode("utf-8")) + 1 > self.control_max_bytes:
                if not out:
                    return _clip(line, self.control_max_bytes)
                return _clip(f"{out}{suffix}\n", self.control_max_bytes)
            out = f"{out}{line}\n"
        return out.rstrip("\n")

    def _output_preview(self, receipt: CheckReceipt, budget: int) -> str:
        """真实 stdout/stderr 有界预览; 完整输出仍在工件, 这里只给尾部事实."""
        text = self._previews.get(receipt.receipt_id)
        if not text or budget <= 0:
            return ""
        header = "[output preview]"
        room = budget - len(header.encode("utf-8")) - 1
        if room <= 0:
            return ""
        return f"{header}\n{_clip(text, room)}"

    def _output_facts(self, receipt: CheckReceipt) -> str:
        """输出/工件事实: 每条字段自身有界, 因此不会在文末被截成半截字段."""
        lines = [
            f"[output_truncated={str(receipt.output_truncated).lower()} "
            f"artifact_truncated={str(receipt.artifact_truncated).lower()} "
            f"artifact_error={receipt.artifact_error}]",
            # 工件路径也走同一 JSON 编码器, 与 command/cwd 的路径口径一致。
            "[output artifact: "
            f"{_json_preview(str(receipt.output_artifact_path), CWD_PREVIEW_MAX_BYTES)}]"
            if receipt.output_artifact_path else "[output artifact: unavailable]",
        ]
        if receipt.output_artifact_sha256:
            lines.append(f"[output artifact sha256: {receipt.output_artifact_sha256}]")
        if receipt.error:
            lines.append(f"[error: {_one_line(receipt.error, _REASON_MAX_BYTES)}]")
        return "\n".join(lines)

    def _state_line(self, state: CheckState) -> str:
        receipt = state.receipt
        if receipt is None:
            return f"[check {state.check_id} status=not_run freshness={state.freshness}]"
        return (
            f"[check {state.check_id} status={state.status} freshness={state.freshness} "
            f"receipt_id={receipt.receipt_id} failure_kind={receipt.failure_kind}]")

    def _history_block(self, receipts: tuple[CheckReceipt, ...], budget: int) -> str:
        if budget <= 0 or not receipts:
            return ""
        lines = ["[receipt history]"]
        lines.extend(
            f"- {r.receipt_id} check={r.check_id} status={r.verification_status} "
            f"exit_code={r.exit_code} elapsed_s={r.elapsed_s:.3f}"
            for r in receipts)
        return _clip("\n".join(lines), budget)


def _read_verified(root: Path, entry: SnapshotEntry, max_bytes: int) -> bytes:
    """重新读取一个已捕获条目并核对 hash; 任何不一致按不可用处理."""
    target = root
    for part in entry.relative_path.split("/"):
        target = target / part
    with open(target, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{entry.relative_path} exceeds the {max_bytes} byte limit")
    if _sha256_bytes(data) != entry.sha256:
        raise ValueError(
            f"{entry.relative_path} changed after the baseline snapshot was captured")
    return data


def _load_entry_bytes(final: Snapshot, relative_path: str) -> bytes:
    """从最终快照装载单个路径的字节并复核已捕获 hash."""
    entry = next((e for e in final.entries if e.relative_path == relative_path), None)
    if entry is None or not entry.exists:
        return b""
    target = final.root
    for part in relative_path.split("/"):
        target = target / part
    with open(target, "rb") as handle:
        data = handle.read(entry.size_bytes + 1 if entry.size_bytes is not None else 0)
    if entry.size_bytes is not None and _sha256_bytes(data) != entry.sha256:
        raise ValueError(f"{relative_path} changed after the final snapshot was captured")
    return data


def _changes(
    files: dict[str, _BaselineFile], final: Snapshot, final_bytes: dict[str, bytes]
) -> tuple[FileChange, ...]:
    """基线字节 vs 最终已捕获字节的逐文件变更清单 (含新增/删除与内容 hash)."""
    changes: list[FileChange] = []
    for entry in final.entries:
        before = files.get(entry.relative_path)
        before_sha = before.sha256 if before is not None else None
        after_sha = entry.sha256 if entry.exists else None
        if (before is None) == (not entry.exists) and before_sha == after_sha:
            continue
        if before is None:
            kind = "added"
        elif not entry.exists:
            kind = "deleted"
        else:
            kind = "modified"
        before_bytes = before.data if before is not None else b""
        after_bytes = final_bytes.get(entry.relative_path, b"") if entry.exists else b""
        changes.append(FileChange(
            relative_path=entry.relative_path,
            kind=kind,
            before_sha256=before_sha,
            after_sha256=after_sha,
            binary=_is_binary(before_bytes) or _is_binary(after_bytes),
            bom_changed=_has_bom(before_bytes) != _has_bom(after_bytes),
            newline_changed=_newline_style(before_bytes) != _newline_style(after_bytes),
        ))
    return tuple(changes)


def _diff_block(
    change: FileChange, files: dict[str, _BaselineFile], final_bytes: dict[str, bytes]
) -> str:
    lines = [
        f"\n=== {change.kind} {change.relative_path}",
        f"--- before_sha256: {change.before_sha256}",
        f"+++ after_sha256: {change.after_sha256}",
    ]
    flags = [
        name for name, active in (
            ("binary", change.binary),
            ("bom_changed", change.bom_changed),
            ("newline_changed", change.newline_changed),
        ) if active
    ]
    if flags:
        lines.append(f"--- flags: {', '.join(flags)}")
    if change.binary:
        lines.append("--- textual diff not produced (binary content)")
        return "\n".join(lines) + "\n"
    before = files.get(change.relative_path)
    before_text = (before.data if before is not None else b"").decode("utf-8-sig")
    after_text = final_bytes.get(change.relative_path, b"").decode("utf-8-sig")
    lines.extend(difflib.unified_diff(
        before_text.splitlines(keepends=False),
        after_text.splitlines(keepends=False),
        fromfile=f"a/{change.relative_path}",
        tofile=f"b/{change.relative_path}",
        lineterm="",
    ))
    return "\n".join(lines) + "\n"


def _is_binary(data: bytes) -> bool:
    """二进制判定: 含 NUL, 或不是合法 UTF-8(-sig); 非 UTF-8 不给伪造的文本 diff."""
    if b"\x00" in data:
        return True
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return True
    return False


def _has_bom(data: bytes) -> bool:
    return data.startswith(b"\xef\xbb\xbf")


def _newline_style(data: bytes) -> str:
    if b"\r\n" in data:
        return "crlf"
    if b"\n" in data:
        return "lf"
    return "none"


def verification_plugin(plan: VerificationPlan) -> PluginDefinition:
    """coding-verification 插件: require workspace+inventory+process, provide coding.verification.

    同一插件同时发布 `coding.snapshot`; 快照实例与服务共用同一账本, 范围单调累计。
    """

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        inventory: GitInventory = ctx.require("coding.inventory")
        runner: PowerShellRunner = ctx.require("coding.process")
        snapshotter = Snapshotter(workspace, inventory)
        ctx.provide(SNAPSHOT_CAPABILITY, snapshotter)
        ctx.provide(VERIFICATION_CAPABILITY, Verifier(workspace, snapshotter, runner, plan))

    return PluginDefinition(
        name="coding-verification",
        setup=setup,
        provides=frozenset({VERIFICATION_CAPABILITY, SNAPSHOT_CAPABILITY}),
        requires=frozenset({"coding.workspace", "coding.inventory", "coding.process"}),
    )
