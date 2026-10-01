"""P4 共享验证契约；只定义事实，不执行检查、不控制内核会话.

回执保存历史执行事实，freshness 由当前 view 单独计算。工件不能恢复权威。
ModelPort 策略可用 view.receipts 按程序 receipt_id 关联所有历史结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import DEFAULT_EXCLUSIONS, POLICY_ID

VERIFICATION_CAPABILITY = "coding.verification"
SCHEMA_VERSION = 1
ExecutionStatus = Literal["exited", "launch_failed", "timed_out", "cancelled", "error"]
VerificationStatus = Literal["passed", "failed", "blocked"]
CheckStatus = Literal["not_run", "passed", "failed", "blocked"]
Freshness = Literal["current", "stale", "unknown"]
ChangeKind = Literal["added", "modified", "deleted"]


@dataclass(frozen=True)
class CheckDefinition:
    check_id: str
    command: str
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise ValueError("check_id must be nonempty")
        if not isinstance(self.command, str) or not self.command.strip() or len(self.command.encode("utf-8")) > 4096:
            raise ValueError("command must be nonempty and at most 4096 UTF-8 bytes")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 300:
            raise ValueError("timeout_s must be finite and in (0, 300]")


@dataclass(frozen=True)
class VerificationPlan:
    verification_run_id: str
    root: Path
    checks: tuple[CheckDefinition, ...]
    policy_id: str = POLICY_ID
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS

    def __post_init__(self) -> None:
        # 不接受可变 list 作为共享计划；文件账本由服务维护，不写回本记录。
        if not isinstance(self.verification_run_id, str) or not self.verification_run_id:
            raise ValueError("verification_run_id must be nonempty")
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("verification root must be an absolute Path")
        if not isinstance(self.checks, tuple) or not 1 <= len(self.checks) <= 8 or not all(isinstance(c, CheckDefinition) for c in self.checks):
            raise ValueError("checks must be a tuple of 1 to 8 CheckDefinition records")
        if len({check.check_id for check in self.checks}) != len(self.checks):
            raise ValueError("check_id values must be unique")
        if not isinstance(self.exclusions, tuple) or not all(isinstance(p, str) for p in self.exclusions):
            raise ValueError("exclusions must be a tuple of strings")


@dataclass(frozen=True)
class SnapshotEntry:
    relative_path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None  # tombstone: exists=False, size_bytes=None, sha256=None。


@dataclass(frozen=True)
class Snapshot:
    snapshot_ref: str
    scope_id: str
    root: Path
    entries: tuple[SnapshotEntry, ...]
    policy_id: str = POLICY_ID
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class SnapshotCapture:
    snapshot: Snapshot | None
    failure_kind: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.snapshot is not None and self.failure_kind is None and self.error is None


@dataclass(frozen=True)
class CheckReceipt:
    verification_run_id: str
    receipt_id: str
    check_id: str
    command: str
    cwd: Path
    timeout_s: float
    elapsed_s: float
    execution_status: ExecutionStatus
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    output_complete: bool
    snapshot_before: str | None
    snapshot_after: str | None
    verification_status: VerificationStatus
    failure_kind: str | None
    output_truncated: bool
    artifact_truncated: bool
    output_artifact_path: Path | None
    output_artifact_sha256: str | None
    artifact_error: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class CheckState:
    check_id: str
    receipt: CheckReceipt | None = None  # 最近一次尝试；不是最近一次成功。
    freshness: Freshness = "unknown"

    @property
    def status(self) -> CheckStatus:
        return self.receipt.verification_status if self.receipt is not None else "not_run"


@dataclass(frozen=True)
class VerificationView:
    verification_run_id: str
    snapshot_ref: str | None
    scope_id: str | None
    checks: tuple[CheckState, ...]
    receipts: tuple[CheckReceipt, ...]  # 全部历史事实，唯一 receipt_id；顺序为尝试顺序。
    process_uncertain: bool
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FileChange:
    relative_path: str
    kind: ChangeKind
    before_sha256: str | None
    after_sha256: str | None
    binary: bool
    bom_changed: bool
    newline_changed: bool


@dataclass(frozen=True)
class ChangeEvidence:
    baseline_snapshot_ref: str | None
    snapshot_ref: str | None
    scope_id: str | None
    changes: tuple[FileChange, ...]
    artifact_path: Path | None
    artifact_sha256: str | None
    complete: bool
    failure_kind: str | None = None
    error: str | None = None


class VerificationService(Protocol):
    @property
    def plan(self) -> VerificationPlan: ...

    async def initialize(self, cancel: CancelToken) -> SnapshotCapture:
        """建立可信基线；不可用时启动层不发模型请求."""
        ...

    async def run_check(self, check_id: str, cancel: CancelToken) -> CheckReceipt:
        """可预期执行故障产生 blocked/failed 回执，不能借模型文字产生事实."""
        ...

    async def refresh(self, cancel: CancelToken) -> VerificationView:
        """捕获失败时给 snapshot_ref=None/unknown view；取消仍协作传播."""
        ...

    def mark_process_uncertain(self, reason: str) -> None:
        """本 run 单调锁存，不可通过后续成功清除."""
        ...

    async def finalize(self, cancel: CancelToken) -> ChangeEvidence:
        """返回已校验的变更证据或完整性故障；不判定模型 stop."""
        ...
