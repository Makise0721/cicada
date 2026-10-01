"""P4 §4 有界代码快照: 单调累计路径账本、两次独立全量扫描、内容 SHA-256 身份.

账本记录**本 run 曾观察到**的全部 Git 可见路径: 已见路径后来被删除或改为 ignored
仍留在账本并按 tombstone/missing 比较, 所以路径不能靠改 ignore 规则消失。
一次捕获做两次独立全量扫描并用同一账本比较: 只有两次观察到的路径集合相同、
候选原始路径上没有 reparse 祖先/gitlink、每个文件内容前后一致、且总预算未超,
才给出不可变 `Snapshot`; 否则 `available=False`, 不回退到半份快照。

这是 checkpoint 一致性检查, 不是操作系统原子快照, 也不能证明检查期间没有发生
"短暂修改后还原"; 它面向单 Agent、无并发修改的验收工作区。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import (
    DEFAULT_EXCLUSIONS,
    POLICY_ID,
    FileInventory,
    GitInventory,
    InventoryEntry,
)
from cicada.plugins.coding.verification_contracts import (
    SCHEMA_VERSION,
    Snapshot,
    SnapshotCapture,
    SnapshotEntry,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginContext, PluginDefinition

SNAPSHOT_CAPABILITY = "coding.snapshot"
SCOPE_SCHEMA_VERSION = 1
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_PATHS = 10_000
CAPTURE_DEADLINE_S = 10.0  # 两次扫描合计
READ_CHUNK_BYTES = 256 * 1024
_CHECKPOINT_EVERY = 128


@dataclass(frozen=True)
class _LedgerScan:
    """一次扫描的观察结果, 尚未提交到账本."""

    observations: dict[str, SnapshotEntry]
    observed_paths: frozenset[str]
    root: Path


class PathScopeLedger:
    """本 run 的路径账本; 单调累计, 只增不减."""

    def __init__(self, *, policy_id: str = POLICY_ID,
                 exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS) -> None:
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be nonempty")
        if not isinstance(exclusions, tuple) or not all(isinstance(item, str) for item in exclusions):
            raise ValueError("exclusions must be a tuple of strings")
        self.policy_id = policy_id
        self.exclusions = exclusions
        self._paths: set[str] = set()

    @property
    def paths(self) -> tuple[str, ...]:
        """稳定排序的账本路径; 排序键与清单一致 (casefold, 原串)."""
        return tuple(sorted(self._paths, key=lambda path: (path.casefold(), path)))

    def observe(self, observed: frozenset[str]) -> None:
        self._paths |= observed

    @property
    def scope_id(self) -> str:
        """策略 + 排除名单 + 账本路径集合的稳定身份; 内容变化不改变范围身份."""
        digest = hashlib.sha256()
        _feed(digest, "cicada-scope", SCOPE_SCHEMA_VERSION, self.policy_id, self.exclusions)
        for path in self.paths:
            _feed(digest, path)
        return f"scope-{digest.hexdigest()}"


class Snapshotter:
    """两次独立全量扫描 + 内容 SHA-256; 任何不稳定/超限/读错/reparse 都 unavailable."""

    def __init__(
        self,
        workspace: Workspace,
        inventory: GitInventory,
        *,
        ledger: PathScopeLedger | None = None,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_paths: int = MAX_PATHS,
        deadline_s: float = CAPTURE_DEADLINE_S,
    ) -> None:
        for name, value, ceiling in (
            ("max_file_bytes", max_file_bytes, MAX_FILE_BYTES),
            ("max_total_bytes", max_total_bytes, MAX_TOTAL_BYTES),
            ("max_paths", max_paths, MAX_PATHS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if isinstance(deadline_s, bool) or not 0 < deadline_s <= CAPTURE_DEADLINE_S:
            raise ValueError(f"deadline_s must be in (0, {CAPTURE_DEADLINE_S}]")
        self.workspace = workspace
        self.inventory = inventory
        self.ledger = ledger if ledger is not None else PathScopeLedger()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_paths = max_paths
        self.deadline_s = deadline_s

    async def capture(self, cancel: CancelToken) -> SnapshotCapture:
        """两次扫描合计在同一 deadline 内; 协作取消与 Task.cancel 都传播."""
        try:
            async with asyncio.timeout(self.deadline_s):
                return await self._capture(cancel)
        except TimeoutError:
            return _unavailable("timeout", "snapshot capture deadline exceeded")

    async def _capture(self, cancel: CancelToken) -> SnapshotCapture:
        cancel.throw_if_cancelled()
        deadline = time.monotonic() + self.deadline_s
        try:
            first = await self._scan(cancel, deadline)
        except _CaptureError as exc:
            return _unavailable(exc.kind, exc.message)
        try:
            second = await self._scan(cancel, deadline)
        except _CaptureError as exc:
            return _unavailable(exc.kind, exc.message)
        if first.root != second.root:
            return _unavailable("unstable", "workspace root changed between snapshot scans")
        if first.observed_paths != second.observed_paths:
            # 两次独立全量扫描的候选集合必须相同; 扫描期间出现新候选即不稳定。
            return _unavailable(
                "unstable", "the two full scans observed different path sets")
        if len(first.observed_paths) > self.max_paths:
            return _unavailable(
                "path_limit", f"snapshot exceeds {self.max_paths} ledger paths")
        entries = tuple(
            first.observations[path]
            if first.observations[path] == second.observations[path]
            # 内容/存在状态两次不一致: 读取前后已 stat 校验, 这里再以第二次为准并标记
            else _unstable_entry(first.observations[path], second.observations[path])
            for path in sorted(first.observations, key=lambda path: (path.casefold(), path))
        )
        if any(entry.sha256 is None and entry.exists for entry in entries):
            return _unavailable("unstable", "file content changed between the two snapshot scans")
        # 只有两次扫描一致才提交账本, 失败捕获不留下半份范围。
        self.ledger.observe(first.observed_paths)
        return SnapshotCapture(
            snapshot=Snapshot(
                snapshot_ref=snapshot_ref(entries),
                scope_id=self.ledger.scope_id,
                root=first.root,
                entries=entries,
                policy_id=self.ledger.policy_id,
                exclusions=self.ledger.exclusions,
            )
        )

    async def _scan(self, cancel: CancelToken, deadline: float) -> _LedgerScan:
        self._checkpoint(cancel, deadline)
        inventory = await self._inventory(cancel)
        observed: set[str] = set()
        for index, entry in enumerate(inventory.files):
            if index % _CHECKPOINT_EVERY == 0:
                await asyncio.sleep(0)
                self._checkpoint(cancel, deadline)
            if entry.is_submodule:
                raise _CaptureError("submodule", f"scope contains a submodule: {entry.relative_path}")
            if entry.is_reparse:
                raise _CaptureError(
                    "reparse",
                    f"scope path crosses a reparse point: {entry.reparse_paths[0] if entry.reparse_paths else entry.relative_path}",
                )
            _validate_relative_path(entry.relative_path)
            observed.add(entry.relative_path)
        observed |= set(self.ledger.paths)
        observations: dict[str, SnapshotEntry] = {}
        total_bytes = 0
        for path in sorted(observed, key=lambda path: (path.casefold(), path)):
            self._checkpoint(cancel, deadline)
            exists, sha256, size = await self._read_content(path, cancel, deadline, total_bytes)
            if size:
                total_bytes += size
            observations[path] = SnapshotEntry(path, exists, size, sha256)
        return _LedgerScan(observations, frozenset(observed), inventory.root)

    async def _inventory(self, cancel: CancelToken) -> FileInventory:
        try:
            return await self.inventory.list_files(cancel)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 清单失败即范围不可用, 不能只比较半份范围
            raise _CaptureError("inventory", f"file inventory failed: {exc}") from exc

    async def _read_content(
        self, path: str, cancel: CancelToken, deadline: float, total_bytes: int
    ) -> tuple[bool, str | None, int | None]:
        """读取一个账本路径; 返回 (exists, sha256, size_bytes).

        读取前后 stat 不一致、超单文件/累计上限、reparse 祖先或读取故障都抛
        `_CaptureError`, 调用方整次捕获 unavailable。
        """
        target = self.workspace.root
        parts = path.split("/")
        for index, part in enumerate(parts):
            target = target / part
            info = self._lstat(target, path)
            if info is None:
                return False, None, None
            if _is_reparse(info):
                raise _CaptureError("reparse", f"reparse point in the scope: {path}")
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise _CaptureError("unreadable", f"scope ancestor is not a directory: {path}")
        before = info
        if not stat.S_ISREG(before.st_mode):
            raise _CaptureError("unreadable", f"scope path is not a regular file: {path}")
        if before.st_size > self.max_file_bytes:
            raise _CaptureError(
                "file_limit",
                f"{path} is {before.st_size} bytes, over the {self.max_file_bytes} byte limit",
            )
        if total_bytes + before.st_size > self.max_total_bytes:
            raise _CaptureError(
                "total_limit", f"snapshot exceeds {self.max_total_bytes} total bytes")
        digest = hashlib.sha256()
        read_bytes = 0
        try:
            with open(target, "rb") as handle:
                while True:
                    self._checkpoint(cancel, deadline)
                    chunk = handle.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    if read_bytes > self.max_file_bytes:
                        raise _CaptureError(
                            "file_limit",
                            f"{path} grew past the {self.max_file_bytes} byte limit while reading",
                        )
                    if total_bytes + read_bytes > self.max_total_bytes:
                        raise _CaptureError(
                            "total_limit", f"snapshot exceeds {self.max_total_bytes} total bytes")
                    digest.update(chunk)
        except _CaptureError:
            raise
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            raise _CaptureError("unreadable", f"cannot read {path}: {exc}") from exc
        after = self._lstat(target, path)
        if after is None:
            raise _CaptureError("unstable", f"{path} disappeared while being read")
        if _identity(before) != _identity(after) or read_bytes != before.st_size:
            raise _CaptureError("unstable", f"{path} changed while being snapshot")
        return True, digest.hexdigest(), read_bytes

    @staticmethod
    def _lstat(target: Path, path: str):
        try:
            return target.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _CaptureError("unreadable", f"cannot stat {path}: {exc}") from exc

    @staticmethod
    def _checkpoint(cancel: CancelToken, deadline: float) -> None:
        cancel.throw_if_cancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError("snapshot capture deadline exceeded")


class _CaptureError(Exception):
    """一次捕获的明确故障; 转成 unavailable 结果, 不外泄半份快照."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(message)


def _unavailable(kind: str, message: str) -> SnapshotCapture:
    return SnapshotCapture(snapshot=None, failure_kind=kind, error=message)


def _unstable_entry(first: SnapshotEntry, second: SnapshotEntry) -> SnapshotEntry:
    """两次扫描不一致的记录: 内容身份作废, 只保留第二次的存在状态与长度."""
    return SnapshotEntry(first.relative_path, second.exists, second.size_bytes, None)


def _is_reparse(info) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _identity(info) -> tuple[int, int, int, int]:
    """读取前后的稳定身份: 设备/inode + 大小 + mtime 纳秒."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _validate_relative_path(path: str) -> None:
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or ":" in path
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise _CaptureError("invalid_path", f"invalid scope path: {path!r}")


def _feed(digest: "hashlib._Hash", *fields: object) -> None:
    """长度前缀的字段编码: 边界明确, 不同字段组合不会拼出同一串."""
    for field_value in fields:
        if isinstance(field_value, tuple):
            payload = json.dumps(list(field_value), ensure_ascii=False).encode("utf-8")
        else:
            payload = str(field_value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def snapshot_ref(entries: tuple[SnapshotEntry, ...]) -> str:
    """schema 版本分隔后的记录序列 SHA-256; 稳定排序, 与扫描顺序无关."""
    digest = hashlib.sha256()
    _feed(digest, "cicada-snapshot", SCHEMA_VERSION)
    for entry in entries:
        _feed(
            digest,
            entry.relative_path,
            "1" if entry.exists else "0",
            "" if entry.size_bytes is None else entry.size_bytes,
            entry.sha256 or "",
        )
    return f"snap-{digest.hexdigest()}"


def snapshot_plugin() -> PluginDefinition:
    """coding-snapshot 插件: 依赖 workspace+inventory, 提供 coding.snapshot."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        inventory: GitInventory = ctx.require("coding.inventory")
        ctx.provide(SNAPSHOT_CAPABILITY, Snapshotter(workspace, inventory))

    return PluginDefinition(
        name="coding-snapshot",
        setup=setup,
        provides=frozenset({SNAPSHOT_CAPABILITY}),
        requires=frozenset({"coding.workspace", "coding.inventory"}),
    )
