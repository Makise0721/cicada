"""进程内文件访问 Module: 稳定读取、版本依据、队列化可靠修改与真实提交结果.

职责 (ADR 0010 / 契约 C1-C4):
- read: 句柄稳定读取 + 成功结果的进程内版本登记 (4096 LRU, 不存正文);
- mutate: 同路径队列内核对依据 -> 纯字节 transform -> 独占暂存 -> 单次原生发布,
  区分 not_written / committed / unknown; 协程取消不掩盖已写事实;
- 版本依据只在当前进程内有效, 不接受模型/ToolResult/存储输入, 不持久化.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import stat
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from cicada.plugins.coding import file_commit_windows as native
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace

if TYPE_CHECKING:
    from cicada.core.cancel import CancelToken

MAX_FILE_BYTES = 16 * 1024 * 1024  # 原文件与 UTF-8 结果各自的硬上限
REGISTRY_LIMIT = 4096

STATE_NOT_WRITTEN = "not_written"
STATE_COMMITTED = "committed"
STATE_UNKNOWN = "unknown"

# 登记失效 (需重读) 的结果; not_written 的可预期拒绝不动登记
_INVALIDATING_REASONS = frozenset({"version_conflict", "conflict_at_commit", "current_unverified"})
_INVALIDATING_STATES = frozenset({STATE_UNKNOWN})


class FileAccessError(Exception):
    """读取失败: 不存在/目录/权限/读取过程不稳定."""


class MutationRejected(Exception):
    """发布前的可预期拒绝 (非法编辑/非 UTF-8), 由 transform 抛出, 映射为未提交."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class FileStamp:
    """版本依据小记录: 身份 (dev/ino) 与字节 (sha256) 同时一致才算相符."""

    key: str  # canonical casefold 路径
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    sha256: str

    def identity(self) -> tuple[int, int]:
        return (self.st_dev, self.st_ino)


@dataclass(frozen=True)
class ReadSnapshot:
    canonical_path: str
    data: bytes  # 完整原始字节; 只在本次处理期间保留, 不进入登记
    stamp: FileStamp


@dataclass(frozen=True)
class MutationOutcome:
    """真实提交结果: committed 是历史事实, current_matches 是当前核对."""

    state: str
    reason: str
    canonical_path: str
    current_matches: bool | None
    caller_cancelled: bool
    winerror: int | None
    retained_paths: tuple[str, ...] = ()
    cleanup_warning: str | None = None
    message: str = ""
    before_stamp: FileStamp | None = None
    after_stamp: FileStamp | None = None


def _outcome(
    state: str, reason: str, canonical_path: str, message: str, *,
    current_matches: bool | None = None, caller_cancelled: bool = False,
    winerror: int | None = None, retained_paths: tuple[str, ...] = (),
    cleanup_warning: str | None = None,
    before_stamp: FileStamp | None = None, after_stamp: FileStamp | None = None,
) -> MutationOutcome:
    return MutationOutcome(
        state=state, reason=reason, canonical_path=canonical_path, message=message,
        current_matches=current_matches, caller_cancelled=caller_cancelled,
        winerror=winerror, retained_paths=retained_paths, cleanup_warning=cleanup_warning,
        before_stamp=before_stamp, after_stamp=after_stamp,
    )


def _read_all(path: Path) -> bytes:
    chunks = []
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            chunks.append(chunk)
    return b"".join(chunks)


class FileAccess:
    """由 Workspace 持有的单一实例; 登记仅进程内, 队列复用 Workspace 同路径锁."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._registry: OrderedDict[str, FileStamp] = OrderedDict()

    # ---- 读取 ----

    async def read(self, raw_path: str, cancel: CancelToken) -> ReadSnapshot:
        """稳定读取: 句柄 fstat 前后与路径核对一致才返回完整字节快照."""
        path = self._workspace.resolve(raw_path).resolve()
        return self._stable_read(path)

    def remember_read(self, snapshot: ReadSnapshot) -> None:
        """成功生成 read 结果后登记版本依据 (调用前后不得插入 await)."""
        self._remember(snapshot.stamp)

    def _remember(self, stamp: FileStamp) -> None:
        self._registry[stamp.key] = stamp
        self._registry.move_to_end(stamp.key)
        while len(self._registry) > REGISTRY_LIMIT:
            self._registry.popitem(last=False)

    def _stable_read(self, path: Path) -> ReadSnapshot:
        try:
            before = os.stat(path)
            if stat.S_ISDIR(before.st_mode):
                raise FileAccessError(f"path is a directory: {path}")
            if not stat.S_ISREG(before.st_mode):
                raise FileAccessError(f"path is not a regular file: {path}")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except FileAccessError:
            raise
        except FileNotFoundError:
            raise FileAccessError(f"file not found: {path}") from None
        except OSError as exc:
            raise FileAccessError(f"cannot read {path}: {exc}") from exc
        try:
            opened = os.fstat(fd)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
            closed = os.fstat(fd)
        except OSError as exc:
            raise FileAccessError(f"cannot read {path}: {exc}") from exc
        finally:
            os.close(fd)
        try:
            after = os.stat(path)
        except OSError as exc:
            raise FileAccessError(f"cannot re-check {path}: {exc}") from exc
        identity = (opened.st_dev, opened.st_ino)
        stable = (
            (before.st_dev, before.st_ino) == identity
            and (closed.st_dev, closed.st_ino) == identity
            and (after.st_dev, after.st_ino) == identity
        )
        if not stable:
            raise FileAccessError(f"file changed while being read: {path}; read it again")
        data = b"".join(chunks)
        stamp = FileStamp(
            key=str(path).casefold(),
            st_dev=opened.st_dev, st_ino=opened.st_ino,
            st_size=opened.st_size, st_mtime_ns=opened.st_mtime_ns,
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return ReadSnapshot(canonical_path=str(path), data=data, stamp=stamp)

    # ---- 修改 ----

    async def mutate(
        self,
        raw_path: str,
        transform: Callable[[bytes | None], bytes],
        cancel: CancelToken,
        *,
        mode: Literal["edit", "write"],
    ) -> MutationOutcome:
        """队列化可靠修改; 外层协程取消不泛化真实结果 (契约 C2)."""
        if mode != "edit":
            # write 分支由票据 02 交付; tool_write 当前仍走旧路径
            raise ValueError(f"mutation mode {mode!r} is not implemented yet")
        stop = threading.Event()
        cancelled = threading.Event()  # caller_cancelled 记录, loop 线程读写

        def note_cancelled() -> None:
            cancelled.set()
            stop.set()

        def on_caller_cancel() -> None:
            note_cancelled()
            cancel.cancel()

        watcher = asyncio.create_task(self._watch_cancel_token(cancel, note_cancelled))
        try:
            outcome = await self._workspace.mutate(
                self._workspace.resolve(raw_path),
                lambda: self._mutation_worker(raw_path, transform, stop),
                preserve_result_on_cancel=True,
                on_caller_cancel=on_caller_cancel,
            )
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if outcome is None:
            # 排队中取消: op 未启动, 令牌已设置
            return _outcome(
                STATE_NOT_WRITTEN, "cancelled_before_commit",
                str(self._workspace.resolve(raw_path)),
                "cancelled while waiting for the per-file queue; file not modified",
                caller_cancelled=True,
            )
        if cancelled.is_set():
            outcome = dataclasses.replace(outcome, caller_cancelled=True)
        return outcome

    @staticmethod
    async def _watch_cancel_token(cancel: CancelToken, on_cancelled: Callable[[], None]) -> None:
        if cancel.cancelled:
            on_cancelled()
            return
        await cancel.wait()
        on_cancelled()

    async def _mutation_worker(
        self, raw_path: str, transform: Callable[[bytes | None], bytes], stop: threading.Event
    ) -> MutationOutcome:
        """持有同路径队列: 核对依据 -> transform -> 线程内暂存与发布."""
        if stop.is_set():
            return _outcome(
                STATE_NOT_WRITTEN, "cancelled_before_commit",
                str(self._workspace.resolve(raw_path)),
                "cancelled before verification; file not modified", caller_cancelled=True,
            )
        try:
            target = self._workspace.resolve_for_write(raw_path)
        except PathNotAllowedError as exc:
            return _outcome(STATE_NOT_WRITTEN, "unsupported_target",
                            str(self._workspace.resolve(raw_path)), str(exc))
        original = self._workspace.resolve(raw_path)
        key = str(target).casefold()

        try:
            target_stat = target.stat()
        except FileNotFoundError:
            entry = self._registry.pop(key, None)
            if entry is not None:
                return _outcome(
                    STATE_NOT_WRITTEN, "version_conflict", str(target),
                    f"file was deleted after it was last read: {target}; "
                    "read it again (or recreate it) before modifying",
                    before_stamp=entry,
                )
            return _outcome(STATE_NOT_WRITTEN, "not_found", str(target),
                            f"file not found: {target}")
        except OSError as exc:
            return _outcome(STATE_NOT_WRITTEN, "unsupported_target", str(target),
                            f"cannot inspect target {target}: {exc}")
        if not stat.S_ISREG(target_stat.st_mode):
            return _outcome(STATE_NOT_WRITTEN, "unsupported_target", str(target),
                            f"target is not a regular file: {target}")

        entry = self._registry.get(key)
        if entry is None:
            return _outcome(
                STATE_NOT_WRITTEN, "read_required", str(target),
                f"existing file must be read with the read tool before editing: {target}",
            )
        try:
            native.check_mutation_target(original, target)
        except native.UnsupportedTargetError as exc:
            return _outcome(STATE_NOT_WRITTEN, "unsupported_target", str(target), str(exc))
        if target_stat.st_size > MAX_FILE_BYTES:
            return _outcome(
                STATE_NOT_WRITTEN, "size_limit", str(target),
                f"file exceeds the {MAX_FILE_BYTES} byte edit limit "
                f"({target_stat.st_size} bytes)",
                before_stamp=entry,
            )
        try:
            snapshot = self._stable_read(target)
        except FileAccessError as exc:
            self._registry.pop(key, None)
            return _outcome(STATE_NOT_WRITTEN, "version_conflict", str(target),
                            f"{exc}; read the file again before editing",
                            before_stamp=entry)
        current = snapshot.stamp
        if current.identity() != entry.identity() or current.sha256 != entry.sha256:
            self._registry.pop(key, None)
            return _outcome(
                STATE_NOT_WRITTEN, "version_conflict", str(target),
                f"file changed since it was last read: {target}; read it again before editing",
                before_stamp=entry,
            )
        try:
            new_bytes = transform(snapshot.data)
        except MutationRejected as exc:
            return _outcome(STATE_NOT_WRITTEN, exc.reason, str(target), str(exc),
                            before_stamp=entry)
        if len(new_bytes) > MAX_FILE_BYTES:
            return _outcome(
                STATE_NOT_WRITTEN, "size_limit", str(target),
                f"edit result exceeds the {MAX_FILE_BYTES} byte limit ({len(new_bytes)} bytes)",
                before_stamp=entry,
            )
        if stop.is_set():
            return _outcome(
                STATE_NOT_WRITTEN, "cancelled_before_commit", str(target),
                "cancelled before publication; file not modified",
                caller_cancelled=True, before_stamp=entry,
            )

        staging = self._workspace.root / ".cicada" / "mutations" / uuid.uuid4().hex
        outcome = await asyncio.to_thread(
            self._publish_replacement, staging, target, original, current, new_bytes, stop
        )
        # 登记策略: 完整核对相符才更新; 冲突/未知/当前不符失效
        verified = (
            outcome.state == STATE_COMMITTED
            and outcome.reason == "applied"
            and outcome.current_matches is True
            and outcome.after_stamp is not None
        )
        if verified:
            self._remember(outcome.after_stamp)
        elif outcome.state == STATE_COMMITTED or outcome.state == STATE_UNKNOWN \
                or outcome.reason in _INVALIDATING_REASONS:
            self._registry.pop(key, None)
        return outcome

    # ---- 原生发布 (工作线程内执行, 不触碰登记) ----

    def _publish_replacement(
        self, staging: Path, target: Path, original: Path,
        before: FileStamp, new_bytes: bytes, stop: threading.Event,
    ) -> MutationOutcome:
        staged = staging / "new.tmp"
        backup = staging / "old.bak"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            if os.stat(staging).st_dev != os.stat(target).st_dev:
                warning = self._cleanup_staging(staging, [staged])
                return _outcome(
                    STATE_NOT_WRITTEN, "unsupported_target", str(target),
                    f"staging directory is not on the target volume: {target}",
                    cleanup_warning=warning, before_stamp=before,
                )
            native.write_staged_file(staged, new_bytes)
        except OSError as exc:
            warning = self._cleanup_staging(staging, [staged])
            return _outcome(
                STATE_NOT_WRITTEN, "publish_failed", str(target),
                f"cannot prepare staged file in {staging}: {exc}",
                winerror=getattr(exc, "winerror", None), cleanup_warning=warning,
                before_stamp=before,
            )
        if stop.is_set():
            warning = self._cleanup_staging(staging, [staged])
            return _outcome(
                STATE_NOT_WRITTEN, "cancelled_before_commit", str(target),
                "cancelled before publication; file not modified",
                caller_cancelled=True, cleanup_warning=warning, before_stamp=before,
            )
        # 发布前重新核对: 原路径绑定、祖先 reparse、目标身份仍未变化
        try:
            if native.reparse_ancestor(original) is not None \
                    or Path(os.path.realpath(original)) != target:
                warning = self._cleanup_staging(staging, [staged])
                return _outcome(
                    STATE_NOT_WRITTEN, "version_conflict", str(target),
                    f"target path changed during the operation: {original} no longer "
                    f"resolves to {target}; read it again before editing",
                    cleanup_warning=warning, before_stamp=before,
                )
            target_stat_now = target.stat()
            if (target_stat_now.st_dev, target_stat_now.st_ino) != before.identity():
                warning = self._cleanup_staging(staging, [staged])
                return _outcome(
                    STATE_NOT_WRITTEN, "version_conflict", str(target),
                    f"target identity changed during the operation: {target}; "
                    "read it again before editing",
                    cleanup_warning=warning, before_stamp=before,
                )
            native.replace_file(target, staged, backup)
        except native.UnsupportedTargetError as exc:
            warning = self._cleanup_staging(staging, [staged])
            return _outcome(STATE_NOT_WRITTEN, "unsupported_target", str(target), str(exc),
                            cleanup_warning=warning, before_stamp=before)
        except native.NativePublishError as exc:
            return self._classify_replace_failure(target, staged, backup, staging, before, exc)
        except OSError as exc:
            warning = self._cleanup_staging(staging, [staged])
            return _outcome(
                STATE_NOT_WRITTEN, "publish_failed", str(target),
                f"pre-publication recheck failed: {exc}",
                winerror=getattr(exc, "winerror", None), cleanup_warning=warning,
                before_stamp=before,
            )
        return self._verify_commit(target, staging, staged, backup, before, new_bytes)

    def _classify_replace_failure(
        self, target: Path, staged: Path, backup: Path, staging: Path,
        before: FileStamp, exc: native.NativePublishError,
    ) -> MutationOutcome:
        backup_exists = backup.exists()
        staged_exists = staged.exists()
        # 目标稳定等于原件: 身份一致; 可读时进一步核对字节, 被锁等不可读时
        # 以 size+mtime 相等作为可达证据 (锁定的目标本身即未被改写的旁证)
        target_unchanged = False
        try:
            info = target.stat()
            if (info.st_dev, info.st_ino) == before.identity():
                try:
                    target_unchanged = (
                        hashlib.sha256(_read_all(target)).hexdigest() == before.sha256
                    )
                except OSError:
                    target_unchanged = (
                        info.st_size == before.st_size and info.st_mtime_ns == before.st_mtime_ns
                    )
        except OSError:
            target_unchanged = False
        provable = (
            exc.winerror in native.NOT_COMMITTED_WINERRORS
            and not backup_exists
            and staged_exists
            and target_unchanged
        )
        if provable:
            warning = self._cleanup_staging(staging, [staged, backup])
            return _outcome(
                STATE_NOT_WRITTEN, "publish_failed", str(target),
                f"publication failed and the target is verified unchanged: {exc}",
                winerror=exc.winerror, cleanup_warning=warning, before_stamp=before,
            )
        retained = tuple(str(p) for p in (staged, backup, staging) if p.exists())
        return _outcome(
            STATE_UNKNOWN, "publish_failed", str(target),
            f"publication failed after entering the native call; "
            f"commit state cannot be proven: {exc}",
            winerror=exc.winerror, retained_paths=retained, before_stamp=before,
        )

    def _verify_commit(
        self, target: Path, staging: Path, staged: Path, backup: Path,
        before: FileStamp, new_bytes: bytes,
    ) -> MutationOutcome:
        expected_sha = hashlib.sha256(new_bytes).hexdigest()
        readback: bytes | None = None
        readback_error: str | None = None
        try:
            readback = _read_all(target)
            after_stat = target.stat()
        except OSError as exc:
            readback_error = f"{type(exc).__name__}: {exc}"
        if readback is not None and readback != new_bytes:
            # 提交成功, 但当前目标已被外部改写: 已写事实保留, 不假报未写
            retained = tuple(str(p) for p in (backup, staging) if p.exists())
            return _outcome(
                STATE_COMMITTED, "applied", str(target),
                f"edit was committed, but the target no longer matches the committed "
                f"content (changed externally after commit): {target}",
                current_matches=False, retained_paths=retained, before_stamp=before,
            )
        backup_ok = False
        backup_error: str | None = None
        try:
            backup_stat = backup.stat()
            backup_ok = (
                (backup_stat.st_dev, backup_stat.st_ino) == before.identity()
                and hashlib.sha256(_read_all(backup)).hexdigest() == before.sha256
            )
        except OSError as exc:
            backup_error = f"{type(exc).__name__}: {exc}"
        if readback_error is not None:
            retained = tuple(str(p) for p in (backup, staging) if p.exists())
            return _outcome(
                STATE_COMMITTED, "current_unverified", str(target),
                f"edit was committed, but the target readback failed ({readback_error}); "
                f"re-read {target} to verify",
                current_matches=None, retained_paths=retained, before_stamp=before,
            )
        if not backup_ok:
            retained = tuple(str(p) for p in (backup, staging) if p.exists())
            detail = backup_error or "backup does not match the verified original"
            return _outcome(
                STATE_COMMITTED, "conflict_at_commit", str(target),
                f"edit was committed, but the replaced original does not match the "
                f"verified version ({detail}); the file may have changed in the commit window",
                current_matches=True, retained_paths=retained, before_stamp=before,
            )
        after_stamp = FileStamp(
            key=before.key, st_dev=after_stat.st_dev, st_ino=after_stat.st_ino,
            st_size=after_stat.st_size, st_mtime_ns=after_stat.st_mtime_ns,
            sha256=expected_sha,
        )
        warning = self._cleanup_staging(staging, [staged, backup])
        message = "edit applied"
        if warning:
            message = f"edit applied; cleanup warning: {warning}"
        return _outcome(
            STATE_COMMITTED, "applied", str(target), message,
            current_matches=True, cleanup_warning=warning,
            before_stamp=before, after_stamp=after_stamp,
        )

    def _cleanup_staging(self, staging: Path, known_files: list[Path]) -> str | None:
        """只删本次记录的确切材料与空 staging 目录; 不递归、不删父目录."""
        warnings: list[str] = []
        for path in known_files:
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"{path}: {exc}")
        try:
            if staging.is_dir() and not any(staging.iterdir()):
                staging.rmdir()
        except OSError as exc:
            warnings.append(f"{staging}: {exc}")
        return "; ".join(warnings) or None
