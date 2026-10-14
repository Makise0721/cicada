"""请求还原点的捕获与存储引擎 (契约 C2-C4): 内容寻址 blob + 每会话追加索引.

职责与边界:
- 只作为 FileAccess 的可选捕获汇被调用: 发布前落预行 (commit_state=unknown),
  发布在同一队列内落定后追加结算行; 捕获 IO 失败向上抛, 由发布路径映射为
  not_written + reason=restore_capture_failed, 不静默降级;
- 只写工作区 `<root>/.cicada/restore/` 自有文件 (blobs/<sha256> 与
  points/<session_id>.jsonl); 不改会话 SQLite schema, 不依赖 git;
- 保留: 每会话最近 100 点、工作区 blob 总量 ≤512 MiB 逐出最旧点、启动机会式清理
  30 天前点; 逐出/缺失标 expired, 不伪装可还原; 清理拒绝 reparse/symlink;
- 本模块不做 UI、不定义中立 DTO (C5 属查看票)、不自动应用还原点.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cicada.plugins.coding.file_access import CaptureTicket
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginContext, PluginDefinition

RESTORE_CAPABILITY = "coding.restore"

INDEX_VERSION = 1
ROW_ENTRY = "entry"
ROW_SETTLE = "settle"
COMMIT_STATES = ("committed", "unknown", "not_written")
SOURCE_STATES = ("committed", "interrupted", "unknown")
OP_CREATE = "create"
OP_MODIFY = "modify"
KIND_REQUEST = "request"
KIND_PRE_RESTORE = "pre_restore"  # 还原备份点 (C3 词表; 由还原执行票写入)

# C3 保留限额 (数值进合同, 不随实现漂移)
POINT_LIMIT_PER_SESSION = 100
BLOB_TOTAL_LIMIT_BYTES = 512 * 1024 * 1024
POINT_MAX_AGE_DAYS = 30
# C2 限额: 每点条目上限; 超出标 incomplete, 已捕获部分仍可用
MAX_ENTRIES_PER_POINT = 10_000

AVAILABLE = "available"
EXPIRED = "expired"
INCOMPLETE = "incomplete"

_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_ROW_KEYS = (
    "version", "row", "point_id", "kind", "request_id", "seq", "epoch_id",
    "session_id", "created_at", "source", "commit_state", "entries", "links",
)
_SETTLE_KEYS = _ROW_KEYS + ("winerror", "availability")
_ENTRY_KEYS = ("path", "op", "before_sha256", "after_sha256", "size_bytes")


class RestoreStoreError(RuntimeError):
    """捕获汇/存储层错误; 捕获路径上的任何异常都由发布路径映射为 not_written."""


@dataclass(frozen=True)
class StoredEntry:
    """聚合后的单文件条目: op/before 取该(点,路径)首份捕获, after/size 取最后一次."""

    path: str
    op: str
    before_sha256: str | None
    after_sha256: str | None
    size_bytes: int


@dataclass(frozen=True)
class StoredPoint:
    """索引聚合后的还原点事实; C5 中立 DTO 由查看票在此之上投影."""

    point_id: str
    kind: str
    request_id: str | None
    seq: int | None
    epoch_id: str | None
    session_id: str
    created_at: str
    source: str
    commit_state: str
    availability: str
    entries: tuple[StoredEntry, ...]
    total_bytes: int
    winerror: int | None
    links: tuple[str, ...]


@dataclass
class _RequestContext:
    """一个请求内的捕获上下文 (应用在 begin 成功后建立, 请求收尾后清除)."""

    request_id: str
    epoch_id: str
    session_id: str
    point_id: str | None = None
    noted_seq: int | None = None
    # path -> 该(点,路径)首份 (op, before_sha256); 同请求重复改同一路径只留首份 before
    captured: dict[str, tuple[str, str | None]] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checked_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
        raise RestoreStoreError(f"invalid session id for the restore index: {session_id!r}")
    return session_id


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _reparse_in_chain(root: Path, path: Path) -> str | None:
    """从 root 自身到 path 逐段 lstat; 缺失即安全停下, 返回第一个 reparse/symlink."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return str(path)
    current = root
    for part in (None, *parts):
        if part is not None:
            current = current / part
        try:
            info = current.lstat()
        except OSError:
            return None
        if _is_reparse(info):
            return str(current)
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def _decode_row(raw: bytes) -> dict | None:
    """严格解析一行索引; 结构不符/截断行返回 None (调用方如实降级, 不猜内容)."""
    try:
        row = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        return None
    if not isinstance(row, dict) or row.get("version") != INDEX_VERSION:
        return None
    if not set(_ROW_KEYS) <= set(row):
        return None
    row_kind = row.get("row")
    allowed = _SETTLE_KEYS if row_kind == ROW_SETTLE else _ROW_KEYS
    if row_kind not in (ROW_ENTRY, ROW_SETTLE) or set(row) - set(allowed):
        return None
    if not isinstance(row.get("point_id"), str) or not row["point_id"]:
        return None
    if row.get("kind") not in (KIND_REQUEST, KIND_PRE_RESTORE):
        return None
    if row.get("request_id") is not None and not isinstance(row["request_id"], str):
        return None
    if row.get("epoch_id") is not None and not isinstance(row["epoch_id"], str):
        return None
    if not isinstance(row.get("session_id"), str):
        return None
    if row.get("commit_state") not in COMMIT_STATES or row.get("source") not in SOURCE_STATES:
        return None
    if row.get("seq") is not None and not isinstance(row["seq"], int):
        return None
    if not isinstance(row.get("created_at"), str) or _row_created_at(row) is None:
        return None
    if row.get("links") is not None and not isinstance(row["links"], (list, tuple)):
        return None
    if row_kind == ROW_SETTLE:
        if row.get("winerror") is not None and not isinstance(row["winerror"], int):
            return None
        if row.get("availability") not in (None, EXPIRED, INCOMPLETE):
            return None
    entries = row.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != set(_ENTRY_KEYS):
            return None
        if entry.get("op") not in (OP_CREATE, OP_MODIFY):
            return None
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            return None
        if entry.get("before_sha256") is not None and not _is_sha256(entry["before_sha256"]):
            return None
        if entry.get("after_sha256") is not None and not _is_sha256(entry["after_sha256"]):
            return None
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return None
    return row


def _row_created_at(row: dict) -> datetime | None:
    try:
        value = datetime.fromisoformat(row.get("created_at", ""))
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _new_point_id() -> str:
    return f"rp-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


class RestoreStore:
    """单一工作区的还原点引擎: 捕获汇 + 存储 + 保留是同一实例 (线程安全)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        self._context: _RequestContext | None = None
        # 收尾来源标记写失败时按会话挂起 (见 end_request): 待本会话下一次成功追加时
        # 一并落盘, 期间读取方也按挂起集合如实标注, 不把中断伪称成已完成。
        self._pending_markers: dict[str, list[dict]] = {}
        self._warnings: list[str] = []

    # ---- 捕获汇 (FileAccess 调用; 均在 FileAccess 队列内, 同步阻塞) ----

    def begin_request(self, request_id: str, epoch_id: str, session_id: str) -> None:
        """请求开始后建立捕获上下文; 请求内缺失上下文属编程错误 (fail-closed)."""
        with self._lock:
            if self._context is not None:
                raise RestoreStoreError(
                    "a restore capture context is already open; overlapping requests "
                    "cannot share one restore point"
                )
            self._context = _RequestContext(
                request_id=str(request_id), epoch_id=str(epoch_id),
                session_id=_checked_session_id(session_id),
            )

    def end_request(self, *, interrupted: bool = False) -> None:
        """清除捕获上下文; 幂等, 可在任何收尾路径重复调用.

        interrupted 为真时先追加一条来源标记行 (无条目): 请求未正常提交即收尾,
        已捕获部分的 before 仍真实可还原, 但该点来源如实标为 interrupted (裁决 4),
        不因部分改动已提交就伪称请求完整完成。

        标记行写失败时不抛出 (收尾路径无从处理), 而是按会话挂起到下次成功写入时补记,
        期间同进程读取仍如实标 interrupted。挂起只在本进程内; 若索引持续不可写直到
        进程结束, 该点的中断标注会随挂起一起丢失 (索引是唯一落盘事实, 写失败时没有
        第二处可记), 这是本设计的已知残留。
        """
        with self._lock:
            context = self._context
            if context is not None and context.point_id is not None and (
                interrupted or context.noted_seq is None
            ):
                marker = _index_row(
                    context, row=ROW_SETTLE, point_id=context.point_id, kind=KIND_REQUEST,
                    commit_state="unknown", source="interrupted" if interrupted else "unknown",
                    entries=(), seq=None,
                    winerror=None,
                )
                try:
                    self._append_row(marker)
                except (OSError, RestoreStoreError):
                    # 标记写失败不能抛 (收尾路径无从处理), 但也不能就此伪称该点已提交:
                    # 挂起待本会话下次成功追加时补记, 期间读取方按挂起集合如实标注 (裁决 4)。
                    self._pending_markers.setdefault(context.session_id, []).append(marker)
            self._context = None
            self._maintain_retention()

    def _maintain_retention(self, *, protected: str | None = None) -> None:
        try:
            self._enforce_retention(self._warnings, protected=protected)
        except (OSError, RestoreStoreError) as exc:
            self._warnings.append(f"restore retention failed; materials retained: {exc}")

    def take_warnings(self) -> tuple[str, ...]:
        with self._lock:
            warnings = tuple(self._warnings)
            self._warnings.clear()
            return warnings

    @property
    def active_point_id(self) -> str | None:
        with self._lock:
            return self._context.point_id if self._context is not None else None

    def capture(self, *, path: str, op: str, before: bytes | None,
                after: bytes) -> CaptureTicket:
        """发布前落盘: before/after blob -> fsync -> 预行 append -> fsync (C2).

        同 (点, 路径) 只保留首份 before 与首份 op; 后续修改重复该 before/op 并更新 after.
        """
        with self._lock:
            context = self._context
            if context is None:
                raise RestoreStoreError(
                    "restore capture context is not set; a request must be open before "
                    "edit/write can capture restore points"
                )
            if op not in (OP_CREATE, OP_MODIFY):
                raise RestoreStoreError(f"unknown capture op {op!r}")
            # Windows 路径大小写不产生第二份 before。
            path = path.replace("/", "\\")
            path_key = path.casefold()
            previous = context.captured.get(path_key)
            if previous is None:
                if len(context.captured) >= MAX_ENTRIES_PER_POINT:
                    self._mark_availability(context.session_id, context.point_id, INCOMPLETE)
                    raise RestoreStoreError("restore point entry limit reached; file not modified")
                first_before = None if before is None else self._store_blob(before)
                previous = (op, first_before)
            after_sha = self._store_blob(after)
            active_blobs = self._inflight_blobs(context.session_id, context.point_id)
            active_blobs.update(sha for sha in (previous[1], after_sha) if sha is not None)
            if sum(self._blob_size(sha) or 0 for sha in active_blobs) > BLOB_TOTAL_LIMIT_BYTES:
                self._mark_availability(context.session_id, context.point_id, INCOMPLETE)
                raise RestoreStoreError("restore point blob limit reached; file not modified")
            if context.point_id is None:
                context.point_id = _new_point_id()
            entry = StoredEntry(path=path, op=previous[0], before_sha256=previous[1],
                                after_sha256=after_sha, size_bytes=len(after))
            self._append_row(_index_row(
                context, row=ROW_ENTRY, point_id=context.point_id, kind=KIND_REQUEST,
                commit_state="unknown", source="unknown", entries=(entry,), seq=None,
                winerror=None,
            ))
            context.captured[path_key] = previous
            self._maintain_retention(protected=context.point_id)
            return CaptureTicket(
                point_id=context.point_id, relative_path=path, op=previous[0],
                before_sha256=previous[1], after_sha256=after_sha, size_bytes=len(after),
            )

    def settle(self, ticket: CaptureTicket, *, commit_state: str,
               winerror: int | None = None) -> None:
        """发布在同一队列内落定后追加结算行; 失败时预行保持 unknown (不假报)."""
        if commit_state not in COMMIT_STATES:
            raise RestoreStoreError(f"unknown commit state {commit_state!r}")
        with self._lock:
            context = self._context
            if context is None:
                raise RestoreStoreError(
                    "cannot settle a restore point without an open request"
                )
            entry = StoredEntry(
                path=ticket.relative_path, op=ticket.op, before_sha256=ticket.before_sha256,
                after_sha256=ticket.after_sha256, size_bytes=ticket.size_bytes,
            )
            self._append_row(_index_row(
                context, row=ROW_SETTLE, point_id=ticket.point_id, kind=KIND_REQUEST,
                commit_state=commit_state,
                source="unknown",
                entries=(entry,), seq=None, winerror=winerror,
            ))

    def note_commit(self, seq: int | None) -> None:
        """请求 commit 取得真实 seq 后补记 (幂等; 不重写 entries, 崩溃遗留点保持 null).

        索引行的 seq 在补记前一律为 null; 补记追加一条无条目的结算行携带真实 seq,
        读取方按点取最后非空 seq。
        """
        with self._lock:
            context = self._context
            if context is None or context.point_id is None or seq is None:
                return
            if context.noted_seq == int(seq):
                return
            self._append_row(_index_row(
                context, row=ROW_SETTLE, point_id=context.point_id, kind=KIND_REQUEST,
                commit_state="committed", source="committed", entries=(), seq=int(seq),
                winerror=None,
            ))
            context.noted_seq = int(seq)

    # ---- 读取 (存储事实; 中立 DTO 由查看票投影) ----

    def read_points(self, session_id: str) -> tuple[StoredPoint, ...]:
        """本会话全部还原点, 新→旧; availability 按 blob 核验与条目上限如实标注."""
        with self._lock:
            session = _checked_session_id(session_id)
            rows = self._read_index(session)
            # 补记尚未落盘的来源标记也在本次读取生效: 索引缺行不等于该点已完成,
            # 否则读出的来源会与收尾事实相反 (裁决 4 要求中断如实标注)。
            pending = frozenset(
                marker["point_id"] for marker in self._pending_markers.get(session, ())
                if marker["source"] == "interrupted"
            )
            points = _points_from_rows(rows, self._blob_size, interrupted=pending)
        points.sort(key=lambda point: (point.created_at, point.point_id), reverse=True)
        return tuple(points)

    def read_blob(self, sha256: str) -> bytes:
        """按内容地址读取 blob; 缺失/越界/reparse 一律拒绝 (查看票的只读取数面)."""
        if not _is_sha256(sha256):
            raise RestoreStoreError(f"not a content address: {sha256!r}")
        target = self._blobs_dir() / sha256
        self._check_own(target)
        with target.open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise RestoreStoreError("blob exceeds the 16 MiB file limit")
        if _sha256(data) != sha256:
            raise RestoreStoreError(f"blob content does not match its name: {target}")
        return data

    # ---- 保留与清理 (C3) ----

    def maintenance(self) -> tuple[str, ...]:
        """启动机会式清理: 30 天前点 -> 保留限额逐出 -> 回收无引用 blob/临时件.

        只动 restore 根自有文件, 拒绝 reparse/symlink; 单个路径失败保留并告警,
        不换方式绕过. 清理失败不阻断启动, 故本方法不抛出。
        """
        warnings: list[str] = []
        with self._lock:
            if not self._restore_root().is_dir():
                return ()
            try:
                self._drop_stale_points(warnings)
                self._enforce_retention(warnings)
            except Exception as exc:  # 清理永不阻断启动; 事实以告警呈现
                warnings.append(f"{self._restore_root()}: maintenance aborted: {exc}")
        return tuple(warnings)

    def _drop_stale_points(self, warnings: list[str]) -> None:
        self._check_own(self._points_dir())
        cutoff = datetime.now(timezone.utc) - timedelta(days=POINT_MAX_AGE_DAYS)
        for path in sorted(self._points_dir().glob("*.jsonl")):
            try:
                self._check_own(path)
                rows = self._read_index(path.stem)
            except (OSError, RestoreStoreError) as exc:
                warnings.append(f"{path}: {exc}")
                continue
            for point in _points_from_rows(rows, self._blob_size):
                if datetime.fromisoformat(point.created_at) < cutoff and point.availability != EXPIRED:
                    try:
                        self._mark_availability(path.stem, point.point_id, EXPIRED)
                    except (OSError, RestoreStoreError) as exc:
                        warnings.append(f"{path}: {exc}")

    def _enforce_retention(self, warnings: list[str], *, protected: str | None = None) -> None:
        """每会话最近 100 点 + 工作区 blob 总量 ≤512 MiB; 逐出点保留索引行 (标 expired)."""
        retained: list[StoredPoint] = []
        self._check_own(self._points_dir())
        self._check_own(self._blobs_dir())
        complete = True
        for path in sorted(self._points_dir().glob("*.jsonl")):
            try:
                available = [point for point in self.read_points(path.stem)
                             if point.availability != EXPIRED]
                retained.extend(available[:POINT_LIMIT_PER_SESSION])
                for point in available[POINT_LIMIT_PER_SESSION:]:
                    self._mark_availability(path.stem, point.point_id, EXPIRED)
            except (OSError, RestoreStoreError) as exc:
                # 读不出的索引不等于没有引用: 此时回收 blob 会把仍被引用的内容删掉,
                # 故放弃本次回收, 只如实告警 (逐出判定也不完整, 一并放弃)。
                complete = False
                warnings.append(f"{path}: {exc}")
        if not complete:
            for stale in sorted(self._points_dir().glob(".tmp-*")):
                self._delete_own_file(stale, warnings)
            return
        # 超限从最旧点逐出: 用量按"保留集中 distinct blob"计, 共享 blob 只算一次。
        # C3 约束的是工作区 blob 总量, 而按各点 total_bytes 累加会把被 N 个点共享的
        # blob 重复计费, 于是远未到限额就逐出 (全共享时逐出更不释放任何字节)。
        retained.sort(key=lambda point: (point.created_at, point.point_id))
        counts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        point_blobs = {
            point.point_id: (self._inflight_blobs(point.session_id, point.point_id)
                             if point.point_id == protected else _point_blobs(point))
            for point in retained
        }
        for point in retained:
            for sha in point_blobs[point.point_id]:
                counts[sha] = counts.get(sha, 0) + 1
                if sha not in sizes:
                    sizes[sha] = self._blob_size(sha) or 0
        total = sum(sizes.values())
        for point in retained:
            if total <= BLOB_TOTAL_LIMIT_BYTES:
                break
            if point.point_id == protected:
                continue
            try:
                self._mark_availability(point.session_id, point.point_id, EXPIRED)
            except (OSError, RestoreStoreError) as exc:
                warnings.append(f"{point.point_id}: {exc}")
                return  # 标记失败不得回收仍可能可用的材料
            for sha in point_blobs[point.point_id]:
                counts[sha] -= 1
                if counts[sha] == 0:
                    total -= sizes[sha]
        referenced = {sha for sha, count in counts.items() if count > 0}
        for target in sorted(self._blobs_dir().glob("*")):
            if not _is_sha256(target.name) and not target.name.startswith(".tmp-"):
                continue
            if target.name in referenced:
                continue
            self._delete_own_file(target, warnings)
        for stale in sorted(self._points_dir().glob(".tmp-*")):
            self._delete_own_file(stale, warnings)

    def _mark_availability(self, session_id: str, point_id: str | None, state: str) -> None:
        if point_id is None:
            return
        rows = self._read_index(session_id)
        head = next((row for row in rows if row["point_id"] == point_id), None)
        if head is None:
            raise RestoreStoreError("cannot mark a missing restore point")
        marker = {**head, "row": ROW_SETTLE, "entries": [], "seq": None,
                  "commit_state": "unknown", "source": "unknown", "winerror": None,
                  "availability": state}
        self._append_row(marker)

    def _inflight_blobs(self, session_id: str, point_id: str | None) -> set[str]:
        # 预行还未结算时，之前成功发布的 after 仍可能是 not_written 后的回退目标。
        # 请求收尾前保留该点所有行引用，避免捕获下一次修改时提前回收它。
        return {
            sha for row in self._read_index(session_id) if row["point_id"] == point_id
            for entry in row["entries"]
            for sha in (entry["before_sha256"], entry["after_sha256"]) if sha is not None
        }

    # ---- 存储细节 ----

    def _restore_root(self) -> Path:
        return self._root / ".cicada" / "restore"

    def _blobs_dir(self) -> Path:
        return self._restore_root() / "blobs"

    def _points_dir(self) -> Path:
        return self._restore_root() / "points"

    def _index_path(self, session_id: str) -> Path:
        return self._points_dir() / f"{_checked_session_id(session_id)}.jsonl"

    def _check_own(self, path: Path) -> None:
        """只接受 restore 根内、且链上无 reparse/symlink 的路径."""
        root = self._restore_root()
        if path != root and root not in path.parents:
            raise RestoreStoreError(f"path is outside the restore root: {path}")
        reparse = _reparse_in_chain(self._root, path)
        if reparse is not None:
            raise RestoreStoreError(f"refusing to follow a reparse point: {reparse}")

    def _ensure_own_dirs(self) -> None:
        for path in (self._restore_root(), self._blobs_dir(), self._points_dir()):
            self._check_own(path)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RestoreStoreError(f"cannot create {path}: {exc}") from exc
            self._check_own(path)

        # 自有数据根内的忽略文件避免改用户 Git 配置/索引，也适用于全新工作区。
        ignore = self._restore_root() / ".gitignore"
        self._check_own(ignore)
        if ignore.exists():
            if ignore.read_bytes() != b"*\n":
                raise RestoreStoreError("restore data ignore file has unexpected content")
        else:
            with open(ignore, "xb") as handle:
                handle.write(b"*\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _store_blob(self, data: bytes) -> str:
        """内容寻址落盘: 临时名 + 同卷原子 rename; 同名先核验内容再复用 (幂等)."""
        digest = _sha256(data)
        self._ensure_own_dirs()
        target = self._blobs_dir() / digest
        self._check_own(target)
        if target.is_file():
            try:
                if target.stat().st_size == len(data) and _sha256(target.read_bytes()) == digest:
                    return digest
            except OSError:
                pass
        temporary = self._blobs_dir() / f".tmp-{uuid.uuid4().hex}"
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(temporary, flags)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        return digest

    def _append_row(self, row: dict) -> None:
        """追加一行严格 JSON 并 fsync; 失败向上抛, 由捕获路径映射为 not_written."""
        path = self._index_path(row["session_id"])
        self._ensure_own_dirs()
        self._check_own(path)
        self._write_line(path, row)
        # 本会话此前挂起的收尾来源标记借这次成功写入一并补记 (裁决 4 的如实标注);
        # 补记失败不影响本次追加的结果, 挂起保留到下次。
        self._flush_pending_markers(row["session_id"])

    def _write_line(self, path: Path, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        try:
            # 崩溃可能留下缺尾换行的半行; 先补回行边界, 否则本次追加会与它粘成一行,
            # 被读取方整行丢弃——预行的落盘证据会静默消失, 发布出去的改动就没有还原点。
            with open(path, "a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                handle.write((line + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise RestoreStoreError(f"cannot append the restore index {path}: {exc}") from exc

    def _flush_pending_markers(self, session_id: str) -> None:
        """把挂起的来源标记补写进索引; 失败保留挂起 (读取方仍按挂起集合标注)."""
        pending = self._pending_markers.get(session_id)
        if not pending:
            return
        try:
            path = self._index_path(session_id)
            self._ensure_own_dirs()
            self._check_own(path)
            for marker in list(pending):
                self._write_line(path, marker)
        except (OSError, RestoreStoreError):
            # 可能已部分写入, 故保留全部挂起: 重写属幂等 (空条目结算行只携带来源)。
            return
        self._pending_markers.pop(session_id, None)

    def _read_index(self, session_id: str) -> list[dict]:
        path = self._index_path(session_id)
        self._check_own(path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise RestoreStoreError(f"cannot read the restore index {path}: {exc}") from exc
        rows: list[dict] = []
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            row = _decode_row(line)
            if row is not None:
                rows.append(row)
            else:
                raise RestoreStoreError(f"unreadable restore index row: {path}; materials retained")
        return rows

    def _delete_own_file(self, path: Path, warnings: list[str]) -> None:
        try:
            self._check_own(path)
            if path.parent not in (self._blobs_dir(), self._points_dir()):
                raise RestoreStoreError(f"refusing to delete outside the restore files: {path}")
            path.unlink(missing_ok=True)
        except (OSError, RestoreStoreError) as exc:
            warnings.append(f"{path}: {exc}")

    def _blob_size(self, sha256: str) -> int | None:
        """blob 实际字节数; 缺失/不可读返回 None (调用方标 expired, 不伪装可还原)."""
        try:
            path = self._blobs_dir() / sha256
            self._check_own(path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
                return None
            return len(self.read_blob(sha256))
        except (OSError, RestoreStoreError):
            return None


def _index_row(context: _RequestContext, *, row: str, point_id: str, kind: str,
               commit_state: str, source: str, entries: tuple[StoredEntry, ...],
               seq: int | None, winerror: int | None) -> dict:
    record = {
        "version": INDEX_VERSION, "row": row, "point_id": point_id, "kind": kind,
        "request_id": context.request_id, "seq": seq, "epoch_id": context.epoch_id,
        "session_id": context.session_id, "created_at": _now_iso(), "source": source,
        "commit_state": commit_state,
        "entries": [
            {"path": entry.path, "op": entry.op, "before_sha256": entry.before_sha256,
             "after_sha256": entry.after_sha256, "size_bytes": entry.size_bytes}
            for entry in entries
        ],
        "links": None,
    }
    # winerror 只属结算行 (C3 行字段): 预行在发布前落盘, 此时尚无发布结果
    if row == ROW_SETTLE:
        record["winerror"] = winerror
    return record


def _points_from_rows(rows: list[dict], blob_size,
                      interrupted: frozenset[str] = frozenset()) -> list[StoredPoint]:
    """按 point_id 聚合: 同(点,路径)取最后一行, not_written 结算行的条目不出现.

    interrupted 是尚未补记落盘的收尾来源标记所在点: 这些点的来源强制标 interrupted,
    否则索引缺行会让中断请求被读成已完成 (标记写失败属收尾期 IO 故障, 不得 fail-open)。
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["point_id"], []).append(row)
    points: list[StoredPoint] = []
    for point_id, group in grouped.items():
        head = group[0]
        entries: dict[str, StoredEntry] = {}
        # 每条存活条目的来源行状态: 点级 commit_state 由最终暴露的条目决定, 否则会出现
        # "点标 not_written 却仍列出已提交条目"的自相矛盾 (多个改动同一请求内成败混合时)。
        entry_state: dict[str, str] = {}
        # 已确认生效的条目 (committed 结算行给出): not_written 结算行表示该次改动没有
        # 发生, 故撤掉它自己的条目——若同一路径此前已有确认生效的改动, 回退到那一份,
        # 不能连先前真实发生的修改一起丢掉 (否则请求确实改过的文件反而不可还原)。
        confirmed: dict[str, StoredEntry] = {}
        seq: int | None = None
        # source: interrupted 一旦出现即吸收 (中断是点的事实, 不得被此前的 committed
        # 结算行掩盖, 裁决 4); 否则出现 committed 即为 committed; 只有预行时为 unknown。
        source = "unknown"
        # winerror 分两个口: 只有"点的可还原条目"来源行 (或点内无条目时) 的失败码才写进
        # 点级 DTO——否则会出现"点标 committed 却带 183"这种与自身条目相反的画面;
        # 被撤掉的 not_written 条目其失败码仍留在索引行供取证, 只是不进点级标签。
        winerror: int | None = None
        winerror_any: int | None = None
        availability_marker: str | None = None
        for row in group:
            if row.get("availability") == EXPIRED:
                availability_marker = EXPIRED
            elif row.get("availability") == INCOMPLETE and availability_marker != EXPIRED:
                availability_marker = INCOMPLETE
            if row["seq"] is not None:
                seq = row["seq"]
            if row["source"] == "interrupted":
                source = "interrupted"
            elif row["source"] == "committed" and source != "interrupted":
                source = "committed"
            if row["row"] == ROW_SETTLE and row.get("winerror") is not None:
                # 补记行不带 winerror: 只被真实携带值的行覆盖, 不把已有失败码抹成 None。
                winerror_any = row["winerror"]
            not_written = row["commit_state"] == "not_written"
            for item in row["entries"]:
                path = item["path"].replace("/", "\\")
                key = path.casefold()
                if key in entries:
                    path = entries[key].path
                if not_written:
                    previous = confirmed.get(key)
                    if previous is None:
                        entries.pop(key, None)
                        entry_state.pop(key, None)
                    else:
                        entries[key] = previous
                        entry_state[key] = "committed"
                    continue
                entry = StoredEntry(
                    path=path, op=item["op"], before_sha256=item["before_sha256"],
                    after_sha256=item["after_sha256"], size_bytes=item["size_bytes"],
                )
                entries[key] = entry
                entry_state[key] = row["commit_state"]
                if row["row"] == ROW_SETTLE and row.get("winerror") is not None:
                    winerror = row["winerror"]
                if row["row"] == ROW_SETTLE and row["commit_state"] == "committed":
                    confirmed[key] = entry
        ordered = tuple(entries.values())[:MAX_ENTRIES_PER_POINT]
        # 尚未补记落盘的收尾标记 (标记写失败): 索引缺行不得让中断读成已完成。
        if point_id in interrupted:
            source = "interrupted"
        # 条目全部来自已提交结算行才算 committed; 有仅预行 (unknown) 的条目则为 unknown;
        # 无条目时退回行序判定 (如只有 not_written 行或被中断的空点)。
        states = {entry_state[entry.path.casefold()] for entry in ordered}
        if ordered:
            commit_state = "committed" if states == {"committed"} else "unknown"
        else:
            commit_state = _point_commit_state(group)
            winerror = winerror_any
        total = 0
        expired = False
        for sha in {sha for entry in ordered for sha in (entry.before_sha256, entry.after_sha256)
                    if sha is not None}:
            size = blob_size(sha)
            if size is None:
                expired = True
            else:
                total += size
        if expired or availability_marker == EXPIRED:
            availability = EXPIRED
        elif len(entries) > MAX_ENTRIES_PER_POINT or availability_marker == INCOMPLETE:
            availability = INCOMPLETE
        else:
            availability = AVAILABLE
        points.append(StoredPoint(
            point_id=point_id, kind=head["kind"], request_id=head["request_id"], seq=seq,
            epoch_id=head["epoch_id"], session_id=head["session_id"],
            created_at=head["created_at"], source=source,
            commit_state=commit_state, availability=availability,
            entries=ordered, total_bytes=total, winerror=winerror,
            links=tuple(head["links"] or ()),
        ))
    return points


def _point_blobs(point: StoredPoint) -> tuple[str, ...]:
    """该点引用的 distinct blob (before/after 去重), 供保留用量与回收判定."""
    return tuple({
        sha for entry in point.entries
        for sha in (entry.before_sha256, entry.after_sha256) if sha is not None
    })


def _point_commit_state(group: list[dict]) -> str:
    """点级提交状态取最后一个带条目的结算行; 只有预行时为 unknown."""
    state = "unknown"
    for row in group:
        if row["entries"] and row["commit_state"] != "unknown":
            state = row["commit_state"]
    return state


def restore_plugin() -> PluginDefinition:
    """coding-restore 插件: 依赖 coding.workspace, 提供 coding.restore 捕获汇.

    只有持久会话 + restore_capture 开启的组合才装配; 非持久/关闭捕获时不安装
    捕获汇, edit/write 行为与既有完全一致 (C2)。
    """

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        store = RestoreStore(workspace.root)
        workspace.install_restore_capture(store)
        ctx.provide(RESTORE_CAPABILITY, store)

    return PluginDefinition(
        name="coding-restore",
        setup=setup,
        provides=frozenset({RESTORE_CAPABILITY}),
        requires=frozenset({"coding.workspace"}),
    )
