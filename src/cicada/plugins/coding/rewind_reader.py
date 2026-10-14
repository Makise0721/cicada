"""还原点只读投影与有界 diff；由应用拥有线程和读取 lease。"""

from __future__ import annotations

import difflib
import hashlib
import uuid
from datetime import datetime, timezone

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.restore_store import RestoreStore, RestoreStoreError, StoredPoint
from cicada.plugins.coding.workspace import Workspace, WorkspaceError
from cicada.restore_points import (
    FILE_LIMIT, PREVIEW_LIMIT, RewindEntry, RewindError, RewindListView,
    RewindPointSummary, RewindPointView, RewindPreview,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RewindReader:
    def __init__(self, store: RestoreStore, workspace: Workspace, session_id: str,
                 epoch_id: str | None) -> None:
        self.store, self.workspace = store, workspace
        self.session_id, self.epoch_id = session_id, epoch_id
        self.points: tuple[StoredPoint, ...] = ()
        self.targets: dict[str, tuple[str, int]] = {}

    def capture(self, cancel: CancelToken) -> RewindListView:
        cancel.throw_if_cancelled()
        self.points = self.store.read_points(self.session_id)
        summaries = []
        for point in self.points:
            cancel.throw_if_cancelled()
            notes = []
            if point.epoch_id != self.epoch_id:
                notes.append("历史代次")
            if point.commit_state == "unknown":
                notes.append("提交结果未知")
            summaries.append(RewindPointSummary(
                point.point_id, point.kind, point.request_id, point.seq, point.epoch_id,
                point.created_at, point.source, point.availability, len(point.entries),
                point.total_bytes, "；".join(notes) or None))
        return RewindListView(now(), str(self.workspace.root), tuple(summaries))

    def _point(self, point_id: str) -> StoredPoint:
        point = next((p for p in self.points if p.point_id == point_id), None)
        if point is None:
            raise RewindError("invalid_target", "还原点不属于当前页面，请刷新。")
        return point

    def read_point(self, point_id: str, cancel: CancelToken) -> RewindPointView:
        point = self._point(point_id)
        self.targets.clear()
        entries = []
        for index, entry in enumerate(point.entries):
            cancel.throw_if_cancelled()
            relation, later_seq = self._relation(point, entry.path, entry.after_sha256, cancel)
            target = uuid.uuid4().hex
            self.targets[target] = point_id, index
            entries.append(RewindEntry(target, entry.path, entry.op, entry.size_bytes,
                                       entry.before_sha256, entry.after_sha256, relation, later_seq))
        return RewindPointView(point_id, now(), point.availability, tuple(entries))

    def _relation(self, point, path, after, cancel):
        try:
            target = self.workspace.resolve_within_root(path)
            with target.open("rb") as stream:
                digest = hashlib.sha256()
                total = 0
                while chunk := stream.read(64 * 1024):
                    cancel.throw_if_cancelled()
                    total += len(chunk)
                    if total > FILE_LIMIT:
                        return "unknown", None
                    digest.update(chunk)
            current = digest.hexdigest()
        except FileNotFoundError:
            return ("missing" if after is not None else "clean"), None
        except (OSError, WorkspaceError):
            return "unknown", None
        if current == after:
            return "clean", None
        # 列表新→旧；反转后从该点之后的最早一点开始匹配。
        later = self.points[:self.points.index(point)]
        for successor in reversed(later):
            cancel.throw_if_cancelled()
            if any(e.path.casefold() == path.casefold() and e.after_sha256 == current
                   for e in successor.entries):
                return "later_request", successor.seq
        return "external", None

    def preview(self, point_id: str, entry_id: str, cancel: CancelToken) -> RewindPreview:
        point = self._point(point_id)
        registered = self.targets.get(entry_id)
        if registered is None or registered[0] != point_id:
            raise RewindError("invalid_target", "文件不属于当前还原点，请重新选择。")
        entry = point.entries[registered[1]]
        def result(state, text="", notice=None):
            return RewindPreview(point_id, entry_id, state, text, notice, now())
        cancel.throw_if_cancelled()
        if point.availability == "expired":
            return result("unavailable", notice="还原点已清理，正文不可用")
        try:
            before = self.store.read_blob(entry.before_sha256) if entry.before_sha256 else b""
            after = self.store.read_blob(entry.after_sha256) if entry.after_sha256 else b""
        except (OSError, RestoreStoreError):
            return result("unavailable", notice="blob 缺失或完整性核验失败，请刷新")
        cancel.throw_if_cancelled()
        if max(len(before), len(after)) > FILE_LIMIT:
            return result("unavailable", notice="文件超过 16 MiB 读取上限")
        if b"\0" in before or b"\0" in after:
            return result("unavailable", notice="二进制文件，仅显示条目元数据")
        try:
            old, new = before.decode("utf-8"), after.decode("utf-8")
        except UnicodeError:
            return result("unavailable", notice="非 UTF-8 文件，仅显示条目元数据")
        # 一行也可能超上限；逐块截断而非先构造整个 diff 正文。
        body = bytearray()
        truncated = False
        for line in difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                         fromfile="before/" + entry.path,
                                         tofile="after/" + entry.path):
            cancel.throw_if_cancelled()
            if not line.endswith("\n"):
                line += "\n\\ No newline at end of file\n"
            encoded = line.encode("utf-8")
            room = PREVIEW_LIMIT - len(body)
            body.extend(encoded[:room])
            if len(encoded) > room:
                truncated = True
                break
        notice = "提交结果未知；after 为捕获候选内容" if point.commit_state == "unknown" else None
        if truncated:
            notice = "预览已截断至 256 KiB" + ("；" + notice if notice else "")
        return result("truncated" if truncated else "complete", body.decode("utf-8", errors="ignore"), notice)
