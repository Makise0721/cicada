"""还原计划与 Windows 文件发布；应用拥有任务及准入，UI 只持有不可变投影。"""

from __future__ import annotations

import ctypes
import hashlib
import time
import uuid
from ctypes import wintypes

from cicada.core.cancel import CancelToken
from cicada.plugins.coding import file_commit_windows as native
from cicada.plugins.coding.rewind_reader import RewindReader, now
from cicada.plugins.coding.restore_store import BLOB_TOTAL_LIMIT_BYTES, RestoreStoreError
from cicada.plugins.coding.workspace import WorkspaceError
from cicada.restore_points import (
    FILE_LIMIT, RewindError, RewindRestoreFileResult, RewindRestorePath,
    RewindRestorePlan, RewindRestoreResult,
)


class RewindRestorer:
    def __init__(self, reader: RewindReader) -> None:
        self.reader = reader
        self.plan: RewindRestorePlan | None = None
        self._created = 0.0

    def _current(self, path: str) -> bytes | None:
        workspace = self.reader.workspace
        target = workspace.resolve_for_write(path)
        original = workspace.resolve(path)
        exists = target.exists()
        native.check_mutation_target(original, target, creating=not exists)
        if not exists:
            return None
        with target.open("rb") as stream:
            data = stream.read(FILE_LIMIT + 1)
        if len(data) > FILE_LIMIT:
            raise RewindError("blocked", "受影响文件超过 16 MiB，无法备份。")
        return data

    @staticmethod
    def _sha(data: bytes | None) -> str | None:
        return hashlib.sha256(data).hexdigest() if data is not None else None

    def prepare(self, point_id: str, cancel: CancelToken) -> RewindRestorePlan:
        self.plan = None
        point = self.reader._point(point_id)
        current_point = next((p for p in self.reader.store.read_points(self.reader.session_id)
                              if p.point_id == point_id), None)
        if current_point is None or current_point.availability == "expired" or not point.entries:
            raise RewindError("unavailable", "还原点已清理或无可还原条目。")
        paths, deletions, impacts = [], [], []
        try:
            for entry in point.entries:
                cancel.throw_if_cancelled()
                current = self._current(entry.path)
                if entry.before_sha256 is not None:
                    self.reader.store.read_blob(entry.before_sha256)
                relation, seq = self.reader._relation(point, entry.path, entry.after_sha256, cancel)
                op = "delete" if entry.before_sha256 is None else "restore"
                paths.append(RewindRestorePath(entry.path, op, self._sha(current), relation, seq))
                if op == "delete":
                    deletions.append(entry.path)
                if relation in ("later_request", "external", "unknown"):
                    impacts.append(f"{entry.path}: " + (f"后继请求 #{seq}" if relation == "later_request"
                                                         else "请求外改动"))
        except (OSError, WorkspaceError, native.UnsupportedTargetError, RestoreStoreError) as exc:
            raise RewindError("blocked", "目标不可安全还原或备份材料不可读，请刷新。") from exc
        self.plan = RewindRestorePlan(uuid.uuid4().hex, point_id, now(), tuple(paths),
                                      tuple(deletions), tuple(impacts))
        self._created = time.monotonic()
        return self.plan

    def execute(self, plan_id: str, cancel: CancelToken) -> RewindRestoreResult:
        plan = self.plan
        self.plan = None  # 单次确认权，失败或取消也必须重新计划。
        if plan is None or plan.plan_id != plan_id:
            raise RewindError("invalid_target", "还原计划已失效，请重新计划。")
        if time.monotonic() - self._created > 300:
            raise RewindError("stale", "还原计划已过期，请重新计划。")
        point = self.reader._point(plan.point_id)
        materials = []
        blob_sizes = {}
        try:
            for item, entry in zip(plan.paths, point.entries, strict=True):
                cancel.throw_if_cancelled()
                current = self._current(item.relative_path)
                if self._sha(current) != item.current_sha256:
                    raise RewindError("stale", "文件内容已变化，请重新计划并确认。")
                desired = self.reader.store.read_blob(entry.before_sha256) if entry.before_sha256 else None
                for data in (current, desired):
                    if data is not None:
                        blob_sizes[self._sha(data)] = len(data)
                if sum(blob_sizes.values()) > BLOB_TOTAL_LIMIT_BYTES:
                    raise RestoreStoreError("restore backup exceeds blob quota")
                materials.append((item.relative_path, current, desired))
            cancel.throw_if_cancelled()
            backup = self.reader.store.backup_restore(self.reader.session_id, self.reader.epoch_id,
                                                     point.point_id, tuple(materials))
        except RewindError:
            raise
        except (OSError, WorkspaceError, native.UnsupportedTargetError, RestoreStoreError):
            return RewindRestoreResult(point.point_id, None, tuple(
                RewindRestoreFileResult(p.relative_path, "failed", notice="备份失败，未执行还原")
                for p in plan.paths), "备份失败，未执行任何工作区写入")
        results = []
        for path, current, desired in materials:
            if cancel.cancelled:
                results.append(RewindRestoreFileResult(path, "cancelled"))
                continue
            try:
                # 备份完成后、每次发布前再核对，拒绝覆盖已漂移的目标。
                if self._sha(self._current(path)) != self._sha(current):
                    raise RewindError("stale", "文件在备份后变化，未发布")
                state = self._publish(path, current, desired, cancel)
                results.append(RewindRestoreFileResult(path, state))
            except (OSError, WorkspaceError, native.UnsupportedTargetError,
                    native.NativePublishError, RewindError) as exc:
                results.append(RewindRestoreFileResult(path, "failed", getattr(exc, "winerror", None),
                                                       str(exc)[:512]))
        notice = None
        try:
            self.reader.store.finish_restore(self.reader.session_id, backup, tuple(results))
        except (OSError, RestoreStoreError):
            notice = "执行事实追加失败；备份点已保留，逐文件结果以本次显示为准"
        return RewindRestoreResult(point.point_id, backup, tuple(results), notice)

    def _publish(self, path: str, current: bytes | None, desired: bytes | None,
                 cancel: CancelToken) -> str:
        target = self.reader.workspace.resolve_for_write(path)
        if desired is None:
            if cancel.cancelled:
                return "cancelled"
            if current is not None:
                if self._current(path) != current:
                    raise RewindError("stale", "删除前文件变化，未删除")
                delete = ctypes.WinDLL("kernel32", use_last_error=True).DeleteFileW
                delete.argtypes, delete.restype = [wintypes.LPCWSTR], wintypes.BOOL
                if not delete(str(target)):
                    raise native.NativePublishError("DeleteFileW", ctypes.get_last_error())
            return "deleted"
        # 缺失文件恢复用 MoveFileExW 创建；存在文件沿用 ReplaceFileW。
        marker = uuid.uuid4().hex
        staged = target.with_name(f".cicada-restore-{marker}.tmp")
        backup = target.with_name(f".cicada-restore-{marker}.bak")
        native.write_staged_file(staged, desired)
        if cancel.cancelled:
            staged.unlink()
            return "cancelled"
        try:
            # fsync 可能长时间阻塞；发布前重新做完整目标约束与内容核对。
            if self._sha(self._current(path)) != self._sha(current):
                raise RewindError("stale", "暂存期间文件变化，未发布")
            original_identity = target.stat() if current is not None else None
        except BaseException:
            staged.unlink()
            raise
        if current is None:
            native.create_file(target, staged)
        else:
            native.replace_file(target, staged, backup)
            info = backup.stat()
            with backup.open("rb") as stream:
                old = stream.read(FILE_LIMIT + 1)
            if ((info.st_dev, info.st_ino) != (original_identity.st_dev, original_identity.st_ino)
                    or old != current):
                raise RewindError("unavailable", f"已发布但原件核验冲突；保留材料：{backup}")
        if self._current(path) != desired:
            raise RewindError("unavailable", f"已发布但当前内容核验失败；保留材料：{backup}")
        # 只清理确认发布成功的本次材料；原生失败材料保留供核对。
        if backup.exists():
            try:
                backup.unlink()
            except OSError:
                pass
        return "restored"
