"""还原执行的长期关键接缝：原生发布/删除/撤销、漂移拒绝、备份失败与取消事实。"""

import asyncio
import ctypes
import json
import threading
from dataclasses import replace
from ctypes import wintypes

import pytest

from cicada.application import RewindRestoreCompleted, open_application
from cicada.core.cancel import CancelToken
from cicada.plugins.coding.rewind_restore import RewindRestorer
from cicada.plugins.coding import file_commit_windows as native
from cicada.restore_points import RewindError
from cicada.ui.conversation_view import build
from test_persistence_application import _config
from test_review_checks import _plan
from tests.support.app import repo, script


def edits():
    return script(
        {"tool_calls": [{"id": "r", "name": "read", "arguments": {"path": "sample.py"}}]},
        {"tool_calls": [
            {"id": "e", "name": "edit", "arguments": {"path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]}},
            {"id": "w", "name": "write", "arguments": {"path": "new.txt", "content": "new"}}]},
        {"text": "done", "stop": True})


async def test_native_restore_confirmation_undo_and_receipt_freshness(tmp_path):
    root = repo(tmp_path)
    session = await open_application(replace(_config(root, edits(), session_id=None), plan=_plan(root)))
    view = build(session)["view"]
    updates = []
    session.subscribe(updates.append)
    try:
        await session.execute("modify and create")
        assert (root / "sample.py").read_bytes() == b"VALUE = 2\r\n"
        assert (root / "new.txt").read_bytes() == b"new"
        # 使用真实离线 check 回执证明同一 freshness 机制，不另造失效标记。
        await session._service.run_check("check-1", CancelToken())
        before = await session._service.refresh(CancelToken())
        assert before.checks[-1].freshness == "current"
        history, budget = session.history, session.budget
        receipts = session._service.receipts
        page = view.rewind_view
        assert page.open()
        await page._task
        point_id = page.snapshot.points[0].point_id
        assert page.execute(page._action("point", point_id))
        await page._task
        assert page.execute(page._action("plan", point_id))
        await page._task
        assert page.plan.deletions == ("new.txt",)
        assert (root / "new.txt").exists()  # 计划阶段零写
        assert page.execute(page._action("confirm", page.plan.plan_id))
        await page._task
        result = page.result
        assert {r.state for r in result.files} == {"restored", "deleted"}
        assert (root / "sample.py").read_bytes() == b"VALUE = 1\r\n"
        assert not (root / "new.txt").exists()
        workspace = session._app.runtime.capability("coding.workspace")
        assert not workspace.files._registry
        after = await session._service.refresh(CancelToken())
        assert after.checks[-1].freshness == "stale"
        assert session._service.receipts == receipts
        assert session.history == history and session.budget == budget
        assert sum(isinstance(u, RewindRestoreCompleted) for u in updates) == 1
        assert "既有检查回执已过期" in "\n".join(view.document_lines(100))
        session.cancel_rewind()
        await session.open_rewind()
        plan = await session.plan_rewind_restore(result.backup_point_id)
        undo = await session.execute_rewind_restore(plan.plan_id)
        assert all(r.state == "restored" for r in undo.files)
        assert (root / "sample.py").read_bytes() == b"VALUE = 2\r\n"
        assert (root / "new.txt").read_bytes() == b"new"
    finally:
        view.close()
        session.cancel_rewind()
        await session.aclose()


@pytest.mark.parametrize("window", ["staging_drift", "staging_cancel", "commit_drift"])
async def test_restore_publication_windows_preserve_materials(tmp_path, monkeypatch, window):
    root = repo(tmp_path)
    session = await open_application(_config(root, edits(), session_id=None))
    try:
        await session.execute("modify and create")
        listing = await session.open_rewind()
        plan = await session.plan_rewind_restore(listing.points[0].point_id)
        original_stage, original_replace = native.write_staged_file, native.replace_file
        def stage(target, data):
            original_stage(target, data)
            if window == "staging_drift":
                (root / "sample.py").write_bytes(b"external")
            elif window == "staging_cancel":
                session._rewind_token.cancel()
        def publish(target, staged, backup):
            if window == "commit_drift":
                target.write_bytes(b"last-window-external")
            original_replace(target, staged, backup)
        monkeypatch.setattr(native, "write_staged_file", stage)
        monkeypatch.setattr(native, "replace_file", publish)
        result = await session.execute_rewind_restore(plan.plan_id)
        if window == "staging_cancel":
            assert [r.state for r in result.files] == ["cancelled", "cancelled"]
            assert (root / "sample.py").read_bytes() == b"VALUE = 2\r\n"
        elif window == "staging_drift":
            assert result.files[0].state == "failed"
            assert (root / "sample.py").read_bytes() == b"external"
        else:
            assert result.files[0].state == "failed" and "已发布" in result.files[0].notice
            backups = list(root.glob(".cicada-restore-*.bak"))
            assert len(backups) == 1 and backups[0].read_bytes() == b"last-window-external"
        assert not list(root.glob(".cicada-restore-*.tmp"))
    finally:
        session.cancel_rewind()
        await session.aclose()


async def test_restore_stale_latches_backup_failure_and_missing_tombstone(tmp_path, monkeypatch):
    root = repo(tmp_path)
    session = await open_application(_config(root, edits(), session_id=None))
    try:
        await session.execute("modify and create")
        listing = await session.open_rewind()
        point_id = listing.points[0].point_id
        for field in ("_process_uncertain", "_storage_failed", "_view_only_reason"):
            setattr(session, field, True)
            with pytest.raises(RewindError) as error:
                await session.plan_rewind_restore(point_id)
            assert error.value.kind == "blocked"
            setattr(session, field, False if field == "_process_uncertain" else None)
        plan = await session.plan_rewind_restore(point_id)
        (root / "sample.py").write_bytes(b"external")
        with pytest.raises(RewindError) as error:
            await session.execute_rewind_restore(plan.plan_id)
        assert error.value.kind == "stale" and (root / "new.txt").exists()
        plan = await session.plan_rewind_restore(point_id)
        assert plan.successor_impacts
        store = session._restore_store
        original = store.backup_restore
        def fail(*args):
            raise OSError("fsync failed")
        monkeypatch.setattr(store, "backup_restore", fail)
        failed = await session.execute_rewind_restore(plan.plan_id)
        assert failed.backup_point_id is None and all(r.state == "failed" for r in failed.files)
        assert (root / "sample.py").read_bytes() == b"external" and (root / "new.txt").exists()
        monkeypatch.setattr(store, "backup_restore", original)
        (root / "sample.py").unlink()
        plan = await session.plan_rewind_restore(point_id)
        restored = await session.execute_rewind_restore(plan.plan_id)
        session.cancel_rewind()
        await session.open_rewind()
        plan = await session.plan_rewind_restore(restored.backup_point_id)
        await session.execute_rewind_restore(plan.plan_id)
        assert not (root / "sample.py").exists()  # tombstone 撤销恢复出的文件
        assert (root / "new.txt").read_bytes() == b"new"
    finally:
        session.cancel_rewind()
        await session.aclose()


async def test_restore_cancel_keeps_published_and_owns_worker_until_settled(tmp_path, monkeypatch):
    root = repo(tmp_path)
    session = await open_application(_config(root, edits(), session_id=None))
    entered, release = threading.Event(), threading.Event()
    original = RewindRestorer._publish
    updates = []
    session.subscribe(updates.append)
    def blocked(self, *args):
        state = original(self, *args)
        entered.set()
        assert release.wait(5)
        return state
    monkeypatch.setattr(RewindRestorer, "_publish", blocked)
    try:
        await session.execute("modify and create")
        listing = await session.open_rewind()
        plan = await session.plan_rewind_restore(listing.points[0].point_id)
        task = asyncio.create_task(session.execute_rewind_restore(plan.plan_id))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.rewind_active
        close = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await close
        result = next(u.result for u in updates if isinstance(u, RewindRestoreCompleted))
        assert [r.state for r in result.files] == ["restored", "cancelled"]
        assert (root / "sample.py").read_bytes() == b"VALUE = 1\r\n"
        assert (root / "new.txt").exists()
        index = root / ".cicada/restore/points" / f"{session.persistence.session_id}.jsonl"
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        assert rows[-1]["restore_results"][1]["state"] == "cancelled"
        assert not session.rewind_active
    finally:
        release.set()
        session.cancel_rewind()
        await session.aclose()


async def test_restore_delete_locked_file_reports_native_failure(tmp_path):
    root = repo(tmp_path)
    session = await open_application(_config(root, edits(), session_id=None))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    handle = None
    try:
        await session.execute("modify and create")
        listing = await session.open_rewind()
        plan = await session.plan_rewind_restore(listing.points[0].point_id)
        handle = create(str(root / "new.txt"), 0x80000000, 1, None, 3, 0, None)
        assert handle not in (None, ctypes.c_void_p(-1).value)
        result = await session.execute_rewind_restore(plan.plan_id)
        assert result.files[0].state == "restored"
        assert (result.files[1].state, result.files[1].winerror) == ("failed", 32)
        assert (root / "new.txt").read_bytes() == b"new"
        assert result.backup_point_id
    finally:
        if handle not in (None, ctypes.c_void_p(-1).value):
            close(handle)
        session.cancel_rewind()
        await session.aclose()
