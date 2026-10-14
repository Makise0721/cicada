"""还原点查看的关键接缝：真实捕获→只读页面，以及在途读取的取消/关闭准入。"""

import asyncio
import threading
from dataclasses import replace

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.application import open_application
from cicada.plugins.coding.rewind_reader import RewindReader
from cicada.restore_points import PREVIEW_LIMIT, RewindError
from cicada.ui.commands import SlashCommandMenu
from cicada.ui.conversation_view import build
from cicada.ui.display_style import DisplayStyle
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.main_screen import visible_width
from cicada.ui.terminal_io import FakeTerminal, PointerEvent
from test_persistence_application import _config
from tests.support.app import repo, script


def writes():
    return script({"tool_calls": [{"id": "w", "name": "write", "arguments": {
        "path": "a.txt", "content": "hello\n"}}]}, {"text": "done", "stop": True})


async def test_rewind_captured_request_page_is_readonly_and_batch_isolated(tmp_path):
    root = repo(tmp_path)
    session = await open_application(_config(root, writes(), session_id=None))
    await session.execute("create a")
    history = session.history
    view = build(session)["view"]
    terminal = FakeTerminal(100, 26)
    frontend = MainScreenFrontend(
        terminal, document_lines=view.document_lines, layout_document=view.layout_document,
        editor=view.editor, on_submit=view.on_submit, on_display_action=view.execute_display_action,
        review=view.review_view, rewind=view.rewind_view, command_menu=SlashCommandMenu())
    enter = KeyPress(Keys.ControlM, "\r")
    try:
        view.editor.set_text("/rewind")
        frontend._on_input_batch([enter, KeyPress("x", "x"), enter])
        await view.rewind_view._task
        assert view.rewind_view.active and view.editor.is_empty()
        frontend._do_render()
        span = next(h for h in frontend._frame_hits if h.action.target_id[1] == "point")
        frontend._on_input_batch([
            PointerEvent("down", span.start_column, span.row, terminal.geometry, button="left"),
            PointerEvent("up", span.start_column, span.row, terminal.geometry, button="left"),
            KeyPress(Keys.BracketedPaste, "ignored"), KeyPress(Keys.BackTab, "\x1b[Z")])
        await view.rewind_view._task
        point = view.rewind_view.point
        assert point.entries[0].current_relation == "clean"
        frontend._do_render()
        span = next(h for h in frontend._frame_hits if h.action.target_id[1] == "file")
        assert view.execute_display_action(span.action)
        await view.rewind_view._task
        assert "+hello" in view.rewind_view.preview.text
        with DisplayStyle(plain=True).use():
            for width, height in ((100, 26), (22, 12), (1, 1)):
                layout = view.rewind_view.layout(width, height)
                assert len(layout.lines) <= height
                assert all(visible_width(line) <= width for line in layout.lines)
                assert all("\x1b" not in line for line in layout.lines)
        frontend._on_input_batch([KeyPress(Keys.Escape, "\x1b"), KeyPress("y", "y"), enter])
        assert not view.rewind_view.active and not session.rewind_active
        assert view.editor.is_empty() and session.history == history
        assert (root / "a.txt").read_bytes() == b"hello\n"
    finally:
        frontend._stopped = True
        view.close()
        await session.aclose()


async def test_rewind_relations_history_unknown_and_blob_boundaries(tmp_path):
    root = tmp_path / "no-git"
    root.mkdir()
    session = await open_application(_config(root, script(), session_id=None))
    store = session._restore_store
    sid = session.persistence.session_id
    def capture(data, seq, *, settle=True):
        store.begin_request("r" + str(seq), "old-epoch", sid)
        ticket = store.capture(path="a.txt", op="modify", before=b"before\n", after=data)
        if settle:
            store.settle(ticket, commit_state="committed")
            store.note_commit(seq)
        store.end_request(interrupted=not settle)
        return ticket.point_id
    first = capture(b"one\n", 1)
    capture(b"two\n", 2)
    unknown = capture(b"maybe\n", 3, settle=False)
    (root / "a.txt").write_bytes(b"two\n")
    try:
        listing = await session.open_rewind()
        assert "历史代次" in listing.points[0].note
        assert listing.points[0].source == "interrupted"
        assert "提交结果未知" in listing.points[0].note
        async def relation(expected, later=None):
            point = await session.read_rewind_point(first)
            assert (point.entries[0].current_relation, point.entries[0].later_seq) == (expected, later)
            return point
        await relation("later_request", 2)
        (root / "a.txt").write_bytes(b"one\n")
        await relation("clean")
        (root / "a.txt").write_bytes(b"external\n")
        await relation("external")
        (root / "a.txt").unlink()
        await relation("missing")
        point = await session.read_rewind_point(unknown)
        preview = await session.read_rewind_preview(unknown, point.entries[0].entry_id)
        assert "提交结果未知" in preview.notice and "+maybe" in preview.text
        with pytest.raises(RewindError):
            await session.read_rewind_point("not-from-this-session")
        session.cancel_rewind()
        huge = capture(("行" * 100000).encode(), 4)
        binary = capture(b"\0binary", 5)
        non_utf8 = capture(b"\xff\xfe", 6)
        await session.open_rewind()
        for pid, notice in ((huge, "截断"), (binary, "二进制"), (non_utf8, "非 UTF-8")):
            point = await session.read_rewind_point(pid)
            preview = await session.read_rewind_preview(pid, point.entries[0].entry_id)
            assert notice in preview.notice
            assert len(preview.text.encode()) <= PREVIEW_LIMIT
        (root / ".cicada/restore/blobs" / point.entries[0].after_sha256).unlink()
        preview = await session.read_rewind_preview(pid, point.entries[0].entry_id)
        assert preview.state == "unavailable" and "blob" in preview.notice
        listing = await session.open_rewind()
        assert listing.points[0].availability == "expired"
    finally:
        session.cancel_rewind()
        await session.aclose()


@pytest.mark.parametrize("timeout", [False, True])
async def test_rewind_cancel_or_timeout_retains_lease_until_worker_settles(tmp_path, monkeypatch, timeout):
    import cicada.application as module
    root = repo(tmp_path)
    session = await open_application(_config(root, script(), session_id=None))
    entered, release = threading.Event(), threading.Event()
    original = RewindReader.capture
    def blocked(self, token):
        entered.set()
        assert release.wait(10)
        return original(self, token)
    monkeypatch.setattr(RewindReader, "capture", blocked)
    if timeout:
        monkeypatch.setattr(module, "REWIND_TIMEOUT_S", 0.02)
    task = asyncio.create_task(session.open_rewind())
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        if timeout:
            with pytest.raises(RewindError, match="超时"):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        session.cancel_rewind()
        session.cancel_rewind()
        assert session.rewind_active
        with pytest.raises(RewindError, match="空闲"):
            await session.open_rewind()
        with pytest.raises(RuntimeError):
            await session.execute("blocked")
        with pytest.raises(Exception):
            await session.open_review()
        close = asyncio.create_task(session.aclose())
        await asyncio.sleep(0.01)
        assert not close.done() and session.rewind_active
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await session.aclose()
    assert not session.rewind_active and session.state == "closed"


async def test_rewind_disabled_capture_reads_history_and_failed_admission_keeps_draft(tmp_path):
    root = repo(tmp_path)
    session = await open_application(_config(root, writes(), session_id=None))
    await session.execute("create")
    sid = session.persistence.session_id
    await session.aclose()
    session = await open_application(replace(_config(root, script(), session_id=sid), restore_capture=False))
    view = build(session)["view"]
    try:
        listing = await session.open_rewind()
        assert listing.points and "历史代次" in listing.points[0].note
        session.cancel_rewind()
        session._begin_in_flight = True
        view.editor.set_text("/rewind")
        assert view.on_submit("/rewind")
        await view.rewind_view._task
        await asyncio.sleep(0)
        assert not view.rewind_view.active
        assert view.editor.get_text() == "/rewind"
    finally:
        session._begin_in_flight = False
        view.close()
        await session.aclose()
