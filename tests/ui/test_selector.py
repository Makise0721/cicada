"""票据 05: 会话选择器聚焦检查 (A12/A16——无模型活动、有界输出、取消与选择)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui.selector import run_session_selector
from cicada.ui.terminal_io import FakeTerminal

UP = KeyPress(Keys.Up, "\x1b[A")
DOWN = KeyPress(Keys.Down, "\x1b[B")
RIGHT = KeyPress(Keys.Right, "\x1b[C")
LEFT = KeyPress(Keys.Left, "\x1b[D")
PAGE_DOWN = KeyPress(Keys.PageDown, "\x1b[6~")
PAGE_UP = KeyPress(Keys.PageUp, "\x1b[5~")
ESCAPE = KeyPress(Keys.Escape, "\x1b")
ENTER = KeyPress(Keys.ControlM, "\r")


async def _make_session(directory: Path, *, title: str, requests: int = 1) -> str:
    import sqlite3

    from cicada.core.events import ModelCallCompleted
    from cicada.core.messages import AssistantMessage, UserMessage
    from cicada.persistence.contracts import (
        LaunchIdentity,
        RequestOutcomeFacts,
        StoredIdentity,
    )
    from cicada.plugins.session_store import open_session_store

    store = open_session_store(directory=directory, session_id=None, workspace=directory)
    session_id = store.session_id
    identity = StoredIdentity(
        launch=LaunchIdentity(
            provider="fake", model="fake", base_url=None, think=None, options={},
            max_turns=6, prompt_version="t", prompt_sha256="a" * 64,
            instructions_sha256=None, checks_signature=None, sources={},
        ),
        tool_fingerprint="fp" + "0" * 30,
    )
    epoch = await store.start_epoch(identity)
    for index in range(requests):
        prompt = f"{title} 问题 {index}"
        request_id = await store.begin_request(epoch, prompt)
        await store.commit_request(
            request_id,
            messages=(UserMessage(text=prompt), AssistantMessage(text=f"回答 {index}")),
            outcome=RequestOutcomeFacts(
                run_id=f"run-{index + 1}", stop_reason="stop", error=None, exit_code=0,
                finalized=True,
                model_calls=(ModelCallCompleted(
                    run_id=f"run-{index + 1}", turn_index=0, stop_reason="stop",
                    metrics=None, elapsed_s=0.01,
                ),),
            ),
        )
    await store.aclose()
    if title:
        connection = sqlite3.connect(str(directory / f"{session_id}.sqlite3"))
        connection.execute("UPDATE session SET short_title = ?", (title,))
        connection.commit()
        connection.close()
    return session_id


async def _drive(terminal: FakeTerminal, directory: Path, keys, *, limit: int = 25):
    """起跑选择器, 逐键注入, 等待其终态 (超时即失败)。"""
    task = asyncio.create_task(run_session_selector(terminal, directory, limit=limit))
    await asyncio.sleep(0.05)
    for key in keys:
        terminal.emit([key])
        await asyncio.sleep(0.01)
    return await asyncio.wait_for(task, timeout=10)


def _run_selector(terminal: FakeTerminal, directory: Path, keys=(), *, limit: int = 25):
    return asyncio.run(_drive(terminal, directory, list(keys), limit=limit))


def test_selector_escapes_with_none(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    terminal = FakeTerminal(columns=60, rows=12)
    session_id, errors = _run_selector(terminal, directory, keys=[ESCAPE])
    assert session_id is None and errors == []
    text = terminal.output_text
    assert "选择要恢复的会话" in text
    assert "暂无会话存档" in text


def test_selector_lists_and_selects(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    first = asyncio.run(_make_session(directory, title="第一场会话"))
    second = asyncio.run(_make_session(directory, title="第二场会话"))

    terminal = FakeTerminal(columns=100, rows=12)
    session_id, _ = _run_selector(terminal, directory, keys=[DOWN, ENTER])
    assert session_id == first, "↓ 从最新行移到次新行, Enter 选定较早会话"

    terminal = FakeTerminal(columns=100, rows=12)
    session_id, _ = _run_selector(terminal, directory, keys=[ENTER])
    assert session_id == second, "默认选中最近提交的会话"

    text = terminal.output_text
    assert "第一场会话" in text and "第二场会话" in text
    assert "终结事实未读取" in text
    assert "请求=" not in text and "阻断" not in text
    for line in text.splitlines():
        assert len(re.sub(r"\x1b\[[0-9;]*m", "", line)) <= 100, "选择器输出行有界"


def test_selector_bad_entries_visible_but_not_selectable(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    good = asyncio.run(_make_session(directory, title="好的会话"))
    (directory / "00000000-1111-4222-8333-444444444444.sqlite3").write_bytes(b"garbage")

    terminal = FakeTerminal(columns=100, rows=12)
    session_id, _ = _run_selector(terminal, directory, keys=[ENTER])
    text = terminal.output_text
    assert "00000000 [坏档:" in text
    assert "坏档" in text, "坏档如实标注"
    assert session_id == good, "坏档不可选, 默认选定好会话"


def test_selector_pagination(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    ids = [asyncio.run(_make_session(directory, title=f"会话 {index}")) for index in range(3)]

    terminal = FakeTerminal(columns=60, rows=12)
    session_id, _ = _run_selector(terminal, directory, keys=[PAGE_DOWN, ENTER], limit=2)
    assert "第 2/2 页" in terminal.output_text
    assert session_id == ids[0], "第二页含最旧会话 (按最近提交降序)"

    terminal = FakeTerminal(columns=60, rows=12)
    session_id, _ = _run_selector(terminal, directory, keys=[PAGE_DOWN, PAGE_UP, ENTER], limit=2)
    assert session_id == ids[2], "翻回第一页后选定最近会话"


def test_selector_bad_archive_enter_refused_until_escape(tmp_path):
    """审查 05-M1: 坏档行 Enter 被拒 (选择器不结束), Escape 才取消——守卫真实驱动."""
    directory = tmp_path / "sessions"
    directory.mkdir()
    asyncio.run(_make_session(directory, title="好的会话"))
    (directory / "00000000-1111-4222-8333-444444444444.sqlite3").write_bytes(b"garbage")

    async def drive():
        terminal = FakeTerminal(columns=100, rows=12)
        task = asyncio.create_task(run_session_selector(terminal, directory))
        await asyncio.sleep(0.05)
        terminal.emit([DOWN])  # 坏档排最后 (无提交记录)
        await asyncio.sleep(0.01)
        terminal.emit([ENTER])
        await asyncio.sleep(0.05)
        assert not task.done(), "坏档行 Enter 被拒, 选择器保持运行"
        assert "ID: 00000000-1111-4222-8333-444444444444" in terminal.output_text
        terminal.emit([ESCAPE])
        return await asyncio.wait_for(task, timeout=10)

    session_id, errors = asyncio.run(drive())
    assert session_id is None and errors == []


async def test_selector_refresh_rejects_late_result_and_drains_on_cancel(tmp_path):
    from cicada.persistence.contracts import SessionCatalogEntry, SessionCatalogSnapshot

    entered = asyncio.Event()
    release = asyncio.Event()
    second = asyncio.Event()
    calls = 0

    async def observe(cancel):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            # 故意模拟 reader 返回迟到成功；UI 必须失效它。
            return SessionCatalogSnapshot('old', str(tmp_path), (
                SessionCatalogEntry('old-id', '迟到旧观察', '', None, None, False),), True)
        second.set()
        await cancel.wait()
        await release.wait()
        raise asyncio.CancelledError

    terminal = FakeTerminal()
    task = asyncio.create_task(run_session_selector(terminal, tmp_path, observe=observe))
    await entered.wait()
    terminal.emit([KeyPress(Keys.ControlR, '\x12')])
    terminal.emit([KeyPress(Keys.ControlR, '\x12')])
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    await second.wait()
    assert calls == 2 and '迟到旧观察' not in terminal.output_text
    release.clear()
    terminal.emit([ESCAPE])
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not terminal.stopped
    release.set()
    import pytest
    with pytest.raises(asyncio.CancelledError):
        await task
    count = len(terminal.written)
    terminal.emit([ENTER])
    terminal.set_size(20, 4)
    assert len(terminal.written) == count and terminal.stopped
