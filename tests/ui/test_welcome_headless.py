"""tui-startup-welcome 票据 02 无头前端检查: 宽窗标志、resize 与低高度编辑器可用性.

A07 resize / A11 (02 范围): 真实 ApplicationSession + ConversationView.build 接缝 +
MainScreenFrontend + FakeTerminal/VtScreen; fake 模型, 零真实模型调用。
宽窗 28×6 标志进入真实 VT 画面; 96→48→96 resize 与 8/12/24 行低高度下
输入/草稿/光标保留, 光标不移入标志区, resize 不增加欢迎条目。
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.key_processor import KeyPress

from cicada.application import open_application
from cicada.ui.conversation_view import build
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.terminal_io import FakeTerminal
from cicada.ui.welcome import WelcomeEntry
from tests.ui.vt_screen import VtScreen

from test_conversation_view import _config, _repo, _script, _wait_until

ENTER = KeyPress(Keys.ControlM, "\r")
_LOGO_CHARS = "█▄▀"


def text_keys(text: str) -> list[KeyPress]:
    return [KeyPress(ch, ch) for ch in text]


class TuiHarness:
    """真实会话 + build 组合 + 无头前端; VT 屏镜像用于画面/光标断言。"""

    def __init__(self, session, *, columns: int = 96, rows: int = 24) -> None:
        self.session = session
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.frontend: MainScreenFrontend | None = None
        self.bundle = build(session, on_exit=lambda: None, on_update=self._request_render)
        self.view = self.bundle["view"]
        self.view.editor.focused = True  # 同 run_tui_session: 硬件光标跟随编辑光标
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=self.bundle["document_lines"],
            editor=self.bundle["editor"],
            on_submit=self.bundle["on_submit"],
            on_escape=self.bundle["on_escape"],
            on_ctrl_c=self.bundle["on_ctrl_c"],
            on_exit=self.bundle["on_exit"],
            footer_lines=self.bundle["footer_lines"],
        )
        self.runner: asyncio.Task | None = None

    def _request_render(self) -> None:
        if self.frontend is not None:
            self.frontend.request_render()

    async def start(self) -> None:
        self.runner = asyncio.ensure_future(self.frontend.run_async())
        await self.settle()

    async def settle(self, ticks: int = 3) -> None:
        await asyncio.sleep(0.05)
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def stop(self) -> list[str]:
        self.frontend.request_exit()
        return await asyncio.wait_for(self.runner, timeout=5)

    def find(self, needle: str) -> bool:
        return self.vt.find_row(needle) is not None

    def welcome_count(self) -> int:
        return sum(isinstance(e, WelcomeEntry) for e in self.view.transcript.entries())


def _force_color_env(monkeypatch) -> None:
    """欢迎 plain 策略捕获自环境: 测试固定有色路径, 不受运行环境 NO_COLOR/TERM 影响。"""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


# --- A09/A11: 宽窗标志进入真实画面, 编辑器与光标不受标志影响 -------------------------------------


@pytest.mark.asyncio
async def test_wide_window_logo_beside_text_editor_cursor_usable(tmp_path, monkeypatch):
    _force_color_env(monkeypatch)
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = TuiHarness(session, columns=96, rows=24)
    try:
        await h.start()
        title_row = h.vt.find_row("新会话 · 启动快照")
        assert title_row is not None
        title_line = h.vt.line_text(title_row)
        assert any(mark in title_line[:32] for mark in _LOGO_CHARS), "标题行左侧为标志"
        assert title_line.index("Cicada ") == 32, "文字从第 32 列起 (1 缩进 + 28 标志 + 3 间隔)"

        h.terminal.emit(text_keys("草稿一"))
        await h.settle()
        draft_row = h.vt.find_row("草稿一")
        assert draft_row is not None, "宽窗标志下草稿可输入"
        assert h.vt.cursor_row == draft_row, "硬件光标在编辑器草稿行, 不移入标志区"

        h.terminal.emit([ENTER])
        assert await _wait_until(lambda: h.find("答")), "提交经原准入路径完成"
        assert h.welcome_count() == 1
    finally:
        errors = await h.stop()
        h.view.close()
        await session.aclose()
    assert errors == []


# --- A11: 低高度 8/12/24 行画面, 编辑器优先 ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [8, 12, 24])
async def test_low_heights_keep_editor_draft_and_cursor(tmp_path, monkeypatch, rows):
    _force_color_env(monkeypatch)
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = TuiHarness(session, columns=96, rows=rows)
    try:
        await h.start()
        if rows >= 24:
            assert h.find("新会话 · 启动快照")
        # 低高度允许欢迎滚出可见区, 编辑器/光标优先
        h.terminal.emit(text_keys("低高度草稿"))
        await h.settle()
        draft_row = h.vt.find_row("低高度草稿")
        assert draft_row is not None
        assert h.vt.cursor_row == draft_row
        h.terminal.emit([ENTER])
        assert await _wait_until(lambda: h.find("答"))
        assert h.welcome_count() == 1
    finally:
        errors = await h.stop()
        h.view.close()
        await session.aclose()
    assert errors == []


# --- A07/A11: 96→48→96 resize 重排, 草稿保留且欢迎不重复 ------------------------------------------


@pytest.mark.asyncio
async def test_resize_96_48_96_rewraps_and_keeps_draft(tmp_path, monkeypatch):
    _force_color_env(monkeypatch)
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = TuiHarness(session, columns=96, rows=24)
    try:
        await h.start()
        title_row = h.vt.find_row("新会话 · 启动快照")
        assert title_row is not None
        assert any(mark in h.vt.line_text(title_row)[:32] for mark in _LOGO_CHARS)

        h.terminal.emit(text_keys("保留我"))
        await h.settle()
        assert h.find("保留我")

        h.terminal.set_size(48, 24)  # 48 列: 省略标志的窄窗四行块
        await h.settle()
        assert h.find("保留我"), "resize 后草稿保留"
        assert h.find("/help 帮助")
        screen = h.vt.screen_text()
        assert all(len(line) <= 48 for line in screen)
        assert not any(mark in line for line in screen for mark in _LOGO_CHARS), "窄窗无标志"

        h.terminal.set_size(96, 24)  # 回到宽窗: 标志恢复并排
        await h.settle()
        title_row = h.vt.find_row("新会话 · 启动快照")
        assert title_row is not None
        assert any(mark in h.vt.line_text(title_row)[:32] for mark in _LOGO_CHARS)
        assert h.find("保留我")
        draft_row = h.vt.find_row("保留我")
        assert h.vt.cursor_row == draft_row, "resize 后光标回到编辑器草稿行"
        assert h.welcome_count() == 1, "resize 不增加欢迎条目"

        h.terminal.emit([ENTER])
        assert await _wait_until(lambda: h.find("答")), "resize 后提交仍沿原准入路径"
    finally:
        errors = await h.stop()
        h.view.close()
        await session.aclose()
    assert errors == []
