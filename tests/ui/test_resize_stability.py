"""稳定性04：生产主屏全状态宽高往返；组件出口与VT回放分别验收。"""

import asyncio
from dataclasses import replace
import json

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

import cicada.ui as ui
from cicada.__main__ import parse_script
from cicada.application import open_application
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.core.ports import TextDelta
from cicada.models import ModelOption
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui.main_screen import CURSOR_MARKER, strip_terminal_sequences, visible_width
from cicada.ui.welcome import WelcomeEntry
from tests.e2e.tui_fixture import StageTerminal, text_keys, wait_until, ENTER, CTRL_C, CTRL_D, CTRL_O
from test_conversation_view import _config, _repo

DOWN = KeyPress(Keys.Down, "\x1b[B")
UP = KeyPress(Keys.Up, "\x1b[A")
ESCAPE = KeyPress(Keys.Escape, "\x1b")
# 复用P08的40–160列×8–40行；另含欢迎96→40→96和纯高度变化。
SIZES = [(96, 24), (40, 24), (96, 24),
         *[(width, rows) for width in (40, 60, 96, 160) for rows in (8, 12, 24, 40)],
         (96, 24)]


@pytest.fixture(params=[False, True], ids=["color", "plain"])
def style(request, monkeypatch):
    monkeypatch.setenv("TERM", "dumb" if request.param else "xterm-256color")
    if request.param:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    return request.param


class ProductionHarness:
    """捕获公开run_tui_session实际组合，不另建简化前端。"""

    def __init__(self, session, monkeypatch):
        self.stage = StageTerminal(96, 24)
        self.session = session
        self.views = []
        self.frontends = []
        original_view, original_frontend = ui.ConversationView, ui.MainScreenFrontend

        def view(*args, **kwargs):
            # 只冻结已有显示时钟接缝，避免回放/静态比较跨越spinner帧。
            kwargs["clock"] = lambda: 0.0
            instance = original_view(*args, **kwargs)
            self.views.append(instance)
            return instance

        def frontend(*args, **kwargs):
            instance = original_frontend(*args, **kwargs)
            self.frontends.append(instance)
            return instance

        monkeypatch.setattr(ui, "ConversationView", view)
        monkeypatch.setattr(ui, "MainScreenFrontend", frontend)
        self.task = asyncio.create_task(ui.run_tui_session(session, terminal=self.stage.terminal))

    @property
    def view(self):
        return self.views[0]

    @property
    def frontend(self):
        return self.frontends[0]

    async def start(self):
        assert await wait_until(lambda: self.stage.terminal.started and bool(self.frontends))

    async def resize(self, columns, rows, *, editor=True):
        before = self.frontend._screen.full_redraws
        old = (self.stage.columns, self.stage.rows)
        self.stage.set_size(columns, rows)
        if old == (columns, rows):
            # StageTerminal更换VT镜像；同尺寸强制回放以建立比较画面。
            self.frontend.request_render(force=True)
        assert await wait_until(lambda: self.frontend._screen.full_redraws > before)
        frame = self.frontend._render_frame_lines(columns)
        # 文档/组件出口证明行宽，不能用VT的天然截断作证。
        assert all(visible_width(line) <= columns for line in self.view.document_lines(columns))
        assert all(visible_width(line) <= columns for line in frame)
        assert self.frontend.ui_failures == []
        assert self.frontend._screen.ui_failures == []
        expected = [strip_terminal_sequences(line).rstrip() for line in frame]
        visible = expected[-rows:]
        screen = [self.stage.vt.line_text(i).rstrip() for i in range(rows)]
        assert screen[:len(visible)] == visible, "新尺寸回放与同尺寸静态帧一致，无旧行残影"
        if editor:
            row, line = next((i, line) for i, line in enumerate(frame) if CURSOR_MARKER in line)
            col = visible_width(line.split(CURSOR_MARKER)[0])
            assert self.stage.vt.cursor_visible
            assert (self.stage.vt.cursor_row, self.stage.vt.cursor_col) == (row - max(0, len(frame) - rows), col)
        else:
            # 登录向导无编辑器硬件光标标记，隐藏聊天光标。
            assert not self.stage.vt.cursor_visible
        assert sum(isinstance(e, WelcomeEntry) for e in self.view.transcript.entries()) == 1

    async def close(self):
        if not self.task.done():
            self.frontend.request_exit()
        assert await asyncio.wait_for(self.task, 10) == 0
        assert self.session.state == "closed"
        assert self.stage.terminal.stopped and self.stage.vt.cursor_visible
        assert not self.stage.vt.sync_output_active
        assert self.frontend.ui_failures == self.frontend._screen.ui_failures == []


async def test_running_and_expanded_document_roundtrip(style, tmp_path, monkeypatch):
    blocked, release = asyncio.Event(), asyncio.Event()
    answer = "## 长文档\n\n" + "中文正文 `code` 与连续长词 abcdefghijklmnopqrstuvwxyz。" * 20
    script = parse_script(json.dumps([
        {"tool_calls": [{"id": "resize-tool", "name": "powershell", "arguments": {
            "command": "foreach ($i in 1..14) { Write-Output ('resize-output-{0:D2}-中文长输出' -f $i) }",
            "timeout": 15}}], "tool_use": True},
        {"text": answer, "stop": True},
    ]))

    class PausedModel(FakeModel):
        async def stream(self, request, cancel):
            async for event in super().stream(request, cancel):
                yield event
                if isinstance(event, TextDelta):
                    blocked.set()
                    await release.wait()

    model = PausedModel(script)
    session = await open_application(replace(_config(_repo(tmp_path), []), model_plugin=fake_model_plugin(model)))
    h = ProductionHarness(session, monkeypatch)
    try:
        await h.start()
        h.stage.emit(text_keys("resize首问") + [ENTER])
        await asyncio.wait_for(blocked.wait(), 15)
        assert len(model.requests) == 2
        assert session.state != "idle" and h.view._gate.busy()
        h.stage.emit([CTRL_O])
        assert h.view.tools.expanded
        h.stage.emit([KeyPress(Keys.BracketedPaste, "中文草稿\n第二行草稿abc")])
        # 光标置于非末尾，缩放不能重置编辑位置。
        h.view.editor.move_cursor(0, -2)
        draft = h.view.editor.get_text()
        position = (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col)
        entries = h.view.transcript.entries()
        for width, rows in SIZES:
            await h.resize(width, rows)
            assert h.view.status_lines(width)
            document = "".join(h.view.document_lines(width))
            assert all(f"resize-output-{i:02}" in document for i in range(1, 15))
            assert h.view.editor.get_text() == draft
            assert (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col) == position
            assert h.view.transcript.entries() == entries
            assert len(model.requests) == 2
            h.stage.emit([ENTER])  # busy Enter始终静默，不新增请求。
            assert h.view.editor.get_text() == draft and len(model.requests) == 2
        release.set()
        assert await wait_until(lambda: session.state == "idle" and not h.view._gate.busy())
        history = session.history
        assert sum(isinstance(m, UserMessage) for m in history) == 1
        assert sum(isinstance(m, AssistantMessage) and m.text == answer for m in history) == 1
        for width, rows in reversed(SIZES):
            await h.resize(width, rows)
            assert h.view.status_lines(width) == []
            assert h.view.editor.get_text() == draft
            assert (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col) == position
            assert session.history == history and len(model.requests) == 2
        h.stage.emit([CTRL_C, CTRL_D])
        await h.close()
    finally:
        release.set()
        if not h.task.done():
            await h.close()


async def test_menus_and_login_roundtrip(style, tmp_path, monkeypatch):
    model = FakeModel([])
    session = await open_application(replace(_config(_repo(tmp_path), []), model_plugin=fake_model_plugin(model)))
    h = ProductionHarness(session, monkeypatch)
    try:
        await h.start()
        h.stage.emit(text_keys("/") + [DOWN, DOWN])
        selected = h.frontend._command_menu.selected
        for width, rows in SIZES:
            await h.resize(width, rows)
            assert h.frontend._command_menu.selected == selected
            assert h.view.editor.get_text() == "/"
            assert "→ " in "\n".join(h.stage.vt.screen_text())
        h.stage.emit([ESCAPE])
        assert await wait_until(lambda: h.frontend._command_menu.selected is None)
        assert h.view.editor.get_text() == "/"
        h.stage.emit([CTRL_C])
        h.view.editor.set_text("模型菜单前草稿")
        # Fake没有模型目录：用安全ModelOption展示接缝，真实选择事务由既有/model测试证明。
        h.view.model_menu.open((ModelOption("one", "模型一"), ModelOption("two", "模型二")), current_id="one")
        h.stage.emit([DOWN])
        for width, rows in SIZES:
            await h.resize(width, rows)
            assert h.view.model_menu.selected.id == "two"
            assert h.view.editor.get_text() == "模型菜单前草稿"
            assert "→ 模型二" in "\n".join(h.stage.vt.screen_text())
        h.stage.emit([UP])
        assert h.view.model_menu.selected.id == "one"
        h.stage.emit([ESCAPE])
        assert await wait_until(lambda: not h.view.model_menu.is_open)
        draft = h.view.editor.get_text()
        position = (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col)
        # Fake不保存登录配置；注入安全read_view，走真实弹窗按键/渲染，取消后无文件保存。
        store = FileUserConfigStore(tmp_path / "synthetic-user")
        h.view.login_modal.open(store.read_view())
        for page in ("provider", "credential", "thinking", "confirm"):
            wizard = h.view.login_modal.wizard
            assert wizard.step == page
            for width, rows in SIZES:
                await h.resize(width, rows, editor=False)
                assert wizard.step == page
                assert h.view.editor.get_text() == draft
                assert (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col) == position
            if page == "credential":
                h.stage.emit([KeyPress(Keys.BracketedPaste, "synthetic-resize-key")])
                for width, rows in reversed(SIZES):
                    await h.resize(width, rows, editor=False)
                    assert "synthetic-resize-key" not in h.stage.terminal.output_text
                    assert "API key: " + "*" * 20 in "\n".join(h.stage.vt.screen_text())
            if page != "confirm":
                h.stage.emit([ENTER])
        h.stage.emit([DOWN])
        selected = wizard.selection
        for width, rows in SIZES:
            await h.resize(width, rows, editor=False)
            assert wizard.selection == selected
        h.stage.emit([ESCAPE])
        assert not h.view.login_modal.is_open
        await h.resize(96, 24)
        assert h.view.editor.get_text() == draft
        assert not model.requests and session.history == ()
        assert not store.root.exists()
        h.stage.emit([CTRL_C, CTRL_D])
        await h.close()
    finally:
        if not h.task.done():
            await h.close()
