"""稳定性 03：四环境、组件与生产组合、会话冻结及缓存隔离。"""

import asyncio
from dataclasses import replace
import re

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.application import open_application
from cicada.core.events import ToolCompleted, ToolStarted
from cicada.core.messages import ToolResult
from cicada.core.ports import StreamDone, TextDelta
from cicada.models import ModelOption
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.user_config import FileUserConfigStore
from cicada.skills import SkillInvocation
from cicada.ui import theme
from cicada.ui.commands import SlashCommandMenu
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.editor import Editor, EditorStyle
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.login_wizard import LoginWizard
from cicada.ui.main_screen import CURSOR_MARKER, strip_terminal_sequences, visible_width
from cicada.ui.model_menu import ModelMenu
from cicada.ui.terminal_io import FakeTerminal
from cicada.ui.tool_view import ToolDocument, ToolGroupEntry
from cicada.ui.transcript import AssistantEntry, InformationEntry, SkillRequestEntry, UserEntry
from test_conversation_view import _config, _repo, _wait_until

SGR = re.compile(r"\x1b\[[0-9;]*m")
MODES = [(None, "xterm-256color", False), ("1", "xterm-256color", True),
         (None, "dumb", True), ("", "xterm-256color", True)]


@pytest.fixture(params=MODES, ids=["color", "no-color", "dumb", "empty-no-color"])
def mode(request, monkeypatch):
    no_color, term, plain = request.param
    monkeypatch.setenv("TERM", term)
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    return plain


def assert_style(lines, plain, *, renderer=False):
    codes = SGR.findall("\n".join(lines))
    if plain:
        assert not codes or (renderer and set(codes) == {"\x1b[0m"})
    else:
        assert any(code != "\x1b[0m" for code in codes), "有色分支必须确有样式"


@pytest.mark.parametrize("width", [18, 60])
def test_all_components_follow_explicit_environment(mode, width, tmp_path):
    style = DisplayStyle.capture()
    assert style.plain is mode
    editor = Editor(border_color=theme.editor_border(), padding_x=2,
                    style=EditorStyle(prompt_glyph="›", placeholder="输入消息…"))
    editor.focused = True
    menu = SlashCommandMenu()
    menu.sync("/", at_end=True)
    models = ModelMenu()
    models.open((ModelOption("first", "模型一"), ModelOption("second", "模型二")), current_id="first")
    tools = ToolDocument()
    tools.apply(ToolStarted("r", "pending", "read"))
    tools.apply(ToolCompleted("r", ToolResult("ok", "glob", "输出正文")))
    tools.apply(ToolCompleted("r", ToolResult("bad", "edit", "失败原因", is_error=True)))
    # 紧凑组 (票据 01): 打开组列表与两张终态卡详情, 输出正文才进入显示行
    tools.toggle_expand()
    tools.toggle_tool(("live", "r", 0, "ok"))
    tools.toggle_tool(("live", "r", 0, "bad"))
    wizard = LoginWizard(FileUserConfigStore(tmp_path / "user").read_view())
    with style.use():
        groups = {
            "user": UserEntry("用户正文").render(width),
            "markdown": theme.assistant_markdown_lines("### 标题\n\n**强调** 与 `code`", width),
            "tools": tools.render_lines(width),
            "information": InformationEntry("检查", [("结果", ["状态: PASS", "来源: 本次检查"])]).render(width),
            "editor": editor.render(width),
            "slash": menu.render(width, 5),
            "models": models.render(width, 5),
            "footer": theme.FooterComponent("workspace", "fake", lambda: (8, False, 3, False)).render(width, 24),
        }
        for step in ("provider", "credential", "thinking", "confirm"):
            wizard.step = step
            groups["login-" + step] = wizard.render_lines(width, 24)
        for name, lines in groups.items():
            assert_style(lines, mode)
            assert all(visible_width(line) <= width for line in lines), name
        flat = "\n".join(strip_terminal_sequences(line) for line in groups["tools"])
        assert all(glyph in flat for glyph in ("+", "-"))  # 折叠标记一列可辨认
        assert "输出正文" in flat and "失败原因" in flat
        assert any("→ " in line for line in groups["slash"])
        assert any("→ " in line for line in groups["models"])
        assert CURSOR_MARKER in "".join(groups["editor"])
        assert "›" in "".join(groups["editor"])
        assert any("─" in line for line in groups["editor"])
        menu.move(3)
        models.move(1)
        assert "model" in strip_terminal_sequences(next(line for line in menu.render(60, 8) if "→ " in line))
        assert "模型二" in strip_terminal_sequences(next(line for line in models.render(width, 5) if "→ " in line))


def test_render_caches_do_not_mix_styles_or_environment(monkeypatch):
    entry = AssistantEntry("r", 0)
    entry.text = "### 缓存标题\n\n**正文**"
    tools = ToolDocument()
    tools.apply(ToolCompleted("r", ToolResult("ok", "read", "缓存输出")))
    card = ToolGroupEntry(tools, 0)
    skill = SkillRequestEntry("record", SkillInvocation(1, "demo", "C:/demo/SKILL.md", "C:/demo",
                                                       "a" * 64, "正文", "b" * 64, "任务"), lambda: False)
    rendered = {}
    for plain in (False, True, False, True):
        # 环境刻意与作用域策略相反，Rich 也必须服从冻结策略。
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "dumb")
        with DisplayStyle(plain=plain).use():
            lines = [*entry.render(40), *card.render(40), *tools.render_lines(40), *skill.render(40)]
            assert_style(lines, plain)
            if plain in rendered:
                assert lines == rendered[plain]
            rendered[plain] = lines
    assert "缓存标题" in "".join(rendered[True])


async def test_render_scopes_are_isolated_between_tasks():
    async def render(plain):
        with DisplayStyle(plain=plain).use():
            await asyncio.sleep(0)
            assert current_style().plain is plain
            lines = theme.assistant_markdown_lines("### 同一正文", 40)
            assert_style(lines, plain)
    await asyncio.gather(render(False), render(True))


async def test_production_freezes_style_across_environment_changes(mode, tmp_path, monkeypatch):
    import cicada.ui as ui

    blocked = asyncio.Event()
    release = asyncio.Event()

    class PausedModel(FakeModel):
        async def stream(self, request, cancel):
            self.requests.append(request)
            yield TextDelta("### 标题\n\n正文\x1b[35;46m保留文字")
            blocked.set()
            await release.wait()
            yield StreamDone("stop")

    model = PausedModel([])
    config = replace(_config(_repo(tmp_path), []), model_plugin=fake_model_plugin(model))
    session = await open_application(config)
    captured = []
    original = ui.ConversationView

    def capture(*args, **kwargs):
        view = original(*args, **kwargs)
        captured.append(view)
        return view

    monkeypatch.setattr(ui, "ConversationView", capture)
    terminal = FakeTerminal(80, 28)
    task = asyncio.create_task(ui.run_tui_session(session, initial_prompt="首问", terminal=terminal))
    try:
        await asyncio.wait_for(blocked.wait(), 5)
        view = captured[0]
        assert view.display_style.plain is mode
        assert_style(view.document_lines(80), mode)
        assert_style(view.status_lines(80), mode)
        welcome = view.transcript.entries()[0].render(80)
        assert_style(welcome, mode)
        if mode:
            assert "\x1b" not in "".join(welcome)
        # 正文、状态、菜单、弹窗及底栏都在启动后切换环境时继续原策略。
        if mode:
            monkeypatch.delenv("NO_COLOR", raising=False)
            monkeypatch.setenv("TERM", "xterm-256color")
        else:
            monkeypatch.setenv("NO_COLOR", "1")
            monkeypatch.setenv("TERM", "dumb")
        assert_style(view.document_lines(80), mode)
        assert_style(view.status_lines(80), mode)
        assert_style(view.footer.render(80, 28), mode)
        release.set()
        assert await _wait_until(lambda: session.state == "idle" and not view._gate.busy())
        terminal.emit([KeyPress("/", "/")])
        assert await _wait_until(lambda: any("→ " in line for line in terminal.written))
        assert_style(terminal.written, mode, renderer=True)
        assert "\x1b[35;46m" not in "".join(terminal.written)
        assert "\x1b[35;46m" in "".join(getattr(m, "text", "") for m in session.history)
        # 同一前端帧组合验证登录弹窗继承启动策略；只打开安全视图，不保存。
        view.login_modal.open(FileUserConfigStore(tmp_path / "user").read_view())
        assert_style(view.login_modal.render(80, 28), mode)
        preview = MainScreenFrontend(terminal, document_lines=view.document_lines, editor=view.editor,
                                     model_menu=view.model_menu, modal=view.login_modal,
                                     display_style=view.display_style)
        assert_style(preview._render_frame_lines(80), mode)
        view.login_modal.dismiss()
        view.model_menu.open((ModelOption("one", "模型一"), ModelOption("two", "模型二")), current_id="one")
        assert_style(preview._render_frame_lines(80), mode)
        view.model_menu.dismiss()
        terminal.emit([KeyPress(Keys.ControlC, "\x03")])
        terminal.emit([KeyPress(Keys.ControlD, "\x04")])
        assert await asyncio.wait_for(task, 5) == 0
        assert session.state == "closed" and len(model.requests) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if session.state != "closed":
            await session.aclose()
