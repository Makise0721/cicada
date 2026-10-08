"""DeepSeek 阶段票据 04 检查: /model 菜单、键位、屏障、状态与真实应用集成.

生产组件 VT 核对 (MainScreenFrontend + ConversationView + ModelMenu +
SlashCommandMenu + FakeTerminal + VtScreen): 菜单整行高亮/宽度/光标位置、
默认选当前、箭头循环、Enter 经应用选择、Esc 收起、编辑不丢字符、busy/
选择中屏障静默、失败有界、closing 不重开; 真实 ApplicationSession + mock
HTTP 证明确认后继请求使用新模型, 思考不进入普通显示。组合层语义
(屏障/事件/失败) 另有不带渲染的聚焦用例。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.application import open_application
from cicada.models import ModelOption
from cicada.ui.commands import SLASH_COMMANDS, SlashCommandMenu
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.model_menu import MODEL_MENU_HINT, ModelMenu
from cicada.ui.terminal_io import FakeTerminal
from test_application import _repo
from test_conversation_view import Harness as ViewHarness
from test_conversation_view import _config as fake_config
from test_conversation_view import _script
from test_conversation_view import _wait_until
from test_deepseek_application import config as ds_config
from test_deepseek_application import response as ds_response
from tests.ui.test_frontend_headless import ENTER, ESCAPE, text_keys
from tests.ui.vt_screen import VtScreen

UP = KeyPress(Keys.Up, "\x1b[A")
DOWN = KeyPress(Keys.Down, "\x1b[B")

SEL_BG = "\x1b[48;2;45;45;61m"       # MENU_SEL_BG #2d2d3d (同斜线菜单规格 §D)
ARROW_FG = "\x1b[38;2;178;148;187m"  # EDITOR_BORDER #b294bb
NAME_FG = "\x1b[38;2;129;162;190m"   # MD_LINK #81a2be


async def _wait_model_settled(h, timeout: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not h.view._gate.model_selecting:
            return True
        await asyncio.sleep(0.01)
    return False


def _transport(bodies: list, answers):
    """记录请求体并按序作答的 mock HTTP (同 test_model_selection 接缝)。"""

    async def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))

    return httpx.MockTransport(handle)


class TUIHarness:
    """生产组合 headless 驱动: 与 run_tui_session 同一接线 (含模型菜单)。"""

    def __init__(self, session, *, columns: int = 60, rows: int = 16) -> None:
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.render_requests = 0
        self.exit_requested = 0
        from cicada.ui.conversation_view import build

        self.bundle = build(session, on_exit=self._exit, on_update=self._update)
        self.view = self.bundle["view"]
        self.view.editor.focused = True
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=self.bundle["document_lines"],
            editor=self.bundle["editor"],
            on_submit=self.bundle["on_submit"],
            on_escape=self.bundle["on_escape"],
            on_ctrl_c=self.bundle["on_ctrl_c"],
            on_exit=self.bundle["on_exit"],
            footer_lines=self.bundle["footer_lines"],
            command_menu=SlashCommandMenu(),
            status_lines=self.bundle["status_lines"],
            model_menu=self.bundle["model_menu"],
            on_model_confirm=self.bundle["on_model_confirm"],
            modal=self.bundle["modal"],
        )
        self.runner: asyncio.Task | None = None

    def _exit(self) -> None:
        self.exit_requested += 1
        self.frontend.request_exit()

    def _update(self) -> None:
        self.render_requests += 1
        self.frontend.request_render()

    async def start(self) -> None:
        self.runner = asyncio.ensure_future(self.frontend.run_async())
        await self.settle()

    async def settle(self, ticks: int = 3) -> None:
        await asyncio.sleep(0.05)  # 真实短睡眠: 节流渲染/Escape 定时器需要真实时间
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def stop(self) -> list[str]:
        self.frontend.request_exit()
        return await asyncio.wait_for(self.runner, timeout=5)

    def screen(self) -> list[str]:
        return self.vt.screen_text()

    def find(self, needle: str) -> bool:
        return self.vt.find_row(needle) is not None

    def menu_row(self, needle: str) -> str | None:
        for row in self.screen():
            if needle in row:
                return row
        return None


async def _ds_session(tmp_path, bodies, answers):
    root = _repo(tmp_path)
    session = await open_application(ds_config(root, _transport(bodies, answers)))
    return session


# --- ModelMenu 单元: 默认当前/循环/高亮/宽度 -------------------------------------------------


def _options() -> tuple[ModelOption, ...]:
    return (ModelOption("deepseek-flash", "deepseek-flash"),
            ModelOption("deepseek-v4-pro", "deepseek-v4-pro"))


def test_open_defaults_to_current_and_move_wraps():
    menu = ModelMenu()
    menu.open(_options(), current_id="deepseek-v4-pro")
    assert menu.selected.id == "deepseek-v4-pro"
    assert menu.move(1) and menu.selected.id == "deepseek-flash"  # 循环回绕
    assert menu.move(-1) and menu.selected.id == "deepseek-v4-pro"
    assert menu.dismiss() and not menu.is_open
    assert menu.selected is None and not menu.move(1)


def test_open_empty_options_stays_closed():
    menu = ModelMenu()
    menu.open((), current_id=None)
    assert not menu.is_open and menu.render(60, 4) == []


def test_selected_row_full_width_highlight_and_current_marker():
    menu = ModelMenu()
    menu.open(_options(), current_id="deepseek-flash")
    lines = menu.render(50, 4)
    assert strip_terminal_sequences(lines[0]) == MODEL_MENU_HINT
    selected = lines[1]
    assert strip_terminal_sequences(selected).startswith("→ deepseek-flash")
    assert "（当前）" in strip_terminal_sequences(selected)
    assert selected.startswith(SEL_BG) and selected.endswith("\x1b[0m")
    assert ARROW_FG in selected and NAME_FG in selected
    assert visible_width(strip_terminal_sequences(selected)) == 50  # 整行背景补齐
    assert selected[:-4].count("\x1b[0m") == 0  # 行内无内层复位 (规格 §D)
    plain = strip_terminal_sequences(lines[2])
    assert plain.startswith("  deepseek-v4-pro") and "（当前）" not in plain
    assert not lines[2].startswith(SEL_BG)


def test_render_fits_narrow_width_and_row_budget():
    menu = ModelMenu()
    menu.open(_options(), current_id=None)
    for line in menu.render(12, 4):
        assert visible_width(strip_terminal_sequences(line)) <= 12
    assert len(menu.render(50, 2)) == 2  # 行预算: 提示行 + 1 条
    assert menu.render(50, 0) == []


def test_slash_menu_discovers_model_command():
    assert any(item.command == "/model" for item in SLASH_COMMANDS)
    menu = SlashCommandMenu()
    menu.sync("/m", at_end=True)
    assert menu.selected is not None and menu.selected.command == "/model"


# --- 生产前端 VT: 打开/键位/收起/宽度/光标 ------------------------------------------------------


async def test_model_menu_opens_highlights_and_confirms_through_vt(tmp_path):
    """A13 主用例: 生产前端按键 -> 菜单 -> 确认 -> 后继真实请求体用新模型."""
    bodies: list = []
    answers = iter([ds_response("思考一", "第一答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = TUIHarness(session)
    try:
        await h.start()
        h.terminal.emit(text_keys("/model") + [ENTER])
        await h.settle()
        # 菜单可见: 提示行 + 默认选中当前模型 flash, Pro 行未选中
        assert h.find("选择模型")
        assert h.menu_row("→ deepseek-flash") is not None
        assert "（当前）" in h.menu_row("→ deepseek-flash")
        assert h.menu_row("deepseek-v4-pro") is not None
        # 硬件光标停留在编辑器提示行, 不被菜单行夺走
        assert "›" in h.screen()[h.vt.cursor_row]
        # 全部可见行不超宽度
        assert all(visible_width(line) <= 60 for line in h.screen())
        h.terminal.emit([DOWN])
        await h.settle()
        assert h.menu_row("→ deepseek-v4-pro") is not None
        h.terminal.emit([ENTER])  # 确认 Pro: 屏障在键分派内同步置位
        assert h.view._gate.model_selecting, "确认时同步屏障先于任务"
        await h.settle()
        assert await _wait_model_settled(h)
        assert not h.view.model_menu.is_open
        assert not h.find("选择模型")
        # 成功事件刷新底栏有效模型 + 确认条目; 启动事实不改写
        model_rows = [i for i, row in enumerate(h.screen()) if "deepseek-v4-pro" in row]
        assert len(model_rows) >= 2  # 确认条目 + 底栏
        footer_row = h.screen()[model_rows[-1]]
        assert model_rows[-1] >= len(h.screen()) - 2  # 底栏区 (末尾两行内)
        assert footer_row.lstrip().startswith("~")  # 模型与工作区同一行 (水平对齐)
        assert h.find("已切换模型")
        assert session.config.model_label == "deepseek-v4-pro"
        # 确认后继请求使用新模型 (A13): 经前端键位提交, mock HTTP 体核对
        h.terminal.emit(text_keys("切换后请求") + [ENTER])
        assert await _wait_until(lambda: session.state == "idle" and not h.view._gate.busy())
        await h.settle()  # 完成事件已落显示文档, 等节流渲染上屏
        assert bodies[-1]["model"] == "deepseek-v4-pro"
        # 思考不进入屏幕; 答案正文照常
        assert not h.find("思考一")
        assert h.find("第一答")
        assert h.frontend.ui_failures == []
    finally:
        errors = await h.stop()
        await session.aclose()
    assert errors == []  # 退出恢复序列干净


async def test_model_menu_escape_dismiss_and_typing_keeps_characters(tmp_path):
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = TUIHarness(session)
    try:
        await h.start()
        h.terminal.emit(text_keys("/model") + [ENTER])
        await h.settle()
        assert h.find("选择模型")
        h.terminal.emit([ESCAPE])
        await h.settle()
        assert not h.view.model_menu.is_open
        assert not h.find("选择模型")
        assert session.state == "idle"  # Esc 收起菜单不触发会话取消
        assert h.bundle["editor"].get_text() == ""
        # 重新打开后打字: 菜单关闭且字符保留
        h.terminal.emit(text_keys("/model") + [ENTER])
        await h.settle()
        h.terminal.emit(text_keys("改"))
        await h.settle()
        assert not h.view.model_menu.is_open
        assert h.bundle["editor"].get_text() == "改"
        assert bodies == []  # 全过程零模型请求
        assert h.frontend.ui_failures == []
    finally:
        await h.stop()
        await session.aclose()


async def test_model_menu_alt_enter_is_edit_not_confirm(tmp_path):
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = TUIHarness(session)
    try:
        await h.start()
        h.terminal.emit(text_keys("/model") + [ENTER])
        await h.settle()
        h.terminal.emit([ESCAPE, ENTER])  # 同批 Alt+Enter: 仅换行, 不确认
        await h.settle()
        assert not h.view.model_menu.is_open
        assert not h.view._gate.model_selecting
        assert h.bundle["editor"].get_text() == "\n"
        assert session.config.model_label == "deepseek-flash"
    finally:
        await h.stop()
        await session.aclose()


@pytest.mark.parametrize("columns,rows", [(30, 8), (60, 16), (120, 30)])
async def test_model_menu_fits_window_and_cursor_stays_in_editor(tmp_path, columns, rows):
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = TUIHarness(session, columns=columns, rows=rows)
    try:
        await h.start()
        h.terminal.emit(text_keys("/model") + [ENTER])
        await h.settle()
        assert h.find("选择模型")
        assert all(visible_width(line) <= columns for line in h.screen())
        assert "›" in h.screen()[h.vt.cursor_row]
        assert h.frontend.ui_failures == []
    finally:
        await h.stop()
        await session.aclose()


async def test_slash_menu_enter_on_model_opens_menu(tmp_path):
    """斜线菜单选中 /model 经统一提交路径只打开模型菜单 (契约 C7)。"""
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = TUIHarness(session)
    try:
        await h.start()
        h.terminal.emit(text_keys("/m"))
        await h.settle()
        assert h.find("→ model")
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.view.model_menu.is_open
        assert h.bundle["editor"].get_text() == ""
        assert bodies == []
    finally:
        await h.stop()
        await session.aclose()


# --- 组合层语义: 屏障/事件/失败/同选/closing -------------------------------------------------


async def test_busy_and_selecting_barrier_silent_with_editable_draft(tmp_path):
    """请求执行中 /model 静默拒绝; 选择进行中 Enter/斜线静默, 草稿可编辑."""
    bodies: list = []
    release = asyncio.Event()
    selection_gate = asyncio.Event()

    async def handle(req):
        bodies.append(json.loads(req.content))
        await release.wait()  # 门控响应: 请求保持 executing
        return httpx.Response(200, content=ds_response("思考", "答"))

    session = await open_application(ds_config(_repo(tmp_path), httpx.MockTransport(handle)))
    h = ViewHarness(session)
    try:
        assert h.submit("第一问")
        assert h.view._gate.busy()  # reservation 同步窗口
        assert not h.submit("/model"), "reservation/executing 内 /model 静默拒绝"
        assert not h.view.model_menu.is_open
        release.set()
        await h.wait_request_done()

        # 打开菜单并确认 Pro; 用门控 select_model 确定性地保持选择进行中
        real_select = session.select_model

        async def gated(option_id: str) -> None:
            await selection_gate.wait()
            await real_select(option_id)

        session.select_model = gated
        assert h.submit("/model")
        assert h.view.model_menu.is_open
        h.view.confirm_model_selection("deepseek-v4-pro")
        assert h.view._gate.model_selecting, "确认时同步屏障先于任务创建"
        assert "正在切换模型" in h.view.status_lines(80)[0]
        # 选择进行中: Enter/斜线/退出静默, 草稿可编辑
        h.bundle["editor"].set_text("选择中的草稿")
        assert not h.submit("选择中的草稿")
        assert h.bundle["editor"].get_text() == "选择中的草稿"
        assert not h.submit("/clear")
        h.view.handle_exit()
        assert h.exit_requested == 0
        assert h.view.handle_ctrl_c() is None  # 静默, 不清草稿
        assert h.bundle["editor"].get_text() == "选择中的草稿"
        selection_gate.set()
        assert await _wait_model_settled(h)
        assert session.config.model_label == "deepseek-v4-pro"
        rendered = chr(10).join(h.lines(80))
        assert "已切换模型：deepseek-flash → deepseek-v4-pro" in rendered
        assert h.view.footer._model_label == "deepseek-v4-pro"
    finally:
        release.set()
        selection_gate.set()
        h.close()
        await session.aclose()


async def test_selection_submits_only_option_id_without_request_or_history(tmp_path):
    """选择不产生 UserMessage/intent/模型轮; 历史与 transcript 用户条目不变 (C7/D8)."""
    bodies: list = []
    answers = iter([ds_response("思考一", "第一答"), ds_response("思考二", "第二答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = ViewHarness(session)
    try:
        assert h.submit("第一问")
        await h.wait_request_done()
        history_before = session.history
        users_before = sum(type(e).__name__ == "UserEntry" for e in h.entries())
        requests_before = len(bodies)
        assert h.submit("/model")
        h.view.confirm_model_selection("deepseek-v4-pro")
        assert await _wait_model_settled(h)
        assert len(bodies) == requests_before, "选择不新增模型调用"
        assert session.history == history_before
        assert sum(type(e).__name__ == "UserEntry" for e in h.entries()) == users_before
        # 确认后继请求使用新模型 (A13): 真实 HTTP 体逐字段核对
        assert h.submit("第二问")
        await h.wait_request_done()
        assert bodies[-1]["model"] == "deepseek-v4-pro"
        assistants = [m for m in bodies[-1]["messages"] if m["role"] == "assistant"]
        assert assistants[0]["reasoning_content"] == "思考一"
        # 思考不进入普通显示; 答案正文照常
        rendered = chr(10).join(h.lines(80))
        assert "思考一" not in rendered and "思考二" not in rendered
        assert "第二答" in rendered
    finally:
        h.close()
        await session.aclose()


async def test_status_and_diagnostics_after_selection(tmp_path):
    """/status 读取有效模型与失效后的预算锚; 启动诊断保留启动事实 (C7/D8)."""
    bodies: list = []
    answers = iter([ds_response("思考一", "第一答")])
    session = await _ds_session(tmp_path, bodies, answers)
    from cicada.ui.conversation_view import build

    view = build(
        session,
        diagnostics=lambda: ["工作区: 启动目录", "模型: deepseek-flash"],
    )["view"]
    try:
        assert view.on_submit("/model")
        view.confirm_model_selection("deepseek-v4-pro")
        deadline = asyncio.get_running_loop().time() + 10.0
        while view._gate.model_selecting and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert not view._gate.model_selecting
        assert view.on_submit("/status")
        rendered = strip_terminal_sequences(chr(10).join(view.document_lines(90)))
        assert "模型: deepseek-v4-pro" in rendered  # 会话段: 有效模型
        assert rendered.count("模型: deepseek-flash") == 1  # 启动诊断: 保留启动事实
        assert "锚: 未立锚" in rendered or "占用估计不可得" in rendered  # 锚已失效
    finally:
        view.close()
        await session.aclose()


async def test_selection_failure_view_only_keeps_model_and_shows_reason(tmp_path):
    """read-only 目录可查看, 选择被拒绝且有界成条; 不解除恢复阻断, 草稿保留 (D8)."""
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = ViewHarness(session)
    try:
        session._view_only_reason = "test view-only"
        assert h.submit("/model")  # 目录仍可查看
        assert h.view.model_menu.is_open
        h.bundle["editor"].set_text("保留我")
        h.view.confirm_model_selection("deepseek-v4-pro")
        assert await _wait_model_settled(h)
        rendered = chr(10).join(h.lines(80))
        assert "模型选择失败" in rendered
        assert "blocked" in rendered  # 有界原因, 无凭据/历史
        assert session.config.model_label == "deepseek-flash"
        assert session._view_only_reason == "test view-only"  # 不解除阻断
        assert h.bundle["editor"].get_text() == "保留我"  # 草稿不被清
        assert not h.view.model_menu.is_open
    finally:
        session._view_only_reason = None
        h.close()
        await session.aclose()


async def test_same_option_is_noop_without_barrier_or_task(tmp_path):
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = ViewHarness(session)
    try:
        assert h.submit("/model")
        before = h.render_requests
        h.view.confirm_model_selection("deepseek-flash")  # 当前模型
        await asyncio.sleep(0.05)
        assert not h.view._gate.model_selecting
        assert h.view._model_task is None, "同选不建任务"
        assert h.render_requests == before, "同选无事件无渲染请求"
        assert len(bodies) == 0
        assert session.persistence.epoch_id is not None
        rendered = chr(10).join(h.lines(80))
        assert "已切换模型" not in rendered  # 无事件无确认条目
    finally:
        h.close()
        await session.aclose()


async def test_model_command_unavailable_without_selection_capability(tmp_path):
    """fake/Ollama 会话无选择能力: /model 给有界说明, 不开菜单 (C7)."""
    session = await open_application(fake_config(_repo(tmp_path), _script({"text": "答", "stop": True})))
    h = ViewHarness(session)
    try:
        assert session.model_options == ()
        assert h.submit("/model")
        assert not h.view.model_menu.is_open
        rendered = chr(10).join(h.lines(80))
        assert "不提供模型选择" in rendered
    finally:
        h.close()
        await session.aclose()


async def test_double_slash_escape_sends_text_without_opening_menu(tmp_path):
    """// 转义保持: //model 作为普通文本发送, 不打开菜单 (D8)."""
    session = await open_application(fake_config(_repo(tmp_path), _script({"text": "答", "stop": True})))
    h = ViewHarness(session)
    try:
        assert h.submit("//model")
        await h.wait_request_done()
        assert not h.view.model_menu.is_open
        model = h.model()
        assert model.requests[0].messages[0].text == "/model"
    finally:
        h.close()
        await session.aclose()


async def test_model_command_and_confirm_after_close_never_reopen(tmp_path):
    """closing/closed 不重开: 关闭后 /model 与确认都静默 (C7 屏障语义)."""
    bodies: list = []
    answers = iter([ds_response("", "答")])
    session = await _ds_session(tmp_path, bodies, answers)
    h = ViewHarness(session)
    try:
        await session.aclose()
        assert not h.submit("/model"), "closed 后 /model 静默拒绝"
        assert not h.view.model_menu.is_open
        h.view.confirm_model_selection("deepseek-v4-pro")
        await asyncio.sleep(0.02)
        assert not h.view._gate.model_selecting
        assert h.view._model_task is None
    finally:
        h.close()


async def test_restored_reasoning_not_in_restore_projection(tmp_path):
    """恢复显示不含思考正文 (A13/D2): restore_view 与 transcript 只渲染 message.text."""
    bodies: list = []
    answers = iter([ds_response("恢复前思考", "恢复前答案")])
    root = _repo(tmp_path)
    session = await open_application(ds_config(root, _transport(bodies, answers)))
    await session.execute("第一问")
    session_id = session.persistence.session_id
    await session.aclose()
    # 重新打开同一存档: 恢复投影覆盖含完整思考的 assistant 消息
    restored = await open_application(ds_config(root, _transport(bodies, iter(())), session_id))
    try:
        assert restored.restored_requests, "应有恢复请求"
        reasoning = restored.history[1].reasoning
        assert reasoning is not None and reasoning.text == "恢复前思考" and reasoning.complete
        h = ViewHarness(restored)
        try:
            rendered = chr(10).join(h.lines(80))
            assert "恢复前思考" not in rendered, "思考正文不得进入恢复显示"
            assert "恢复前答案" in rendered
        finally:
            h.close()
    finally:
        await restored.aclose()
