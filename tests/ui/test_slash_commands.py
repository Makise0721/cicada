"""通过终端按键与实际 VT 屏幕核对斜线命令选择，不依赖菜单内部状态。"""

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui.commands import SlashCommandMenu
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from tests.ui.test_frontend_headless import BACKSPACE, ENTER, ESCAPE, Harness, text_keys

UP = KeyPress(Keys.Up, "\x1b[A")
DOWN = KeyPress(Keys.Down, "\x1b[B")


async def test_menu_wraps_selection_and_filters_typed_prefix():
    h = Harness(commands=True)
    try:
        await h.start()
        h.terminal.emit(text_keys("/") + [UP])
        await h.settle()
        assert h.find("→ exit")
        h.terminal.emit([DOWN])
        await h.settle()
        assert h.find("→ help")
        h.terminal.emit(text_keys("c"))
        await h.settle()
        # 02: /c 前缀现有 compact/clear 两个候选, 选中首个
        assert h.find("→ compact")
        h.terminal.emit([DOWN])
        await h.settle()
        assert h.find("→ clear")
        assert not h.find("显示状态与计量")
        h.terminal.emit([ENTER])
        assert h.submitted == ["/clear"]
        assert h.editor.is_empty()
    finally:
        await h.stop()


async def test_escape_hides_menu_without_cancelling_or_losing_draft():
    h = Harness(commands=True, escape_timeout=0.001)
    try:
        await h.start()
        h.terminal.emit(text_keys("/") + [DOWN])
        await h.settle()
        h.terminal.emit([ESCAPE])
        await h.settle()
        assert h.editor.get_text() == "/"
        assert not h.find("显示状态与计量")
        assert h.escape_count == 0
        h.frontend.request_render()
        await h.settle()
        assert not h.find("显示状态与计量")
        h.terminal.emit([BACKSPACE] + text_keys("/"))
        await h.settle()
        assert h.find("→ help")
    finally:
        await h.stop()


@pytest.mark.parametrize("text", ["//内容", "/unknown", "普通 /help", "/help 参数"])
async def test_non_command_prefix_keeps_original_editing_and_submission(text):
    h = Harness(commands=True)
    try:
        await h.start()
        h.terminal.emit(text_keys(text))
        await h.settle()
        assert not h.find("→ help")
        h.terminal.emit([ENTER])
        assert h.submitted == [text]
    finally:
        await h.stop()


async def test_menu_rejected_enter_keeps_draft_and_selection():
    submitted = []
    ready = False

    def admit(text):
        if not ready:
            return False
        submitted.append(text)
        return True

    h = Harness(commands=True, on_submit=admit)
    try:
        await h.start()
        h.terminal.emit(text_keys("/") + [DOWN, ENTER])
        await h.settle()
        assert h.editor.get_text() == "/"
        assert h.find("→ status")
        assert submitted == []
        ready = True
        h.terminal.emit([ENTER])
        assert submitted == ["/status"]
        assert h.editor.is_empty()
    finally:
        await h.stop()


async def test_alt_enter_and_multiline_paste_do_not_confirm_menu():
    h = Harness(commands=True)
    try:
        await h.start()
        h.terminal.emit(text_keys("/") + [ESCAPE, ENTER])
        await h.settle()
        assert h.editor.get_text() == "/\n"
        assert h.escape_count == 0
        assert h.submitted == []
        h.editor.clear()
        h.terminal.emit([KeyPress(Keys.BracketedPaste, "/\n内容")])
        await h.settle()
        assert h.editor.get_text() == "/\n内容"
        assert not h.find("→ help")
        assert h.submitted == []
    finally:
        await h.stop()


async def test_moving_cursor_out_of_prefix_returns_to_text_editing():
    h = Harness(commands=True)
    try:
        await h.start()
        h.terminal.emit(text_keys("/h") + [KeyPress(Keys.Home, "\x1b[H")])
        await h.settle()
        assert not h.find("→ help")
        h.terminal.emit(text_keys("普通"))
        assert h.editor.get_text() == "普通/h"
    finally:
        await h.stop()


@pytest.mark.parametrize("columns,rows", [(10, 8), (40, 8), (80, 24), (160, 40)])
async def test_menu_fits_window_and_keeps_hardware_cursor_in_editor(columns, rows):
    h = Harness(commands=True, columns=columns, rows=rows)
    try:
        await h.start()
        h.doc_lines = ["历史记录"] * 30
        h.terminal.emit(text_keys("/") + [UP])
        await h.settle()
        assert h.find("→ exit")
        assert h.screen()[h.vt.cursor_row].strip() == "/"
        assert h.frontend.ui_failures == []
    finally:
        await h.stop()


# --- tui-aesthetics 04: 选中行整行高亮 (规格 §D, 字面量断言) ------------------------------------

SEL_BG = "\x1b[48;2;45;45;61m"       # MENU_SEL_BG #2d2d3d
ARROW_FG = "\x1b[38;2;178;148;187m"  # #b294bb
NAME_FG = "\x1b[38;2;129;162;190m"   # #81a2be
DESC_FG = "\x1b[38;2;128;128;128m"   # #808080


def test_v2_selected_row_full_width_highlight():
    menu = SlashCommandMenu()
    menu.sync("/", at_end=True)
    lines = menu.render(60, 4)
    selected = lines[0]
    assert strip_terminal_sequences(selected).startswith("→ help")
    assert selected.startswith(SEL_BG) and selected.endswith("\x1b[0m")
    assert ARROW_FG in selected and NAME_FG in selected and DESC_FG in selected
    assert visible_width(strip_terminal_sequences(selected)) == 60  # 整行背景补齐
    # 选中行内部不出现复位 (内层 \x1b[0m 会清掉整行背景)
    assert selected[:-4].count("\x1b[0m") == 0
    for line in lines[1:]:
        if "(" in strip_terminal_sequences(line):
            continue  # 分页指示行
        assert not line.startswith(SEL_BG)  # 未选中行无背景


def test_v2_selected_row_narrow_width_truncates_description():
    menu = SlashCommandMenu()
    menu.sync("/", at_end=True)
    for line in menu.render(12, 3):
        assert visible_width(strip_terminal_sequences(line)) <= 12
