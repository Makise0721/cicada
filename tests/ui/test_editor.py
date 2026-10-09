"""票据 01 聚焦检查: Pi 式编辑器 (几何、图素光标、换行滚动、撤销)。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("color_env")

from cicada.ui.editor import Editor, create_scroll_border
from cicada.ui.main_screen import CURSOR_MARKER, visible_width


def stripped_lines(lines: list[str]) -> list[str]:
    """去掉样式与光标标记, 取可见文本。"""
    from cicada.ui.main_screen import strip_terminal_sequences

    return [strip_terminal_sequences(line).replace(CURSOR_MARKER, "").rstrip() for line in lines]


def test_render_geometry_borders_and_padding():
    editor = Editor()
    editor.focused = True
    editor.insert_text("hello")
    lines = editor.render(20)
    assert len(lines) == 3  # 上横线 + 1 正文 + 下横线
    assert stripped_lines([lines[0]])[0] == "─" * 20
    assert stripped_lines([lines[2]])[0] == "─" * 20
    # 光标在行尾: 反显空格 + 聚焦时硬件光标标记; 反显用 \x1b[27m 收尾,
    # 不复位全部属性 (背景行内嵌套光标时 \x1b[0m 会连带清掉外层背景)
    assert CURSOR_MARKER in lines[1]
    assert "\x1b[7m \x1b[27m" in lines[1]


def test_cursor_marker_only_when_focused():
    editor = Editor()
    editor.focused = False
    editor.insert_text("x")
    lines = editor.render(10)
    assert CURSOR_MARKER not in lines[1]
    editor.focused = True
    lines = editor.render(10)
    assert CURSOR_MARKER in lines[1]


def test_grapheme_cursor_backspace_and_move():
    editor = Editor()
    editor.insert_text("中文")
    editor.move_cursor(0, -1)  # 左移一图素 (整字)
    editor.insert_text("间")
    assert editor.get_text() == "中间文"
    editor.backspace()
    assert editor.get_text() == "中文"
    # ZWJ emoji 作为一个图素: 一次退格整体删除
    editor.move_to_line_end()
    editor.insert_text("\U0001F468\u200D\U0001F469")
    editor.backspace()
    assert editor.get_text() == "中文"


def test_multiline_insert_and_newline():
    editor = Editor()
    editor.insert_text("第一行")
    editor.insert_newline()
    editor.insert_text("第二行")
    assert editor.get_text() == "第一行\n第二行"
    for _ in range(3):  # 光标逐图素左移到行首
        editor.move_cursor(0, -1)
    editor.backspace()  # 合并上一行
    assert editor.get_text() == "第一行第二行"


def test_wrapped_line_cursor_layout():
    editor = Editor()
    editor.insert_text("一二三四五六七八九十")  # 20 显示列
    lines = editor.render(12)
    # 12 列编辑器: 无侧边距时排版宽 11 列 -> 每行 5 个双列字
    assert len(lines) == 4
    body = stripped_lines(lines[1:3])
    assert body[0] == "一二三四五"
    assert body[1] == "六七八九十"


def test_internal_scrollback_with_indicators():
    editor = Editor()
    editor.insert_text("行一\n行二\n行三\n行四\n行五\n行六")
    editor.move_to_line_end()  # 光标在最后一行
    lines = editor.render(20, max_visible_rows=3)
    assert "↑" in stripped_lines([lines[0]])[0]  # 上方有隐藏行
    visible_body = stripped_lines(lines[1:4])
    assert visible_body == ["行四", "行五", "行六"]  # 滚动保持光标可见
    assert "↓" not in stripped_lines([lines[4]])[0]
    # 光标上移到第一可视行以上 -> 视口跟随
    editor.move_cursor(-3, 0)
    lines = editor.render(20, max_visible_rows=3)
    assert "行三" in "".join(stripped_lines(lines))


def test_undo_restores_previous_state():
    editor = Editor()
    editor.insert_text("abc")
    editor.insert_text("def")
    editor.undo()
    assert editor.get_text() == "abc"
    editor.undo()
    assert editor.get_text() == ""
    editor.undo()  # 空栈: 无动作
    assert editor.get_text() == ""


def test_submit_clears_and_returns_text():
    editor = Editor()
    editor.insert_text("  提问内容  ")
    result = editor.submit()
    assert result == "提问内容"
    assert editor.is_empty()
    editor.disable_submit = True
    editor.insert_text("草稿")
    assert editor.submit() is None
    assert editor.get_text() == "草稿"


def test_scroll_border_formats():
    border = create_scroll_border("↑", 5, 20)
    assert visible_width(border) == 20
    assert "↑ 5 more" in border
    narrow = create_scroll_border("↓", 12, 6)
    assert visible_width(narrow) == 6


def test_editor_shrink_clears_with_clear_on_shrink_source():
    """编辑器行数变化经 renderer 差分; clear_on_shrink 是 renderer 开关, 编辑器只提供行。"""
    editor = Editor()
    editor.insert_text("a\nb")
    assert len(editor.render(10)) == 4
    editor.backspace()
    editor.backspace()
    assert len(editor.render(10)) == 3  # 空编辑器: 上下横线 + 1 空正文


# --- tui-aesthetics 03: 编辑器 v2 (规格 §C, 字面量断言不复算实现) -----------------------------

V2_BG = "\x1b[48;2;31;31;40m"       # 已移除的输入区底色 (#1f1f28), 用于断言"不再出现"
V2_PROMPT_FG = "\x1b[38;2;178;148;187m"  # 提示符 #b294bb
V2_PLACEHOLDER_FG = "\x1b[38;2;106;106;106m"  # 占位 #6a6a6a


def v2_editor() -> Editor:
    from cicada.ui.editor import EditorStyle

    return Editor(
        padding_x=2,
        style=EditorStyle(
            prompt_glyph="›",
            prompt_color="#b294bb",
            placeholder="输入消息… Enter 发送, Shift+Enter 换行",
            placeholder_color="#6a6a6a",
        ),
    )


def test_v2_empty_editor_prompt_cursor_and_placeholder():
    editor = v2_editor()
    editor.focused = True
    lines = editor.render(60)
    body = lines[1]
    assert "\x1b[48;2;" not in body  # 输入区无底色 (2026-10-07 用户裁决)
    plain = strip_terminal_sequences_local(body)
    assert plain.startswith("› ")  # 提示符占左边距
    assert "输入消息… Enter 发送, Shift+Enter 换行" in plain  # 空态占位
    assert "\x1b[7m \x1b[27m" in body  # 光标仍在行首
    assert V2_PROMPT_FG in body and V2_PLACEHOLDER_FG in body
    assert visible_width(strip_terminal_sequences_local(body)) == 60


def test_v2_draft_hides_placeholder_and_keeps_prompt():
    editor = v2_editor()
    editor.focused = True
    editor.insert_text("你好")
    body = editor.render(40)[1]
    plain = strip_terminal_sequences_local(body)
    assert plain.startswith("› 你好")
    assert "输入消息" not in plain
    assert V2_PLACEHOLDER_FG not in body


def test_v2_multiline_only_first_line_has_prompt():
    editor = v2_editor()
    editor.focused = True
    editor.insert_text("第一行")
    editor.insert_newline()
    editor.insert_text("第二行")
    lines = editor.render(40)
    first, second = strip_terminal_sequences_local(lines[1]), strip_terminal_sequences_local(lines[2])
    assert first.startswith("› 第一行")
    assert second.startswith("  第二行")  # 其余行左边距留空对齐
    for line in lines[1:3]:
        assert "\x1b[48;2;" not in line  # 无底色


def test_default_editor_without_style_unchanged():
    """无 style 的编辑器不产生底色/提示符/占位 (兼容既有调用方)。"""
    editor = Editor()
    editor.focused = True
    lines = editor.render(20)
    assert not lines[1].startswith("\x1b[48;2;")
    assert "›" not in strip_terminal_sequences_local(lines[1])


def strip_terminal_sequences_local(line: str) -> str:
    from cicada.ui.main_screen import strip_terminal_sequences

    return strip_terminal_sequences(line).replace(CURSOR_MARKER, "")
