"""票据 01 聚焦检查: Pi 式编辑器 (几何、图素光标、换行滚动、撤销)。"""

from __future__ import annotations

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
    # 光标在行尾: 反显空格 + 聚焦时硬件光标标记
    assert CURSOR_MARKER in lines[1]
    assert "\x1b[7m \x1b[0m" in lines[1]


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
