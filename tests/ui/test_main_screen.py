"""票据 01 聚焦检查: ANSI 列宽工具与主屏差分 renderer (P08/P10/P11 最小场景)."""

from __future__ import annotations

import pytest

from cicada.ui.main_screen import (
    CURSOR_MARKER,
    MainScreen,
    iter_graphemes,
    slice_by_column,
    truncate_to_width,
    visible_width,
    word_wrap_line,
)
from tests.ui.vt_screen import VtScreen


# --- P11: 列宽与切片 --------------------------------------------------------------


def test_visible_width_cjk_combining_zwj_emoji():
    assert visible_width("abc") == 3
    assert visible_width("中文") == 4  # 双列
    # 组合字符: 基字符 + combining 零宽
    assert visible_width("e\u0301") == 1
    # ZWJ emoji (家庭 emoji): 保守按一个 2 列图素, 后续 ZWJ/附标零宽
    assert visible_width("\U0001F468\u200D\U0001F469\u200D\U0001F467") == 2
    # ANSI 样式不计宽
    assert visible_width("\x1b[31m红\x1b[0m") == 2
    assert visible_width(CURSOR_MARKER) == 0
    assert visible_width("a\tb") == 5  # tab 3 列


def test_iter_graphemes_boundaries():
    assert iter_graphemes("中文") == ["中", "文"]
    assert iter_graphemes("e\u0301") == ["e\u0301"]
    assert iter_graphemes("\U0001F468\u200D\U0001F469") == ["\U0001F468\u200D\U0001F469"]
    assert iter_graphemes("\U0001F1E8\U0001F1F3x") == ["\U0001F1E8\U0001F1F3", "x"]


def test_slice_by_column_ansi_and_cjk():
    line = "\x1b[31m中文abc\x1b[0m"
    sliced = slice_by_column(line, 0, 2)
    assert "\x1b[31m" in sliced and visible_width(sliced) == 2  # 样式前缀随内容保留
    assert visible_width(slice_by_column(line, 2, 3)) == 3  # 文abc
    # strict: 边界处宽字符不越界; 非严格允许整图素跨边界 (Pi 语义)
    assert slice_by_column("中文", 0, 3, strict=True) == "中"
    assert slice_by_column("中文", 0, 3) == "中文"


def test_word_wrap_line_cjk_and_long_word():
    chunks = word_wrap_line("你好世界测试", 6)  # 3 个双列字每行
    assert [c.text for c in chunks] == ["你好世", "界测试"]
    chunks = word_wrap_line("hello 世界 world", 8)
    assert [c.text for c in chunks] == ["hello 世", "界 world"]
    # 超长单词按列强制断
    chunks = word_wrap_line("aaaaaaaaaaaa", 5)
    assert all(visible_width(c.text) <= 5 for c in chunks)
    assert "".join(c.text for c in chunks) == "aaaaaaaaaaaa"


def test_truncate_to_width():
    assert truncate_to_width("中文内容", 5) == "中文…"
    assert truncate_to_width("short", 10) == "short"


# --- renderer: 用真实 VT 屏验证 -----------------------------------------------------


class Doc:
    """可变文档源: 模拟条目行列表。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, width: int) -> list[str]:
        return list(self.lines)


def build(vt_columns=20, vt_rows=10):
    vt = VtScreen(vt_columns, vt_rows)
    writes: list[str] = []
    doc = Doc()

    def write(data: str) -> None:
        writes.append(data)
        vt.feed(data)

    screen = MainScreen(doc, write)
    return vt, writes, doc, screen


def render(screen: MainScreen, columns: int, rows: int) -> None:
    screen.render_once(columns, rows)


def test_first_render_appends_without_clear():
    vt, writes, doc, screen = build()
    doc.lines = ["第一行", "second"]
    render(screen, 20, 10)
    assert "\x1b[2J" not in "".join(writes)  # 首帧不清屏
    assert vt.screen_text() == ["第一行", "second"]


def test_no_change_frame_writes_nothing():
    vt, writes, doc, screen = build()
    doc.lines = ["a", "b"]
    render(screen, 20, 10)
    assert len(writes) == 2  # 帧 + 无光标标记时的隐藏光标写
    writes.clear()
    render(screen, 20, 10)
    assert writes == []  # P10: 无修改帧不重复写屏


def test_append_and_modify_last_line_diff():
    vt, writes, doc, screen = build()
    doc.lines = ["alpha", "beta"]
    render(screen, 20, 10)
    writes.clear()
    doc.lines = ["alpha", "beta", "gamma"]
    render(screen, 20, 10)
    assert vt.screen_text() == ["alpha", "beta", "gamma"]
    # 追加帧只写增量, 不含清屏
    frame = "".join(writes)
    assert "\x1b[2J" not in frame
    writes.clear()
    doc.lines = ["alpha", "gam"]  # 最后一行缩短: 清尾重写
    render(screen, 20, 10)
    assert vt.screen_text() == ["alpha", "gam"]  # P10: 缩短行清除旧尾, 无残影
    assert "\x1b[2J" not in "".join(writes)


def test_shrink_content_clears_extra_lines():
    vt, writes, doc, screen = build()
    doc.lines = ["one", "two", "three"]
    render(screen, 20, 10)
    doc.lines = ["one"]
    render(screen, 20, 10)
    assert vt.screen_text() == ["one"]  # 多余行被清除


def test_width_change_full_redraw_replays_document():
    # 同宽重复渲染无写; 宽度变化触发 2J/3J 完整回放 (P08/P09 最小场景)
    vt, writes, doc, screen = build()
    doc.lines = ["宽9527", "second"]
    render(screen, 20, 10)
    assert screen.full_redraws == 1  # 首帧即 full_render(False)
    writes.clear()
    render(screen, 20, 10)
    assert writes == []

    vt2, writes2, doc2, screen2 = build(vt_columns=12, vt_rows=10)
    doc2.lines = ["宽9527", "second"]
    render(screen2, 20, 10)  # 20 列首帧
    writes2.clear()
    doc2.lines = ["宽9527", "second"]
    render(screen2, 12, 10)  # 收窄: full render
    assert screen2.full_redraws == 2
    assert "\x1b[2J\x1b[H\x1b[3J" in "".join(writes2)
    # 回放后的屏幕与新宽度静态排版一致
    assert vt2.screen_text() == ["宽9527", "second"]


def test_height_change_triggers_full_redraw():
    vt, writes, doc, screen = build()
    doc.lines = ["a", "b"]
    render(screen, 20, 10)
    writes.clear()
    doc.lines = ["a", "b", "c"]
    render(screen, 20, 8)
    assert screen.full_redraws == 2  # 首帧 + 高度变化
    assert "\x1b[3J" in "".join(writes)


def test_cursor_marker_positions_hardware_cursor():
    vt, writes, doc, screen = build()
    doc.lines = ["user input here", "assistant answer"]
    render(screen, 20, 10)
    # 在最后一行中部插入光标标记
    doc.lines = ["user input here", "assistant " + CURSOR_MARKER + "answer"]
    render(screen, 20, 10)
    frame = "".join(writes)
    assert CURSOR_MARKER not in frame  # 标记被剥除, 不进输出
    # 硬件光标定位到标记列 (col 10 -> CHA 11G)
    assert "\x1b[11G" in frame
    assert vt.cursor_visible


def test_no_marker_hides_cursor():
    vt, writes, doc, screen = build()
    doc.lines = ["plain"]
    render(screen, 20, 10)
    assert "\x1b[?25l" in "".join(writes)  # 无标记: 隐藏光标
    writes.clear()
    doc.lines = ["plain2"]
    render(screen, 20, 10)
    assert "\x1b[?25l" not in "".join(writes)  # 已隐藏则不重复写 (P10)


def test_overwide_line_truncated_with_failure_record():
    vt, writes, doc, screen = build()
    doc.lines = ["这一行远远超过二十列的宽度限制会被截断记录"]
    render(screen, 20, 10)
    assert screen.ui_failures and "exceeds width" in screen.ui_failures[0]
    assert all(visible_width(vt.line_text(r)) <= 20 for r in range(10))


def test_lines_above_viewport_force_full_redraw():
    vt, writes, doc, screen = build(vt_columns=20, vt_rows=4)
    doc.lines = [f"line{i}" for i in range(10)]  # 10 行 > 4 行视口
    render(screen, 20, 4)
    assert screen.full_redraws == 1
    writes.clear()
    # 视口上方 (已滚出) 的行变化: 差分写不到 -> fullRender
    doc.lines = ["CHANGED"] + [f"line{i}" for i in range(1, 10)]
    render(screen, 20, 4)
    assert screen.full_redraws == 2
    assert "\x1b[2J" in "".join(writes)
