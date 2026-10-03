"""票据 04 检查: Pi 主题、消息排版、底栏与前端布局 (P01/P08/P09/P11/P13).

P01 用户背景/边距/无角色助手/上下细线编辑器/两行底栏对齐固定 Pi 场景;
P08 尺寸矩阵重排; P09 完整重绘回放与静态排版一致; P11 宽字符列宽;
P13 计量归属与未知语义。真实终端观感由用户人工核对 (P16)。
"""

from __future__ import annotations

import pytest

from cicada.ui import theme
from cicada.ui.editor import Editor
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.terminal_io import FakeTerminal
from cicada.ui.theme import (
    EDITOR_BORDER,
    FooterComponent,
    assistant_markdown_lines,
    bg,
    fg,
    format_tokens,
    format_workspace,
    user_message_lines,
)
from tests.ui.vt_screen import VtScreen


def plain(lines: list[str]) -> list[str]:
    return [strip_terminal_sequences(line).rstrip() for line in lines]


# --- P01: 用户消息背景盒 --------------------------------------------------------------


def test_user_message_box_geometry_and_background():
    lines = user_message_lines("你好", 20)
    assert len(lines) == 3  # 上下 1 行整宽背景
    assert lines[0] == bg(theme.USER_MESSAGE_BG, " " * 20)
    body = plain(lines[1:2])[0]
    assert body == " 你好"
    assert visible_width(lines[1]) == 20  # 整行背景 (含填充)
    assert lines[1].startswith("\x1b[48;2;52;53;65m")


def test_user_message_multiline_and_cjk_narrow():
    lines = user_message_lines("第一行\n第二行", 12)
    assert [x.strip() for x in plain(lines) if x.strip()] == ["第一行", "第二行"]
    assert all(visible_width(strip_terminal_sequences(line)) <= 12 for line in lines)
    # 极窄: 有界退化, 不抛错不超宽
    narrow = user_message_lines("中文内容", 2)
    assert all(visible_width(strip_terminal_sequences(line)) <= 2 for line in narrow)


# --- P01/P11: 助手 Markdown 排版 ------------------------------------------------------


def test_assistant_markdown_heading_colors_and_indent():
    text = "# 标题\n\n正文段落，包含**强调**。"
    lines = assistant_markdown_lines(text, 40)
    body = [line for line in lines if line.strip()]
    assert visible_width("# 标题") == 6
    assert all(visible_width(strip_terminal_sequences(line)) <= 40 for line in body)
    assert all(line.startswith(" ") for line in body)  # 左右各 1 列缩进
    heading_line = body[0]
    assert "38;2;240;198;116m" in heading_line  # mdHeading #f0c674 (可能与 bold 合成序列)
    # 缓存: 同 (正文, 宽度) 二次渲染返回等价内容
    assert assistant_markdown_lines(text, 40) == lines


def test_assistant_markdown_empty_and_cache_invalidation():
    assert assistant_markdown_lines("", 40) == []
    first = assistant_markdown_lines("第一版", 40)
    second = assistant_markdown_lines("第一版第二版", 40)
    assert first != second  # 正文变化后缓存失效重排


# --- P01: 编辑器边框颜色 --------------------------------------------------------------


def test_editor_border_uses_fixed_appearance_color():
    editor = Editor(border_color=theme.editor_border())
    editor.insert_text("草稿")
    lines = editor.render(20)
    r, g, b = (178, 148, 187)  # #b294bb
    assert f"\x1b[38;2;{r};{g};{b}m" in lines[0]
    assert strip_terminal_sequences(lines[0]) == "─" * 20


# --- P13: 计量紧凑格式与底栏 ----------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "0"), (999, "999"), (1500, "1.5k"), (25000, "25k"), (1500000, "1.5M"), (25000000, "25M")],
)
def test_format_tokens_compact(count, expected):
    assert format_tokens(count) == expected


def test_footer_two_lines_workspace_and_right_aligned_model():
    footer = FooterComponent(r"C:\repo\demo", "qwen3.5:9b", lambda: (1500, False, 300, False))
    lines = footer.render(40, 24)
    assert len(lines) == 2
    assert "累计 ↑1.5k ↓300" in strip_terminal_sequences(lines[1])
    # 模型右对齐 (去掉 dim 样式后行尾是模型名)
    assert strip_terminal_sequences(lines[1]).endswith("qwen3.5:9b")
    assert "\x1b[38;2;102;102;102m" in lines[0]  # dim #666666
    # 工作区行含工作区路径
    assert "demo" in strip_terminal_sequences(lines[0])


def test_footer_unknown_metrics_not_zero_and_partial_marked():
    # 无任何计量: 不显示箭头不虚构 0
    footer = FooterComponent("w", "m", lambda: (None, False, None, False))
    stats = strip_terminal_sequences(footer.render(60, 24)[1])
    assert "↑" not in stats and "↓" not in stats
    assert stats.strip().endswith("m")
    # 字段缺失: 有值照显 + (部分), 缺失标不可得
    footer2 = FooterComponent("w", "m", lambda: (100, True, None, True))
    stats2 = strip_terminal_sequences(footer2.render(60, 24)[1])
    assert "↑100(部分)" in stats2
    assert "↓不可得" in stats2


def test_footer_low_height_keeps_stats_only():
    footer = FooterComponent(r"C:\repo\demo", "m", lambda: (10, False, 5, False))
    lines = footer.render(40, 8)
    assert len(lines) == 1  # 低高度: 工作区行让位
    assert "↑10" in strip_terminal_sequences(lines[0])


def test_footer_narrow_truncates_model_not_dropping_stats():
    footer = FooterComponent("w", "很长的模型名称" * 10, lambda: (10, False, 5, False))
    lines = footer.render(20, 24)
    assert visible_width(strip_terminal_sequences(lines[1])) <= 20
    assert "↑10" in strip_terminal_sequences(lines[1])  # 计量优先保留


def test_format_workspace_home_abbreviation(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    inside = tmp_path / "sub" / "repo"
    assert format_workspace(str(inside)) == "~/sub/repo"
    outside = tmp_path.parent / "elsewhere"
    assert format_workspace(str(outside)) == str(outside)


# --- P08/P09/P06: 前端布局组合 (文档 + 编辑器 + 底栏) 与缩放回放 -------------------------


class LayoutHarness:
    """MainScreenFrontend + VT 屏: 固定文档内容, 检查布局与缩放回放。"""

    def __init__(self, columns=80, rows=24):
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.doc_lines = ["用户问题在背景盒中", "回答正文第一行", "回答正文第二行"]
        footer = FooterComponent("ws", "test-model", lambda: (1200, False, 300, False))
        editor = Editor(border_color=theme.editor_border())
        editor.focused = True
        self.editor = editor
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=lambda width: list(theme.user_message_lines(self.doc_lines[0], width)) + self.doc_lines[1:],
            editor=editor,
            footer_lines=footer.render,
        )
        self.runner = None

    def resize(self, columns: int, rows: int) -> None:
        """同步换尺寸并重建 VT 屏 (renderer 会完整回放, 新屏从零解释下一帧)。"""
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.terminal.set_size(columns, rows)

    async def start(self):
        import asyncio

        self.runner = asyncio.ensure_future(self.frontend.run_async())
        for _ in range(4):
            await asyncio.sleep(0.01)

    async def stop(self):
        self.frontend.request_exit()
        import asyncio

        await asyncio.wait_for(self.runner, 5)


def _visible(line: str) -> str:
    return strip_terminal_sequences(line).rstrip()


@pytest.mark.asyncio
async def test_layout_composition_document_editor_footer():
    h = LayoutHarness(columns=80, rows=24)
    try:
        await h.start()
        screen = h.vt.screen_text()
        joined = "\n".join(screen)
        assert "用户问题在背景盒中" in joined  # P01: 用户背景盒
        assert "回答正文第一行" in joined
        assert "累计 ↑1.2k ↓300" in joined  # P13: 计量底栏
        assert any("test-model" in line for line in screen)  # 模型右对齐
        # 编辑器上下细线存在 (P01)
        assert any(set(strip_terminal_sequences(line)) == {"─"} for line in screen)
    finally:
        await h.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [40, 80, 120, 160])
@pytest.mark.parametrize("rows", [8, 24, 40])
async def test_layout_resize_matrix_replays_consistently(width, rows):
    """P08/P09: 尺寸矩阵下重排当前文档; 完整重绘回放与新宽度静态排版一致。"""
    h = LayoutHarness(columns=61, rows=23)  # 初始尺寸异于所有目标: 保证 resize 必触发重绘
    try:
        await h.start()
        h.resize(width, rows)
        for _ in range(4):
            import asyncio

            await asyncio.sleep(0.01)
        # 静态排版基准在 resize 后取 (低高度底栏/编辑器随当前尺寸退让)
        baseline_static = h.frontend._render_frame_lines(width)
        # VT 屏可见内容与新宽度静态排版一致 (P09): 逐行剥样式比较
        static_plain = [_visible(line) for line in baseline_static]
        visible_rows = min(rows, len(static_plain))
        screen_plain = h.vt.screen_text(strip_blank_tail=False)
        top = len(static_plain) - visible_rows  # 滚出视口的部分不进屏幕
        for i in range(visible_rows):
            assert _visible(screen_plain[i]) == static_plain[top + i], f"row {i} at {width}x{rows}"
        assert all(visible_width(strip_terminal_sequences(line)) <= width for line in baseline_static)
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_layout_low_height_editor_three_rows():
    """设计 §4: 8 行时编辑器正文 1–3 行, 底栏单行, 剩余给对话。"""
    h = LayoutHarness(columns=60, rows=8)
    try:
        await h.start()
        static = h.frontend._render_frame_lines(60)
        # 底栏 1 行 + 编辑器 (上下边框 + ≤3 正文) + 文档
        footer_lines = [line for line in static if "累计" in strip_terminal_sequences(line)]
        assert len(footer_lines) == 1
        border_rows = [line for line in static if set(strip_terminal_sequences(line)) == {"─"}]
        assert len(border_rows) == 2
    finally:
        await h.stop()
