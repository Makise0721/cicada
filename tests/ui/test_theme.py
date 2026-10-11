"""票据 04 检查: Pi 主题、消息排版、底栏与前端布局 (P01/P08/P09/P11/P13).

P01 用户背景/边距/无角色助手/上下细线编辑器/两行底栏对齐固定 Pi 场景;
P08 尺寸矩阵重排; P09 完整重绘回放与静态排版一致; P11 宽字符列宽;
P13 计量归属与未知语义。真实终端观感由用户人工核对 (P16)。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("color_env")

from cicada.budget import BudgetReport, BudgetSnapshot, ComponentEstimate
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
    footer = FooterComponent(r"C:\repo\demo", "deepseek-flash", lambda: (1500, False, 300, False))
    lines = footer.render(40, 24)
    assert len(lines) == 2
    first = strip_terminal_sequences(lines[0])
    # 行 1: 工作区左 + 模型右 (同一行水平对齐)
    assert "demo" in first and first.endswith("deepseek-flash")
    # 行 2: 计量左 (规格 §E: · 分隔)
    assert "累计 ↑1.5k · ↓300" in strip_terminal_sequences(lines[1])
    assert "\x1b[38;2;128;128;128m" in lines[0]  # dim 提亮 #808080 (规格 §E)


def test_footer_unknown_metrics_not_zero_and_partial_marked():
    # 无任何计量: 不显示箭头不虚构 0
    footer = FooterComponent("w", "m", lambda: (None, False, None, False))
    lines = footer.render(60, 24)
    assert strip_terminal_sequences(lines[0]).strip().endswith("m")  # 模型在工作区行
    stats = strip_terminal_sequences(lines[1])
    assert "↑" not in stats and "↓" not in stats
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
    assert visible_width(strip_terminal_sequences(lines[0])) <= 20  # 工作区+模型行
    assert visible_width(strip_terminal_sequences(lines[1])) <= 20
    assert "↑10" in strip_terminal_sequences(lines[1])  # 计量优先保留


def test_format_workspace_home_abbreviation(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    inside = tmp_path / "sub" / "repo"
    assert format_workspace(str(inside)) == "~/sub/repo"
    outside = tmp_path.parent / "elsewhere"
    assert format_workspace(str(outside)) == str(outside)


# --- thinking-budget-visibility 03: 底栏最近预算段 (A08/A10) ----------------------------


def _budget_snapshot(
    total: int | None, limit: int, source: str = "estimated", *, snapshot_limit: int | None = None,
) -> BudgetSnapshot:
    """真实 BudgetReport/Snapshot: total = 输入 + 预留 4096 + 余量 256。"""
    if total is None:
        return BudgetSnapshot(limit=snapshot_limit if snapshot_limit is not None else limit,
                              report=None, anchor=None)
    report = BudgetReport(
        limit=limit,
        components=(ComponentEstimate("system_prompt", total - 4096 - 256, source),),
        input_estimate=total - 4096 - 256,
        output_reserve=4096,
        safety_margin=256,
        total_estimate=total,
        source=source,
    )
    return BudgetSnapshot(limit=snapshot_limit if snapshot_limit is not None else limit,
                          report=report, anchor=None)


def _no_metrics() -> tuple[int | None, bool, int | None, bool]:
    return None, False, None, False


@pytest.mark.parametrize("source,long_label", [
    ("estimated", "输入估计"), ("anchored", "输入锚定"),
    ("measured", "输入实测"), ("unknown", "输入未知"),
])
def test_footer_budget_full_segment_approximate_and_source(source, long_label):
    footer = FooterComponent("w", "m", _no_metrics,
                             budget=lambda: _budget_snapshot(8300, 131072, source))
    segment = strip_terminal_sequences(footer.render(80, 24)[1])
    # 总需求始终带 ~ (measured 也不例外); 百分比原始整数一位小数 (规格 D5 示例级)
    assert segment == f"最近预算 ~8.3k/131k · 6.3% · {long_label}"


def test_footer_budget_states_not_fabricated_and_pending_shows_known_limit():
    # 未启用: 快照 None 不补 0, 累计照常同段显示
    footer = FooterComponent("w", "m", lambda: (100, False, None, False), budget=lambda: None)
    assert strip_terminal_sequences(footer.render(80, 24)[1]) == "预算未启用 · 累计 ↑100"
    # 待评估: 无报告显示已知上限, 不伪造占用
    pending = FooterComponent("w", "m", _no_metrics, budget=lambda: _budget_snapshot(None, 131072))
    segment = strip_terminal_sequences(pending.render(80, 24)[1])
    assert segment == "最近预算 待评估 · 上限 131k"
    # 快照与报告上限不一致: 按待评估, 不拼接不同身份的数值
    mismatch = FooterComponent(
        "w", "m", _no_metrics,
        budget=lambda: _budget_snapshot(8300, 131072, snapshot_limit=999999),
    )
    assert strip_terminal_sequences(mismatch.render(80, 24)[1]) == "最近预算 待评估"


def test_footer_budget_over_limit_by_raw_integers_not_capped():
    # 原始整数超限判定, 百分比不封顶 (A08)
    over = FooterComponent("w", "m", _no_metrics,
                           budget=lambda: _budget_snapshot(262144, 131072))
    assert strip_terminal_sequences(over.render(80, 24)[1]) == \
        "最近预算 超限 ~262k/131k · 200.0% · 输入估计"
    # 舍入显示 100.0% 时仍保留超限标记
    rounded = FooterComponent("w", "m", _no_metrics,
                              budget=lambda: _budget_snapshot(100049, 100000))
    segment = strip_terminal_sequences(rounded.render(80, 24)[1])
    assert "100.0%" in segment and "超限" in segment


def test_footer_budget_levels_pick_by_width_and_keep_numbers_whole():
    footer = FooterComponent("w", "m", _no_metrics,
                             budget=lambda: _budget_snapshot(8300, 131072))

    def segment(width: int) -> str:
        return strip_terminal_sequences(footer.render(width, 24)[1])

    assert segment(37) == "最近预算 ~8.3k/131k · 6.3% · 输入估计"  # 阈值取边界
    assert segment(22) == "最近预算 估 ~8.3k/131k"
    assert segment(18) == "预算 估 ~8.3k/131k"
    assert segment(7) == "预算 估"
    assert segment(4) == "预算"
    assert all(visible_width(segment(width)) <= width for width in (1, 2, 4, 7, 18, 22, 37))


def test_footer_budget_over_limit_levels_keep_flag_and_numbers_whole():
    over = FooterComponent("w", "m", _no_metrics,
                           budget=lambda: _budget_snapshot(262144, 131072))

    def segment(width: int) -> str:
        return strip_terminal_sequences(over.render(width, 24)[1])

    assert segment(44) == "最近预算 超限 ~262k/131k · 200.0% · 输入估计"
    assert segment(43) == "最近预算 超限 ~262k/131k · 输入估计"
    assert segment(34) == "预算 超限 ~262k/131k · 估"
    assert segment(24) == "预算 超限 ~262k/131k"
    assert segment(19) == "预算 超限"
    assert segment(4) == "超限"


def test_footer_budget_low_height_budget_first_model_only_when_preserved():
    budget = lambda: _budget_snapshot(8300, 131072)
    footer = FooterComponent("ws", "test-model", lambda: (10, False, 5, False), budget=budget)
    # 低高度单行: 累计让位; 40 列放不下完整预算+模型, 预算为先
    line = strip_terminal_sequences(footer.render(40, 8)[0])
    assert line == "最近预算 ~8.3k/131k · 6.3% · 输入估计"
    assert "累计" not in line
    # 52 列预算段完整保留后模型加入
    wide = strip_terminal_sequences(footer.render(52, 8)[0])
    assert wide.startswith("最近预算") and wide.endswith("test-model")
    # 30 列: 完整级放不下退次级, 模型让位、预算段不截断
    narrow = FooterComponent("ws", "very-long-model-name",
                             lambda: (10, False, 5, False), budget=budget)
    line2 = strip_terminal_sequences(narrow.render(30, 8)[0])
    assert line2 == "最近预算 估 ~8.3k/131k"


def test_footer_budget_callback_read_once_per_render():
    calls: list[int] = []

    def budget() -> BudgetSnapshot:
        calls.append(1)
        return _budget_snapshot(8300, 131072)

    footer = FooterComponent("w", "m", _no_metrics, budget=budget)
    footer.render(80, 24)
    footer.render(80, 8)
    footer.render(20, 24)
    assert len(calls) == 3, "契约 C5: 每 render 至多读取一次预算回调"


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
        assert "累计 ↑1.2k · ↓300" in joined  # P13: 计量底栏 (规格 §E: · 分隔)
        assert any("test-model" in line for line in screen)  # 模型右对齐
        # 编辑器上下细线存在 (P01)
        assert any(set(strip_terminal_sequences(line)) == {"─"} for line in screen)
    finally:
        await h.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("width,rows", [(40, 8), (80, 24), (160, 40)])
async def test_layout_resize_matrix_replays_consistently(width, rows):
    """P08/P09: 尺寸矩阵下重排当前文档; 完整重绘回放与新宽度静态排版一致。

    (12 组合矩阵精简为窄低/常规/宽高三代表; 全状态宽高往返由
    test_resize_stability 的生产组合覆盖。)"""
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
