"""Pi dark 主题、消息排版与底栏 (票据 04).

来源: pi-mono@1a584a7a packages/coding-agent/src/modes/interactive/theme/dark.json、
components/footer.ts、user-message.ts (MIT, Copyright (c) 2025 Mario Zechner);
只取固定外观事实: 颜色值、用户消息 Box(outputPad,1) 背景盒、底栏左右对齐/退让
规则。thinking 级别配色、费用、cache、上下文百分比、扩展状态在本期没有对应
事实, 不显示、不伪造 (规格 P13)。底栏布局已于 2026-10-19 按用户裁决改为单行
分段 (见 FooterComponent), 不再沿用上游两行布局。

边界:
- 本模块是纯排版: 不订阅事件、不写终端; 会话事实经 provider 回调注入
  (conversation_view.build 接线, 05 生产组合消费)。
- 行产出一律先对纯文本排版再套样式, 每行 visible_width <= width;
  样式序列不计列宽 (main_screen.visible_width 已剥离)。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cicada.budget import BudgetSnapshot

from cicada.ui.main_screen import truncate_to_width, visible_width, word_wrap_line, wrap_text_ansi
from cicada.ui.rendering import sanitize_display
from cicada.ui.display_style import DisplayStyle, current_style

# --- dark.json 固定颜色 (var 引用已解析) ----------------------------------------------

USER_MESSAGE_BG = "#343541"
USER_MESSAGE_TEXT = "#d4d4d4"
ASSISTANT_TEXT = "#d4d4d4"
EDITOR_BORDER = "#b294bb"  # 用户截图固定的外观色 (dark.json thinkingHigh 同值, 语义无关)
EDITOR_PLACEHOLDER = "#6a6a6a"  # 空态占位提示 (规格 §C)
MENU_SEL_BG = "#2d2d3d"         # 斜线菜单选中行背景 (规格 §D)
FOOTER_DIM = "#808080"  # 底栏低亮 (tui-aesthetics 规格 §E 提亮, 原 #666666)
# 单行分段底栏配色 (2026-10-19 用户裁决): 品牌最暗, 档位/模型强调黄,
# 工作区蓝, ctx 与累计为中性亮灰; 越靠右越次要, 让位顺序与亮度一致。
FOOTER_BRAND = "#6f6f6f"
FOOTER_ACCENT = "#ffd75f"
FOOTER_PATH = "#6ba3d4"
FOOTER_TEXT = "#c9cdd6"
MUTED_GRAY = "#808080"
MD_HEADING = "#f0c674"
MD_LINK = "#81a2be"
MD_LINK_URL = "#666666"
MD_CODE = "#8abeb7"
MD_CODE_BLOCK = "#b5bd68"
DIFF_ADDED = "#b5bd68"
DIFF_REMOVED = "#cc6666"
DIFF_ADDED_BG = "#1f2419"  # 与成功工具卡一致
DIFF_REMOVED_BG = "#2b1c1c"  # 与失败工具卡一致

# --- 工具卡 v2 (tui-aesthetics 规格 §B, 2026-10-07 起为本项目自有外观, 不再回引 dark.json) -----

TOOL_ACCENT_RUNNING = MD_LINK        # #81a2be
TOOL_ACCENT_SUCCESS = MD_CODE_BLOCK  # #b5bd68
TOOL_ACCENT_ERROR = DIFF_REMOVED     # #cc6666
TOOL_OUTPUT_FG = "#a8a8a8"
TOOL_HINT_FG = "#7f7f7f"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def fg(hex_color: str, text: str) -> str:
    """真彩前景 SGR 包裹; 样式不改变可见列宽。"""
    r, g, b = _rgb(hex_color)
    return current_style().sgr(f"38;2;{r};{g};{b}") + text + current_style().sgr("0")


def bg(hex_color: str, text: str) -> str:
    """真彩背景 SGR 包裹。"""
    r, g, b = _rgb(hex_color)
    return current_style().sgr(f"48;2;{r};{g};{b}") + text + current_style().sgr("0")


def fg_open(hex_color: str) -> str:
    """只开不复位的前景序列: 背景行内部换色用 (内层 \\x1b[0m 会连带清掉外层背景)。"""
    r, g, b = _rgb(hex_color)
    return current_style().sgr(f"38;2;{r};{g};{b}")


def bg_open(hex_color: str) -> str:
    """只开不复位的背景序列。"""
    r, g, b = _rgb(hex_color)
    return current_style().sgr(f"48;2;{r};{g};{b}")


def editor_border() -> Callable[[str], str]:
    """Editor.border_color 注入: 上下细线按固定外观色渲染。"""
    return lambda line: fg(EDITOR_BORDER, line)


def information_lines(
    title: str, sections: list[tuple[str, list[str]]], width: int,
) -> list[str]:
    """Pi 信息块: 强调标题、低亮标签、分组留白与一列缩进。

    来源: pi-mono@1a584a7a interactive-mode.ts handleSessionCommand (MIT)。
    只排版已提供的事实; 标题和分组不参与字段限额或模型历史。
    """
    pad = " " if width > 4 else ""
    content_width = max(1, width - len(pad) * 2)
    lines = [""]

    def append_heading(text: str) -> None:
        for chunk in wrap_text_ansi(sanitize_display(text), content_width):
            lines.append(pad + current_style().sgr("1") + chunk + current_style().sgr("0"))

    append_heading(title)
    for heading, rows in sections:
        if not rows:
            continue
        lines.append("")
        if heading:
            append_heading(heading)
        for row in rows:
            for logical in sanitize_display(row).split("\n"):
                command = logical.startswith("/")
                label, separator, _ = logical.partition(" " if command else ": ")
                label_end = len(label) if command else (len(label + separator) if separator else 0)
                for chunk in word_wrap_line(logical, content_width):
                    split_at = max(0, min(len(chunk.text), label_end - chunk.start_index))
                    prefix, body = chunk.text[:split_at], chunk.text[split_at:]
                    styled = fg(MD_CODE if command else MUTED_GRAY, prefix)
                    styled += fg(MUTED_GRAY, body) if command else body
                    lines.append(pad + styled)
    return lines


# --- 用户消息背景盒 (Pi user-message.ts: Box(outputPad, 1) + userMessageBg) ------------


def user_message_lines(text: str, width: int) -> list[str]:
    """用户消息 Pi 样式: 默认左右 1 列、上下 1 行整宽背景, 原文换行保留。

    窄窗退化: 内容宽不足 1 列时整盒只剩单列背景行 (有界, 不抛错)。
    """
    cleaned = sanitize_display(text)
    content_width = max(1, width - 2)
    content_lines: list[str] = []
    if cleaned:
        for logical in cleaned.split("\n"):
            content_lines.extend(wrap_text_ansi(logical, content_width))
    else:
        content_lines = [""]
    blank = bg(USER_MESSAGE_BG, " " * width)
    lines: list[str] = [blank]
    for line in content_lines:
        body = truncate_to_width(" " + line, width)  # 窄窗退化: 单列放不下双列字符时有界截断
        pad = " " * max(0, width - visible_width(body))
        lines.append(bg(USER_MESSAGE_BG, body + pad))
    lines.append(blank)
    return lines


# --- 助手 Markdown 离屏排版 (Pi assistant-message.ts: 无背景, 前置空行, 左右 1 列) --------

_RICH_MD_THEME = {
    "markdown.h1": f"bold {MD_HEADING}",
    "markdown.h2": f"bold {MD_HEADING}",
    "markdown.h3": f"bold {MD_HEADING}",
    "markdown.h4": MD_HEADING,
    "markdown.h5": MD_HEADING,
    "markdown.h6": MD_HEADING,
    "markdown.link": MD_LINK,
    "markdown.link_url": MD_LINK_URL,
    "markdown.code": MD_CODE,
    "markdown.item.bullet": MD_CODE,
    "markdown.block.quote": MUTED_GRAY,
    "markdown.hr": MUTED_GRAY,
}

_MD_CACHE_LIMIT = 64
_md_cache: dict[tuple[str, int, DisplayStyle], list[str]] = {}


def assistant_markdown_lines(text: str, width: int) -> list[str]:
    """助手正文离屏 Markdown 排版: 左右各 1 列缩进, 每行 <= width。

    Rich 只排版不打印 (现有 rendering.rich_to_ansi 同款约束, 真彩主题映射
    Pi md* 颜色; 代码块语法色用 Rich 缺省, 布局以固定场景为准——规格允许
    Rich 与 Pi Markdown 的细节差异)。缓存按 (正文, 宽度, 策略) 复用, 流式增量
    未变化时不重复排版。宽度 1-2 的极端窄窗退化为单列 (有界截断)。
    """
    cleaned = sanitize_display(text)
    if not cleaned:
        return []
    style = current_style()
    key = (cleaned, width, style)
    cached = _md_cache.get(key)
    if cached is not None:
        return list(cached)
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme

    pad = 1 if width > 4 else 0
    content_width = max(1, width - pad * 2)
    console = Console(
        width=content_width,
        height=24,
        file=StringIO(),
        force_terminal=True,
        color_system=None if style.plain else "truecolor",
        no_color=style.plain,
        legacy_windows=False,
        markup=False,
        highlight=False,
        emoji=False,
        theme=Theme(_RICH_MD_THEME),
    )
    console.print(Markdown(cleaned, hyperlinks=False))
    raw = console.file.getvalue()
    # Rich 产物可能含 OSC (超链接等): 剥离后按行切, 每行补 1 列缩进
    from cicada.ui.rendering import _OSC_RE

    ansi = _OSC_RE.sub("", raw)
    lines: list[str] = []
    for line in ansi.split("\n"):
        line = line.rstrip("\x1b") if line.endswith("\x1b") else line
        indent = " " * pad
        body = indent + line
        if visible_width(body) > width:
            body = truncate_to_width(body, width)
        lines.append(body)
    while lines and lines[-1].strip() == "":
        lines.pop()  # Rich 尾部空白行不进入文档 (条目间距由调用方统一)
    if len(_md_cache) >= _MD_CACHE_LIMIT:
        _md_cache.pop(next(iter(_md_cache)))
    _md_cache[key] = list(lines)
    return lines


# --- 底栏 (2026-10-19 用户裁决: 单行分段; 来源 pi footer.ts 的两行布局已退场) ----------


def format_tokens(count: int) -> str:
    """底栏 token 紧凑格式 (Pi formatTokens 移植)。"""
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    if count < 10000000:
        return f"{count / 1000000:.1f}M"
    return f"{round(count / 1000000)}M"


def format_workspace(workspace: str) -> str:
    """工作区显示: HOME 前缀替换为 ~ (Pi formatCwdForFooter 移植)。"""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if not home:
        return workspace
    try:
        root = Path(workspace).absolute()
        home_path = Path(home).absolute()
        if root.is_relative_to(home_path):
            rel = root.relative_to(home_path)
            return "~" if not str(rel) else f"~/{rel.as_posix()}"
    except (ValueError, OSError):
        pass
    return workspace


_CTX_SOURCE_LABELS = {
    # ctx 段输入来源短标 (契约 C5): unknown 即使有数值也保留"未", 不称为实测
    "estimated": "估",
    "anchored": "锚",
    "measured": "测",
    "unknown": "未",
}

_SEGMENT_SEPARATOR = " · "
_SEGMENT_SEPARATOR_FG = "#585858"  # 分隔点比正文更低亮 (外观对齐参考稿)


def _ctx_levels(snapshot: BudgetSnapshot | None) -> list[str]:
    """底栏 ctx 段候选级别, 从完整到简略 (思考可见性规格 D6、契约 C5)。

    近似标记/百分比/分母只整段出现; 待评估/未启用不补 0; 快照与报告上限不一致
    按待评估显示, 不拼接不同身份的数值; 超限由原始整数判定, 标记不被 format_tokens
    舍入或列宽截断抹掉。
    """
    if snapshot is None:
        return ["ctx: 未启用"]
    report = snapshot.report
    if report is None:
        # 无报告: 空间充足时显示已知上限, 不伪造占用 (C5)
        return [f"ctx: 待评估/{format_tokens(snapshot.limit)}", "ctx: 待评估"]
    if snapshot.limit != report.limit:
        return ["ctx: 待评估"]
    total, limit = report.total_estimate, report.limit
    percent = f"~{total * 100 / limit:.1f}%"  # 原始整数计算, 一位小数, 不封顶
    limit_text = format_tokens(limit)
    source = _CTX_SOURCE_LABELS.get(report.source, "未")
    flag = "超限 " if total > limit else ""  # 超限优先保留 (D5/D6)
    return [
        f"ctx: {flag}{percent}/{limit_text} {source}",
        f"ctx: {flag}{percent}/{limit_text}",
        f"ctx: {flag}{percent}".rstrip(),
    ]


class FooterComponent:
    """单行分段底栏 (2026-10-19 用户裁决, 外观对齐参考稿):

        cicada · [high] deepseek-flash · ~/repo · ctx: ~26.9%/131k 估 · 累计 ↑5.8k · ↓849

    四类事实按固定顺序、低亮分隔点连接; 品牌最暗, 档位与模型强调黄, 工作区蓝,
    ctx 与累计中性亮灰。窄窗按段整体降级/丢弃, 绝不在段内截出半个数值。

    metrics 回调返回 (输入累计, 输入含缺失, 输出累计, 输出含缺失);
    None 字段按规格省略, 有值但部分调用缺失时标注 (部分)。
    budget 回调 (契约 C5) 返回应用只读预算快照, 每次 render 至多读取一次;
    ctx 段展示最近一次预算评估 (近似标记/百分比/上限/输入来源短标);
    未注入回调时按"ctx: 未启用"呈现, 不补 0。
    effort 回调 (03, C6) 返回 (生效档位|None=关闭, 事务目标档位) 或 None=
    不支持; None 时不绘制档位徽标。档位缩写不含事务目标 (不伪造已生效)。
    实际渲染的档位徽标列区间记入 badge_spans 供帧命中。
    """

    _BRAND = "cicada"

    def __init__(
        self,
        workspace: str,
        model_label: str | None,
        metrics: Callable[[], tuple[int | None, bool, int | None, bool]],
        *,
        budget: Callable[[], BudgetSnapshot | None] | None = None,
        effort: Callable[[], tuple[str | None, str | None] | None] | None = None,
        display_style: DisplayStyle | None = None,
    ) -> None:
        self.display_style = display_style if display_style is not None else current_style()
        self._workspace = workspace
        self._model_label = model_label
        self._metrics = metrics
        self._budget = budget
        self._effort = effort
        self.badge_spans: tuple[tuple[int, int, int], ...] = ()

    def set_model_label(self, label: str | None) -> None:
        """模型选择发布后更新底栏为有效配置 (DeepSeek 04/契约 C7); 启动诊断不受影响。"""
        self._model_label = label

    def _stats_left(self) -> str:
        input_total, input_unknown, output_total, output_unknown = self._metrics()
        parts: list[str] = []
        if input_total is not None:
            parts.append(f"↑{format_tokens(input_total)}" + ("(部分)" if input_unknown else ""))
        elif input_unknown:
            parts.append("↑不可得")
        if output_total is not None:
            parts.append(f"↓{format_tokens(output_total)}" + ("(部分)" if output_unknown else ""))
        elif output_unknown:
            parts.append("↓不可得")
        joined = " · ".join(parts)  # 规格 §E: 计量多段 · 分隔
        if joined:
            joined = f"累计 {joined}"
        return joined

    def _badge_text(self) -> str:
        """思考强度档位徽标文本 (C6); 不支持思考时为空串 (不绘制也不占列)。"""
        facts = self._effort() if self._effort is not None else None
        if facts is None:
            return ""
        current, pending = facts
        text = "off" if current is None else current
        if pending is not None:
            text = f"{text}→{pending}"  # 旧值仍为当前值, 不伪造已生效 (C6)
        return f"[{text}]"

    def _segments(self) -> dict[str, tuple[str, str, int]]:
        """本帧各段事实: 种类 -> (文本, 前景色, 段内徽标列宽)。

        档位徽标并入模型段 ("[high] deepseek-flash"), 与参考稿一致不留分隔点;
        空文本段由调用方整体省略。
        """
        badge = self._badge_text()
        model = self._model_label or "不可得"
        return {
            "brand": (self._BRAND, FOOTER_BRAND, 0),
            "model": (f"{badge} {model}" if badge else model, FOOTER_ACCENT, len(badge)),
            "workspace": (format_workspace(self._workspace), FOOTER_PATH, 0),
            "stats": (self._stats_left(), FOOTER_TEXT, 0),
        }

    def _compose(
        self,
        kinds: tuple[str, ...],
        segments: dict[str, tuple[str, str, int]],
        ctx_text: str,
    ) -> tuple[str, int, int, int]:
        """按段拼一行; 返回 (行, 可见宽, 徽标起列, 徽标列宽), 未渲染徽标时列宽 0。"""
        pieces: list[str] = []
        column = 0
        badge_start = 0
        badge_width = 0
        for kind in kinds:
            text, color = (ctx_text, FOOTER_TEXT) if kind == "ctx" else segments[kind][:2]
            if not text:
                continue
            if pieces:
                column += visible_width(_SEGMENT_SEPARATOR)
                pieces.append(fg(_SEGMENT_SEPARATOR_FG, _SEGMENT_SEPARATOR))
            if kind == "model":
                badge_width = segments[kind][2]
                badge_start = column
            pieces.append(fg(color, text))
            column += visible_width(text)
        return "".join(pieces), column, badge_start, badge_width

    def render(self, width: int, rows: int) -> list[str]:
        with self.display_style.use():
            return self._render(width, rows)

    def _render(self, width: int, rows: int) -> list[str]:
        self.badge_spans = ()  # 命中列只属于本次渲染; 每帧重算 (D5 同源)
        # 契约 C5: 每 render 至多读取一次预算; 未注入回调的老调用者按"未启用"呈现,
        # 与预算停用时同一文案, 不另开布局分支
        snapshot = self._budget() if self._budget is not None else None
        ctx_levels = _ctx_levels(snapshot)
        segments = self._segments()
        # 降级顺序 (2026-10-19 裁决): 累计最先让位, 其次整个工作区段, 然后 ctx
        # 明细逐级简化 (去来源→去分母→只留百分比), 最后才动档位/模型 (最长且最
        # 常看, 只在极窄时缩写为 [off]/[high] 之外的形式并最终省略)。每级整段
        # 保留或整段消失, 段内不截半个数值, 也不退化成没有数值的空标签 ("ctx:")。
        candidates: list[tuple[tuple[str, ...], int]] = [
            (("brand", "model", "workspace", "ctx", "stats"), 0),
            (("brand", "model", "workspace", "ctx"), 0),
            (("brand", "model", "workspace", "ctx"), 1),
            (("brand", "model", "workspace", "ctx"), 2),
            (("brand", "model", "workspace"), -1),
            (("brand", "model", "ctx"), 0),
            (("brand", "model", "ctx"), 1),
            (("brand", "model", "ctx"), 2),
            (("brand", "model"), -1),
            (("brand",), -1),
        ]
        for kinds, index in candidates:
            ctx_text = ctx_levels[min(index, len(ctx_levels) - 1)] if "ctx" in kinds else ""
            line, line_width, badge_start, badge_width = self._compose(
                kinds, segments, ctx_text
            )
            if line_width > width:
                continue
            if badge_width:
                self._record_badge(badge_start, badge_width, width)
            return [line]
        # 极窄兜底: 品牌 + 无档位的模型名按列截断 (截断后档位不呈现, 也就无命中);
        # 连模型名都放不下才只留品牌。始终保证帧宽不超过终端。
        brand = fg(FOOTER_BRAND, self._BRAND)
        if width > visible_width(_SEGMENT_SEPARATOR) + visible_width(self._BRAND):
            room = width - visible_width(_SEGMENT_SEPARATOR) - visible_width(self._BRAND)
            model = truncate_to_width(self._model_label or "不可得", room)
            if model:
                return [
                    brand
                    + fg(_SEGMENT_SEPARATOR_FG, _SEGMENT_SEPARATOR)
                    + fg(FOOTER_ACCENT, model)
                ]
        return [fg(FOOTER_BRAND, truncate_to_width(self._BRAND, width))]

    def _record_badge(self, offset: int, badge_width: int, width: int) -> None:
        """左对齐行内徽标列区间 (C6): 徽标整段保留才可点, 被裁掉无 hit。"""
        if offset < 0 or badge_width <= 0 or offset + badge_width > width:
            return
        self.badge_spans = ((0, offset, offset + badge_width),)


__all__ = [
    "ASSISTANT_TEXT",
    "DIFF_ADDED",
    "DIFF_REMOVED",
    "EDITOR_BORDER",
    "EDITOR_PLACEHOLDER",
    "FOOTER_ACCENT",
    "FOOTER_BRAND",
    "FOOTER_DIM",
    "FOOTER_PATH",
    "FOOTER_TEXT",
    "FooterComponent",
    "MD_CODE",
    "MD_CODE_BLOCK",
    "MD_HEADING",
    "MD_LINK",
    "MD_LINK_URL",
    "MENU_SEL_BG",
    "MUTED_GRAY",
    "TOOL_ACCENT_ERROR",
    "TOOL_ACCENT_RUNNING",
    "TOOL_ACCENT_SUCCESS",
    "TOOL_HINT_FG",
    "TOOL_OUTPUT_FG",
    "USER_MESSAGE_BG",
    "USER_MESSAGE_TEXT",
    "assistant_markdown_lines",
    "bg",
    "bg_open",
    "editor_border",
    "fg",
    "fg_open",
    "format_tokens",
    "format_workspace",
    "information_lines",
    "user_message_lines",
]
