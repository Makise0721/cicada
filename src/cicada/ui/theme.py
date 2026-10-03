"""Pi dark 主题、消息排版与底栏 (票据 04).

来源: pi-mono@1a584a7a packages/coding-agent/src/modes/interactive/theme/dark.json、
components/footer.ts、user-message.ts (MIT, Copyright (c) 2025 Mario Zechner);
只取固定外观事实: 颜色值、用户消息 Box(outputPad,1) 背景盒、底栏两行布局与
左右对齐/退让规则。thinking 级别配色、费用、cache、上下文百分比、扩展状态
在本期没有对应事实, 不显示、不伪造 (规格 P13)。

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

from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display

# --- dark.json 固定颜色 (var 引用已解析) ----------------------------------------------

USER_MESSAGE_BG = "#343541"
USER_MESSAGE_TEXT = "#d4d4d4"
ASSISTANT_TEXT = "#d4d4d4"
EDITOR_BORDER = "#b294bb"  # 用户截图固定的外观色 (dark.json thinkingHigh 同值, 语义无关)
FOOTER_DIM = "#666666"
MUTED_GRAY = "#808080"
TOOL_BG_PENDING = "#282832"
TOOL_BG_SUCCESS = "#283228"
TOOL_BG_ERROR = "#3c2828"
MD_HEADING = "#f0c674"
MD_LINK = "#81a2be"
MD_LINK_URL = "#666666"
MD_CODE = "#8abeb7"
MD_CODE_BLOCK = "#b5bd68"
DIFF_ADDED = "#b5bd68"
DIFF_REMOVED = "#cc6666"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def fg(hex_color: str, text: str) -> str:
    """真彩前景 SGR 包裹; 样式不改变可见列宽。"""
    r, g, b = _rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


def bg(hex_color: str, text: str) -> str:
    """真彩背景 SGR 包裹。"""
    r, g, b = _rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m{text}\x1b[0m"


def editor_border() -> Callable[[str], str]:
    """Editor.border_color 注入: 上下细线按固定外观色渲染。"""
    return lambda line: fg(EDITOR_BORDER, line)


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
_md_cache: dict[tuple[str, int], list[str]] = {}


def assistant_markdown_lines(text: str, width: int) -> list[str]:
    """助手正文离屏 Markdown 排版: 左右各 1 列缩进, 每行 <= width。

    Rich 只排版不打印 (现有 rendering.rich_to_ansi 同款约束, 真彩主题映射
    Pi md* 颜色; 代码块语法色用 Rich 缺省, 布局以固定场景为准——规格允许
    Rich 与 Pi Markdown 的细节差异)。缓存按 (正文, 宽度) 复用, 流式增量
    未变化时不重复排版。宽度 1-2 的极端窄窗退化为单列 (有界截断)。
    """
    cleaned = sanitize_display(text)
    if not cleaned:
        return []
    cached = _md_cache.get((cleaned, width))
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
        color_system="truecolor",
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
    _md_cache[(cleaned, width)] = list(lines)
    return lines


# --- 底栏 (Pi footer.ts: 工作区行 + 计量左/模型右, #666666 低亮度) ----------------------


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


class FooterComponent:
    """两行底栏: 工作区行 + 计量左/模型右 (dim #666666)。

    metrics 回调返回 (输入累计, 输入含缺失, 输出累计, 输出含缺失);
    None 字段按规格省略, 有值但部分调用缺失时标注 (部分)。低高度
    (rows < 12) 只保留计量行, 工作区行让位给对话 (设计 §4)。
    """

    def __init__(
        self,
        workspace: str,
        model_label: str | None,
        metrics: Callable[[], tuple[int | None, bool, int | None, bool]],
    ) -> None:
        self._workspace = workspace
        self._model_label = model_label
        self._metrics = metrics

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
        joined = " ".join(parts)
        if joined:
            joined = f"累计 {joined}"
        return joined

    def render(self, width: int, rows: int) -> list[str]:
        dim = FOOTER_DIM
        pwd = format_workspace(self._workspace)
        pwd_line = truncate_to_width(fg(dim, pwd), width, fg(dim, "..."))

        stats_left = self._stats_left()
        model = self._model_label or "不可得"
        stats_left_width = visible_width(stats_left)
        if stats_left_width > width:
            stats_left = truncate_to_width(stats_left, width, "...")
            stats_left_width = visible_width(stats_left)
        min_padding = 2
        total = stats_left_width + min_padding + visible_width(model)
        if total <= width:
            padding = " " * (width - stats_left_width - visible_width(model))
            stats_line = stats_left + padding + model
        else:
            available = width - stats_left_width - min_padding
            if available > 0:
                truncated_right = truncate_to_width(model, available, "")
                pad = " " * max(0, width - stats_left_width - visible_width(truncated_right))
                stats_line = stats_left + pad + truncated_right
            else:
                stats_line = stats_left

        lines = [pwd_line, fg(dim, stats_line)]
        if rows < 12:
            return lines[1:]  # 低高度: 工作区行让位
        return lines


__all__ = [
    "ASSISTANT_TEXT",
    "DIFF_ADDED",
    "DIFF_REMOVED",
    "EDITOR_BORDER",
    "FOOTER_DIM",
    "FooterComponent",
    "MD_CODE",
    "MD_CODE_BLOCK",
    "MD_HEADING",
    "MD_LINK",
    "MD_LINK_URL",
    "MUTED_GRAY",
    "TOOL_BG_ERROR",
    "TOOL_BG_PENDING",
    "TOOL_BG_SUCCESS",
    "USER_MESSAGE_BG",
    "USER_MESSAGE_TEXT",
    "assistant_markdown_lines",
    "bg",
    "editor_border",
    "fg",
    "format_tokens",
    "format_workspace",
    "user_message_lines",
]
