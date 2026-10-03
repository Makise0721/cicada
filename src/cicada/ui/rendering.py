"""Rich -> prompt_toolkit styled fragments 渲染 (TUI 第一版 04).

边界:
- 终端控制权属于 prompt_toolkit (03/05); 本模块只把 PresentationRecord /
  ActivePreview / StatusSnapshot 排版为 styled fragments (FormattedText),
  不 print、不直接写终端.
- ANSI 串不作为普通文本输出: Rich 排版产物经 prompt_toolkit ANSI() 解析为
  样式片段 (做法与 01 探针一致), 调用方 (03/05) 用 print_formatted_text 输出.
- T13: 不可信文本先做显示清理 (去 CSI/OSC/控制字符, 除换行与 tab), 再交给 Rich;
  外部纯文本不当 Rich markup (用 Text 而非 markup 字符串), 不生成可执行终端控制
  序列, 不主动打开链接. presenter 记录中的规范文本 (原始正文) 不受清理影响.
- None 计量显示"不可得", 不是 0; 累计计量标注"累计", 不标为单次上下文峰值.
"""

from __future__ import annotations

import io
import re

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.utils import get_cwidth

from cicada.ui.contracts import (
    ActivePreview,
    PresentationRecord,
    StatusSnapshot,
)

RENDER_WIDTH = 100

# 显示清理: CSI 序列、OSC 序列、未终结的截断段、除 \n \t 外的 C0/C1 控制字符
_CSI_RE = re.compile(r"\x1b\[[0-9;:?<=>!]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_UNTERMINATED_OSC_RE = re.compile(r"\x1b\][^\x1b]*")
_TRUNCATED_CSI_RE = re.compile(r"\x1b\[[0-9;:?<=>!]*[ -/]*$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")

_ROLE_STYLES = {
    "user": "bold",
    "assistant": "",
    "tool": "ansicyan",
    "system": "ansimagenta",
    "error": "ansired bold",
}
_STATUS_STYLES = {
    "pending": "ansiyellow",
    "running": "ansibrightblue",
    "done": "ansigreen",
    "failed": "ansired",
    "cancelled": "ansiyellow",
}


def sanitize_display(text: str) -> str:
    """显示副本清理: 移除 CSI/OSC/未终结截断段与控制字符 (保留 \n 与 \t); 只用于渲染输入."""
    text = _UNTERMINATED_OSC_RE.sub("", _OSC_RE.sub("", text))
    text = _TRUNCATED_CSI_RE.sub("", _CSI_RE.sub("", text))
    return _CONTROL_RE.sub("", text)


def rich_to_ansi(renderable, width: int = RENDER_WIDTH) -> str:
    """Rich 只排版: renderable -> ANSI 串; 调用方经 ANSI() 解析, 不作普通文本输出."""
    from rich.console import Console

    console = Console(
        # 离屏排版的几何尺寸由调用者决定. 同时给出 height, 避免 TERM=dumb
        # 时 Rich.size 忽略单独给出的 width 而回退到 80 列.
        width=max(1, width), height=25, file=io.StringIO(), force_terminal=True,
        color_system="standard", legacy_windows=False,
        markup=False, highlight=False, emoji=False,
    )
    console.print(renderable)
    # Rich 的 OSC 8 超链接类序列不进入最终产物: 链接文本保留, 不主动打开链接
    return _OSC_RE.sub("", console.file.getvalue())


def ansi_fragments(ansi: str) -> FormattedText:
    """把 Rich 产物解析为 prompt_toolkit styled fragments (不做普通文本 print)."""
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    return to_formatted_text(ANSI(ansi))


def _markdown_fragments(text: str, width: int) -> FormattedText:
    from rich.markdown import Markdown

    return ansi_fragments(rich_to_ansi(Markdown(sanitize_display(text)), width))


def _text_fragments(text: str, width: int, style: str = "") -> FormattedText:
    from rich.text import Text

    fragments = ansi_fragments(rich_to_ansi(Text(sanitize_display(text)), width))
    return FormattedText([(style, value) for _, value in fragments])


_DIFF_LINE_RE = re.compile(r"^(?:diff --git |index [0-9a-f]{7,}|\+\+\+ |--- |@@ )")


def _looks_like_diff(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for line in lines if _DIFF_LINE_RE.match(line) or line[:1] in "+-")
    return hits / len(lines) >= 0.6


def _tool_fragments(body: str, width: int) -> FormattedText:
    from rich.syntax import Syntax
    from rich.text import Text

    cleaned = sanitize_display(body)
    if _looks_like_diff(cleaned):
        renderable = Syntax(cleaned, "diff", word_wrap=True, background_color="default")
    else:
        # 外部纯文本: Text 不解释 markup, 不伪造样式
        renderable = Text(cleaned)
    return ansi_fragments(rich_to_ansi(renderable, width))


def record_header(record: PresentationRecord) -> tuple[str, str]:
    """记录头 (文本, prompt_toolkit 样式); 样式只来自程序的角色/状态映射."""
    role_style = _ROLE_STYLES.get(record.role, "")
    status_style = _STATUS_STYLES.get(record.status, "")
    header = f"[{record.role}] {record.title} ({record.status})"
    style = " ".join(part for part in (role_style, status_style) if part)
    return header, style


def record_fragments(record: PresentationRecord, width: int = RENDER_WIDTH) -> FormattedText:
    """把一条记录排版为 styled fragments; 正文经显示清理, 不读工件文件."""
    from prompt_toolkit.formatted_text import FormattedText as _FT

    header, style = record_header(record)
    fragments = _text_fragments(header, width, style)
    body = "\n".join(record.body)
    if not body.strip():
        return fragments
    if record.role == "assistant":
        fragments.extend(_markdown_fragments(body, width))
    elif record.role == "tool":
        fragments.extend(_tool_fragments(body, width))
    else:
        fragments.extend(_text_fragments(body, width))
    return _FT(fragments)


def preview_fragments(preview: ActivePreview, width: int = RENDER_WIDTH) -> FormattedText:
    """活动预览: 有界文本 + 截断事实; 不含最终记录 (最终回答以完成事件为准)."""
    from prompt_toolkit.formatted_text import FormattedText as _FT
    header = f"▌ 活动预览 ({preview.run_id} turn {preview.turn_index})"
    fragments = _text_fragments(header, width, "ansibrightblue")
    fragments.extend(_text_fragments(preview.text, width))
    if preview.truncated:
        fragments.extend(_text_fragments("…[预览已截断，完整文本以最终回答为准]", width, "ansiyellow"))
    return _FT(fragments)


def _metric_text(value: int | None) -> str:
    return "不可得" if value is None else str(value)


def _clip_cells(text: str, width: int) -> str:
    """单行省略按显示列计算, 中文双列字符不会越过右边界."""
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    result = []
    used = 0
    for char in text:
        cells = get_cwidth(char)
        if used + cells > width - 1:
            break
        result.append(char)
        used += cells
    return "".join(result) + "…"


def _bounded_status(parts: list[str], width: int) -> str:
    # 窄窗优先保留 phase/check; 长工作区不会把关键状态挤到屏幕外.
    phase, check = parts[2], parts[6]
    separator = " | "
    critical = separator.join((phase, check))
    if get_cwidth(critical) > width:
        available = width - get_cwidth(separator)
        if available < 2:
            return _clip_cells(critical, width)
        phase_width = min(get_cwidth(phase), available // 2)
        check_width = available - phase_width
        # 检查摘要必须整体可见. 截断 passed/current 或多个检查的前缀会隐藏
        # stale/failed 等后续事实; 放不下时只提示省略, 完整事实走 /status.
        check_text = check if get_cwidth(check) <= check_width else "检查: 省略(/status)"
        return separator.join((_clip_cells(phase, phase_width), _clip_cells(check_text, check_width)))
    line = critical
    for part in (parts[0], parts[3], parts[4], parts[5], parts[1], *parts[7:]):
        available = width - get_cwidth(line) - get_cwidth(separator)
        if available <= 0:
            break
        clipped = _clip_cells(part, available)
        if clipped:
            line += separator + clipped
        if clipped != part:
            break
    return line


def status_fragments(status: StatusSnapshot, width: int | None = None) -> FormattedText:
    """状态栏: None 显示不可得; 计量标注累计口径; 检查事实来自 presenter 投影."""
    model = status.model or "不可得"
    workspace = status.workspace or "不可得"
    phase = status.phase or "不可得"
    turn = "不可得" if status.turn_index is None else str(status.turn_index)
    check = status.check_status
    if check is None:
        check = "未启用检查" if status.verification_run_id is None else "不可得"
    parts = [
        f"模型: {model}",
        f"工作区: {workspace}",
        f"阶段: {phase}",
        f"轮次: {turn}",
        # 计量是跨调用累计口径, 不是单次上下文峰值, 也不是实际模型分配
        f"累计输入: {_metric_text(status.input_tokens)}",
        f"累计输出: {_metric_text(status.output_tokens)}",
        f"检查: {check}",
    ]
    if status.verification_run_id is not None:
        parts.append(f"验证运行: {status.verification_run_id}")
    if status.change_scope is not None:
        parts.append(status.change_scope)
    parts = [sanitize_display(part).replace("\n", " ").replace("\t", " ") for part in parts]
    line = " | ".join(parts) if width is None else _bounded_status(parts, max(1, width))
    return FormattedText([("reverse", line)])  # FormattedText 是 list 子类, 需实例化
