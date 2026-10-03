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
        width=width, file=io.StringIO(), force_terminal=True,
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


def _markdown_fragments(text: str) -> FormattedText:
    from rich.markdown import Markdown

    return ansi_fragments(rich_to_ansi(Markdown(sanitize_display(text))))


_DIFF_LINE_RE = re.compile(r"^(?:diff --git |index [0-9a-f]{7,}|\+\+\+ |--- |@@ )")


def _looks_like_diff(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for line in lines if _DIFF_LINE_RE.match(line) or line[:1] in "+-")
    return hits / len(lines) >= 0.6


def _tool_fragments(body: str) -> FormattedText:
    from rich.syntax import Syntax
    from rich.text import Text

    cleaned = sanitize_display(body)
    if _looks_like_diff(cleaned):
        renderable = Syntax(cleaned, "diff", word_wrap=True, background_color="default")
    else:
        # 外部纯文本: Text 不解释 markup, 不伪造样式
        renderable = Text(cleaned)
    return ansi_fragments(rich_to_ansi(renderable))


def record_header(record: PresentationRecord) -> tuple[str, str]:
    """记录头 (文本, prompt_toolkit 样式); 样式只来自程序的角色/状态映射."""
    role_style = _ROLE_STYLES.get(record.role, "")
    status_style = _STATUS_STYLES.get(record.status, "")
    header = f"[{record.role}] {record.title} ({record.status})"
    style = " ".join(part for part in (role_style, status_style) if part)
    return header, style


def record_fragments(record: PresentationRecord) -> FormattedText:
    """把一条记录排版为 styled fragments; 正文经显示清理, 不读工件文件."""
    from prompt_toolkit.formatted_text import FormattedText as _FT

    header, style = record_header(record)
    fragments: FormattedText = _FT([(style, header), ("", "\n")])
    body = "\n".join(record.body)
    if not body.strip():
        return fragments
    if record.role == "assistant":
        fragments.extend(_markdown_fragments(body))
    elif record.role == "tool":
        fragments.extend(_tool_fragments(body))
    else:
        from rich.text import Text

        fragments.extend(ansi_fragments(rich_to_ansi(Text(sanitize_display(body)))))
    return _FT(fragments)


def preview_fragments(preview: ActivePreview) -> FormattedText:
    """活动预览: 有界文本 + 截断事实; 不含最终记录 (最终回答以完成事件为准)."""
    from prompt_toolkit.formatted_text import FormattedText as _FT
    from rich.text import Text

    header = f"▌ 活动预览 ({preview.run_id} turn {preview.turn_index})"
    fragments: FormattedText = _FT([("ansibrightblue", header), ("", "\n")])
    fragments.extend(ansi_fragments(rich_to_ansi(Text(sanitize_display(preview.text)))))
    if preview.truncated:
        fragments.append(("ansiyellow", "…[预览已截断，完整文本以最终回答为准]"))
    return _FT(fragments)


def _metric_text(value: int | None) -> str:
    return "不可得" if value is None else str(value)


def status_fragments(status: StatusSnapshot) -> FormattedText:
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
    line = " | ".join(parts)
    return FormattedText([("reverse", line)])  # FormattedText 是 list 子类, 需实例化
