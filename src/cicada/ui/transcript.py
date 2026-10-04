"""显示文档 (票据 02): 真实会话消息条目的原位流式状态与排版.

职责 (规格 §原位文本和工具更新 / 设计 §显示文档与事件):
- 保存用户/助手/系统/错误/取消条目, 顺序与身份稳定; assistant 条目以
  (run_id, turn_index) 定位, AssistantTextDelta 累积、AssistantCompleted 校准
  同一条目, 完成后不二次追加, 迟到增量不复活已完成条目。
- 不沿用旧 12 行/4096 bytes 活动预览与 64 行正文裁剪: 保存已收到的完整
  正文, 渲染按当前宽度全量换行, 不擅自删掉显示历史。
- 外部文本先显示清理 (sanitize_display) 再排版; 每一行产出保证
  visible_width <= width, 超宽即本模块缺陷 (renderer 只做防御截断)。
- 文档不是规范历史, 不参与模型请求; 工具条目 (票据 03) 可经 add_entry
  挂载本文档, 条目协议只有 render(width) -> list[str] 一个方法。
- ModelCallCompleted 计量在此累计: None 字段显示"不可得", 不当作 0;
  展示出口 (底栏) 由票据 04 组装, 本模块只提供口径文本。
"""

from __future__ import annotations

from typing import Protocol

from cicada.core.messages import AssistantMessage
from cicada.core.ports import ModelMetrics
from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display

ERROR_PREFIX = "错误: "
CANCELLED_TEXT = "已取消: 本次请求在完成前被中止。"
LENGTH_NOTE = "（回答因达到输出长度上限被截断）"
STREAM_PLACEHOLDER = "…"


class TranscriptEntry(Protocol):
    """显示文档条目协议 (票据 03 的工具条目同样实现)。"""

    def render(self, width: int) -> list[str]: ...


def _wrap_lines(text: str, width: int) -> list[str]:
    """清理后的多行文本按宽度换行; 每行可见列宽 <= width。"""
    lines: list[str] = []
    for logical in text.split("\n"):
        lines.extend(wrap_text_ansi(logical, max(1, width)))
    return lines


def _wrap_with_prefix(prefix: str, text: str, width: int) -> list[str]:
    """带前缀的换行: 前缀占列, 正文按剩余宽度换行。"""
    content_width = max(1, width - visible_width(prefix))
    lines: list[str] = []
    for logical in text.split("\n"):
        for chunk in wrap_text_ansi(logical, content_width):
            lines.append(prefix + chunk)
    return lines if lines else [prefix.rstrip()]


def _fit(lines: list[str], width: int) -> list[str]:
    """逐行防御: 任何超宽行截断到 width (正常换行后不应触发)。"""
    result: list[str] = []
    for line in lines:
        if visible_width(line) > width:
            line = truncate_to_width(line, width)
        result.append(line)
    return result


class UserEntry:
    """用户消息条目: Pi 样式背景盒 (上下 1 行、左右 1 列), 原文换行保留。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return theme.user_message_lines(self.text, width)


class AssistantEntry:
    """助手消息条目: 以 (run_id, turn_index) 定位, 流式累积, 完成校准。

    渲染为离屏 Markdown (Pi 样式, 前置空行, 左右 1 列); 缓存按
    (正文, 宽度) 复用, 流式增量未变化时不重复排版。
    """

    def __init__(self, run_id: str, turn_index: int) -> None:
        self.run_id = run_id
        self.turn_index = turn_index
        self.text = ""
        self.finalized = False
        self.stop_reason: str | None = None
        self.error: str | None = None
        self._render_cache: tuple[str, int, list[str]] | None = None

    def render(self, width: int) -> list[str]:
        cleaned = sanitize_display(self.text)
        lines: list[str]
        if cleaned:
            cached = self._render_cache
            if cached is not None and cached[0] == cleaned and cached[1] == width:
                lines = list(cached[2])
            else:
                lines = theme.assistant_markdown_lines(cleaned, width)
                self._render_cache = (cleaned, width, list(lines))
            lines = ["", *lines]  # Pi assistant 前置空行
        elif not self.finalized:
            lines = [STREAM_PLACEHOLDER]  # 已建条目尚未收到任何增量
        else:
            lines = []  # 完成但无正文 (如错误路径): 事实由错误/取消条目表达
        if self.finalized and self.stop_reason == "length":
            lines.append(theme.fg(theme.MUTED_GRAY, LENGTH_NOTE))
        return lines


class SystemEntry:
    """系统条目: 命令输出等程序事实, 无角色前缀。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        prefix = " " if width > 4 else ""
        return ["", *_wrap_with_prefix(prefix, sanitize_display(self.text), width)]


class InformationEntry(SystemEntry):
    """命令信息块: 事实正文与标题/分组装饰分开保存。"""

    def __init__(self, title: str, sections: list[tuple[str, list[str]]]) -> None:
        super().__init__("\n".join(row for _, rows in sections for row in rows))
        self.title = title
        self.sections = [(heading, list(rows)) for heading, rows in sections]

    def render(self, width: int) -> list[str]:
        return theme.information_lines(self.title, self.sections, width)


class ErrorEntry:
    """错误条目: 如实显示错误事实 (模型错误/请求被拒/异常)。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return _wrap_with_prefix(ERROR_PREFIX, sanitize_display(self.text), width)


class CancelledEntry:
    """取消条目: 本次请求被用户中止的事实。"""

    def render(self, width: int) -> list[str]:
        return _wrap_lines(CANCELLED_TEXT, width)


class Transcript:
    """显示文档: 条目有序集合 + 助手条目定位索引 + 计量累计。"""

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._assistant_index: dict[tuple[str, int], AssistantEntry] = {}
        # 计量累计: None = 尚无任何数值 (显示不可得); unknown = 有调用但该字段缺失
        self._input_total: int | None = None
        self._output_total: int | None = None
        self._input_unknown = False
        self._output_unknown = False

    # --- 条目管理 ----------------------------------------------------------------

    def entries(self) -> list[TranscriptEntry]:
        """当前条目快照 (顺序即文档顺序), 供测试与诊断。"""
        return list(self._entries)

    def add_entry(self, entry: TranscriptEntry) -> None:
        """追加任意条目 (票据 03 的工具条目经此挂载, 保持文档顺序稳定)。"""
        self._entries.append(entry)

    def append_user(self, text: str) -> UserEntry:
        entry = UserEntry(text)
        self._entries.append(entry)
        return entry

    def append_system(self, text: str) -> SystemEntry:
        entry = SystemEntry(text)
        self._entries.append(entry)
        return entry

    def append_information(self, title: str, sections: list[tuple[str, list[str]]]) -> None:
        self._entries.append(InformationEntry(title, sections))

    def append_error(self, text: str) -> ErrorEntry:
        entry = ErrorEntry(text)
        self._entries.append(entry)
        return entry

    def append_cancelled(self) -> CancelledEntry:
        entry = CancelledEntry()
        self._entries.append(entry)
        return entry

    # --- 助手条目原位更新 -----------------------------------------------------------

    def stream_assistant_delta(self, run_id: str, turn_index: int, text: str) -> None:
        """AssistantTextDelta: 累积到 (run_id, turn_index) 条目; 已完成条目丢弃迟到增量。"""
        if not text:
            return
        entry = self._assistant_entry(run_id, turn_index)
        if entry.finalized:
            return  # 迟到增量不复活已完成条目
        entry.text += text

    def complete_assistant(self, run_id: str, turn_index: int, message: AssistantMessage) -> None:
        """AssistantCompleted: 校准同一条目的完整正文与终结事实, 不二次追加。

        message 本身不带 turn_index, 由调用方按内核 TurnStarted 跟踪的当前轮传入。
        """
        entry = self._assistant_entry(run_id, turn_index)
        entry.text = message.text  # 完成事件是权威: 校准 (非追加) 同一条目
        entry.finalized = True
        entry.stop_reason = message.stop_reason
        entry.error = message.error

    def _assistant_entry(self, run_id: str, turn_index: int) -> AssistantEntry:
        key = (run_id, turn_index)
        entry = self._assistant_index.get(key)
        if entry is None:
            entry = AssistantEntry(run_id, turn_index)
            self._assistant_index[key] = entry
            self._entries.append(entry)
        return entry

    # --- 请求终结与计量 ---------------------------------------------------------------

    def record_request_result(self, stop_reason: str, error: str | None) -> None:
        """请求级终结事实: 错误/取消如实成条; 正常完成不产生任何条目 (安静界面)。

        length 不单独立条: 截断事实标注在对应 assistant 条目上。
        """
        if stop_reason == "aborted":
            self.append_cancelled()
        elif stop_reason == "error":
            self.append_error(error or "模型请求以错误结束，但未提供错误信息。")

    def add_metrics(self, metrics: ModelMetrics | None) -> None:
        """累计一次模型调用的计量; None 字段记不可得, 不当作 0。"""
        if metrics is None:
            self._input_unknown = True
            self._output_unknown = True
            return
        if metrics.input_tokens is None:
            self._input_unknown = True
        else:
            self._input_total = (self._input_total or 0) + metrics.input_tokens
        if metrics.output_tokens is None:
            self._output_unknown = True
        else:
            self._output_total = (self._output_total or 0) + metrics.output_tokens

    def metrics_summary(self) -> str:
        """计量口径文本: 跨调用累计; 无任何数值时为不可得, 不虚构 0。

        部分调用的字段缺失时如实标注, 已收到的累计不作废也不夸大为完整值。
        """
        if self._input_total is None and self._output_total is None:
            return "计量: 不可得"

        def fmt(label: str, total: int | None, unknown: bool) -> str:
            if total is None:
                return f"{label} 不可得"
            if unknown:
                return f"{label} {total}（部分调用不可得）"
            return f"{label} {total}"

        return f"计量: 累计{fmt('输入', self._input_total, self._input_unknown)} · 累计{fmt('输出', self._output_total, self._output_unknown)}"

    def metrics_fact(self) -> tuple[int | None, bool, int | None, bool]:
        """底栏用计量事实: (输入累计, 输入含缺失, 输出累计, 输出含缺失)。"""
        return (self._input_total, self._input_unknown, self._output_total, self._output_unknown)

    # --- 渲染 ----------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """整份文档按当前宽度排版; 每行可见列宽 <= width。"""
        lines: list[str] = []
        for entry in self._entries:
            lines.extend(entry.render(width))
        return _fit(lines, width)


__all__ = [
    "CANCELLED_TEXT",
    "ERROR_PREFIX",
    "LENGTH_NOTE",
    "STREAM_PLACEHOLDER",
    "AssistantEntry",
    "CancelledEntry",
    "ErrorEntry",
    "InformationEntry",
    "SystemEntry",
    "Transcript",
    "TranscriptEntry",
    "UserEntry",
]
