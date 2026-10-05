"""工具原位卡片与输出展开 (票据 03): 显示文档片段, 事件驱动原位更新。

来源: pi-mono@1a584a7a packages/coding-agent/src/modes/interactive/components/tool-execution.ts
(MIT, Copyright (c) 2025 Mario Zechner); 本文件移植其单条目原位更新、
partial/success/error 背景 (Pi dark.json: #282832/#283228/#3c2828 真彩 SGR)、
fallback 前 10 行预览 + "... (N more lines)" 折叠提示, 以及 Ctrl+O 全局
展开标志 (Pi setToolsExpanded 遍历全部条目切换)。kitty 图片、鼠标区域与
按工具自定义 renderer 不移植; diff/代码高亮排版归票据 04, 输出先按纯文本行。

边界 (见 shared-boundary-handoff 与票据 03):
- 纯状态 + 排版: apply() 接收 ApplicationUpdate / 内核事件更新条目,
  render_lines(width) 产出显示行, toggle_expand() 只翻转展示标志。
  不读取检查工件、不执行工具、不触发模型请求; 展开只显示已收到的
  ToolResult.content, 不显示未曾收到的字节。
- 条目身份是 (run_id, turn_index, call_id) (规格 §原位文本和工具更新)。
  ToolStarted/ToolCompleted 不携带 turn 与参数: turn_index 从 TurnStarted
  事件同步, 参数摘要取同 run 前一条 AssistantCompleted 携带的 tool_calls
  arguments_json — 两者都是真实应用事件事实, 不伪造字段。
- 取消/失败/工具自身截断按 ToolResult 原文如实标注: 内核取消哨兵
  ("cancelled before execution"/"cancelled during execution", core/agent.py)
  只用于把错误卡标注为已取消, 不改写内容; read 等工具写进 content 尾部
  的截断脚注原样保留, 展开不恢复未收到的字节。
- 外部文本先显示清理 (ui/rendering.sanitize_display), 外部 CSI/OSC 不作为
  可信控制序列执行; 行内样式 (粗体/背景/灰字) 只由程序状态生成。
  换行一律先对纯文本排版再套样式 (word_wrap_line 不感知 ANSI)。
  本模块保证每行 visible_width <= width; 超宽即本模块缺陷 (renderer 只做
  防御截断), 有界失败记入 ui_failures 而非抛异常。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeAlias

from cicada.core.events import (
    AssistantCompleted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.ui.checks import artifact_line as _artifact_line_from_details
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display

# Pi FALLBACK_PREVIEW_LINES: 工具输出默认预览行数
PREVIEW_LINES = 10

# Pi dark.json 工具状态背景 (真彩 SGR): pending #282832 / success #283228 / error #3c2828
TOOL_BG_PENDING = "\x1b[48;2;40;40;50m"
TOOL_BG_SUCCESS = "\x1b[48;2;40;50;40m"
TOOL_BG_ERROR = "\x1b[48;2;60;40;40m"

BOLD = "\x1b[1m"
BOLD_OFF = "\x1b[22m"
MUTED = "\x1b[90m"  # Pi dark.json muted -> gray
RESET = "\x1b[0m"

# 内核取消路径固定哨兵前缀 (core/agent.py _dispatch / _finish_pending_calls)
_CANCELLED_PREFIX = "cancelled"

# 参数摘要上限 (字符): 摘要本身有界, 展示行再按列宽换行
_ARGS_SUMMARY_MAX_CHARS = 200
_PREFERRED_ARG_KEYS = (
    "path",
    "file_path",
    "command",
    "pattern",
    "action",
    "check_id",
    "query",
    "url",
)

_UI_FAILURES_MAX = 100

CardStatus: TypeAlias = Literal["running", "success", "error"]

_STATUS_BG: dict[str, str] = {
    "running": TOOL_BG_PENDING,
    "success": TOOL_BG_SUCCESS,
    "error": TOOL_BG_ERROR,
}


def _clean_text(text: str) -> str:
    """单行显示副本: 显示清理 + 换行折叠为空格 (标题/摘要用)。"""
    return sanitize_display(str(text)).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _clean_output(content: str) -> str:
    """输出区显示副本: 规范换行后清理; 已收到的字节除控制序列外不增删。"""
    normalized = str(content).replace("\r\n", "\n").replace("\r", "")
    return sanitize_display(normalized)


@dataclass(frozen=True)
class ToolCardView:
    """工具条目只读快照 (测试与组合方观察用); output_text 为已收到的完整输出。"""

    run_id: str
    turn_index: int
    call_id: str
    name: str
    status: CardStatus
    is_error: bool
    cancelled: bool
    output_text: str


class _ToolCard:
    """单张工具卡的内部可变状态; 终态只写一次, 不回退。"""

    __slots__ = (
        "run_id",
        "turn_index",
        "call_id",
        "name",
        "args_summary",
        "status",
        "is_error",
        "cancelled",
        "content",
        "terminal",
        "artifact",
    )

    def __init__(self, *, run_id: str, turn_index: int, call_id: str, name: str, args_summary: str) -> None:
        self.run_id = run_id
        self.turn_index = turn_index
        self.call_id = call_id
        self.name = name
        self.args_summary = args_summary
        self.status: CardStatus = "running"
        self.is_error = False
        self.cancelled = False
        self.content = ""
        self.terminal = False
        # 回执输出工件行 (路径 + hash, 只显示不读入; 同旧 presenter 卡头口径)
        self.artifact: str | None = None

    def complete(self, content: str, is_error: bool) -> None:
        self.content = _clean_output(content)
        self.is_error = bool(is_error)
        self.status = "error" if is_error else "success"
        # 取消哨兵只用于标注: 内容原文照旧显示
        self.cancelled = self.is_error and self.content.lstrip().startswith(_CANCELLED_PREFIX)
        self.terminal = True


class ToolDocument:
    """工具显示文档片段: 应用事件 -> 原位条目 -> 当前宽度显示行。

    apply() 接受 ApplicationSession 通知的 ApplicationUpdate (取其 .event
    字段, 即 AgentEventReceived 包装), 也直接接受内核 Event; 其余更新类型
    属于消息/检查等组件, 本模块忽略。条目按到达序稳定排列, 原位更新不改变
    条目位置; 展开标志是全局的 (与 Pi setToolsExpanded 一致), 只影响展示。
    """

    def __init__(self) -> None:
        self._cards: list[_ToolCard] = []
        self._by_key: dict[tuple[str, int, str], _ToolCard] = {}
        self._current_turn: dict[str, int] = {}
        self._args_by_call: dict[tuple[str, str], str] = {}
        self._expanded = False
        self._version = 0
        self._cache: tuple[int, int, list[str]] | None = None
        self.ui_failures: list[str] = []

    # --- 事件入口 ---------------------------------------------------------------

    def apply(self, update: object) -> None:
        """消费一条应用更新; ToolStarted/ToolCompleted 更新条目, 其余忽略。"""
        event = getattr(update, "event", update)
        if isinstance(event, TurnStarted):
            self._current_turn[event.run_id] = event.turn_index
        elif isinstance(event, AssistantCompleted):
            for call in event.message.tool_calls:
                self._args_by_call[(event.run_id, call.id)] = call.arguments_json
        elif isinstance(event, ToolStarted):
            self._on_tool_started(event)
        elif isinstance(event, ToolCompleted):
            self._on_tool_completed(event)
        # 其余事件类型 (RunStarted/AssistantTextDelta/RunFinished/应用状态通知等)
        # 不是工具卡的事实来源, 静默忽略。

    def _on_tool_started(self, event: ToolStarted) -> None:
        turn = self._current_turn.get(event.run_id, 0)
        key = (event.run_id, turn, event.call_id)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing.terminal:
                # 迟到开始事件不能恢复已完成条目 (规格 §原位文本和工具更新)
                self._record_failure(f"stale ToolStarted for terminal card {key}; ignored")
                return
            existing.name = _clean_text(event.name)  # 重复开始: 幂等刷新
            return
        card = _ToolCard(
            run_id=event.run_id,
            turn_index=turn,
            call_id=event.call_id,
            name=_clean_text(event.name),
            args_summary=self._summarize_args(event.run_id, event.call_id),
        )
        self._append(card)

    def _on_tool_completed(self, event: ToolCompleted) -> None:
        result = event.result
        run_id = event.run_id
        artifact = _artifact_line_from_details(result.details)
        # 1) 同 run 同 call 的未终结条目: 原位更新 (开始→终态同一条目)
        for card in reversed(self._cards):
            if card.run_id == run_id and card.call_id == result.call_id and not card.terminal:
                card.complete(result.content, result.is_error)
                card.artifact = artifact
                self._touch()
                return
        # 2) 缺少 ToolStarted 的闭合结果: 以当前轮建终态卡 (跨轮复用 call_id 不合并)
        turn = self._current_turn.get(run_id, 0)
        key = (run_id, turn, result.call_id)
        if key in self._by_key:
            self._record_failure(f"duplicate ToolCompleted for terminal card {key}; ignored")
            return
        self._add_terminal_card(
            run_id=run_id, turn_index=turn, call_id=result.call_id, name=result.name,
            args_raw=self._args_by_call.get((run_id, result.call_id)),
            content=result.content, is_error=result.is_error, artifact=artifact,
        )

    def add_completed_card(
        self, *, run_id: str, turn_index: int, call_id: str, name: str,
        arguments_json: str | None, content: str, is_error: bool,
        artifact: str | None = None,
    ) -> int:
        """恢复投影入口 (票据 05): 直接以终态挂一张完成卡, 返回其到达序下标。

        run_id 由调用方命名空间化 (如 restored:<request_id>), 与本进程新 run
        不碰撞; 同键重复是调用方缺陷, 记 ui_failures 并返回现有卡下标。
        """
        key = (run_id, turn_index, call_id)
        existing = self._by_key.get(key)
        if existing is not None:
            self._record_failure(f"duplicate restored card {key}; ignored")
            return self._cards.index(existing)
        self._add_terminal_card(
            run_id=run_id, turn_index=turn_index, call_id=call_id, name=name,
            args_raw=arguments_json, content=content, is_error=is_error, artifact=artifact,
        )
        return len(self._cards) - 1

    def _add_terminal_card(
        self, *, run_id: str, turn_index: int, call_id: str, name: str,
        args_raw: str | None, content: str, is_error: bool, artifact: str | None,
    ) -> None:
        card = _ToolCard(
            run_id=run_id,
            turn_index=turn_index,
            call_id=call_id,
            name=_clean_text(name),
            args_summary=self._summarize_raw(args_raw),
        )
        card.complete(content, is_error)
        card.artifact = artifact
        self._append(card)

    # --- 只读视图 ---------------------------------------------------------------

    @property
    def expanded(self) -> bool:
        return self._expanded

    @property
    def entries(self) -> tuple[ToolCardView, ...]:
        """当前条目快照 (按到达序); 修改快照不影响文档。"""
        return tuple(
            ToolCardView(
                run_id=card.run_id,
                turn_index=card.turn_index,
                call_id=card.call_id,
                name=card.name,
                status=card.status,
                is_error=card.is_error,
                cancelled=card.cancelled,
                output_text=card.content,
            )
            for card in self._cards
        )

    # --- 展示 -------------------------------------------------------------------

    def toggle_expand(self) -> None:
        """Ctrl+O 接缝: 翻转全局展开标志; 只改展示, 不触任何请求或工具。"""
        self._expanded = not self._expanded
        self._touch()

    @property
    def version(self) -> int:
        """单调内容版本 (条目/输出/展开变化即递增); 组合层条目缓存键用 (票据 05)。"""
        return self._version

    def render_card(self, index: int, width: int) -> list[str]:
        """单卡渲染 (票据 05 组合): 到达序第 index 张卡的显示行。

        越界是组合层缺陷: 有界处理 (记 ui_failures 返回空), 不抛异常杀渲染。
        """
        width = max(1, int(width))
        if not 0 <= index < len(self._cards):
            self._record_failure(f"render_card index out of range: {index}")
            return []
        return self._render_card(self._cards[index], width)

    def render_lines(self, width: int) -> list[str]:
        """按当前列宽产出全部工具卡的显示行 (含返回副本, 修改不影响缓存)。"""
        width = max(1, int(width))
        cached = self._cache
        if cached is not None and cached[0] == width and cached[1] == self._version:
            return list(cached[2])
        lines: list[str] = []
        for card in self._cards:
            lines.extend(self._render_card(card, width))
        self._cache = (width, self._version, list(lines))
        return lines

    def _render_card(self, card: _ToolCard, width: int) -> list[str]:
        """单卡布局 (Pi ToolExecutionComponent 子集): 前置空行 + Box(1,1) 背景 +
        标题 (粗体工具名/参数摘要) + 输出预览区 (10 行折叠与展开提示)。"""
        bg = _STATUS_BG[card.status]
        # Pi Text 窄窗规则: 边距放不下时收窄, 保证每行可见宽度不超限
        pad_x = min(1, max(0, (width - 1) // 2))
        content_w = max(1, width - pad_x * 2)

        def row(text: str) -> str:
            return self._bg_row(bg, text, width, pad_x)

        # Pi 组件先 addChild(Spacer(1)): 卡片之间隔一个无背景空行
        lines: list[str] = [""]
        lines.append(row(""))  # Box paddingY=1 顶部背景行

        title_text = f"{card.name} {card.args_summary}".strip()
        pos = 0
        for chunk in wrap_text_ansi(title_text, content_w):
            lines.append(row(self._bold_name_span(chunk, pos, card.name)))
            pos += len(chunk)

        # 回执输出工件 (路径 + hash): 只显示来源事实, 不读入内容
        if card.artifact:
            for chunk in wrap_text_ansi(card.artifact, content_w):
                lines.append(row(MUTED + chunk))

        if card.terminal and card.content:
            output_lines = card.content.split("\n")
            shown = output_lines if self._expanded else output_lines[:PREVIEW_LINES]
            for line in shown:
                for chunk in wrap_text_ansi(line, content_w):
                    lines.append(row(chunk))
            remaining = len(output_lines) - len(shown)
            if remaining > 0:
                hint = f"... 还有 {remaining} 行 (Ctrl+O 展开)"
                for chunk in wrap_text_ansi(hint, content_w):
                    lines.append(row(MUTED + chunk))

        lines.append(row(""))  # Box paddingY=1 底部背景行
        return lines

    @staticmethod
    def _bold_name_span(chunk: str, chunk_start: int, name: str) -> str:
        """对换行块中与工具名区间 [0, len(name)) 重叠的部分套粗体。

        样式在纯文本换行之后按区间补齐 (word_wrap_line 不感知 ANSI),
        名字被换行截断时各块分别开合粗体; 样式序列零可见宽度。
        """
        if not name:
            return chunk
        overlap_start = max(0, chunk_start)
        overlap_end = min(len(name), chunk_start + len(chunk))
        if overlap_start >= overlap_end:
            return chunk
        left = chunk[: overlap_start - chunk_start]
        mid = chunk[overlap_start - chunk_start : overlap_end - chunk_start]
        right = chunk[overlap_end - chunk_start :]
        return f"{left}{BOLD}{mid}{BOLD_OFF}{right}"

    # --- 内部 -------------------------------------------------------------------

    def _summarize_args(self, run_id: str, call_id: str) -> str:
        """参数摘要: 优先常用主参数, 否则单行 JSON; 全部来自真实事件文本。"""
        return self._summarize_raw(self._args_by_call.get((run_id, call_id)))

    def _summarize_raw(self, raw: str | None) -> str:
        if raw is None:
            return ""
        text = raw.strip()
        if not text or text == "{}":
            return ""
        try:
            obj = json.loads(text)
        except ValueError:
            return self._cap_args(_clean_text(text))
        if isinstance(obj, dict) and obj:
            for key in _PREFERRED_ARG_KEYS:
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return self._cap_args(_clean_text(value))
                if value is not None and not isinstance(value, (dict, list)):
                    return self._cap_args(_clean_text(str(value)))
        if isinstance(obj, (dict, list)):
            compact = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
            return self._cap_args(_clean_text(compact))
        return self._cap_args(_clean_text(str(obj)))

    @staticmethod
    def _cap_args(summary: str) -> str:
        if len(summary) <= _ARGS_SUMMARY_MAX_CHARS:
            return summary
        return summary[:_ARGS_SUMMARY_MAX_CHARS] + "…"

    def _bg_row(self, bg: str, text: str, width: int, pad_x: int) -> str:
        """背景行: 状态背景 + 左边距 + 内容 + 填充到整行; 每行可见宽度 <= width。"""
        available = width - pad_x
        if visible_width(text) > available:
            # 防御: 换行后仍超宽 (如 1 列窗里的双列字符); 截断保版式并记录
            self._record_failure(f"tool card row exceeds width ({visible_width(text)} > {available}); truncated")
            text = truncate_to_width(text, available)
        pad = max(0, available - visible_width(text))
        return bg + " " * pad_x + text + " " * pad + RESET

    def _append(self, card: _ToolCard) -> None:
        self._cards.append(card)
        self._by_key[(card.run_id, card.turn_index, card.call_id)] = card
        self._touch()

    def _touch(self) -> None:
        self._version += 1

    def _record_failure(self, message: str) -> None:
        if len(self.ui_failures) < _UI_FAILURES_MAX:
            self.ui_failures.append(message)
        self._touch()


class ToolCardEntry:
    """单卡 transcript 条目适配 (票据 05 组合): 按到达序委托 ToolDocument 渲染。

    每张工具卡在创建时经 transcript.add_entry 挂载一个本适配条目, 保持文档
    顺序与消息交错一致; 渲染按 (宽度, 文档版本) 缓存, 展开切换或内容更新
    (版本递增) 后重排。
    """

    def __init__(self, document: "ToolDocument", index: int) -> None:
        self._document = document
        self._index = index
        self._cache: tuple[int, int, list[str]] | None = None

    def render(self, width: int) -> list[str]:
        cached = self._cache
        version = self._document.version
        if cached is not None and cached[0] == width and cached[1] == version:
            return list(cached[2])
        lines = self._document.render_card(self._index, width)
        self._cache = (width, version, list(lines))
        return lines


__all__ = [
    "BOLD",
    "BOLD_OFF",
    "MUTED",
    "PREVIEW_LINES",
    "RESET",
    "TOOL_BG_ERROR",
    "TOOL_BG_PENDING",
    "TOOL_BG_SUCCESS",
    "ToolCardEntry",
    "ToolCardView",
    "ToolDocument",
]
