"""内核会话事件: typed, 与插件运行时生命周期通知无关."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from cicada.core.messages import AssistantMessage, StopReason, ToolResult
from cicada.core.ports import ModelMetrics


@dataclass(frozen=True)
class RunStarted:
    run_id: str


@dataclass(frozen=True)
class TurnStarted:
    run_id: str
    turn_index: int


@dataclass(frozen=True)
class AssistantCompleted:
    run_id: str
    message: AssistantMessage


@dataclass(frozen=True)
class AssistantTextDelta:
    """只读文本增量: 供 UI 活动预览; thinking 与参数增量不发布.

    完成事件 (AssistantCompleted) 仍是回答与工具调用的权威, 预览不是
    可供模型续写的历史.
    """

    run_id: str
    turn_index: int
    text: str


@dataclass(frozen=True)
class AssistantReasoningDelta:
    """只读思考增量: 供 UI 活动预览; 空字符串同样发布 (区分未返回字段).

    完成事件 (AssistantCompleted.message.reasoning) 仍是思考权威, 预览不是
    规范记录, 也不参与模型续写历史。
    """

    run_id: str
    turn_index: int
    text: str


@dataclass(frozen=True)
class ModelCallCompleted:
    """一次模型调用的计量记录; 取消/错误路径同样产生记录."""

    run_id: str
    turn_index: int
    stop_reason: StopReason
    metrics: ModelMetrics | None
    elapsed_s: float


@dataclass(frozen=True)
class ToolStarted:
    run_id: str
    call_id: str
    name: str


@dataclass(frozen=True)
class ToolCompleted:
    run_id: str
    result: ToolResult


@dataclass(frozen=True)
class RunFinished:
    run_id: str
    stop_reason: StopReason
    error: str | None = None


Event: TypeAlias = (
    RunStarted
    | TurnStarted
    | AssistantTextDelta
    | AssistantReasoningDelta
    | AssistantCompleted
    | ModelCallCompleted
    | ToolStarted
    | ToolCompleted
    | RunFinished
)
