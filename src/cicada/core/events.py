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
    | AssistantCompleted
    | ModelCallCompleted
    | ToolStarted
    | ToolCompleted
    | RunFinished
)
