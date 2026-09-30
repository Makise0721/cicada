"""内核消息协议: 用户/助手/工具结果消息与工具调用配对类型."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

StopReason: TypeAlias = Literal["stop", "tool_use", "length", "error", "aborted"]


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用; arguments_json 为模型原始参数文本."""

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结果; 以 call_id 与 ToolCall 配对."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class UserMessage:
    text: str
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantMessage:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = "stop"
    error: str | None = None
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True)
class ToolResultMessage:
    result: ToolResult
    role: Literal["tool"] = "tool"


Message: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage
