"""内核消息协议: 用户/助手/工具结果消息与工具调用配对类型."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
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


def isolate_tool_result(result: ToolResult) -> ToolResult:
    """复制嵌套可变 details, 其余字段本就不可变; 类型结构保持原记录.

    frozen dataclass 只保护外层; 不隔离 details 会让订阅者/快照改动穿透
    规范历史与 RunPolicy 判定.
    """
    if result.details is None:
        return result
    return replace(result, details=copy.deepcopy(result.details))


def isolate_message(message: Message) -> Message:
    """返回隔离嵌套可变数据后的等价消息; 其余类型直接返回原对象."""
    if isinstance(message, ToolResultMessage):
        return ToolResultMessage(result=isolate_tool_result(message.result))
    return message
