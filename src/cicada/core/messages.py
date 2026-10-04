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


def validate_history_structure(messages: Any) -> None:
    """校验结构化历史: 类型、组内身份、闭合与次序; 违规 ValueError.

    Agent 执行前与持久化恢复共用同一拒绝时机与内容: 校验只读不复制,
    不强制每个用户请求恰好一个 assistant, 以保留多轮工具循环.
    """
    if not isinstance(messages, tuple):
        raise ValueError(f"history must be a tuple of messages, got {type(messages).__name__}")
    for index, message in enumerate(messages):
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise ValueError(f"history[{index}] is not a supported message type")
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if isinstance(message, ToolResultMessage):
            raise ValueError(f"history[{index - 1}] is a tool result without a preceding open tool call group")
        if not isinstance(message, AssistantMessage) or not message.tool_calls:
            continue
        ids: set[str] = set()
        for call in message.tool_calls:
            if (
                not isinstance(call, ToolCall)
                or not isinstance(call.id, str)
                or not call.id
                or not isinstance(call.name, str)
                or not call.name
                or not isinstance(call.arguments_json, str)
            ):
                raise ValueError(
                    f"history[{index - 1}] contains a tool call with invalid identity; "
                    "id/name must be nonempty strings and arguments_json a string"
                )
            if call.id in ids:
                raise ValueError(
                    f"history[{index - 1}] contains duplicate tool call id {call.id!r} within one group"
                )
            ids.add(call.id)
        if index + len(message.tool_calls) > len(messages):
            raise ValueError(f"history[{index - 1}] ends with unclosed tool calls")
        for call in message.tool_calls:
            follow = messages[index]
            if not isinstance(follow, ToolResultMessage):
                raise ValueError(
                    f"history[{index}] does not close tool call {call.id!r} in declared order"
                )
            if follow.result.call_id != call.id or follow.result.name != call.name:
                raise ValueError(
                    f"history[{index}] does not match its declared tool call {call.id!r} "
                    f"({call.name!r}) by call id and tool name"
                )
            index += 1
