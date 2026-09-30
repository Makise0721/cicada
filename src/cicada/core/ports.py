"""内核能力端口: 模型与工具的抽象接口. 插件实现这些协议, 内核不知道注册机制."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, TypeAlias

from cicada.core.cancel import CancelToken
from cicada.core.messages import Message, StopReason, ToolResult


@dataclass(frozen=True)
class ToolSpec:
    """工具声明; 每次模型请求携带当前快照."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class ModelMetrics:
    """provider 报告的计量; None 表示不可得, 不是 0."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_duration_s: float | None = None


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    system_prompt: str = ""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """完整的一次工具调用事件. 增量参数事件是后续扩展, 不改变本接口."""

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class StreamDone:
    """流终结事件; 适配器协议错误必须经由此事件表达, 不得裸抛."""

    stop_reason: StopReason
    error: str | None = None
    metrics: ModelMetrics | None = None


StreamEvent: TypeAlias = TextDelta | ToolCallEvent | StreamDone


class ModelPort(Protocol):
    def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        """产出流事件; 必须以 StreamDone 终结 (正常/错误/取消)."""
        ...


@dataclass(frozen=True)
class ToolContext:
    call_id: str
    cancel: CancelToken


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """arguments 已通过 JSON Schema 校验; 取消经 ctx.cancel 协作传播."""
        ...
