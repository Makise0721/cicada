"""假工具: echo 成功 / fail 抛错 / slow 等待取消, 覆盖内核配对与取消路径."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext


class EchoTool:
    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="回显 text 参数",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.invocations.append(arguments)
        return ToolResult(call_id=ctx.call_id, name="echo", content=arguments["text"])


class FailTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="fail", description="总是抛错", parameters={"type": "object", "properties": {}})

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise RuntimeError("fail tool always raises")


class SlowTool:
    """等待取消或超时; 用于取消路径验收."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow",
            description="等待 cancel 或 timeout 秒",
            parameters={"type": "object", "properties": {"timeout": {"type": "number"}}},
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.started.set()
        try:
            await asyncio.wait_for(ctx.cancel.wait(), timeout=arguments.get("timeout", 30))
        except asyncio.TimeoutError:
            return ToolResult(call_id=ctx.call_id, name="slow", content="completed without cancel")
        return ToolResult(call_id=ctx.call_id, name="slow", content="observed cancel", is_error=True)


def fake_tools_plugin(tools: list) -> PluginDefinition:
    tool_list = list(tools)

    def setup(ctx: PluginContext) -> None:
        ctx.provide("tools", tool_list)

    return PluginDefinition(name="fake-tools", setup=setup, provides=frozenset({"tools"}))
