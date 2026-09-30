"""write 工具: UTF-8 整文件写入, 自动建父目录, 经同文件队列串行."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext


class WriteTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write",
            description=(
                "以 UTF-8 写入文件, 自动创建父目录, 已存在则覆盖. "
                "不继承原文件 BOM/换行约定; 非原子替换."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = self._workspace.resolve_for_write(arguments["path"])
        except PathNotAllowedError as exc:
            return ToolResult(
                call_id=ctx.call_id, name="write", content=str(exc), is_error=True
            )

        async def op() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding="utf-8", newline="")

        await self._workspace.mutate(path, op)
        try:
            display = str(path.relative_to(self._workspace.root))
        except ValueError:
            display = str(path)
        return ToolResult(
            call_id=ctx.call_id,
            name="write",
            content=f"wrote {len(arguments['content'].encode('utf-8'))} bytes to {display}",
        )


def write_plugin() -> PluginDefinition:
    """tool-write 插件: 依赖 coding.workspace, 提供 tool.write."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        ctx.provide("tool.write", WriteTool(workspace))

    return PluginDefinition(
        name="tool-write",
        setup=setup,
        provides=frozenset({"tool.write"}),
        requires=frozenset({"coding.workspace"}),
    )
