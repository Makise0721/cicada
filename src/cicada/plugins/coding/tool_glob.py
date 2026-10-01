"""glob 工具: 在 Git 可见清单内按限定 glob 查找文件."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import (
    GLOB_DEFAULT_LIMIT,
    GLOB_MAX_LIMIT,
    SearchError,
    run_glob,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

TOOL_NAME = "glob"


class GlobTool:
    def __init__(self, workspace: Workspace, inventory: GitInventory) -> None:
        self._workspace = workspace
        self._inventory = inventory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=TOOL_NAME,
            description=(
                "在 Git 可见清单 (tracked + 未被忽略的 untracked) 内按限定 glob 查找文件, "
                "只返回文件、不返回目录. pattern 逐段支持 * 与 ?, 独立段 ** 匹配零个或多个路径段; "
                "path 默认 '.', 指定单文件时只匹配该文件; 结果含绝对路径, 可直接交给 read. "
                "content 严格不超过 8192 UTF-8 bytes, 超出 limit 时明确标记截断并要求收窄."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": GLOB_MAX_LIMIT,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            rendered, _skipped = await run_glob(
                self._workspace,
                self._inventory,
                pattern=arguments["pattern"],
                raw_path=arguments.get("path", "."),
                limit=arguments.get("limit", GLOB_DEFAULT_LIMIT),
                cancel=ctx.cancel,
            )
        except SearchError as exc:
            return ToolResult(
                call_id=ctx.call_id, name=TOOL_NAME, content=str(exc), is_error=True
            )
        return ToolResult(
            call_id=ctx.call_id,
            name=TOOL_NAME,
            content=rendered.content,
            details={
                "shown": rendered.shown,
                "truncated": rendered.truncated,
                "complete": not rendered.truncated,
                "truncation_reason": rendered.reason,
            },
        )


def glob_plugin() -> PluginDefinition:
    """tool-glob 插件: 依赖 coding.workspace 与 coding.inventory, 提供 tool.glob."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        inventory: GitInventory = ctx.require("coding.inventory")
        ctx.provide("tool.glob", GlobTool(workspace, inventory))

    return PluginDefinition(
        name="tool-glob",
        setup=setup,
        provides=frozenset({"tool.glob"}),
        requires=frozenset({"coding.workspace", "coding.inventory"}),
    )
