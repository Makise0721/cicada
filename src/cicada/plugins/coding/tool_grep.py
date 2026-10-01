"""grep 工具: 在 Git 可见清单内做字面量子串搜索 (无 regex 开关)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import (
    GREP_DEFAULT_LIMIT,
    GREP_MAX_CONTEXT,
    GREP_MAX_LIMIT,
    SearchError,
    run_grep,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

TOOL_NAME = "grep"


class GrepTool:
    def __init__(self, workspace: Workspace, inventory: GitInventory) -> None:
        self._workspace = workspace
        self._inventory = inventory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=TOOL_NAME,
            description=(
                "在 Git 可见清单内的文本文件中做字面量子串搜索, 输出含绝对路径与 1 起始行号, "
                "可直接交给 read (不支持正则). include 为相对 path 的限定 glob; context 给出上下文行; "
                "ignore_case 用 Unicode casefold. 每条匹配行算一个命中, shown 数匹配行; "
                "content 严格不超过 8192 UTF-8 bytes, 放不下时标记 truncated 并要求收窄查询."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "include": {"type": "string", "minLength": 1},
                    "ignore_case": {"type": "boolean"},
                    "context": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": GREP_MAX_CONTEXT,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": GREP_MAX_LIMIT,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            rendered, skipped = await run_grep(
                self._workspace,
                self._inventory,
                pattern=arguments["pattern"],
                raw_path=arguments.get("path", "."),
                include=arguments.get("include", "**/*"),
                ignore_case=arguments.get("ignore_case", False),
                context=arguments.get("context", 0),
                limit=arguments.get("limit", GREP_DEFAULT_LIMIT),
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
                "skipped": {name: count for name, count in skipped.items() if count},
            },
        )


def grep_plugin() -> PluginDefinition:
    """tool-grep 插件: 依赖 coding.workspace 与 coding.inventory, 提供 tool.grep."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        inventory: GitInventory = ctx.require("coding.inventory")
        ctx.provide("tool.grep", GrepTool(workspace, inventory))

    return PluginDefinition(
        name="tool-grep",
        setup=setup,
        provides=frozenset({"tool.grep"}),
        requires=frozenset({"coding.workspace", "coding.inventory"}),
    )
