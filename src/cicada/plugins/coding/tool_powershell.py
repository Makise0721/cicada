"""powershell 工具: 经 ProcessRunner 在 workspace 内执行 pwsh 命令."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.process import PowerShellRunner
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

DEFAULT_TIMEOUT = 120.0


class PowerShellTool:
    def __init__(self, workspace: Workspace, runner: PowerShellRunner | None = None) -> None:
        self._workspace = workspace
        self._runner = runner or PowerShellRunner()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="powershell",
            description=(
                "在 workspace 内以 pwsh -NoProfile -NonInteractive 执行命令. "
                "stdout/stderr 按到达序合并; 输出超限时 tail 截断并把完整输出写入工作区 .cicada/outputs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        try:
            result = await self._runner.run(
                command=arguments["command"],
                cwd=self._workspace.root,
                timeout=timeout,
                cancel=ctx.cancel,
                output_dir=self._workspace.output_dir,
            )
        except FileNotFoundError:
            return ToolResult(
                call_id=ctx.call_id,
                name="powershell",
                content="pwsh executable not found on PATH",
                is_error=True,
            )
        bounded = result.output
        status = (
            f"[exit_code={result.exit_code} timed_out={result.timed_out} "
            f"cancelled={result.cancelled} truncated={bounded.truncated}]"
        )
        if bounded.truncated and bounded.full_output_path is not None:
            status += f" full output: {bounded.full_output_path}"
        content = f"{bounded.text}\n{status}" if bounded.text else status
        is_error = result.exit_code != 0 or result.timed_out or result.cancelled
        return ToolResult(
            call_id=ctx.call_id,
            name="powershell",
            content=content,
            is_error=is_error,
            details={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
                "truncated": bounded.truncated,
                "full_output_path": (
                    str(bounded.full_output_path) if bounded.full_output_path else None
                ),
            },
        )


def powershell_plugin() -> PluginDefinition:
    """tool-powershell 插件: 依赖 coding.workspace 与 coding.process, 提供 tool.powershell."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        runner: PowerShellRunner = ctx.require("coding.process")
        ctx.provide("tool.powershell", PowerShellTool(workspace, runner))

    return PluginDefinition(
        name="tool-powershell",
        setup=setup,
        provides=frozenset({"tool.powershell"}),
        requires=frozenset({"coding.workspace", "coding.process"}),
    )
