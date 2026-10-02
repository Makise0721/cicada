"""powershell 工具: 经 ProcessRunner 在 workspace 内执行 pwsh 命令."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.process import BoundedText, PowerShellRunner
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

DEFAULT_TIMEOUT = 120.0
MAX_CONTENT_BYTES = 50 * 1024  # 状态行/footer/工件提示全部计入的 UTF-8 总预算
_STATUS_RESERVE_BYTES = 256  # 状态行的保守预算, 实际状态行远短于此
_TAIL_DROP_NOTE = "[tail omitted to fit the 50 KiB content cap]"


def _tail_fitting(text: str, max_bytes: int) -> str:
    """按 UTF-8 边界从左侧裁剪, 只保留不超过 max_bytes 的后缀."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[-max_bytes:].decode("utf-8", "ignore")


def _shorten_footer(
    bounded: BoundedText, path_text: str | None, notes: list[str], budget: int
) -> str:
    """footer 在预算内逐级降级: 完整说明 > 去掉工件路径 > 极简事实, 始终保留截断事实."""
    full = _truncation_notes(bounded, path_text, notes)
    if len(full.encode("utf-8")) <= budget:
        return full
    without_path = _truncation_notes(bounded, None, notes)
    if len(without_path.encode("utf-8")) <= budget:
        return without_path
    minimal = f"[output_truncated=true artifact_truncated={bounded.artifact_truncated}]"
    if len(minimal.encode("utf-8")) <= budget:
        return minimal
    return ""


def _truncation_notes(
    bounded: BoundedText, path_text: str | None, notes: list[str]
) -> str:
    reasons = list(notes)
    if bounded.artifact_error is not None:
        reasons.append(f"output artifact unavailable: {bounded.artifact_error}")
    if bounded.artifact_truncated:
        reasons.append("output artifact truncated at 16 MiB (prefix only)")
    elif path_text is not None:
        reasons.append("full output written to the artifact path above")
    return (
        f"[truncated=true reasons: {'; '.join(reasons)}; "
        f"raw total {bounded.total_bytes} bytes; re-run a narrower command if you need more]"
    )


def _output_footer(
    bounded: BoundedText,
    artifact: str | None,
    output_complete: bool,
    notes: list[str],
    budget: int,
) -> str:
    if bounded.artifact_error is not None:
        # 没有可读工件时必须明说, 不能把有界 tail 说成完整输出
        return _shorten_footer(bounded, None, notes, budget)
    if not bounded.truncated:
        if output_complete:
            return ""
        return (
            "[output_complete=false; the output pipe did not reach EOF, "
            "so termination is not established]"
        )
    return _shorten_footer(bounded, artifact, notes, budget)


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
                "stdout/stderr 按到达序合并; 最终 content (含状态行与提示) 严格不超过 50 KiB UTF-8, "
                "超限时只保留有界 tail, 完整输出持续写入工作区 .cicada/outputs 下的工件 (最多 16 MiB)."
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
        artifact = str(bounded.full_output_path) if bounded.full_output_path else None
        # 未采到 EOF 但已有可见输出说明命令未必终结, 明确要求按未完成处理
        incomplete_notes = (
            ["output pipe did not reach EOF; termination is not established"]
            if not result.output_complete
            else []
        )
        parts = [
            f"[exit_code={result.exit_code} timed_out={result.timed_out} "
            f"cancelled={result.cancelled} truncated={bounded.truncated} "
            f"output_complete={result.output_complete} "
            f"artifact_truncated={bounded.artifact_truncated}]"
        ]
        if artifact is not None:
            parts.append(f"[full output: {artifact}]")
        content = self._assemble(parts, bounded, artifact, result.output_complete, incomplete_notes)
        # 采集失败 (含 reader 异常: 此时 timed_out 仍是 false) 也是工具错误, 不能把不完整
        # 输出当成功结果交给模型; Verifier/Policy 另按 output_complete 保守阻断。
        is_error = (
            result.exit_code != 0
            or result.timed_out
            or result.cancelled
            or not result.output_complete
        )
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
                "full_output_path": artifact,
                "full_output_bytes": bounded.total_bytes,
                "artifact_truncated": bounded.artifact_truncated,
                "artifact_error": bounded.artifact_error,
                "output_complete": result.output_complete,
            },
        )

    @staticmethod
    def _assemble(
        parts: list[str],
        bounded: BoundedText,
        artifact: str | None,
        output_complete: bool,
        notes: list[str],
    ) -> str:
        """有界 tail + 状态行 + footer 严格落在 MAX_CONTENT_BYTES 内.

        普通尾部视图语义保持不变: 正文在前, 状态行在后。
        """
        status_block = "".join(f"\n{part}" for part in parts)
        status_bytes = len(status_block.encode("utf-8"))
        body_budget = MAX_CONTENT_BYTES - status_bytes - _STATUS_RESERVE_BYTES
        text = _tail_fitting(bounded.text, max(body_budget, 0))
        if bounded.truncated and len(text.encode("utf-8")) < len(bounded.text.encode("utf-8")):
            text = f"{_TAIL_DROP_NOTE}\n{text}" if text else _TAIL_DROP_NOTE
        assembled = f"{text}{status_block}" if text else status_block.lstrip("\n")
        footer = _output_footer(
            bounded,
            artifact,
            output_complete,
            notes,
            MAX_CONTENT_BYTES - len(assembled.encode("utf-8")) - 1,
        )
        if footer:
            assembled = f"{assembled}\n{footer}"
        if len(assembled.encode("utf-8")) <= MAX_CONTENT_BYTES:
            return assembled
        # 状态行自身异常膨胀 (仅超长路径可达): 只保留有界尾部, 绝不返回半截字段或越界内容
        return _tail_fitting(assembled, MAX_CONTENT_BYTES)


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
