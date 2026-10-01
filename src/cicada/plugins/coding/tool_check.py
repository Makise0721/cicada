"""check 工具: 模型只能选 check_id, 命令/cwd/timeout 由启动者在计划里固定.

模型可见 content 严格 ≤16 KiB (控制字段预留预算), 完整回执进入 `details`。
`details` 里的程序唯一 `receipt_id` 是 K/应用层关联历史回执的键, 不能用模型 call_id。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.verification import Verifier, VerificationError
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

TOOL_NAME = "check"


class CheckTool:
    def __init__(self, service: Verifier) -> None:
        self._service = service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=TOOL_NAME,
            description=(
                "运行启动者指定的固定检查, 或查看当前检查状态. action=run 需要 check_id; "
                "action=status 不接受 check_id. 模型不能提供或改写命令、cwd、timeout: "
                "只有计划里声明的 check_id 可以运行, 其它值被拒绝. "
                "run 会先捕获代码快照、原样执行该检查、再捕获快照: 退出码 0 且输出完整终结、"
                "前后快照一致才是 passed; 正常非零且终结完整是 failed; 超时、取消、启动失败、"
                "输出未 EOF、快照不可用或代码在检查中被改动都是 blocked. "
                "代码变化后旧回执变 stale, 最新一次尝试取代旧结果; freshness 为 "
                "current/stale/unknown, action=status 给出当前 freshness 与不能交付的原因. "
                "结果 content 有界 (≤16 KiB), 完整回执与输出工件在 details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["run", "status"]},
                    "check_id": {"type": "string", "minLength": 1},
                },
                "required": ["action"],
                "additionalProperties": False,
                "if": {"properties": {"action": {"const": "run"}}},
                "then": {"required": ["check_id"]},
                "else": {"not": {"required": ["check_id"]}},
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if arguments["action"] == "status":
            return await self._status(ctx)
        return await self._run(arguments["check_id"], ctx)

    async def _run(self, check_id: str, ctx: ToolContext) -> ToolResult:
        try:
            receipt = await self._service.run_check(check_id, ctx.cancel)
        except VerificationError as exc:
            return self._error(ctx, str(exc))
        # freshness 与回执分开: 执行事实是历史事实, 有效性依赖当前快照。
        view = await self._service.refresh(ctx.cancel)
        state = next((s for s in view.checks if s.check_id == check_id), None)
        freshness = state.freshness if state is not None else "unknown"
        details = self._service.receipt_details(receipt, freshness)
        details["snapshot_ref"] = view.snapshot_ref
        details["baseline_snapshot_ref"] = self._service.baseline_snapshot_ref
        details["process_uncertain"] = view.process_uncertain
        details["blocking_reasons"] = list(view.blocking_reasons)
        return ToolResult(
            call_id=ctx.call_id,
            name=TOOL_NAME,
            content=self._service.render_receipt_content(receipt),
            is_error=False,
            details=details,
        )

    async def _status(self, ctx: ToolContext) -> ToolResult:
        view = await self._service.refresh(ctx.cancel)
        engagement = _engagement(self._service, view.snapshot_ref)
        return ToolResult(
            call_id=ctx.call_id,
            name=TOOL_NAME,
            content=self._service.render_status_content(view, engagement),
            is_error=False,
            details={
                "action": "status",
                "verification_run_id": view.verification_run_id,
                "snapshot_ref": view.snapshot_ref,
                "scope_id": view.scope_id,
                "baseline_snapshot_ref": self._service.baseline_snapshot_ref,
                "process_uncertain": view.process_uncertain,
                "blocking_reasons": list(view.blocking_reasons),
                "checks": [
                    {
                        "check_id": state.check_id,
                        "status": state.status,
                        "freshness": state.freshness,
                        "receipt_id": state.receipt.receipt_id if state.receipt else None,
                        "failure_kind": state.receipt.failure_kind if state.receipt else None,
                    }
                    for state in view.checks
                ],
                "receipt_ids": [receipt.receipt_id for receipt in view.receipts],
            },
        )

    @staticmethod
    def _error(ctx: ToolContext, message: str) -> ToolResult:
        return ToolResult(call_id=ctx.call_id, name=TOOL_NAME, content=message, is_error=True)


def _engagement(service: Verifier, snapshot_ref: str | None) -> str | None:
    """快照不可用本身就说明检查无法针对当前代码成立; 这是状态, 不是工具故障."""
    if snapshot_ref is None:
        return "the current code snapshot is unavailable, so no receipt can be current"
    return None


def check_plugin() -> PluginDefinition:
    """tool-check 插件: require coding.verification, provide tool.check."""

    def setup(ctx: PluginContext) -> None:
        service: Verifier = ctx.require("coding.verification")
        ctx.provide("tool.check", CheckTool(service))

    return PluginDefinition(
        name="tool-check",
        setup=setup,
        provides=frozenset({"tool.check"}),
        requires=frozenset({"coding.verification"}),
    )
