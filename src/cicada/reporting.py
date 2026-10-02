"""运行摘要与交付判定: 由 RunResult / VerificationView / ChangeEvidence 直接生成.

契约 (P3 计划 §7): 没有报告值显示 unknown; 0 计入已知, None 不转成 0;
provider duration 与本地 elapsed 分别标注不相加; input token 跨调用求和是
重复处理在内的计量, 不是唯一上下文 token 数; 摘要不输出任务验收结论.

P4 §6: 检查模式的交付判定由程序执行 (decide_delivery), 不解析模型自然语言,
也不把 tools_ok/tool_errors 当验收; build_run_summary 的三行契约不变,
交付事实以独立 delivery section 输出 (PROMPT/摘要不做验收结论的语义保持).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cicada.core.agent import RunResult
from cicada.core.messages import ToolResultMessage

if TYPE_CHECKING:
    from cicada.plugins.coding.verification_contracts import (
        ChangeEvidence,
        VerificationPlan,
        VerificationView,
    )


def _sum_known_int(values: list[int | None]) -> tuple[str, int]:
    """(总和显示或 unknown, 已知个数); 0 是合法已知值."""
    known = [value for value in values if value is not None]
    return (str(sum(known)), len(known)) if known else ("unknown", 0)


def _sum_known_float(values: list[float | None]) -> tuple[str, int]:
    known = [value for value in values if value is not None]
    return (f"{sum(known):.2f}", len(known)) if known else ("unknown", 0)


def build_run_summary(result: RunResult) -> str:
    """生成三行运行摘要: 运行状态 / 耗时 / token 计量与覆盖率."""
    calls = result.model_calls
    total = len(calls)
    tool_results = [m.result for m in result.messages if isinstance(m, ToolResultMessage)]
    tool_errors = sum(1 for item in tool_results if item.is_error)

    model_time = f"{sum(call.elapsed_s for call in calls):.2f}" if calls else "unknown"
    provider_time, provider_known = _sum_known_float(
        [call.metrics.provider_duration_s if call.metrics else None for call in calls]
    )
    input_sum, input_known = _sum_known_int(
        [call.metrics.input_tokens if call.metrics else None for call in calls]
    )
    output_sum, output_known = _sum_known_int(
        [call.metrics.output_tokens if call.metrics else None for call in calls]
    )

    return "\n".join(
        [
            f"[run_summary run_id={result.run_id} stop_reason={result.stop_reason} "
            f"model_calls={total} tools={len(tool_results)} tool_errors={tool_errors}]",
            # provider_duration 是服务端自报, 与本地 elapsed 分别标注, 不相加当总运行时间
            f"[model_time_s={model_time} provider_time_s={provider_time} "
            f"provider_reported_calls={provider_known}/{total}]",
            f"[input_tokens_known={input_sum} input_reported_calls={input_known}/{total} "
            f"output_tokens_known={output_sum} output_reported_calls={output_known}/{total}]",
        ]
    )


@dataclass(frozen=True)
class DeliveryDecision:
    """程序交付判定: 只来自 stop_reason/验证视图/变更证据/工件核对事实."""

    model_stopped: bool
    can_deliver: bool
    blocking_reasons: tuple[str, ...]


def decide_delivery(
    result: RunResult,
    view: "VerificationView | None",
    evidence: "ChangeEvidence | None",
    artifact_problems: tuple[str, ...] = (),
) -> DeliveryDecision:
    """按 P4 §6 的交付条件独立判定; 不读模型文本, 不用 tools_ok/tool_errors.

    条件: 模型 stop; 无 process_uncertain; 最终快照可用; 全部指定检查的最近回执
    passed 且 freshness=current; 变更证据完整且其快照与最终检查视图一致; 回执引用的
    输出工件与记录 hash 相符 (artifact_problems 由应用层核对后传入)。
    """
    reasons: list[str] = []
    stopped = result.stop_reason == "stop"
    if not stopped:
        suffix = f" ({result.error})" if result.error else ""
        reasons.append(f"model ended with stop_reason={result.stop_reason}{suffix}")
    if view is None:
        reasons.append("the final verification view is unavailable")
    else:
        if view.process_uncertain:
            reasons.append(
                "process_uncertain: a process ended without proven termination in this run")
        if view.snapshot_ref is None:
            reasons.append("the final code snapshot is unavailable")
        for state in view.checks:
            if state.receipt is None:
                reasons.append(f"check {state.check_id} has not been run in this verification run")
                continue
            if state.status != "passed":
                detail = f" ({state.receipt.failure_kind})" if state.receipt.failure_kind else ""
                reasons.append(f"check {state.check_id} is {state.status}{detail}")
            elif state.freshness != "current":
                reasons.append(
                    f"check {state.check_id} passed but its receipt is {state.freshness}; "
                    "it does not match the final code state")
    if evidence is None:
        reasons.append("the change evidence is unavailable")
    else:
        if not evidence.complete:
            detail = f" ({evidence.failure_kind}): {evidence.error}" if evidence.error else ""
            reasons.append(f"the change evidence is incomplete{detail}")
        elif view is not None and view.snapshot_ref != evidence.snapshot_ref:
            reasons.append(
                "the code changed after the final check view: the check receipts do not "
                "belong to the snapshot in the change evidence")
    reasons.extend(artifact_problems)
    if view is not None:
        # 服务 blocking 原因 (基线/快照等) 与本地推导合并去重; 宁可重复披露也不少报。
        reasons.extend(view.blocking_reasons)
    unique = tuple(dict.fromkeys(reasons))
    return DeliveryDecision(model_stopped=stopped, can_deliver=stopped and not unique, blocking_reasons=unique)


def build_delivery_section(
    decision: DeliveryDecision,
    view: "VerificationView | None",
    evidence: "ChangeEvidence | None",
    plan: "VerificationPlan | None" = None,
) -> str:
    """独立 delivery section: 范围政策、检查/回执/有效性、快照、真实变更与工件 hash.

    只陈述程序事实; can_deliver=false 时 blocking_reasons 给出每个未满足条件。
    """
    lines = [
        "[delivery "
        f"can_deliver={str(decision.can_deliver).lower()} "
        f"model_stopped={str(decision.model_stopped).lower()}]"
    ]
    if view is not None:
        lines.append(f"[verification_run_id={view.verification_run_id}]")
        lines.append(
            f"[snapshot_ref={view.snapshot_ref}]"
            + (f" scope_id={view.scope_id}" if view.scope_id else "")
        )
        lines.append(f"[process_uncertain={str(view.process_uncertain).lower()}]")
    if plan is not None:
        lines.append(
            f"[scope policy_id={plan.policy_id} exclusions={','.join(plan.exclusions)} "
            f"checks={len(plan.checks)}]")
    if evidence is not None:
        lines.append(
            f"[baseline_snapshot_ref={evidence.baseline_snapshot_ref}]")
        artifact = evidence.artifact_path
        if artifact is None:
            lines.append("[change artifact: unavailable]")
        else:
            sha = evidence.artifact_sha256 or "unverified"
            lines.append(f"[change artifact: {artifact} sha256={sha}]")
    if view is not None and view.checks:
        lines.append("[checks]")
        for state in view.checks:
            receipt = state.receipt
            if receipt is None:
                lines.append(
                    f"- {state.check_id} status={state.status} freshness={state.freshness}")
                continue
            lines.append(
                f"- {state.check_id} status={state.status} freshness={state.freshness} "
                f"receipt_id={receipt.receipt_id} execution={receipt.execution_status} "
                f"exit_code={receipt.exit_code} failure_kind={receipt.failure_kind}"
            )
    if evidence is not None:
        lines.append(f"[changed files: {len(evidence.changes)}]")
        for change in evidence.changes:
            flags = ",".join(
                name for name, active in (
                    ("binary", change.binary),
                    ("bom_changed", change.bom_changed),
                    ("newline_changed", change.newline_changed),
                ) if active
            )
            suffix = f" flags={flags}" if flags else ""
            lines.append(
                f"- {change.kind} {change.relative_path} "
                f"before_sha256={change.before_sha256} after_sha256={change.after_sha256}"
                f"{suffix}"
            )
    lines.append("[blocking_reasons]")
    if decision.blocking_reasons:
        lines.extend(f"- {reason}" for reason in decision.blocking_reasons)
    else:
        lines.append("- none")
    return "\n".join(lines)
