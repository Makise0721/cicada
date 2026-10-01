"""P4 §6 应用层 ModelPort 投影策略: 每轮 stream 前刷新验证视图并重投影历史检查结果.

只构造发给模型的投影, 不修改内核会话记录, 不从输出文字推断成功或进程终结:

- 每轮先按 check/powershell 的**终结点事实** (details/receipt) 扫描原始历史, 超时/取消/
  非 EOF/缺少可靠终结事实的异常结果保守调用 `mark_process_uncertain`, 单调锁存.
  可靠的 `output_complete=False` (非 EOF drain) 独立锁存, 不因 exit_code=0 放行.
- 再 `refresh` 取得不可变 `VerificationView`, 构造 ≤4096 UTF-8 bytes 的系统状态段,
  追加在原有 system prompt 之后 (不改原提示词).
- 历史 check 结果的回执身份只在 `refresh` 之后按程序唯一 `receipt_id` 核对
  `view.receipts` (不按模型 call_id): 账本里找不到, 或回执自身没有可靠终结事实,
  都保守锁存; 存在 receipt_id 字符串不是跳过核对的理由.
- 原文一字不改, 只按账本在末尾追加一行有界 freshness 标记; 缺失可靠身份的保守标
  `unknown`. 工具调用的 id/name/配对顺序和消息对象本身都不重写.

未从模型可见文字解析任何“通过/完成”信号。文字只用于识别程序自己写的
**执行前/启动失败**哨兵 (内核在调用工具之前就拒绝, 或命令根本没起来), 这些结果与进程
终结无关, 不锁存; 其余无可靠终结事实的异常一律保守锁存。这是 bytes 保护, 不是 token
预算或 compaction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

from cicada.core.cancel import CancelToken
from cicada.core.messages import Message, ToolResultMessage
from cicada.core.ports import ModelPort, ModelRequest, StreamDone, StreamEvent
from cicada.plugins.coding.verification_contracts import (
    CheckReceipt,
    Freshness,
    VerificationService,
    VerificationView,
)

CONTROL_SECTION_MAX_BYTES = 4096
FRESHNESS_MARKER_MAX_BYTES = 256
IDENTIFIER_MAX_BYTES = 128
REASON_MAX_BYTES = 512
CHECK_TOOL_NAME = "check"
PROCESS_TOOL_NAME = "powershell"
FRESHNESS_TOOLS = frozenset({CHECK_TOOL_NAME, PROCESS_TOOL_NAME})

# 程序自己写的哨兵: 内核在调用工具**之前**就拒绝, 进程不可能起来.
PRE_EXECUTION_MARKERS = (
    "cancelled before execution",
    "unknown tool:",
    "invalid arguments JSON:",
    "arguments must be a JSON object",
    "arguments failed validation:",
    "model output truncated; tool call not executed",
)
# 命令根本没起来 (明确的启动失败), 与"进程终止未证"是不同事实.
LAUNCH_FAILURE_MARKERS = ("pwsh executable not found on PATH",)
# 工具执行阶段抛错/被取消: 进程可能已经起来且没有可靠的终结事实.
_EXECUTION_FAILURE_MARKERS = ("tool raised:", "cancelled during execution")


@dataclass(frozen=True)
class RunPolicyConfig:
    """投影策略配置; 只影响发给模型的投影, 不改内核会话."""

    control_section_max_bytes: int = CONTROL_SECTION_MAX_BYTES
    identifier_max_bytes: int = IDENTIFIER_MAX_BYTES
    reason_max_bytes: int = REASON_MAX_BYTES

    def __post_init__(self) -> None:
        for name in ("control_section_max_bytes", "identifier_max_bytes", "reason_max_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _ToolFacts:
    """一条 check/powershell 工具结果里可用于判定的结构化事实."""

    tool: str
    is_error: bool
    has_details: bool
    receipt_id: str | None
    output_complete: bool | None
    timed_out: bool
    cancelled: bool
    exit_code: int | None
    error_text: str

    @property
    def termination_known(self) -> bool:
        """是否有可靠的终结事实: 记录了退出码且采集到 EOF, 未超时/取消."""
        return (
            self.output_complete is True
            and not self.timed_out
            and not self.cancelled
            and self.exit_code is not None
        )

    @property
    def process_not_started(self) -> bool:
        """是否有程序自证的执行前/启动失败事实: 与进程终结无关, 不锁存."""
        text = self.error_text
        return _starts_with_any(text, PRE_EXECUTION_MARKERS) or _starts_with_any(
            text, LAUNCH_FAILURE_MARKERS
        )

    @property
    def execution_failure(self) -> bool:
        return _starts_with_any(self.error_text, _EXECUTION_FAILURE_MARKERS)


@dataclass(frozen=True)
class _TerminalScan:
    """一次历史扫描的结果: 立即锁存的原因 + 留待账本核对的回执身份."""

    reasons: tuple[str, ...]
    deferred_receipts: tuple[tuple[str, str], ...]  # (tool, receipt_id)



def verification_policy(
    model: ModelPort,
    service: VerificationService,
    config: RunPolicyConfig | None = None,
) -> ModelPort:
    """bootstrap `model_policy(model, runtime)` 可用的同步工厂."""
    return RunPolicy(model, service, config)


class RunPolicy:
    """ModelPort 装饰器: 包装原适配器并投影当前验证状态. 默认不启用 (由启动层显式组入)."""

    def __init__(
        self,
        inner: ModelPort,
        service: VerificationService,
        config: RunPolicyConfig | None = None,
    ) -> None:
        if not callable(getattr(inner, "stream", None)):
            raise TypeError("inner must be a ModelPort")
        if not callable(getattr(service, "refresh", None)):
            raise TypeError("service must be a VerificationService")
        self.inner = inner
        self.service = service
        self.config = config or RunPolicyConfig()

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        # 取消优先: 已取消则不发请求, 也不做刷新
        if cancel.cancelled:
            yield StreamDone("aborted")
            return
        scan = _scan_terminal_facts(request.messages)
        for reason in scan.reasons:
            self.service.mark_process_uncertain(reason)
        view = await self.service.refresh(cancel)
        # 回执身份只能在 refresh 之后核对真实账本; 未知/不可信一律保守锁存
        for reason in _reconcile_receipts(scan, view):
            self.service.mark_process_uncertain(reason)
        projected = ModelRequest(
            messages=tuple(_project_message(message, view, self.config) for message in request.messages),
            tools=request.tools,
            system_prompt=_with_control_section(request.system_prompt, view, self.config),
        )
        async for event in self.inner.stream(projected, cancel):
            yield event

    def _scan_terminal_facts(self, messages: tuple[Message, ...]) -> None:
        """按终结点事实保守锁存 process_uncertain (账本可核对的部分留给 refresh 之后)."""
        scan = _scan_terminal_facts(messages)
        for reason in scan.reasons:
            self.service.mark_process_uncertain(reason)


def _scan_terminal_facts(messages: tuple[Message, ...]) -> _TerminalScan:
    """只读结构化字段, 逐条判定立即锁存或留待账本核对; 不解析任意输出文字."""
    reasons: list[str] = []
    deferred: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, ToolResultMessage):
            continue
        facts = _tool_facts(message)
        if facts is None:
            continue
        deferred_receipt = _deferred_receipt_id(facts)
        if deferred_receipt is not None:
            deferred.append((facts.tool, deferred_receipt))
            continue
        reason = _uncertain_reason(facts)
        if reason is not None:
            reasons.append(reason)
    return _TerminalScan(reasons=tuple(reasons), deferred_receipts=tuple(deferred))


def _reconcile_receipts(scan: _TerminalScan, view: VerificationView) -> tuple[str, ...]:
    """按账本核对回执身份: 找不到回执或回执自身终结不可信时保守锁存."""
    if not scan.deferred_receipts:
        return ()
    index: dict[str, CheckReceipt] = {}
    for receipt in view.receipts:
        if not isinstance(receipt, CheckReceipt) or not isinstance(receipt.receipt_id, str):
            continue
        index[receipt.receipt_id] = receipt  # 重复身份无法分辨最近一次尝试
    reasons: list[str] = []
    for tool, receipt_id in scan.deferred_receipts:
        receipt = index.get(receipt_id)
        if receipt is None:
            reasons.append(
                f"{tool} result receipt_id is not in the verification ledger; termination unproven"
            )
        elif not _receipt_terminates(receipt):
            reasons.append(
                f"{tool} result receipt {receipt_id} does not record a trustworthy terminal fact "
                "in the verification ledger"
            )
    return tuple(reasons)


def _receipt_terminates(receipt: CheckReceipt) -> bool:
    """回执自身是否有可靠终结事实; 启动失败属于程序记录的明确事实, 与终结未知不同."""
    if receipt.cancelled or receipt.timed_out:
        return False
    if receipt.execution_status == "launch_failed":
        return receipt.exit_code is None
    return (
        receipt.execution_status == "exited"
        and receipt.output_complete is True
        and receipt.exit_code is not None
    )


def _tool_facts(message: ToolResultMessage) -> _ToolFacts | None:
    result = message.result
    if result.name not in FRESHNESS_TOOLS:
        return None
    details = result.details if isinstance(result.details, dict) else None
    error_text = result.content if result.is_error else ""
    if details is None:
        return _ToolFacts(
            tool=result.name,
            is_error=result.is_error,
            has_details=False,
            receipt_id=None,
            output_complete=None,
            timed_out=False,
            cancelled=False,
            exit_code=None,
            error_text=error_text,
        )
    receipt_id = details.get("receipt_id")
    output_complete = details.get("output_complete")
    exit_code = details.get("exit_code")
    return _ToolFacts(
        tool=result.name,
        is_error=result.is_error,
        has_details=True,
        receipt_id=receipt_id if isinstance(receipt_id, str) and receipt_id else None,
        output_complete=output_complete if isinstance(output_complete, bool) else None,
        timed_out=details.get("timed_out") is True,
        cancelled=details.get("cancelled") is True,
        exit_code=exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
        error_text=error_text,
    )


def _deferred_receipt_id(facts: _ToolFacts) -> str | None:
    """该结果只能用真实账本核对终结事实时, 返回它的回执身份.

    已经由 runner 直接记录的未终结事实 (超时/取消/执行异常) 不需要账本, 立即锁存;
    程序自证的执行前/启动失败与进程终结无关, 也不锁存。
    """
    if facts.receipt_id is None:
        return None
    if facts.timed_out or facts.cancelled or facts.execution_failure:
        return None
    return None if facts.process_not_started else facts.receipt_id


def _uncertain_reason(facts: _ToolFacts) -> str | None:
    """返回需要立即锁存的保守原因; 无可靠终结事实且进程可能已启动才锁存."""
    if facts.timed_out:
        return f"{facts.tool} timed out before EOF"
    if facts.cancelled:
        return f"{facts.tool} was cancelled; process termination unproven"
    if facts.process_not_started:
        return None  # 程序自证的执行前/启动失败: 与进程终结无关
    if facts.execution_failure:
        return f"{facts.tool} result reports a termination failure"
    if not facts.has_details:
        if facts.is_error:
            return f"{facts.tool} result has no structured details; termination unproven"
        return None
    # 非 EOF drain 的事实独立于退出码: exit_code=0 不能抵消 output_complete=false
    if facts.output_complete is False:
        return f"{facts.tool} ended without reaching EOF; termination unproven"
    if not facts.termination_known and facts.is_error:
        return f"{facts.tool} error carries no trustworthy terminal fact"
    return None


def _starts_with_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(text.startswith(marker) for marker in markers)


def _project_message(message: Message, view: VerificationView | None, config: RunPolicyConfig) -> Message:
    if not isinstance(message, ToolResultMessage):
        return message
    result = message.result
    if result.name not in FRESHNESS_TOOLS:
        return message
    freshness, reason = _freshness_of(result.details, view)
    marker = _freshness_marker(result.name, freshness, reason, config)
    if marker in result.content:
        return message  # 同一视图重复投影不叠加
    return ToolResultMessage(result=replace(result, content=f"{result.content}\n{marker}"))


def _freshness_of(details: Any, view: VerificationView | None) -> tuple[Freshness, str]:
    if view is None:
        return "unknown", "verification view unavailable"
    if not isinstance(details, dict):
        return "unknown", "no structured receipt in tool result"
    receipt_id = details.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        return "unknown", "tool result carries no receipt_id"
    index = {receipt.receipt_id: receipt for receipt in view.receipts}
    if receipt_id not in index:
        return "unknown", "receipt_id is not in the verification ledger"
    return _state_freshness(view, index[receipt_id])


def _state_freshness(view: VerificationView, receipt: CheckReceipt) -> tuple[Freshness, str]:
    for state in view.checks:
        if state.check_id != receipt.check_id:
            continue
        if state.receipt is None or state.receipt.receipt_id != receipt.receipt_id:
            return "stale", "a newer attempt for this check exists or the receipt is superseded"
        return state.freshness, state.receipt.failure_kind or state.receipt.error or ""
    return "unknown", "check_id is not part of the current verification plan"


def _freshness_marker(tool: str, freshness: Freshness, reason: str, config: RunPolicyConfig) -> str:
    head = (
        f"Cicada freshness: stale={str(freshness != 'current').lower()} "
        f"freshness={freshness} tool={tool}"
    )
    if freshness != "current" and reason:
        head = f"{head} reason={_short(reason, config.reason_max_bytes)}"
    return _clip_str(head, min(config.control_section_max_bytes, FRESHNESS_MARKER_MAX_BYTES))


def _with_control_section(prompt: str, view: VerificationView, config: RunPolicyConfig) -> str:
    section = _control_section(view, config)
    if not section:
        return prompt
    if not prompt:
        return section
    return f"{prompt}\n{section}"


def _control_section(view: VerificationView, config: RunPolicyConfig) -> str:
    """构造有界系统状态段; 关键字段 (snapshot/status/freshness/blocking) 优先保留."""
    ident, reason = config.identifier_max_bytes, config.reason_max_bytes
    lines = [
        "<cicada-verification-state>",
        f"verification_run_id: {_short(view.verification_run_id, ident)}",
    ]
    if view.snapshot_ref is None:
        lines.append("snapshot: unavailable")
        lines.append("freshness: unknown (the current code state could not be captured)")
    else:
        lines.append(f"snapshot_ref: {_short(view.snapshot_ref, ident)}")
        lines.append(f"scope_id: {_short(view.scope_id or 'unavailable', ident)}")
    if view.process_uncertain:
        lines.append(
            "process_uncertain: true (a process ended without proven termination; "
            "this run cannot become deliverable)"
        )
    if not view.checks:
        lines.append("required checks: none declared")
    for state in view.checks:
        lines.append(
            f"check {_short(state.check_id, ident)}: status={state.status} freshness={state.freshness}"
        )
        if state.receipt is not None and state.receipt.failure_kind:
            lines.append(f"  failure_kind: {_short(state.receipt.failure_kind, reason)}")
    lines.append("blocking_reasons:")
    if not view.blocking_reasons:
        lines.append("- none reported by the verification view")
    for item in view.blocking_reasons:
        lines.append(f"- {_short(item, reason)}")
    for state in view.checks:
        if state.freshness == "stale":
            lines.append(
                f"note: historical '{_short(state.check_id, ident)}' results below "
                "are stale because the code changed after them"
            )
    return _drain(lines, config.control_section_max_bytes)


def _drain(lines: list[str], limit: int) -> str:
    """按行累积到 limit 字节; 单行超限即截断并停止, 末尾留有界标记."""
    out = ""
    suffix = "\n[control section truncated at byte limit]"
    if limit <= len(suffix.encode("utf-8")):
        return _clip_str("".join(lines), limit)
    for line in lines:
        if len(out.encode("utf-8")) + len(line.encode("utf-8")) + 1 > limit - len(suffix.encode("utf-8")):
            budget = limit - len(out.encode("utf-8")) - len(suffix.encode("utf-8")) - 1
            rest = _clip_str(line, budget)
            if rest:
                out = f"{out}{rest}\n"
            out = f"{out}{suffix}"
            break
        out = f"{out}{line}\n"
    return _clip_str(out, limit)


def _short(text: str, limit: int) -> str:
    """有界单行文本: 去掉换行并截断到 limit UTF-8 bytes (不切半个字符)."""
    flat = " ".join(text.split())
    if len(flat.encode("utf-8")) <= limit:
        return flat
    return _clip_str(flat, limit)


def _clip_str(text: str, limit: int) -> str:
    """按 UTF-8 字节截断字符串; 只退到完整字符边界, 保证解码不产生半个字符."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    if limit <= 3:
        return encoded[:limit].decode("utf-8", errors="ignore")
    return encoded[: limit - 3].decode("utf-8", errors="ignore") + "..."
