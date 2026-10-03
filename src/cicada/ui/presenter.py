"""TuiPresenter 正式实现: ApplicationUpdate 流 -> 展示投影 (TUI 第一版 04).

职责边界 (ui/contracts.py 冻结):
- 只消费 ApplicationUpdate 与 RequestOutcome, 只改自己的 UI 状态; 不执行工具、
  不写会话、不计算自制的 PASS、不反向调用应用执行接口.
- 不产生第二套验证权威: 检查/交付事实一律取自 outcome.view / outcome.evidence /
  outcome.decision (程序判定); presenter 只做投影与来源标注.
- 显示清理不在这里: 本模块记录的正文保留原始文本 (规范事实), 清理由 rendering 在
  显示副本上完成; 因此 presenter 状态不受"显示清理"影响.
- mailbox (1024 条) 与 30 FPS 合并属于终端执行器/05 组合层的通知管道; 本实现负责
  文本合并语义 (增量合并进当前预览, 最终事实不丢) 与有界性 (预览/工具卡限额).

T06: 工具卡 key=(run_id, turn_index, call_id); 参数取同 turn AssistantCompleted 的
arguments_json; 无 ToolStarted 的 ToolCompleted (内核固定文本闭合) 建立 failed/未执行卡;
跨 turn 重用 call ID 各自成卡.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cicada.application import (
    AgentEventReceived,
    ApplicationClosed,
    ApplicationUpdate,
    HistoryCleared,
    RequestCompleted,
    RequestPhaseChanged,
)
from cicada.core.events import (
    AssistantCompleted,
    AssistantTextDelta,
    Event,
    ModelCallCompleted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.ui.contracts import (
    ActivePreview,
    PresentationRecord,
    PresentationUpdate,
    StatusSnapshot,
)

PREVIEW_MAX_LINES = 12
PREVIEW_MAX_BYTES = 4096
TOOL_CARD_MAX_LINES = 32
TOOL_CARD_MAX_BYTES = 8192
BODY_MAX_LINES = 64
BODY_MAX_BYTES = 16384

CHECK_TOOL_NAME = "check"

# presenter 产出的标注文案 (程序事实的中文标签; 不含 PASS/FAIL 之类自制判定)
NORMAL_MODE_CHECK_LABEL = "未启用检查"
HISTORICAL_PASS_LABEL = "此前检查，仍有效"
HISTORICAL_NEUTRAL_LABEL = "此前检查"
CURRENT_RUN_CHECK_LABEL = "本次检查"
PROCESS_UNCERTAIN_LABEL = "进程终结未知（锁存）"
NOT_RUN_CHECK_LABEL = "未运行"


def clip_lines(text: str, max_lines: int, max_bytes: int) -> tuple[tuple[str, ...], int, int, bool]:
    """按 UTF-8 bytes 与行数截断; 返回 (保留行, 原行数, 原 bytes, 是否截断).

    单行超限时退到完整字符边界; 只在确有内容被丢弃时标记截断.
    """
    lines = text.splitlines() or [""]
    original_bytes = len(text.encode("utf-8"))
    kept: list[str] = []
    budget = max_bytes
    truncated = False
    for line in lines:
        if len(kept) >= max_lines:
            truncated = True
            break
        encoded = line.encode("utf-8")
        if len(encoded) > budget:
            clipped = encoded[: max(budget, 0)].decode("utf-8", errors="ignore")
            kept.append(clipped)
            truncated = True
            break
        kept.append(line)
        budget -= len(encoded) + 1  # 预留换行符
    return tuple(kept), len(lines), original_bytes, truncated


def _truncation_note(kept: tuple[str, ...], original_lines: int, original_bytes: int, truncated: bool) -> tuple[str, ...]:
    if not truncated:
        return ()
    kept_bytes = len("\n".join(kept).encode("utf-8"))
    return (f"…[已截断: 显示 {len(kept)}/{original_lines} 行, {kept_bytes}/{original_bytes} UTF-8 bytes]",)


def _bounded_body(text: str) -> tuple[str, ...]:
    kept, original_lines, original_bytes, truncated = clip_lines(text, BODY_MAX_LINES, BODY_MAX_BYTES)
    return kept + _truncation_note(kept, original_lines, original_bytes, truncated)


def _sum_known(values: list[int | None]) -> int | None:
    # None 表示不可得: 任一调用该计量不可得时, 该累计就是不可得, 不显示部分和
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _card_status_for(result_content: str, is_error: bool) -> str:
    if result_content.startswith("cancelled"):
        return "cancelled"
    return "failed" if is_error else "done"


@dataclass(frozen=True)
class _ToolCardState:
    """一张工具卡的中间状态; 终结后转为 PresentationRecord."""

    run_id: str
    turn_index: int
    call_id: str
    name: str
    arguments_json: str
    started: bool
    result_content: str | None = None
    is_error: bool = False
    artifact_line: str | None = None
    # 本轮注册过卡才有资格断言"未执行"; 订阅前开始的卡只能标"状态未知"
    registered: bool = True


class ProjectionPresenter:
    """消费 ApplicationUpdate 的只读展示投影; 满足冻结的 TuiPresenter Protocol."""

    def __init__(
        self,
        *,
        model: str | None = None,
        workspace: str | None = None,
        verification_run_id: str | None = None,
    ) -> None:
        self._model = model
        self._workspace = workspace
        self._verification_run_id = verification_run_id
        self._records: dict[str, PresentationRecord] = {}
        self._order: list[str] = []
        self._cards: dict[str, _ToolCardState] = {}
        self._open_calls: dict[tuple[str, str], str] = {}  # (run_id, call_id) -> 卡 key
        self._preview_buffers: dict[tuple[str, int], str] = {}
        self._current_turn: int = -1
        self._current_preview: ActivePreview | None = None
        self._status = StatusSnapshot(
            model=model, workspace=workspace, verification_run_id=verification_run_id
        )
        self._metrics_input: list[int | None] = []
        self._metrics_output: list[int | None] = []
        self._check_receipts_this_request: set[str] = set()
        self._process_uncertain = False
        self._request_counter = 0

    # --- 只读快照 ------------------------------------------------------------------

    @property
    def records(self) -> tuple[PresentationRecord, ...]:
        """按追加顺序的全部记录; 消费方按键 upsert."""
        return tuple(self._records[key] for key in self._order)

    @property
    def status(self) -> StatusSnapshot:
        return self._status

    @property
    def current_preview(self) -> ActivePreview | None:
        return self._current_preview

    def has_process_uncertain(self) -> bool:
        """展示层锁存: 一旦为真不再清除 (与单调锁存事实一致的显示口径)."""
        return self._process_uncertain

    # --- accept --------------------------------------------------------------------

    def accept(self, update: ApplicationUpdate) -> PresentationUpdate:
        if isinstance(update, AgentEventReceived):
            return self._accept_event(update.event)
        if isinstance(update, RequestPhaseChanged):
            if update.phase == "executing" and update.run_id is None:
                self._begin_request()
            status = self._replace_status(phase=update.phase)
            return PresentationUpdate(status=status)
        if isinstance(update, RequestCompleted):
            return self._accept_completed(update)
        if isinstance(update, HistoryCleared):
            return self._accept_history_cleared()
        if isinstance(update, ApplicationClosed):
            role = "system" if update.error is None else "error"
            body = ("应用已关闭。",) if update.error is None else (
                "应用已关闭（清理未全部成功）:", f"  {update.error}",
            )
            record = self._upsert(PresentationRecord(
                key="app:closed", role=role, title="应用关闭", status="done" if update.error is None else "failed",
                body=body,
            ))
            return PresentationUpdate(appended=(record,))
        return PresentationUpdate()

    def accept_error(self, message: str) -> PresentationUpdate:
        """把应用层拒绝/异常文本 (如 session.execute 的 RuntimeError) 如实呈现为错误记录."""
        kept, lines, size, truncated = clip_lines(message, TOOL_CARD_MAX_LINES, TOOL_CARD_MAX_BYTES)
        record = self._upsert(PresentationRecord(
            key=f"error:runtime:{self._request_counter}:{len(self._order)}",
            role="error", title="应用拒绝", status="failed",
            body=kept + _truncation_note(kept, lines, size, truncated),
        ))
        return PresentationUpdate(appended=(record,))

    # --- 内核事件 ------------------------------------------------------------------

    def _accept_event(self, event: Event) -> PresentationUpdate:
        if isinstance(event, TurnStarted):
            self._current_turn = event.turn_index
            status = self._replace_status(turn_index=event.turn_index)
            return PresentationUpdate(status=status)
        if isinstance(event, AssistantTextDelta):
            self._current_turn = event.turn_index
            return self._accept_delta(event)
        if isinstance(event, AssistantCompleted):
            return self._accept_assistant(event)
        if isinstance(event, ModelCallCompleted):
            return self._accept_metrics(event)
        if isinstance(event, ToolStarted):
            return self._accept_tool_started(event)
        if isinstance(event, ToolCompleted):
            return self._accept_tool_completed(event)
        return PresentationUpdate()

    def _begin_request(self) -> None:
        self._request_counter += 1
        self._metrics_input = []
        self._metrics_output = []
        self._check_receipts_this_request = set()

    def _accept_delta(self, event: AssistantTextDelta) -> PresentationUpdate:
        key = (event.run_id, event.turn_index)
        merged = self._preview_buffers.get(key, "") + event.text
        self._preview_buffers[key] = merged
        kept, lines, size, truncated = clip_lines(merged, PREVIEW_MAX_LINES, PREVIEW_MAX_BYTES)
        self._current_preview = ActivePreview(
            run_id=event.run_id, turn_index=event.turn_index,
            text="\n".join(kept), truncated=truncated,
        )
        return PresentationUpdate(preview=self._current_preview)

    def _accept_assistant(self, event: AssistantCompleted) -> PresentationUpdate:
        message = event.message
        # AssistantCompleted 不携带 turn_index: 按 contracts 从当前 TurnStarted 取
        turn = self._current_turn
        appended: list[PresentationRecord] = []
        if message.text or message.error:
            key = f"{event.run_id}:{turn}:assistant"
            body_text = message.text + (f"\n错误: {message.error}" if message.error else "")
            # 最终回答是完整规范文本 (契约: 完整文本以最终 AssistantCompleted 为准),
            # 不套预览/卡片限额; 截断标注只属于活动预览与工具卡
            appended.append(self._upsert(PresentationRecord(
                key=key, role="assistant", title="助手", status="done" if message.error is None else "failed",
                body=tuple(body_text.split("\n")),
            )))
        for call in message.tool_calls:
            card_key = f"{event.run_id}:{turn}:{call.id}"
            self._cards[card_key] = _ToolCardState(
                run_id=event.run_id, turn_index=turn, call_id=call.id,
                name=call.name, arguments_json=call.arguments_json, started=False,
            )
            self._open_calls[(event.run_id, call.id)] = card_key
            appended.append(self._upsert(self._card_record(card_key, status="pending")))
        # 最终完成替换预览: 置空内部状态, 消费方以"最终记录已追加"收起活动预览
        self._current_preview = None
        return PresentationUpdate(appended=tuple(appended))

    def _accept_metrics(self, event: ModelCallCompleted) -> PresentationUpdate:
        self._metrics_input.append(event.metrics.input_tokens if event.metrics else None)
        self._metrics_output.append(event.metrics.output_tokens if event.metrics else None)
        status = self._replace_status(
            turn_index=event.turn_index,
            input_tokens=_sum_known(self._metrics_input),
            output_tokens=_sum_known(self._metrics_output),
        )
        return PresentationUpdate(status=status)

    def _accept_tool_started(self, event: ToolStarted) -> PresentationUpdate:
        card_key = self._open_calls.get((event.run_id, event.call_id))
        if card_key is None:
            return PresentationUpdate()
        card = self._cards[card_key]
        self._cards[card_key] = replace(card, started=True)
        record = self._upsert(self._card_record(card_key, status="running"))
        return PresentationUpdate(appended=(record,))

    def _accept_tool_completed(self, event: ToolCompleted) -> PresentationUpdate:
        result = event.result
        card_key = self._open_calls.pop((event.run_id, result.call_id), None)
        if card_key is None:
            # 无注册卡的 ToolCompleted (订阅晚加入等): 仍投影, 不丢事实;
            # 但没有本轮注册事实, 不能断言"未执行"
            card_key = f"{event.run_id}:unknown:{result.call_id}"
            self._cards[card_key] = _ToolCardState(
                run_id=event.run_id, turn_index=-1, call_id=result.call_id,
                name=result.name, arguments_json="", started=False, registered=False,
            )
        card = self._cards[card_key]
        self._cards[card_key] = replace(
            card, result_content=result.content, is_error=result.is_error,
            artifact_line=self._artifact_line(result.details),
        )
        if result.name == CHECK_TOOL_NAME and result.details:
            receipt_id = result.details.get("receipt_id")
            if isinstance(receipt_id, str) and receipt_id:
                self._check_receipts_this_request.add(receipt_id)
        record = self._upsert(self._card_record(
            card_key, status=_card_status_for(result.content, result.is_error)
        ))
        return PresentationUpdate(appended=(record,))

    @staticmethod
    def _artifact_line(details: dict | None) -> str | None:
        """回执引用的输出工件: 只显示路径与 hash, 不读入内容."""
        if not isinstance(details, dict):
            return None
        path = details.get("output_artifact_path")
        if not isinstance(path, str) or not path:
            return None
        sha = details.get("output_artifact_sha256")
        sha_text = sha if isinstance(sha, str) and sha else "unverified"
        return f"输出工件: {path} sha256={sha_text}"

    def _card_record(self, card_key: str, status: str) -> PresentationRecord:
        card = self._cards[card_key]
        title = f"工具 {card.name} ({card.call_id})"
        head: list[str] = []
        if card.turn_index >= 0:
            head.append(f"run={card.run_id} turn={card.turn_index}")
        if not card.started:
            head.append(
                "执行状态未知（订阅前开始）" if not card.registered else "未执行（无 ToolStarted）"
            )
        if card.artifact_line:
            head.append(card.artifact_line)
        # 参数与结果合并后整体受 32 行 / 8192 bytes 卡片限额约束
        detail = card.arguments_json
        if card.result_content is not None:
            detail = f"{detail}\n{card.result_content}" if detail else card.result_content
        remaining_lines = max(TOOL_CARD_MAX_LINES - len(head), 1)
        kept, lines, size, truncated = clip_lines(detail, remaining_lines, TOOL_CARD_MAX_BYTES)
        body = tuple(head) + kept + _truncation_note(kept, lines, size, truncated)
        return PresentationRecord(
            key=card_key, role="tool", title=title, status=status, body=body  # type: ignore[arg-type]
        )

    # --- 请求终结 ------------------------------------------------------------------

    def _accept_completed(self, update: RequestCompleted) -> PresentationUpdate:
        outcome = update.outcome
        result = outcome.result
        appended: list[PresentationRecord] = []
        view = outcome.view
        if view is not None:
            # 无快照的 view (收尾监督超时哨兵) 不携带真实 verification_run_id:
            # 保持既有真实 id, 不让哨兵值覆盖状态栏
            if view.snapshot_ref is not None:
                self._verification_run_id = view.verification_run_id
            if view.process_uncertain:
                self._process_uncertain = True
        # 运行结果记录: stop_reason / 错误 (普通模式与检查模式一致)
        summary_lines = [
            f"run={result.run_id} stop_reason={result.stop_reason} exit_code={outcome.exit_code}",
        ]
        if result.error:
            summary_lines.append(f"错误: {result.error}")
        error_in_run = result.stop_reason not in ("stop", "tool_use")
        appended.append(self._upsert(PresentationRecord(
            key=f"{result.run_id}:summary", role="error" if error_in_run else "system",
            title="运行结束", status="failed" if error_in_run else "done",
            body=_bounded_body("\n".join(summary_lines)),
        )))
        if outcome.decision is not None:
            appended.append(self._upsert(self._delivery_record(outcome)))
        status = self._replace_status(
            phase="idle",
            verification_run_id=self._verification_run_id,
            check_status=self._check_status_text(outcome),
            change_scope=self._change_scope_text(outcome),
        )
        return PresentationUpdate(appended=tuple(appended), status=status)

    def _delivery_record(self, outcome) -> PresentationRecord:
        decision = outcome.decision
        view = outcome.view
        evidence = outcome.evidence
        lines: list[str] = ["程序判定（非模型自然语言）: " + ("可交付" if decision.can_deliver else "不可交付")]
        if view is not None:
            lines.append(f"verification_run_id={view.verification_run_id}")
            if view.process_uncertain or self._process_uncertain:
                lines.append(PROCESS_UNCERTAIN_LABEL)
            for state in view.checks:
                if state.receipt is None:
                    source = NOT_RUN_CHECK_LABEL
                elif state.receipt.receipt_id in self._check_receipts_this_request:
                    source = CURRENT_RUN_CHECK_LABEL
                elif state.status == "passed" and state.freshness == "current":
                    # "仍有效" 只属于快照没变且仍为最新尝试的旧 PASS
                    source = HISTORICAL_PASS_LABEL
                else:
                    # 失败/stale/未知 freshness 的历史回执不得宣称为仍有效
                    source = HISTORICAL_NEUTRAL_LABEL
                line = f"check {state.check_id}: status={state.status} freshness={state.freshness} 来源={source}"
                if state.receipt is not None:
                    line += f" receipt_id={state.receipt.receipt_id} exit_code={state.receipt.exit_code}"
                lines.append(line)
        if evidence is not None:
            if evidence.artifact_path is not None:
                sha = evidence.artifact_sha256 or "unverified"
                lines.append(f"累计变更工件: {evidence.artifact_path} sha256={sha}")
            lines.append(f"相对启动基线累计改动: {len(evidence.changes)} 项")
        if not decision.can_deliver:
            kept, total_lines, size, truncated = clip_lines(
                "\n".join(decision.blocking_reasons), TOOL_CARD_MAX_LINES, TOOL_CARD_MAX_BYTES
            )
            lines.append("不能交付的原因:")
            lines.extend(f"- {reason}" for reason in kept)
            lines.extend(_truncation_note(kept, total_lines, size, truncated))
        return PresentationRecord(
            key=f"{outcome.result.run_id}:delivery",
            role="system" if decision.can_deliver else "error",
            title="检查结论",
            status="done" if decision.can_deliver else "failed",
            body=_bounded_body("\n".join(lines)),
        )

    def _check_status_text(self, outcome) -> str | None:
        if outcome.view is None:
            return NORMAL_MODE_CHECK_LABEL
        parts: list[str] = []
        if self._process_uncertain:
            parts.append(PROCESS_UNCERTAIN_LABEL)
        if outcome.view.checks:
            parts.extend(
                f"{state.check_id}={state.status}/{state.freshness}" for state in outcome.view.checks
            )
        else:
            parts.append(NOT_RUN_CHECK_LABEL)
        return " ".join(parts)

    @staticmethod
    def _change_scope_text(outcome) -> str | None:
        evidence = outcome.evidence
        if evidence is None:
            return None
        if not evidence.complete:
            detail = f"（不完整: {evidence.failure_kind}）" if evidence.failure_kind else "（不完整）"
            return f"累计改动口径: 相对启动基线{detail}"
        return f"累计改动口径: 相对启动基线 {len(evidence.changes)} 项变更"

    def _accept_history_cleared(self) -> PresentationUpdate:
        lines = ["模型历史已清空；验证基线/账本与既有工件保留。"]
        if self._process_uncertain:
            lines.append(PROCESS_UNCERTAIN_LABEL + "：清空历史不解锁。")
        record = self._upsert(PresentationRecord(
            key="app:history-cleared", role="system", title="历史清空", status="done",
            body=tuple(lines),
        ))
        return PresentationUpdate(appended=(record,))

    # --- 状态与记录簿记 -------------------------------------------------------------

    def _replace_status(self, **changes) -> StatusSnapshot:
        current = self._status
        fields = {
            "model": current.model, "workspace": current.workspace, "phase": current.phase,
            "turn_index": current.turn_index, "input_tokens": current.input_tokens,
            "output_tokens": current.output_tokens,
            "verification_run_id": current.verification_run_id,
            "check_status": current.check_status, "change_scope": current.change_scope,
        }
        fields.update(changes)
        self._status = StatusSnapshot(**fields)
        return self._status

    def _upsert(self, record: PresentationRecord) -> PresentationRecord:
        if record.key not in self._records:
            self._order.append(record.key)
        self._records[record.key] = record
        return record
