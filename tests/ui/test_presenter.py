"""04 聚焦: ProjectionPresenter 对 ApplicationUpdate 的投影 (T06/T08/T09/T10/T15).

用手工构造的 ApplicationUpdate 序列做确定性单元检查; 真实应用/工具/Git/PowerShell
的投影检查在 tests/e2e/test_tui_verification.py。
"""

from __future__ import annotations

from pathlib import Path

from cicada.application import (
    AgentEventReceived,
    HistoryCleared,
    RequestCompleted,
    RequestOutcome,
    RequestPhaseChanged,
)
from cicada.core.agent import RunResult
from cicada.core.events import (
    AssistantCompleted,
    AssistantTextDelta,
    ModelCallCompleted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.messages import AssistantMessage, ToolCall, ToolResult
from cicada.core.ports import ModelMetrics
from cicada.plugins.coding.verification_contracts import (
    ChangeEvidence,
    CheckReceipt,
    CheckState,
    FileChange,
    VerificationView,
)
from cicada.reporting import DeliveryDecision
from cicada.ui.contracts import PresentationRecord
from cicada.ui.presenter import (
    HISTORICAL_NEUTRAL_LABEL,
    HISTORICAL_PASS_LABEL,
    NORMAL_MODE_CHECK_LABEL,
    PREVIEW_MAX_BYTES,
    PREVIEW_MAX_LINES,
    PROCESS_UNCERTAIN_LABEL,
    TOOL_CARD_MAX_BYTES,
    TOOL_CARD_MAX_LINES,
    ProjectionPresenter,
    clip_lines,
)

RUN = "run-1"


def _event(event) -> AgentEventReceived:
    return AgentEventReceived(event=event)


def _outcome(
    run_id: str = RUN,
    stop: str = "stop",
    *,
    view: VerificationView | None = None,
    evidence: ChangeEvidence | None = None,
    decision: DeliveryDecision | None = None,
    exit_code: int = 0,
    error: str | None = None,
) -> RequestOutcome:
    return RequestOutcome(
        result=RunResult(run_id, stop, error, (), ()),
        view=view,
        evidence=evidence,
        decision=decision,
        exit_code=exit_code,
    )


def _receipt(receipt_id: str = "r1", check_id: str = "check-1", status: str = "passed") -> CheckReceipt:
    return CheckReceipt(
        verification_run_id="vrun", receipt_id=receipt_id, check_id=check_id,
        command="exit 0", cwd=Path("."), timeout_s=1.0, elapsed_s=0.1,
        execution_status="exited", exit_code=0, timed_out=False, cancelled=False,
        output_complete=True, snapshot_before="snap-a", snapshot_after="snap-b",
        verification_status=status, failure_kind=None, output_truncated=False,
        artifact_truncated=False, output_artifact_path=None, output_artifact_sha256=None,
    )


def _view(
    checks: tuple[CheckState, ...] = (),
    receipts: tuple[CheckReceipt, ...] = (),
    process_uncertain: bool = False,
) -> VerificationView:
    return VerificationView(
        verification_run_id="vrun", snapshot_ref="snap-b", scope_id="scope",
        checks=checks, receipts=receipts, process_uncertain=process_uncertain,
        blocking_reasons=(),
    )


def _assistant_call_event(call_id: str = "c1", turn: int = 0, name: str = "read") -> AssistantCompleted:
    return AssistantCompleted(RUN, AssistantMessage(
        text="", tool_calls=(ToolCall(call_id, name, '{"path": "sample.py"}'),),
        stop_reason="tool_use",
    ))


def _body_of(presenter: ProjectionPresenter, key: str) -> str:
    record = next(r for r in presenter.records if r.key == key)
    return "\n".join(record.body)


# --- clip_lines 有界性 ------------------------------------------------------------------------


def test_clip_lines_respects_line_and_byte_bounds():
    text = "\n".join(f"line-{i}" for i in range(40))
    kept, total_lines, total_bytes, truncated = clip_lines(text, 12, 4096)
    assert len(kept) == 12 and total_lines == 40 and truncated
    assert "\n".join(kept).encode("utf-8").count(b"line-") == 12


def test_clip_lines_respects_byte_bound_on_single_huge_line():
    text = "汉" * 5000  # 15000 bytes
    kept, _, _, truncated = clip_lines(text, 12, 4096)
    assert truncated
    assert len("\n".join(kept).encode("utf-8")) <= 4096
    assert kept[0], "截断不产生空串"


def test_clip_lines_no_truncation_flag_when_content_fits():
    kept, _, _, truncated = clip_lines("a\nb\nc", 12, 4096)
    assert kept == ("a", "b", "c") and not truncated


# --- T06: 工具卡 -----------------------------------------------------------------------------


def test_tool_card_keyed_by_run_turn_call_with_arguments_from_completion():
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(_assistant_call_event(call_id="c1", turn=0)))
    presenter.accept(_event(ToolStarted(RUN, "c1", "read")))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult("c1", "read", "file body"))))
    card = next(r for r in presenter.records if r.key == f"{RUN}:0:c1")
    assert card.role == "tool" and card.status == "done"
    assert "run=run-1 turn=0" in "\n".join(card.body)
    assert '{"path": "sample.py"}' in "\n".join(card.body)
    assert "file body" in "\n".join(card.body)


def test_unstarted_tool_completion_builds_failed_unexecuted_card():
    presenter = ProjectionPresenter()
    presenter.accept(_event(_assistant_call_event(call_id="c1")))
    # 无 ToolStarted: 模型流失败的固定文本闭合 (内核权威文本)
    update = presenter.accept(_event(ToolCompleted(
        RUN, ToolResult("c1", "read", "model stream failed; tool call not executed", is_error=True)
    )))
    card = update.appended[0]
    assert card.status == "failed"
    assert "未执行（无 ToolStarted）" in "\n".join(card.body)
    assert "model stream failed; tool call not executed" in "\n".join(card.body)


def test_cross_turn_call_id_reuse_creates_separate_cards():
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(_assistant_call_event(call_id="same", turn=0)))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult("same", "read", "first"))))
    presenter.accept(_event(TurnStarted(RUN, 1)))
    presenter.accept(_event(_assistant_call_event(call_id="same", turn=1)))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult("same", "read", "second"))))
    first = next(r for r in presenter.records if r.key == f"{RUN}:0:same")
    second = next(r for r in presenter.records if r.key == f"{RUN}:1:same")
    assert "first" in "\n".join(first.body)
    assert "second" in "\n".join(second.body)
    assert len([r for r in presenter.records if r.role == "tool" and "same" in r.title]) == 2


def test_tool_started_after_completion_does_not_resurrect():
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(_assistant_call_event(call_id="c1")))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult("c1", "read", "done"))))
    presenter.accept(_event(ToolStarted(RUN, "c1", "read")))
    card = next(r for r in presenter.records if r.key == f"{RUN}:0:c1")
    assert card.status == "done"


def test_tool_card_body_bounded_and_marks_truncation():
    presenter = ProjectionPresenter()
    big = "\n".join(f"输出第{i}行" for i in range(200))
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(_assistant_call_event(call_id="c1")))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult("c1", "read", big))))
    card = next(r for r in presenter.records if r.key == f"{RUN}:0:c1")
    body = "\n".join(card.body)
    assert "已截断" in body
    assert len(card.body) <= TOOL_CARD_MAX_LINES + 2  # 正文 + 截断标注 + run/turn 头
    assert len(body.encode("utf-8")) < TOOL_CARD_MAX_BYTES + 200


def test_check_tool_completed_records_receipt_id_for_source_attribution():
    presenter = ProjectionPresenter()
    presenter.accept(_event(_assistant_call_event(call_id="c1", name="check")))
    presenter.accept(_event(ToolCompleted(RUN, ToolResult(
        "c1", "check", "receipt body", is_error=False,
        details={"receipt_id": "receipt-7"},
    ))))
    view = _view(checks=(CheckState("check-1", _receipt("receipt-7"), "current"),))
    presenter.accept(RequestCompleted(_outcome(
        view=view, evidence=None,
        decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )))
    assert "receipt-7" in _body_of(presenter, f"{RUN}:delivery")
    assert "本次检查" in _body_of(presenter, f"{RUN}:delivery")


# --- T15: 预览合并 / 有界 / 最终事实 -----------------------------------------------------------


def test_preview_merges_deltas_and_is_bounded():
    presenter = ProjectionPresenter()
    deltas = ["第一行\n", "第二行\n"] + [f"填充{i}\n" for i in range(20)]
    for text in deltas:
        update = presenter.accept(_event(AssistantTextDelta(RUN, 0, text)))
    preview = update.preview
    assert preview is not None
    lines = preview.text.splitlines()
    assert len(lines) <= PREVIEW_MAX_LINES
    assert len(preview.text.encode("utf-8")) <= PREVIEW_MAX_BYTES
    assert preview.truncated, "22 行内容应标记截断"
    assert preview.run_id == RUN and preview.turn_index == 0


def test_final_completion_keeps_authoritative_text_after_merge():
    presenter = ProjectionPresenter()
    presenter.accept(_event(AssistantTextDelta(RUN, 0, "增量一")))
    presenter.accept(_event(AssistantTextDelta(RUN, 0, "增量二")))
    update = presenter.accept(_event(AssistantCompleted(RUN, AssistantMessage(
        text="增量一增量二\n权威全文", stop_reason="stop",
    ))))
    record = next(r for r in update.appended if r.key == f"{RUN}:0:assistant")
    assert "权威全文" in "\n".join(record.body)
    assert presenter.current_preview is None, "最终完成替换预览"


def test_metrics_any_unknown_call_makes_that_cumulative_unknown():
    presenter = ProjectionPresenter()
    update = presenter.accept(_event(ModelCallCompleted(
        RUN, 0, "stop", ModelMetrics(input_tokens=100, output_tokens=10), 0.1,
    )))
    assert update.status.input_tokens == 100 and update.status.output_tokens == 10
    update = presenter.accept(_event(ModelCallCompleted(
        RUN, 1, "stop", ModelMetrics(input_tokens=250, output_tokens=None), 0.1,
    )))
    # None 表示不可得: 任一调用该计量不可得, 该累计就是不可得 (不显示部分和);
    # input 全部已知仍可累计 (跨调用求和, 不是单次峰值)
    assert update.status.input_tokens == 350 and update.status.output_tokens is None


def test_metrics_reset_between_requests():
    presenter = ProjectionPresenter()
    presenter.accept(RequestPhaseChanged(None, "executing"))
    presenter.accept(_event(ModelCallCompleted(RUN, 0, "stop", ModelMetrics(500, 50), 0.1)))
    presenter.accept(RequestPhaseChanged(RUN, "idle"))
    presenter.accept(RequestCompleted(_outcome()))
    presenter.accept(RequestPhaseChanged(None, "executing"))
    update = presenter.accept(_event(ModelCallCompleted("run-2", 0, "stop", ModelMetrics(7, 3), 0.1)))
    assert update.status.input_tokens == 7 and update.status.output_tokens == 3


# --- 状态栏 / 普通模式 / 检查模式 --------------------------------------------------------------


def test_normal_mode_reports_check_not_enabled():
    presenter = ProjectionPresenter(model="fake", workspace="w")
    presenter.accept(RequestPhaseChanged(None, "executing"))
    update = presenter.accept(RequestCompleted(_outcome()))
    assert update.status.check_status == NORMAL_MODE_CHECK_LABEL == "未启用检查"
    assert update.status.verification_run_id is None
    assert update.status.change_scope is None
    assert update.status.phase == "idle"


def test_phase_and_turn_flow_through_status():
    presenter = ProjectionPresenter()
    update = presenter.accept(RequestPhaseChanged(None, "executing"))
    assert update.status.phase == "executing"
    update = presenter.accept(_event(TurnStarted(RUN, 3)))
    assert update.status.turn_index == 3
    update = presenter.accept(RequestPhaseChanged(RUN, "finalizing"))
    assert update.status.phase == "finalizing"


def test_check_mode_status_shows_verification_run_and_checks():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    view = _view(checks=(CheckState("check-1", _receipt(), "current"),))
    update = presenter.accept(RequestCompleted(_outcome(
        view=view, evidence=None,
        decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )))
    assert update.status.verification_run_id == "vrun"
    assert "check-1=passed/current" in update.status.check_status


def test_finalizing_fact_and_change_scope_in_status():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    presenter.accept(RequestPhaseChanged(None, "executing"))
    presenter.accept(RequestPhaseChanged(RUN, "finalizing"))
    assert presenter.status.phase == "finalizing"
    evidence = ChangeEvidence(
        baseline_snapshot_ref="base", snapshot_ref="snap", scope_id="scope",
        changes=(FileChange("a.py", "modified", "x", "y", False, False, False),),
        artifact_path=Path("artifact.diff"), artifact_sha256="abc", complete=True,
    )
    update = presenter.accept(RequestCompleted(_outcome(view=_view(), evidence=evidence, decision=None)))
    assert "累计改动口径" in update.status.change_scope
    assert "1 项" in update.status.change_scope


# --- T09: process_uncertain 锁存 --------------------------------------------------------------


def test_process_uncertain_latches_and_survives_clear_and_pass():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    view = _view(checks=(CheckState("check-1", _receipt(status="failed"), "current"),), process_uncertain=True)
    presenter.accept(RequestCompleted(_outcome(view=view, evidence=None, decision=DeliveryDecision(
        model_stopped=True, can_deliver=False,
        blocking_reasons=("process_uncertain: a process ended without proven termination in this run",),
    ))))
    assert presenter.has_process_uncertain()
    update = presenter.accept(HistoryCleared())
    assert presenter.has_process_uncertain(), "清空历史不解除锁存"
    assert PROCESS_UNCERTAIN_LABEL in "\n".join(update.appended[0].body)
    # 后续视图 (模拟) 即使不再报告 process_uncertain, 展示锁存也不被绕过
    presenter.accept(RequestCompleted(_outcome(
        run_id="run-2", view=_view(checks=(CheckState("check-1", _receipt("r2"), "current"),)),
        evidence=None, decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )))
    assert presenter.has_process_uncertain()
    assert PROCESS_UNCERTAIN_LABEL in presenter.status.check_status


def test_runtime_rejection_presented_as_error_record():
    presenter = ProjectionPresenter()
    update = presenter.accept_error(
        "process_uncertain: new coding requests are refused; only status, clear or exit remain"
    )
    record = update.appended[0]
    assert record.role == "error" and record.status == "failed"
    assert "process_uncertain" in "\n".join(record.body)
    assert "应用拒绝" in record.title


# --- T10: 程序判定与工件身份 -------------------------------------------------------------------


def _evidence(artifact: str = "artifact-1.diff", sha: str = "aa") -> ChangeEvidence:
    return ChangeEvidence(
        baseline_snapshot_ref="base", snapshot_ref="snap", scope_id="scope",
        changes=(FileChange("a.py", "modified", "x", "y", False, False, False),),
        artifact_path=Path(artifact), artifact_sha256=sha, complete=True,
    )


def test_delivery_record_shows_program_decision_and_artifact_identity():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    outcome = _outcome(
        view=_view(checks=(CheckState("check-1", _receipt(), "current"),)),
        evidence=_evidence("a1.diff", "hash-1"),
        decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )
    update = presenter.accept(RequestCompleted(outcome))
    record = update.appended[-1]
    assert record.role == "system" and record.status == "done"
    body = "\n".join(record.body)
    assert "程序判定" in body and "可交付" in body
    assert "a1.diff" in body and "hash-1" in body
    assert "相对启动基线累计改动: 1 项" in body


def test_failed_decision_shown_as_error_with_reasons_not_model_text():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    outcome = _outcome(
        view=_view(checks=(CheckState("check-1", _receipt(status="failed"), "current"),)),
        evidence=_evidence(),
        decision=DeliveryDecision(
            model_stopped=True, can_deliver=False,
            blocking_reasons=("check check-1 is failed (exit 2)",),
        ),
        exit_code=3,
    )
    update = presenter.accept(RequestCompleted(outcome))
    record = update.appended[-1]
    assert record.role == "error" and record.status == "failed"
    body = "\n".join(record.body)
    assert "不可交付" in body and "check check-1 is failed" in body


def test_stale_check_not_shown_as_deliverable():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    outcome = _outcome(
        run_id="run-2",
        view=_view(checks=(CheckState("check-1", _receipt("r-old"), "stale"),)),
        evidence=_evidence(),
        decision=DeliveryDecision(
            model_stopped=True, can_deliver=False,
            blocking_reasons=(
                "check check-1 passed but its receipt is stale; "
                "it does not match the final code state",
            ),
        ),
        exit_code=3,
    )
    update = presenter.accept(RequestCompleted(outcome))
    record = update.appended[-1]
    assert record.status == "failed"
    assert "stale" in "\n".join(record.body)


def test_historical_pass_marked_by_source_without_this_run_check():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    # 第二请求没有 check 工具事件: view 中的 passed/current 回执来自此前请求
    outcome = _outcome(
        run_id="run-2",
        view=_view(checks=(CheckState("check-1", _receipt("r-old"), "current"),)),
        evidence=_evidence("a2.diff", "hash-2"),
        decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )
    presenter.accept(RequestCompleted(outcome))
    body = _body_of(presenter, "run-2:delivery")
    assert HISTORICAL_PASS_LABEL in body
    assert "a2.diff" in body, "每请求结果绑定自己的工件"


def test_historical_failed_or_stale_receipt_not_labeled_still_valid():
    """历史回执只有 passed+current 才配"仍有效"; 失败/stale 用中性来源标注."""
    presenter = ProjectionPresenter(verification_run_id="vrun")
    outcome = _outcome(
        run_id="run-2",
        view=_view(checks=(
            CheckState("check-1", _receipt("r-failed", status="failed"), "current"),
            CheckState("check-2", _receipt("r-stale"), "stale"),
        )),
        decision=DeliveryDecision(model_stopped=True, can_deliver=False, blocking_reasons=("x",)),
        exit_code=3,
    )
    presenter.accept(RequestCompleted(outcome))
    body = _body_of(presenter, "run-2:delivery")
    assert "来源=" + HISTORICAL_NEUTRAL_LABEL in body
    assert "仍有效" not in body
    assert "status=failed" in body and "freshness=stale" in body


def test_finalize_timeout_sentinel_view_does_not_overwrite_real_run_id():
    """收尾监督超时的哨兵 view (snapshot_ref=None) 不得把 'unavailable' 写进状态栏."""
    presenter = ProjectionPresenter(verification_run_id="vrun-real")
    sentinel = VerificationView(
        verification_run_id="unavailable", snapshot_ref=None, scope_id=None,
        checks=(), receipts=(), process_uncertain=False,
        blocking_reasons=("finalization timed out",),
    )
    update = presenter.accept(RequestCompleted(
        _outcome(run_id="run-2", view=sentinel, exit_code=3,
                 decision=DeliveryDecision(model_stopped=False, can_deliver=False,
                                           blocking_reasons=("finalization timed out",)))
    ))
    assert update.status.verification_run_id == "vrun-real"
    presenter.accept(RequestCompleted(
        _outcome(run_id="run-3", view=_view(), exit_code=0,
                 decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()))
    ))
    assert presenter.status.verification_run_id == "vrun"  # 真实 view 照常更新


def test_mid_subscribe_unknown_card_not_claimed_unexecuted():
    """订阅前开始的工具: 完成仍投影, 但只标"状态未知", 不断言"未执行"."""
    presenter = ProjectionPresenter()
    presenter.accept(_event(ToolStarted(RUN, "call-late", "echo")))  # 无注册卡: 静默 (无事实可显示)
    presenter.accept(_event(ToolCompleted(
        RUN, ToolResult(call_id="call-late", name="echo", content="done"),
    )))
    card = presenter.records[-1]
    assert card.key == f"{RUN}:unknown:call-late" and card.status == "done"
    body = "\n".join(card.body)
    assert "执行状态未知" in body
    assert "未执行" not in body


def test_final_assistant_record_not_bounded_by_preview_limits():
    """最终回答是完整规范文本, 不套预览/工具卡的有界上限 (完整文本以完成事件为准)."""
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    long_text = "\n".join(f"line-{i:03d}" for i in range(80))
    presenter.accept(_event(AssistantCompleted(
        RUN, AssistantMessage(text=long_text, stop_reason="stop"),
    )))
    record = next(r for r in presenter.records if r.key == f"{RUN}:0:assistant")
    body = "\n".join(record.body)
    assert "line-000" in body and "line-079" in body
    assert not any("已截断" in line for line in record.body)


def test_delta_after_final_answer_does_not_rebuild_preview():
    """已终结轮的迟到增量不重建预览 (批次重排后 flush 的陈旧文本)."""
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(AssistantCompleted(
        RUN, AssistantMessage(text="最终回答", stop_reason="stop"),
    )))
    update = presenter.accept(_event(AssistantTextDelta(RUN, 0, "迟到增量")))
    assert update.preview is None and presenter.current_preview is None
    # 未终结轮的增量照常更新预览
    update = presenter.accept(_event(AssistantTextDelta(RUN, 1, "新一轮")))
    assert update.preview is not None and update.preview.turn_index == 1


# --- 关闭 / 清空 / 去重 -----------------------------------------------------------------------


def test_history_cleared_record_keeps_verification_fact():
    presenter = ProjectionPresenter(verification_run_id="vrun")
    update = presenter.accept(HistoryCleared())
    record = update.appended[0]
    assert record.role == "system"
    assert "基线/账本" in "\n".join(record.body)


def test_application_closed_clean_is_system_record():
    from cicada.application import ApplicationClosed

    presenter = ProjectionPresenter()
    update = presenter.accept(ApplicationClosed())
    record = update.appended[0]
    assert record.role == "system" and record.status == "done"


def test_closed_with_error_is_error_record():
    from cicada.application import ApplicationClosed

    presenter = ProjectionPresenter()
    update = presenter.accept(ApplicationClosed(error="runtime stop failed: RuntimeError()"))
    record = update.appended[0]
    assert record.role == "error" and "runtime stop failed" in "\n".join(record.body)


def test_record_upsert_does_not_duplicate_keys():
    presenter = ProjectionPresenter()
    presenter.accept(_event(TurnStarted(RUN, 0)))
    presenter.accept(_event(_assistant_call_event(call_id="c1")))
    presenter.accept(_event(ToolStarted(RUN, "c1", "read")))
    presenter.accept(_event(ToolStarted(RUN, "c1", "read")))
    cards = [r for r in presenter.records if r.key == f"{RUN}:0:c1"]
    assert len(cards) == 1
    assert cards[0].status == "running"


def test_presentation_record_is_frozen_contract_shape():
    record = PresentationRecord(key="k", role="user", title="t", status="done", body=("b",))
    assert record.role == "user" and record.status == "done"
