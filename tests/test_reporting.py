"""reporting 运行摘要聚焦测试: unknown/partial/零值/求和/覆盖率/不做验收结论."""

from __future__ import annotations

from cicada.core.agent import RunResult
from cicada.core.events import ModelCallCompleted
from cicada.core.messages import ToolResult, ToolResultMessage
from cicada.core.ports import ModelMetrics
from cicada.reporting import build_run_summary


def _call(turn: int, metrics: ModelMetrics | None, elapsed: float = 1.0, stop: str = "stop"):
    return ModelCallCompleted(
        run_id="run-1", turn_index=turn, stop_reason=stop, metrics=metrics, elapsed_s=elapsed
    )


def _result(calls=(), messages=()) -> RunResult:
    return RunResult(
        run_id="run-1", stop_reason="stop", error=None, messages=messages, model_calls=tuple(calls)
    )


def test_all_unknown_when_no_calls():
    summary = build_run_summary(_result())
    assert "model_calls=0" in summary
    assert "model_time_s=unknown" in summary
    assert "provider_time_s=unknown provider_reported_calls=0/0" in summary
    assert "input_tokens_known=unknown input_reported_calls=0/0" in summary
    assert "output_tokens_known=unknown output_reported_calls=0/0" in summary


def test_full_coverage_sums_all_calls():
    metrics_a = ModelMetrics(input_tokens=100, output_tokens=10, provider_duration_s=1.5)
    metrics_b = ModelMetrics(input_tokens=250, output_tokens=20, provider_duration_s=2.25)
    summary = build_run_summary(
        _result([_call(0, metrics_a, 2.0), _call(1, metrics_b, 3.0, "tool_use")])
    )
    assert "model_calls=2" in summary
    assert "model_time_s=5.00" in summary  # 本地 elapsed 求和
    assert "provider_time_s=3.75 provider_reported_calls=2/2" in summary
    assert "input_tokens_known=350 input_reported_calls=2/2" in summary
    assert "output_tokens_known=30 output_reported_calls=2/2" in summary


def test_partial_coverage_shows_known_sum_and_ratio():
    full = ModelMetrics(input_tokens=100, output_tokens=10, provider_duration_s=1.0)
    summary = build_run_summary(
        _result([_call(0, full), _call(1, None), _call(2, ModelMetrics(output_tokens=7))])
    )
    assert "input_tokens_known=100 input_reported_calls=1/3" in summary
    assert "output_tokens_known=17 output_reported_calls=2/3" in summary
    assert "provider_time_s=1.00 provider_reported_calls=1/3" in summary


def test_zero_is_a_known_value():
    summary = build_run_summary(
        _result([_call(0, ModelMetrics(input_tokens=0, output_tokens=0, provider_duration_s=0.0))])
    )
    assert "input_tokens_known=0 input_reported_calls=1/1" in summary
    assert "output_tokens_known=0 output_reported_calls=1/1" in summary


def test_error_and_aborted_calls_are_counted():
    summary = build_run_summary(
        _result([_call(0, None, 0.5, "error"), _call(1, None, 0.5, "aborted")])
    )
    assert "model_calls=2" in summary
    assert "model_time_s=1.00" in summary
    assert "input_tokens_known=unknown input_reported_calls=0/2" in summary


def test_tool_counts_come_from_messages():
    messages = (
        ToolResultMessage(result=ToolResult(call_id="c1", name="read", content="ok")),
        ToolResultMessage(
            result=ToolResult(call_id="c2", name="edit", content="bad", is_error=True)
        ),
    )
    summary = build_run_summary(_result(messages=messages))
    assert "tools=2 tool_errors=1" in summary


def test_summary_makes_no_acceptance_claim():
    summary = build_run_summary(_result([_call(0, ModelMetrics(1, 1, 1.0))]))
    for forbidden in ("PASS", "passed", "验收", "成功完成"):
        assert forbidden not in summary


# --- P4 §6: 程序交付判定 (不读模型文字, 不用 tools_ok/tool_errors) ------------------------

from pathlib import Path

from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    CheckReceipt,
    CheckState,
    ChangeEvidence,
    FileChange,
    VerificationPlan,
    VerificationView,
)
from cicada.reporting import build_delivery_section, decide_delivery


def _plan(checks=("c1",)):
    definitions = tuple(
        CheckDefinition(check_id=cid, command=f"cmd {cid}", timeout_s=60.0) for cid in checks
    )
    return VerificationPlan(
        verification_run_id="vrun-x", root=Path("C:/w"), checks=definitions
    )


def _receipt(check_id="c1", status="passed", after="snap-1", run_id="vrun-x"):
    return CheckReceipt(
        verification_run_id=run_id, receipt_id=f"chk-{check_id}", check_id=check_id,
        command="cmd", cwd=Path("C:/w"), timeout_s=60.0, elapsed_s=0.5,
        execution_status="exited", exit_code=0, timed_out=False, cancelled=False,
        output_complete=True, snapshot_before="snap-1", snapshot_after=after,
        verification_status=status, failure_kind=None if status == "passed" else "nonzero_exit",
        output_truncated=False, artifact_truncated=False,
        output_artifact_path=None, output_artifact_sha256=None,
    )


def _view(checks, snapshot_ref="snap-1", uncertain=False, blocking=()):
    return VerificationView(
        verification_run_id="vrun-x", snapshot_ref=snapshot_ref, scope_id="scope-1",
        checks=tuple(checks), receipts=(), process_uncertain=uncertain,
        blocking_reasons=tuple(blocking),
    )


def _evidence(complete=True, snapshot_ref="snap-1", changes=(), failure=None):
    return ChangeEvidence(
        baseline_snapshot_ref="snap-0", snapshot_ref=snapshot_ref, scope_id="scope-1",
        changes=tuple(changes), artifact_path=Path("C:/w/.cicada/changes.txt") if complete else None,
        artifact_sha256="a" * 64 if complete else None, complete=complete,
        failure_kind=failure, error=None if complete else "boom",
    )


def test_decide_delivery_all_conditions_met():
    result = _result()
    view = _view([CheckState("c1", _receipt(), "current")])
    decision = decide_delivery(result, view, _evidence())
    assert decision.model_stopped and decision.can_deliver
    assert decision.blocking_reasons == ()


def test_decide_delivery_rejects_non_stop_model():
    result = RunResult("run-1", "error", "boom", ())
    decision = decide_delivery(result, _view([CheckState("c1", _receipt(), "current")]), _evidence())
    assert not decision.model_stopped and not decision.can_deliver
    assert any("stop_reason=error" in r for r in decision.blocking_reasons)


def test_decide_delivery_missing_view_or_evidence():
    decision = decide_delivery(_result(), None, None)
    assert not decision.can_deliver
    assert any("verification view" in r for r in decision.blocking_reasons)
    assert any("change evidence" in r for r in decision.blocking_reasons)


def test_decide_delivery_check_not_run_failed_or_stale():
    not_run = _view([CheckState("c1", None, "unknown")])
    decision = decide_delivery(_result(), not_run, _evidence())
    assert not decision.can_deliver
    assert any("has not been run" in r for r in decision.blocking_reasons)

    failed = _view([CheckState("c1", _receipt(status="failed"), "current")])
    decision = decide_delivery(_result(), failed, _evidence())
    assert any("check c1 is failed" in r for r in decision.blocking_reasons)

    stale = _view([CheckState("c1", _receipt(), "stale")])
    decision = decide_delivery(_result(), stale, _evidence())
    assert any("receipt is stale" in r for r in decision.blocking_reasons)


def test_decide_delivery_process_uncertain_and_snapshot_unavailable():
    uncertain = _view([CheckState("c1", _receipt(), "current")], uncertain=True)
    decision = decide_delivery(_result(), uncertain, _evidence())
    assert not decision.can_deliver
    assert any("process_uncertain" in r for r in decision.blocking_reasons)

    no_snapshot = _view([CheckState("c1", _receipt(), "unknown")], snapshot_ref=None)
    decision = decide_delivery(_result(), no_snapshot, _evidence())
    assert not decision.can_deliver
    assert any("final code snapshot is unavailable" in r for r in decision.blocking_reasons)


def test_decide_delivery_incomplete_or_shifted_evidence():
    incomplete = _evidence(complete=False, failure="baseline_corrupt")
    decision = decide_delivery(_result(), _view([CheckState("c1", _receipt(), "current")]), incomplete)
    assert not decision.can_deliver
    assert any("incomplete" in r and "baseline_corrupt" in r for r in decision.blocking_reasons)

    # 终局再次变化: 检查视图快照与证据快照不同
    shifted = _evidence(snapshot_ref="snap-2")
    decision = decide_delivery(_result(), _view([CheckState("c1", _receipt(), "current")]), shifted)
    assert not decision.can_deliver
    assert any("changed after the final check view" in r for r in decision.blocking_reasons)


def test_decide_delivery_consumes_artifact_problems():
    view = _view([CheckState("c1", _receipt(), "current")])
    decision = decide_delivery(_result(), view, _evidence(), ("check c1 output artifact missing",))
    assert not decision.can_deliver
    assert "check c1 output artifact missing" in decision.blocking_reasons


def test_decide_delivery_merges_view_blocking_reasons_deduped():
    view = _view(
        [CheckState("c1", _receipt(), "current")],
        blocking=("baseline snapshot is unavailable: x",),
    )
    decision = decide_delivery(_result(), view, _evidence())
    assert not decision.can_deliver
    assert "baseline snapshot is unavailable: x" in decision.blocking_reasons
    assert len(decision.blocking_reasons) == len(set(decision.blocking_reasons))


def test_delivery_section_lists_facts_and_changes():
    changes = (
        FileChange("app.py", "modified", "b" * 64, "c" * 64, False, False, True),
        FileChange("new.txt", "added", None, "d" * 64, False, False, False),
        FileChange("old.bin", "deleted", "e" * 64, None, True, False, False),
    )
    view = _view([CheckState("c1", _receipt(), "current")])
    decision = decide_delivery(_result(), view, _evidence(changes=changes))
    section = build_delivery_section(decision, view, _evidence(changes=changes), _plan())
    assert "[delivery can_deliver=true model_stopped=true]" in section
    assert "[verification_run_id=vrun-x]" in section
    assert "[scope policy_id=" in section and "exclusions=" in section
    assert "- c1 status=passed freshness=current receipt_id=chk-c1" in section
    assert "[change artifact: " in section and "sha256=" in section
    assert "[changed files: 3]" in section
    assert "- modified app.py" in section and "flags=newline_changed" in section
    assert "- added new.txt" in section
    assert "- deleted old.bin" in section
    assert "[blocking_reasons]\n- none" in section


def test_delivery_section_reports_blocking_and_missing_facts():
    view = _view([CheckState("c1", None, "unknown")], snapshot_ref=None)
    decision = decide_delivery(_result(), view, _evidence(complete=False, failure="no_baseline"))
    section = build_delivery_section(decision, view, _evidence(complete=False, failure="no_baseline"))
    assert "[delivery can_deliver=false" in section
    assert "[snapshot_ref=None]" in section
    assert "[change artifact: unavailable]" in section
    assert any("- check c1 has not been run" in line for line in section.splitlines())
    assert "- the change evidence is incomplete" in section
