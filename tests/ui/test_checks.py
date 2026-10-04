"""票据 05 检查投影单元检查: 来源四分类、回执收集、/status 事实与有界限额.

自旧 test_presenter 的检查语义迁移到新前端投影层; 真实 Git/PowerShell 的
端到端语义在 tests/e2e/test_tui_verification.py 覆盖。
"""

from __future__ import annotations

from pathlib import Path

from cicada.application import RequestOutcome
from cicada.core.agent import RunResult
from cicada.core.events import ToolCompleted
from cicada.core.messages import ToolResult
from cicada.plugins.coding.verification_contracts import (
    CheckReceipt,
    CheckState,
    ChangeEvidence,
    VerificationView,
)
from cicada.reporting import DeliveryDecision
from cicada.ui.checks import (
    CURRENT_RUN_CHECK_LABEL,
    HISTORICAL_NEUTRAL_LABEL,
    HISTORICAL_PASS_LABEL,
    NORMAL_MODE_CHECK_LABEL,
    NOT_RUN_CHECK_LABEL,
    PROCESS_UNCERTAIN_LABEL,
    STATUS_MAX_LINES,
    CheckProjection,
    artifact_line,
    bound_status_lines,
    change_scope_text,
    check_status_text,
    clip_lines,
    delivery_lines,
)


class _StubSession:
    """CheckProjection 只读依赖的最小形态 (verification_run_id/锁存)。"""

    def __init__(self, *, verification_run_id: str | None = None, process_uncertain: bool = False):
        self._verification_run_id = verification_run_id
        self._process_uncertain = process_uncertain

    @property
    def verification_run_id(self) -> str | None:
        return self._verification_run_id

    def has_process_uncertain(self) -> bool:
        return self._process_uncertain


def _receipt(receipt_id: str, check_id: str = "check-1", status: str = "passed") -> CheckReceipt:
    return CheckReceipt(
        verification_run_id="vr-1",
        receipt_id=receipt_id,
        check_id=check_id,
        command="exit 0",
        cwd=Path("."),
        timeout_s=1.0,
        elapsed_s=0.1,
        execution_status="exited",
        exit_code=0 if status == "passed" else 1,
        timed_out=False,
        cancelled=False,
        output_complete=True,
        snapshot_before=None,
        snapshot_after=None,
        verification_status=status,  # type: ignore[arg-type]
        failure_kind=None,
        output_truncated=False,
        artifact_truncated=False,
        output_artifact_path=None,
        output_artifact_sha256=None,
    )


def _view(*checks: CheckState, run_id: str = "vr-1", process_uncertain: bool = False) -> VerificationView:
    return VerificationView(
        verification_run_id=run_id,
        snapshot_ref="snap-1",
        scope_id="scope-1",
        checks=checks,
        receipts=tuple(c.receipt for c in checks if c.receipt is not None),
        process_uncertain=process_uncertain,
        blocking_reasons=(),
    )


def _outcome(
    *,
    view: VerificationView | None = None,
    evidence: ChangeEvidence | None = None,
    decision: DeliveryDecision | None = None,
    run_id: str = "run-1",
) -> RequestOutcome:
    return RequestOutcome(
        result=RunResult(run_id=run_id, stop_reason="stop", error=None, messages=(), model_calls=()),
        view=view,
        evidence=evidence,
        decision=decision,
        exit_code=0,
    )


# --- clip_lines / bound_status_lines (限额与截断标记) ------------------------------------------


def test_clip_lines_respects_line_and_byte_bounds():
    text = "\n".join(f"line-{i}" for i in range(40))
    kept, total_lines, total_bytes, truncated = clip_lines(text, 10, 8192)
    assert len(kept) == 10 and total_lines == 40 and truncated
    from cicada.ui.checks import truncation_note

    note = truncation_note(kept, total_lines, total_bytes, truncated)
    assert note and note[0].startswith("…[已截断: 显示 10/40 行, ")


def test_clip_lines_no_truncation_flag_when_content_fits():
    kept, total_lines, total_bytes, truncated = clip_lines("a\nb", 10, 8192)
    assert kept == ("a", "b") and not truncated and total_lines == 2


def test_clip_lines_single_huge_line_clips_at_byte_bound():
    text = "汉" * 5000  # 15000 bytes > 8192
    kept, total_lines, total_bytes, truncated = clip_lines(text, 32, 8192)
    assert truncated and len(kept) == 1
    assert len(kept[0].encode("utf-8")) <= 8192


def test_bound_status_lines_passthrough_within_limits():
    lines = [f"状态: idle", "模型: fake", "计量: 不可得"]
    assert bound_status_lines(lines) == lines


def test_bound_status_lines_marks_line_overflow():
    lines = [f"row-{i}" for i in range(STATUS_MAX_LINES + 5)]
    bounded = bound_status_lines(lines)
    assert len(bounded) == STATUS_MAX_LINES + 1
    assert bounded[-1].startswith("…[已截断: 显示 20/")
    assert "UTF-8 bytes]" in bounded[-1]


def test_bound_status_lines_marks_field_overflow():
    long_field = "字段: " + "汉" * 400  # > 512 bytes
    bounded = bound_status_lines([long_field])
    assert len(bounded) == 1
    assert "…[已截断: 512/" in bounded[0] and "UTF-8 bytes]" in bounded[0]
    assert bounded[0].count("汉") < 400, "字段本体已被截断"


# --- artifact_line ---------------------------------------------------------------------------


def test_artifact_line_from_check_details():
    details = {"output_artifact_path": "artifacts/x.md", "output_artifact_sha256": "abc123"}
    assert artifact_line(details) == "输出工件: artifacts/x.md sha256=abc123"
    assert artifact_line({"output_artifact_path": "p"}) == "输出工件: p sha256=unverified"
    assert artifact_line(None) is None
    assert artifact_line({"other": 1}) is None


# --- 来源四分类 (delivery_lines) -----------------------------------------------------------------


def test_delivery_lines_source_this_run_check():
    session = _StubSession(verification_run_id="vr-1")
    projection = CheckProjection(session)
    projection.observe_event(ToolCompleted(
        "run-1",
        ToolResult(call_id="c1", name="check", content="ok", is_error=False,
                   details={"receipt_id": "r-1"}),
    ))
    outcome = _outcome(
        view=_view(CheckState("check-1", _receipt("r-1"), "current")),
        decision=DeliveryDecision(model_stopped=True, can_deliver=True, blocking_reasons=()),
    )
    projection.on_request_completed(outcome)
    lines = delivery_lines(outcome, session, projection._receipts_this_request)
    body = "\n".join(lines)
    assert "程序判定（非模型自然语言）: 可交付" in body
    assert f"来源={CURRENT_RUN_CHECK_LABEL}" in body
    assert "receipt_id=r-1" in body


def test_delivery_lines_historical_pass_and_stale_sources():
    session = _StubSession(verification_run_id="vr-1")
    # 仍有效: 历史回执 + passed/current
    current = _outcome(
        view=_view(CheckState("check-1", _receipt("r-old"), "current")),
        decision=DeliveryDecision(True, True, ()),
    )
    lines = delivery_lines(current, session, set())
    assert HISTORICAL_PASS_LABEL in "\n".join(lines)
    assert CURRENT_RUN_CHECK_LABEL not in "\n".join(lines)
    # stale 历史回执: 只标中性来源, 不得宣称为仍有效
    stale = _outcome(
        view=_view(CheckState("check-1", _receipt("r-old"), "stale")),
        decision=DeliveryDecision(True, False, ("check check-1 is stale",)),
    )
    stale_body = "\n".join(delivery_lines(stale, session, set()))
    assert HISTORICAL_NEUTRAL_LABEL in stale_body
    assert HISTORICAL_PASS_LABEL not in stale_body
    assert "不可交付" in stale_body and "不能交付的原因:" in stale_body


def test_delivery_lines_not_run_and_uncertain_labels():
    session = _StubSession(verification_run_id="vr-1", process_uncertain=True)
    outcome = _outcome(
        view=_view(CheckState("check-1", None, "unknown"), process_uncertain=True),
        decision=DeliveryDecision(False, False, ("check check-1 has not run",)),
    )
    body = "\n".join(delivery_lines(outcome, session, set()))
    assert NOT_RUN_CHECK_LABEL in body
    assert PROCESS_UNCERTAIN_LABEL in body


def test_delivery_lines_change_scope_and_artifact():
    session = _StubSession(verification_run_id="vr-1")
    evidence = ChangeEvidence(
        baseline_snapshot_ref="snap-0",
        snapshot_ref="snap-1",
        scope_id="scope-1",
        changes=(),
        artifact_path=Path("artifacts/run-1.md"),
        artifact_sha256="sha-1",
        complete=True,
    )
    outcome = _outcome(evidence=evidence, decision=DeliveryDecision(True, True, ()))
    body = "\n".join(delivery_lines(outcome, session, set()))
    assert "累计变更工件" in body and "run-1.md" in body and "sha256=sha-1" in body
    assert "相对启动基线累计改动: 0 项" in body


# --- CheckProjection: 回执收集 / 请求清零 / 安静界面 ----------------------------------------------


def test_check_projection_collects_receipt_ids_and_resets_per_request():
    projection = CheckProjection(_StubSession(verification_run_id="vr-1"))
    projection.observe_event(ToolCompleted(
        "run-1",
        ToolResult(call_id="c1", name="check", content="ok", is_error=False,
                   details={"receipt_id": "r-1"}),
    ))
    assert projection._receipts_this_request == {"r-1"}
    projection.observe_event(ToolCompleted(
        "run-1",
        ToolResult(call_id="c2", name="read", content="x", is_error=False,
                   details={"receipt_id": "not-check"}),
    ))
    assert projection._receipts_this_request == {"r-1"}, "非 check 工具不收集"
    projection.note_phase(None, "executing")
    assert projection._receipts_this_request == set(), "新请求开始清零"
    projection.note_phase("run-1", "executing")  # 带 run_id 的阶段通知不清零
    projection.observe_event(ToolCompleted(
        "run-1",
        ToolResult(call_id="c3", name="check", content="ok", is_error=False,
                   details={"receipt_id": "r-2"}),
    ))
    assert projection._receipts_this_request == {"r-2"}


def test_check_projection_quiet_in_normal_mode():
    projection = CheckProjection(_StubSession(verification_run_id=None))
    assert projection.on_request_completed(_outcome()) is None, "正常完成无条目"
    # 普通模式完成后 /status 如实显示未启用检查 (口径同旧状态栏)
    assert projection.status_lines() == ["检查: 未启用检查"]


def test_check_projection_status_lines_check_mode():
    projection = CheckProjection(_StubSession(verification_run_id="vr-1"))
    assert projection.status_lines() == ["验证运行: vr-1", "检查: 不可得"]
    outcome = _outcome(
        view=_view(CheckState("check-1", _receipt("r-1"), "current")),
        evidence=ChangeEvidence(
            baseline_snapshot_ref="s0", snapshot_ref="s1", scope_id="scope",
            changes=(), artifact_path=None, artifact_sha256=None, complete=True,
        ),
        decision=DeliveryDecision(True, True, ()),
    )
    assert projection.on_request_completed(outcome) is not None
    status = "\n".join(projection.status_lines())
    assert f"检查: check-1=passed/current" in status
    assert "累计改动口径: 相对启动基线 0 项变更" in status


def test_check_status_text_normal_mode_and_uncertain():
    session = _StubSession(verification_run_id=None)
    assert check_status_text(_outcome(), session) == NORMAL_MODE_CHECK_LABEL
    latch = _StubSession(verification_run_id="vr-1", process_uncertain=True)
    outcome = _outcome(view=_view(CheckState("check-1", _receipt("r-1"), "current")))
    assert PROCESS_UNCERTAIN_LABEL in check_status_text(outcome, latch)


def test_change_scope_text_incomplete_evidence():
    incomplete = ChangeEvidence(
        baseline_snapshot_ref=None, snapshot_ref=None, scope_id=None, changes=(),
        artifact_path=None, artifact_sha256=None, complete=False,
        failure_kind="scan_failed",
    )
    outcome = _outcome(evidence=incomplete)
    assert change_scope_text(outcome) == "累计改动口径: 相对启动基线（不完整: scan_failed）"
    assert change_scope_text(_outcome()) is None
