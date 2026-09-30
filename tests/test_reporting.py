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
