"""budget.py 契约聚焦测试: 估计器、配置校验、报告完整性与序列化 (A01).

估计算法只断言方向与可加性, 不复述实现常量; 报告断言"无静默省略、
各部分之和=总估计、来源等级与算法名+版本齐备、可 JSON 序列化"。
"""

from __future__ import annotations

import json
import math

import pytest

from cicada.budget import (
    BUDGET_ESTIMATOR_NAME,
    BUDGET_ESTIMATOR_VERSION,
    BudgetConfig,
    BudgetReport,
    assemble_report,
    budget_exceeded_text,
    estimate_input_components,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_spec_tokens,
    message_kind,
)
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ToolSpec


def _config(**overrides) -> BudgetConfig:
    return BudgetConfig(num_ctx=100_000, num_predict=10, **overrides)


def _specs(*names: str) -> tuple[ToolSpec, ...]:
    return tuple(
        ToolSpec(name, f"description of {name}", {"type": "object", "properties": {}})
        for name in names
    )


# --- 估计器 ----------------------------------------------------------------------


def test_text_estimation_is_bytes_over_three():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 2  # 4 bytes -> ceil(4/3)
    assert estimate_text_tokens("你好") == 2  # 6 bytes
    assert estimate_text_tokens("a" * 300) == 100


def test_message_estimation_adds_template_overhead():
    config = _config(message_template_overhead=8)
    bare = estimate_text_tokens("abcd")
    assert estimate_message_tokens(UserMessage(text="abcd"), config) == bare + 8


def test_message_estimation_covers_tool_calls_and_results():
    config = _config()
    assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall("c1", "read", '{"path": "a.py"}'),),
    )
    result = ToolResultMessage(
        result=ToolResult(call_id="c1", name="read", content="x" * 30, details={"k": "v"})
    )
    for message in (assistant, result):
        tokens = estimate_message_tokens(message, config)
        assert tokens > config.message_template_overhead
    # details 参与估计: 更大的 details 估计更高
    bigger = ToolResultMessage(
        result=ToolResult(call_id="c1", name="read", content="x" * 30, details={"k": "v" * 100})
    )
    assert estimate_message_tokens(bigger, config) > estimate_message_tokens(result, config)


def test_message_estimation_rejects_unknown_type():
    with pytest.raises(ValueError, match="unsupported message type"):
        estimate_message_tokens("not a message", _config())  # type: ignore[arg-type]


def test_message_kind_labels():
    assert message_kind(UserMessage(text="x")) == "user"
    assert message_kind(AssistantMessage(text="x")) == "assistant"
    assert message_kind(
        ToolResultMessage(result=ToolResult(call_id="c", name="n", content="x"))
    ) == "tool"


def test_tool_schema_estimation_is_deterministic_and_content_sensitive():
    first = estimate_tool_spec_tokens(_specs("t")[0])
    assert first > 0
    assert estimate_tool_spec_tokens(_specs("t")[0]) == first
    big = ToolSpec("t", "d" * 300, {"type": "object"})
    assert estimate_tool_spec_tokens(big) > first


# --- 报告完整性 (A01) --------------------------------------------------------------


def test_report_covers_every_section_and_sums_to_total():
    config = _config()
    messages = (UserMessage(text="hello"), AssistantMessage(text="hi"))
    input_estimate, components = estimate_input_components(
        "system prompt text", _specs("read", "edit"), messages, config
    )
    labels = [component.label for component in components]
    # 完整开销分解: system prompt / 工具 schema 逐个 / 消息逐条, 无静默省略
    assert labels[0] == "system_prompt"
    assert labels[1:3] == ["tool_schema:read", "tool_schema:edit"]
    assert labels[3] == "message:0:user"
    assert labels[4] == "message:1:assistant"
    assert sum(component.tokens for component in components) == input_estimate

    report = assemble_report(
        config, input_estimate=input_estimate, input_components=components, source="estimated"
    )
    assert report.output_reserve == 10
    assert report.safety_margin == max(
        config.safety_margin_min, math.ceil(config.safety_margin_ratio * input_estimate)
    )
    assert report.total_estimate == (
        report.input_estimate + report.output_reserve + report.safety_margin
    )
    assert sum(component.tokens for component in report.components) == report.total_estimate
    assert report.source == "estimated"
    assert report.estimator == BUDGET_ESTIMATOR_NAME
    assert report.estimator_version == BUDGET_ESTIMATOR_VERSION
    assert report.limit == config.num_ctx


def test_report_output_reserve_uses_default_when_num_predict_absent():
    config = BudgetConfig(num_ctx=50_000)
    assert config.effective_output_reserve == config.output_reserve_default
    report = assemble_report(
        config, input_estimate=100, input_components=(), source="estimated"
    )
    reserve = next(c for c in report.components if c.label == "output_reserve")
    assert reserve.tokens == config.output_reserve_default


def test_report_is_json_serializable():
    config = _config()
    _, components = estimate_input_components("sys", _specs("read"), (UserMessage(text="q"),), config)
    report = assemble_report(config, input_estimate=50, input_components=components, source="estimated")
    decoded = json.loads(json.dumps(report.to_dict()))
    assert decoded["total_estimate"] == report.total_estimate
    assert decoded["source"] == "estimated"
    assert decoded["estimator_version"] == BUDGET_ESTIMATOR_VERSION
    assert [c["label"] for c in decoded["components"]] == [c.label for c in report.components]


def test_budget_exceeded_text_carries_numbers_and_components():
    config = BudgetConfig(num_ctx=1000, num_predict=10)
    _, components = estimate_input_components(
        "s" * 3000, _specs("read"), (UserMessage(text="q" * 900),), config
    )
    report = assemble_report(config, input_estimate=1500, input_components=components, source="estimated")
    assert report.total_estimate > report.limit
    text = budget_exceeded_text(report)
    assert f"~{report.total_estimate}" in text
    assert "1000" in text
    assert "largest components:" in text
    # 主要占用组成: 最长消息/工具/系统段以标签出现
    assert "message:0:user" in text or "system_prompt" in text


# --- 配置校验 ----------------------------------------------------------------------


@pytest.mark.parametrize("num_ctx", [0, -1, True, "32768", 1.5])
def test_num_ctx_must_be_positive_int(num_ctx):
    with pytest.raises(ValueError, match="num_ctx"):
        BudgetConfig(num_ctx=num_ctx)


@pytest.mark.parametrize("num_predict", [0, -5, True])
def test_num_predict_must_be_positive_int_or_none(num_predict):
    with pytest.raises(ValueError, match="num_predict"):
        BudgetConfig(num_ctx=100, num_predict=num_predict)


def test_config_rejects_bad_margin_parameters():
    with pytest.raises(ValueError, match="safety_margin_ratio"):
        BudgetConfig(num_ctx=100, safety_margin_ratio=1.5)
    with pytest.raises(ValueError, match="safety_margin_min"):
        BudgetConfig(num_ctx=100, safety_margin_min=-1)
    with pytest.raises(ValueError, match="message_template_overhead"):
        BudgetConfig(num_ctx=100, message_template_overhead=-1)
    with pytest.raises(ValueError, match="output_reserve_default"):
        BudgetConfig(num_ctx=100, output_reserve_default=0)


def test_report_margin_remaining_can_go_negative():
    report = BudgetReport(
        limit=100,
        components=(),
        input_estimate=500,
        output_reserve=10,
        safety_margin=50,
        total_estimate=560,
        source="estimated",
    )
    assert report.margin_remaining == 100 - 560
