"""context_policy.py 聚焦测试: 每轮权威准入、零发送拒绝与锚状态机 (A02/A03).

锚失效矩阵逐条钉住 (spec D3): 缺 usage/0 值/error/aborted 不立锚;
clear (committed_count=0)/投影或提示或工具变化经前缀失配失效。
适配器零发送用 FakeModel.requests 断言。
"""

from __future__ import annotations

import asyncio

import pytest

from cicada.budget import BudgetConfig, estimate_message_tokens
from cicada.context_policy import BudgetPolicy
from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelMetrics, ModelRequest, StreamDone, TextDelta, ToolSpec
from cicada.plugins.fake_model import FakeModel


def _config(**overrides) -> BudgetConfig:
    defaults = dict(
        num_ctx=100_000,
        num_predict=1,
        safety_margin_min=1,
        safety_margin_ratio=0,
        message_template_overhead=1,
    )
    defaults.update(overrides)
    return BudgetConfig(**defaults)


def _specs(*names: str) -> tuple[ToolSpec, ...]:
    return tuple(
        ToolSpec(name, f"description of {name}", {"type": "object"}) for name in names
    )


def _request(*texts: str, system: str = "sys", tools: tuple[ToolSpec, ...] = ()) -> ModelRequest:
    return ModelRequest(
        messages=tuple(UserMessage(text=text) for text in texts),
        tools=tools,
        system_prompt=system,
    )


def _stop(metrics: ModelMetrics | None = None, stop: str = "stop") -> StreamDone:
    return StreamDone(stop, metrics=metrics)


def _measured(input_tokens: int | None = None) -> ModelMetrics | None:
    return ModelMetrics(input_tokens=input_tokens, output_tokens=1) if input_tokens is not None else None


# --- 每轮权威准入 (A02) --------------------------------------------------------------


async def test_oversized_request_rejected_with_zero_send():
    inner = FakeModel([])
    policy = BudgetPolicy(inner, _config(num_ctx=50))
    events = [
        event async for event in policy.stream(_request("x" * 300), CancelToken())
    ]
    assert len(events) == 1
    done = events[0]
    assert done.stop_reason == "error"
    assert "context budget exceeded" in done.error
    assert "50 (num_ctx)" in done.error
    assert "no request was sent" in done.error
    assert inner.requests == []  # 适配器零发送


async def test_admitted_request_passes_through_unchanged():
    inner = FakeModel([[_stop(_measured(10))]])
    policy = BudgetPolicy(inner, _config())
    request = _request("hello", tools=_specs("read"))
    events = [event async for event in policy.stream(request, CancelToken())]
    assert [type(event) for event in events] == [StreamDone]
    assert events[0].stop_reason == "stop"
    assert inner.requests == [request]  # 原请求对象不改动


async def test_policy_forwards_non_terminal_events():
    inner = FakeModel([[TextDelta("hi"), _stop()]])
    policy = BudgetPolicy(inner, _config())
    events = [event async for event in policy.stream(_request("q"), CancelToken())]
    assert [type(event) for event in events] == [TextDelta, StreamDone]


async def test_cancelled_request_is_aborted_before_estimation():
    inner = FakeModel([])
    policy = BudgetPolicy(inner, _config(num_ctx=1))
    token = CancelToken()
    token.cancel()
    events = [event async for event in policy.stream(_request("q"), token)]
    assert events == [StreamDone("aborted")]
    assert inner.requests == []


# --- 实测锚 (A03) ---------------------------------------------------------------------


async def _anchored_policy(script_entry=None, **config_overrides):
    """一轮流结束后返回 (policy, 基准请求); 锚以实测 100 tokens 建立."""
    entry = [script_entry] if script_entry is not None else [_stop(_measured(100))]
    inner = FakeModel([entry])
    policy = BudgetPolicy(inner, _config(**config_overrides))
    request = _request("q1", tools=_specs("read"))
    async for _ in policy.stream(request, CancelToken()):
        pass
    return policy, request


async def test_anchor_established_on_measured_stop():
    policy, _ = await _anchored_policy()
    assert policy.anchor is not None
    assert policy.anchor.measured_input_tokens == 100
    assert policy.anchor.message_count == 1


@pytest.mark.parametrize(
    "entry",
    [
        _stop(),                       # 缺 usage (metrics None)
        _stop(_measured(None)),        # input_tokens 不可得
        _stop(ModelMetrics(input_tokens=0, output_tokens=1)),  # 0 值不立锚
        _stop(_measured(100), stop="error"),
        _stop(_measured(100), stop="aborted"),
    ],
)
async def test_anchor_matrix_refuses_invalid_terminations(entry):
    policy, request = await _anchored_policy(script_entry=entry)
    assert policy.anchor is None
    extended = _request("q1", "q2", tools=request.tools)
    async for _ in policy.stream(extended, CancelToken()):
        pass
    assert policy.last_report.source == "estimated"  # 无锚可依, 纯启发式


async def test_anchored_estimate_is_measured_plus_suffix_increment():
    policy, request = await _anchored_policy()
    suffix = UserMessage(text="q2")
    extended = ModelRequest(
        messages=request.messages + (suffix,), tools=request.tools, system_prompt=request.system_prompt
    )
    async for _ in policy.stream(extended, CancelToken()):
        pass
    report = policy.last_report
    assert report.source == "anchored"
    expected_suffix = estimate_message_tokens(suffix, policy.config)
    assert report.input_estimate == 100 + expected_suffix
    labels = [component.label for component in report.components]
    assert labels[0] == "anchor_measured_prefix"
    assert report.components[0].source == "measured"
    assert report.components[0].tokens == 100
    assert labels[1] == "message:1:user"
    assert report.components[1].source == "estimated"
    assert sum(c.tokens for c in report.components) == report.total_estimate


async def test_identical_request_is_fully_measured():
    policy, request = await _anchored_policy()
    async for _ in policy.stream(request, CancelToken()):
        pass
    report = policy.last_report
    assert report.source == "measured"
    assert report.input_estimate == 100


async def test_note_request_start_zero_drops_anchor():
    policy, request = await _anchored_policy()
    policy.note_request_start(0)
    assert policy.anchor is None
    extended = _request("q1", "q2", tools=request.tools)
    async for _ in policy.stream(extended, CancelToken()):
        pass
    assert policy.last_report.source == "estimated"


async def test_note_request_start_nonzero_keeps_anchor():
    policy, _ = await _anchored_policy()
    policy.note_request_start(3)
    assert policy.anchor is not None


@pytest.mark.parametrize("changed", ["system", "prefix_message", "tools"])
async def test_anchor_invalidated_by_projection_fingerprint_change(changed):
    policy, request = await _anchored_policy()
    if changed == "system":
        drifted = ModelRequest(request.messages, request.tools, system_prompt="sys v2")
    elif changed == "prefix_message":
        drifted = ModelRequest((UserMessage(text="q1 changed"),), request.tools, request.system_prompt)
    else:
        drifted = ModelRequest(request.messages, _specs("read", "edit"), request.system_prompt)
    drifted = ModelRequest(
        drifted.messages + (UserMessage(text="q2"),), drifted.tools, drifted.system_prompt
    )
    async for _ in policy.stream(drifted, CancelToken()):
        pass
    assert policy.last_report.source == "estimated"


async def test_new_anchor_replaces_previous():
    inner = FakeModel([[_stop(_measured(100))], [_stop(_measured(150))]])
    policy = BudgetPolicy(inner, _config())
    request = _request("q1", tools=_specs("read"))
    async for _ in policy.stream(request, CancelToken()):
        pass
    async for _ in policy.stream(request, CancelToken()):
        pass
    assert policy.anchor.measured_input_tokens == 150


# --- 快照出口与构造校验 -----------------------------------------------------------------


async def test_snapshot_carries_report_and_anchor():
    policy, _ = await _anchored_policy()
    snapshot = policy.snapshot()
    assert snapshot.limit == policy.config.num_ctx
    assert snapshot.report is not None
    assert snapshot.report.source == "estimated"  # 立锚的那一轮尚无锚可用
    assert snapshot.anchor.measured_input_tokens == 100


async def test_snapshot_before_any_stream_has_no_report():
    policy = BudgetPolicy(FakeModel([]), _config())
    snapshot = policy.snapshot()
    assert snapshot.report is None
    assert snapshot.anchor is None


def test_constructor_rejects_bad_inner_and_config():
    with pytest.raises(TypeError, match="inner must be a ModelPort"):
        BudgetPolicy("not a port", _config())
    with pytest.raises(TypeError, match="config must be a BudgetConfig"):
        BudgetPolicy(FakeModel([]), "not a config")


def test_constructor_accepts_injected_seams_but_stays_inert():
    classify = lambda text: True  # noqa: E731
    summarize = lambda *args, **kwargs: None  # noqa: E731
    policy = BudgetPolicy(
        FakeModel([]), _config(), summarize=summarize, classify_overflow=classify
    )
    assert policy.inner is not None
    with pytest.raises(TypeError, match="classify_overflow"):
        BudgetPolicy(FakeModel([]), _config(), classify_overflow="not callable")
    with pytest.raises(TypeError, match="summarize"):
        BudgetPolicy(FakeModel([]), _config(), summarize="not callable")


def test_note_request_start_validates_argument():
    policy = BudgetPolicy(FakeModel([]), _config())
    with pytest.raises(ValueError, match="committed_count"):
        policy.note_request_start(-1)
    with pytest.raises(ValueError, match="committed_count"):
        policy.note_request_start(True)
