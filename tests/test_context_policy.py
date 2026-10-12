"""context_policy.py 聚焦测试: 每轮权威准入、零发送拒绝与锚状态机 (A02/A03).

锚失效矩阵逐条钉住 (spec D3): 缺 usage/0 值/error/aborted 不立锚;
clear (committed_count=0)/投影或提示或工具变化经前缀失配失效。
适配器零发送用 FakeModel.requests 断言。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from cicada.budget import estimate_message_tokens
from cicada.context_policy import BudgetPolicy
from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelMetrics, ModelRequest, StreamDone, TextDelta, ToolSpec
from cicada.plugins.fake_model import FakeModel

from tests.support.budget import policy_config as _config


async def test_complete_local_priority_fallback_and_dispatch_schema_snapshot():
    from cicada.budget import LocalTokenCount

    class Counter:
        result = LocalTokenCount(20, "test-profile")
        def count_request(self, request):
            return self.result

    counter = Counter()
    request = _request("hello", tools=_specs("read"))

    class MutatingModel:
        async def stream(self, frozen, cancel):
            assert frozen.tools[0].parameters == {"type": "object"}
            request.tools[0].parameters["required"] = ["new"]
            yield _stop(_measured(100))

    policy = BudgetPolicy(MutatingModel(), _config(), token_counter=counter)
    await anext(policy.stream(request, CancelToken()))
    assert policy.last_report.input_estimate == 20
    assert policy._assess(request).source == "estimated"  # 原对象变化不能复用旧实测
    original = replace(request, tools=_specs("read"))
    assert policy._assess(original).source == "measured"
    extension = replace(original, messages=original.messages + (UserMessage("next"),))
    report = policy._assess(extension)
    assert report.input_estimate == 20 and report.source == "estimated"
    assert report.estimator == "deepseek-local-template"
    assert [c.label for c in report.components] == ["local_template_input", "output_reserve", "safety_margin"]
    assert sum(c.tokens for c in report.components) == report.total_estimate
    counter.result = LocalTokenCount(None, "test-profile", "asset-missing")
    assert policy._assess(extension).source == "anchored"
    assert policy._assess(extension).local_count.reason == "asset-missing"
    counter.result = False
    with pytest.raises(TypeError, match="LocalTokenCount"):
        policy._assess(extension)


async def test_count_qualification_rejection_and_cancel_after_compaction_are_terminal():
    from cicada.budget import LocalTokenCount

    class Counter:
        def count_request(self, request):
            if request.system_prompt == "bad":
                raise ValueError("safe request qualification diagnostic")
            return LocalTokenCount(10000, "test-profile")

    model = FakeModel([])
    policy = BudgetPolicy(model, _config(), token_counter=Counter())
    bad = _request("x", system="bad")
    events = [e async for e in policy.stream(bad, CancelToken())]
    assert events == [StreamDone("error", "safe request qualification diagnostic")]
    policy.config = _config(num_ctx=100)
    async def compact():
        policy.token_counter = Counter()
        policy.token_counter.count_request = lambda request: (_ for _ in ()).throw(ValueError("after compact"))
        return True
    policy.set_auto_compact(compact)
    events = [e async for e in policy.stream(_request("x"), CancelToken())]
    assert events == [StreamDone("error", "after compact")]
    token = CancelToken()
    async def cancel_compact():
        token.cancel()
        return True
    policy.token_counter = Counter()
    policy.set_auto_compact(cancel_compact)
    events = [e async for e in policy.stream(_request("x"), token)]
    assert events == [StreamDone("aborted")]
    assert model.requests == []


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


# --- 带尾部请求的锚条件 (缓存 02, 契约 C4) ---------------------------------------------


async def _tail_anchored_policy(context: str = "tail-v1"):
    """一轮带尾部流结束后返回 (policy, 基准请求); 锚以实测 100 tokens 建立."""
    inner = FakeModel([[_stop(_measured(100))]])
    policy = BudgetPolicy(inner, _config())
    request = replace(_request("q1", tools=_specs("read")), request_context=context)
    async for _ in policy.stream(request, CancelToken()):
        pass
    return policy, request


async def test_tail_request_is_measured_only_on_full_equality():
    policy, request = await _tail_anchored_policy()
    async for _ in policy.stream(request, CancelToken()):
        pass
    report = policy.last_report
    assert report.source == "measured"
    assert report.input_estimate == 100


async def test_tail_request_history_extension_is_estimated_not_anchored():
    """带尾部的请求即使规范历史延长也不是完整 wire 前缀延长: 重新估计, 不标 anchored."""
    policy, request = await _tail_anchored_policy()
    extended = replace(request, messages=request.messages + (UserMessage(text="q2"),))
    async for _ in policy.stream(extended, CancelToken()):
        pass
    report = policy.last_report
    assert report.source == "estimated"
    assert "request_context" in [component.label for component in report.components]


async def test_tail_change_reestimates_even_with_identical_history():
    policy, request = await _tail_anchored_policy()
    drifted = replace(request, request_context="tail-v2")
    async for _ in policy.stream(drifted, CancelToken()):
        pass
    assert policy.last_report.source == "estimated"


async def test_anchor_from_no_tail_request_never_serves_tail_request():
    """无尾锚与新尾部请求 context 不等: 不沿现行前缀条件复用实测."""
    policy, request = await _anchored_policy()
    with_tail = replace(request, request_context="tail-v1")
    async for _ in policy.stream(with_tail, CancelToken()):
        pass
    assert policy.last_report.source == "estimated"


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
    assert snapshot.comparison is None


# --- 同请求计数对照 (票据 03, 契约 C7) --------------------------------------------------


class _LocalCounter:
    """可改写的本地计数替身: 每次调用返回最新结果。"""

    def __init__(self, result):
        self.result = result

    def count_request(self, request):
        return self.result


async def test_comparison_pairs_local_count_with_same_dispatch_usage():
    from cicada.budget import LocalTokenCount, TokenCountComparison

    counter = _LocalCounter(LocalTokenCount(20, "test-profile"))
    inner = FakeModel([
        [_stop(_measured(100))],                                  # 同请求 usage 可得
        [_stop()],                                                # usage 缺失
        [_stop(ModelMetrics(input_tokens=0, output_tokens=1))],   # 已报告的 0 保留
        [_stop(_measured(100))],                                  # 本地不可用那一轮
    ])
    policy = BudgetPolicy(inner, _config(), token_counter=counter)

    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison == TokenCountComparison(
        LocalTokenCount(20, "test-profile"), 100
    )

    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison.input_tokens is None, "缺 usage 保持不可得而非 0"

    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison.input_tokens == 0, "0 与 None 区分"

    counter.result = LocalTokenCount(None, "test-profile", "asset-missing")
    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    snapshot = policy.snapshot()
    assert snapshot.comparison is None, "本地不可用时比较不可得"
    assert snapshot.report.local_count.reason == "asset-missing", "安全原因随报告可得"


async def test_comparison_cleared_by_invalidation_error_and_cancel():
    from cicada.budget import LocalTokenCount, TokenCountComparison

    counter = _LocalCounter(LocalTokenCount(20, "test-profile"))
    inner = FakeModel([[_stop(_measured(100))], [_stop(_measured(100))], [_stop(_measured(100))]])
    policy = BudgetPolicy(inner, _config(), token_counter=counter)
    expected = TokenCountComparison(LocalTokenCount(20, "test-profile"), 100)

    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison == expected
    policy.invalidate()  # clear/压缩/配置变化/恢复的公共失效面
    assert policy.snapshot().comparison is None

    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison == expected
    policy.config = _config(num_ctx=1)
    async for _ in policy.stream(_request("x" * 300), CancelToken()):
        pass
    assert policy.snapshot().comparison is None, "准入拒绝不留下旧偏差"

    policy.config = _config()
    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    assert policy.snapshot().comparison == expected
    token = CancelToken()
    token.cancel()
    async for _ in policy.stream(_request("q1"), token):
        pass
    assert policy.snapshot().comparison is None, "取消不冒充新调用"


async def test_comparison_cleared_at_request_boundary_seam():
    """03: 请求边界接缝清旧对照 (本次尝试可能在进入预算评估前就终结: 取消/
    RunPolicy 拒绝/本地准入拒绝, 不进入 BudgetPolicy.stream)。"""
    from cicada.budget import LocalTokenCount, TokenCountComparison

    counter = _LocalCounter(LocalTokenCount(20, "test-profile"))
    policy = BudgetPolicy(FakeModel([[_stop(_measured(100))]]), _config(), token_counter=counter)
    async for _ in policy.stream(_request("q1"), CancelToken()):
        pass
    expected = TokenCountComparison(LocalTokenCount(20, "test-profile"), 100)
    assert policy.snapshot().comparison == expected

    policy.note_request_start(3)  # 应用在准入通过后、首次预估前调用一次
    assert policy.snapshot().comparison is None, "新尝试开始即丢弃旧偏差"


async def test_comparison_refuses_non_local_count_payload():
    from cicada.budget import LocalTokenCount, TokenCountComparison

    with pytest.raises(TypeError, match="local_count"):
        TokenCountComparison(local_count=20, input_tokens=1)
    with pytest.raises(TypeError, match="input_tokens"):
        TokenCountComparison(local_count=LocalTokenCount(1, "p"), input_tokens=True)
    with pytest.raises(ValueError, match="available count"):
        TokenCountComparison(local_count=LocalTokenCount(None, "p", "asset-missing"), input_tokens=1)


def test_constructor_rejects_bad_inner_and_config():
    with pytest.raises(TypeError, match="inner must be a ModelPort"):
        BudgetPolicy("not a port", _config())
    with pytest.raises(TypeError, match="config must be a BudgetConfig"):
        BudgetPolicy(FakeModel([]), "not a config")


def test_constructor_accepts_injected_seams_but_stays_inert():
    summarize = lambda *args, **kwargs: None  # noqa: E731
    policy = BudgetPolicy(
        FakeModel([]), _config(), summarize=summarize
    )
    assert policy.inner is not None
    with pytest.raises(TypeError, match="summarize"):
        BudgetPolicy(FakeModel([]), _config(), summarize="not callable")


def test_note_request_start_validates_argument():
    policy = BudgetPolicy(FakeModel([]), _config())
    with pytest.raises(ValueError, match="committed_count"):
        policy.note_request_start(-1)
    with pytest.raises(ValueError, match="committed_count"):
        policy.note_request_start(True)
