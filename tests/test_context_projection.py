"""context_policy.py 02 增量聚焦测试: 投影容器、摘要块与摘要生成协议 (D4/D5).

 BudgetPolicy 投影替换用记录适配器断言实际转发消息; 摘要通道用最小脚本替身
 (独立于共享剧本队列, B-F4), 逐类失败分类断言。
"""

from __future__ import annotations

import asyncio

import pytest

from cicada.budget import estimate_message_tokens
from cicada.context_policy import (
    SUMMARY_PROMPT_SHA256,
    SUMMARY_PROMPT_VERSION,
    BudgetPolicy,
    ContextProjection,
    ProjectionRelationError,
    SummaryBlock,
    canonical_prefix_sha256,
    generate_summary,
    summary_block_message,
    summary_prompt_text,
)
from cicada.core.cancel import CancelToken
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.core.ports import (
    ModelMetrics,
    ModelRequest,
    StreamDone,
    TextDelta,
    ToolCallEvent,
)
from tests.support.budget import policy_config as _config


# --- 投影容器 (D4) ---------------------------------------------------------------------


def _block(end: int, *, first_seq: int = 1, last_seq: int = 1, count: int | None = None) -> SummaryBlock:
    return SummaryBlock(
        message=summary_block_message(
            first_request_seq=first_seq, last_request_seq=last_seq,
            message_count=count if count is not None else end,
            text=f"摘要覆盖到 {end}",
        ),
        end_message_index=end,
        first_request_seq=first_seq,
        last_request_seq=last_seq,
        message_count=count if count is not None else end,
        prefix_sha256="a" * 64,
    )


def test_empty_projection_is_identity():
    projection = ContextProjection()
    messages = (UserMessage(text="a"), AssistantMessage(text="b"))
    assert projection.project(messages) == messages
    assert projection.retained_start == 0
    assert projection.blocks == ()


def test_project_prepends_blocks_and_slices_suffix():
    projection = ContextProjection()
    canonical = (UserMessage(text="a"), AssistantMessage(text="b"), UserMessage(text="c"))
    block = _block(2)
    projection.apply_summary(block, canonical_length=len(canonical))
    projected = projection.project(canonical)
    assert projected[0] is block.message
    assert projected[1:] == canonical[2:]
    assert projection.retained_start == 2


def test_apply_summary_rejects_discontiguous_and_out_of_range():
    projection = ContextProjection()
    projection.apply_summary(_block(2), canonical_length=4)
    with pytest.raises(ProjectionRelationError, match="beyond the current retained start"):
        projection.apply_summary(_block(2), canonical_length=4)
    with pytest.raises(ProjectionRelationError, match="exceeds the canonical history"):
        projection.apply_summary(_block(5), canonical_length=4)
    with pytest.raises(ProjectionRelationError):
        projection.apply_summary(_block(1), canonical_length=4)


def test_adopt_validates_strictly_increasing_ranges():
    projection = ContextProjection()
    projection.adopt([_block(2), _block(4, count=2)])
    assert projection.retained_start == 4
    with pytest.raises(ProjectionRelationError):
        projection.adopt([_block(2), _block(2)])
    with pytest.raises(ProjectionRelationError):
        projection.adopt([_block(4), _block(2)])


def test_reset_clears_blocks_for_new_generation():
    projection = ContextProjection()
    projection.apply_summary(_block(2), canonical_length=4)
    projection.reset()
    assert projection.blocks == ()
    assert projection.retained_start == 0


def test_summary_block_message_is_deterministic_and_declares_authority():
    first = summary_block_message(first_request_seq=3, last_request_seq=5, message_count=7, text="正文")
    second = summary_block_message(first_request_seq=3, last_request_seq=5, message_count=7, text="正文")
    assert first == second
    assert "不是原始对话记录" in first.text
    assert "seq 3..5" in first.text
    assert "不携带验证权威" in first.text
    assert first.text.endswith("正文")


def test_canonical_prefix_sha256_pins_fingerprint_recipe():
    """冻结契约: 覆盖前缀指纹 = sha256(encode_messages 的 UTF-8 字节) 的十六进制.

    字面量钉死哈希配方——编码或哈希算法变化必须显式更新本测试 (历史摘要
    指纹依赖该配方, 不允许随实现静默漂移)。
    """
    messages = (UserMessage(text="a"), AssistantMessage(text="b"))
    expected = "2dfb44c815e76d41aec06b061ae9e434a6fc67b00e4369d4f4cc815dcccc0656"
    assert canonical_prefix_sha256(messages) == expected
    assert canonical_prefix_sha256(messages[:1]) == (
        "e5f96f7748bb025531abd7d87193e18fc3b51d6c600e4491ff49f00a10f474e1"
    )
    assert canonical_prefix_sha256(messages[:1]) != expected


def test_summary_prompt_is_versioned_and_stable():
    assert SUMMARY_PROMPT_VERSION == "2"  # v2: 请求末尾追加固定 user 指令 (05 收口修正)
    assert len(SUMMARY_PROMPT_SHA256) == 64
    assert summary_prompt_text() == summary_prompt_text()


def test_summary_input_messages_appends_fixed_user_turn():
    """摘要请求整形 (05 收口修正): 对话切片以 assistant 结尾时模型立即终结
    (模型 stop + content 空 + output_tokens=1)——固定 user
    结尾保证摘要请求有生成起点; 追加消息不属规范历史 (调用方不计覆盖范围)."""
    from cicada.context_policy import SUMMARY_TURN_TEXT, summary_input_messages

    slice_messages = (UserMessage(text="问"), AssistantMessage(text="答"))
    shaped = summary_input_messages(slice_messages)
    assert shaped[:2] == slice_messages, "原始切片逐位保留"
    assert shaped[-1].role == "user" and shaped[-1].text == SUMMARY_TURN_TEXT
    assert shaped == summary_input_messages(list(slice_messages)), "确定性输出"


# --- 摘要生成协议 (D5) -------------------------------------------------------------------


class _Channel:
    """最小 ModelPort 替身: 按剧本产出事件; 校验摘要请求形态 (无工具+专用提示)."""

    def __init__(self, events, *, stop_after_cancel: bool = False) -> None:
        self._events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest, cancel: CancelToken):
        self.requests.append(request)
        for event in self._events:
            await asyncio.sleep(0)
            yield event


def _summary_request_guard(channel: _Channel) -> ModelRequest:
    assert channel.requests, "摘要通道未被调用"
    request = channel.requests[0]
    assert request.tools == (), "摘要调用必须无工具 (D5)"
    assert request.system_prompt == summary_prompt_text()
    return request


async def test_generate_summary_accepts_stop_with_bounded_text():
    channel = _Channel([TextDelta("第一段"), TextDelta("第二段"),
                        StreamDone("stop", metrics=ModelMetrics(input_tokens=10, output_tokens=5))])
    outcome = await generate_summary(
        channel, (UserMessage(text="q"),), CancelToken()
    )
    assert outcome.ok
    assert outcome.text == "第一段第二段"
    assert outcome.metrics is not None and outcome.metrics.input_tokens == 10
    request = _summary_request_guard(channel)
    assert request.messages == (UserMessage(text="q"),)


@pytest.mark.parametrize(
    "events, failure",
    [
        ([StreamDone("length")], "length"),
        ([StreamDone("error", "boom")], "error"),
        ([ToolCallEvent("c1", "t", "{}"), StreamDone("tool_use")], "tool_use"),
        ([StreamDone("aborted")], "aborted"),
        ([StreamDone("stop")], "empty"),
    ],
)
async def test_generate_summary_classifies_failures(events, failure):
    channel = _Channel(events)
    outcome = await generate_summary(channel, (UserMessage(text="q"),), CancelToken())
    assert not outcome.ok
    assert outcome.failure == failure
    assert outcome.text == ""


async def test_generate_summary_rejects_oversized_text():
    events = [TextDelta("x" * (16 * 1024 + 1)), StreamDone("stop")]
    outcome = await generate_summary(_Channel(events), (UserMessage(text="q"),), CancelToken())
    assert not outcome.ok and outcome.failure == "too_long"


async def test_generate_summary_reports_stream_without_terminal():
    outcome = await generate_summary(_Channel([TextDelta("部分")]), (UserMessage(text="q"),), CancelToken())
    assert not outcome.ok and outcome.failure == "error"
    assert outcome.detail is not None


async def test_generate_summary_reports_channel_exception():
    class _Broken:
        async def stream(self, request, cancel):
            raise RuntimeError("channel down")
            yield  # pragma: no cover

    outcome = await generate_summary(_Broken(), (UserMessage(text="q"),), CancelToken())
    assert not outcome.ok and outcome.failure == "error"
    assert "channel down" in (outcome.detail or "")


# --- BudgetPolicy 投影替换 ----------------------------------------------------------------


class _Recording:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest, cancel: CancelToken):
        self.requests.append(request)
        yield StreamDone("stop", metrics=ModelMetrics(input_tokens=50, output_tokens=1))


async def test_policy_forwards_projected_request_when_blocks_exist():
    inner = _Recording()
    policy = BudgetPolicy(inner, _config())
    canonical = (UserMessage(text="x" * 3000), AssistantMessage(text="done"), UserMessage(text="tail"))
    block = summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=2, text="较早请求的摘要"
    )
    policy.projection.apply_summary(
        SummaryBlock(
            message=block, end_message_index=2, first_request_seq=1,
            last_request_seq=1, message_count=2,
            prefix_sha256=canonical_prefix_sha256(canonical[:2]),
        ),
        canonical_length=len(canonical),
    )
    request = ModelRequest(messages=canonical, tools=(), system_prompt="sys")
    events = [event async for event in policy.stream(request, CancelToken())]
    assert events[-1].stop_reason == "stop"
    forwarded = inner.requests[0]
    assert forwarded is not request
    assert forwarded.messages[0].text == block.text
    assert forwarded.messages[1:] == canonical[2:]


async def test_policy_projection_preserves_request_context():
    """投影替换规范消息时 request_context 原样保留; 尾部不是摘要覆盖范围 (契约 C4)."""
    inner = _Recording()
    policy = BudgetPolicy(inner, _config())
    canonical = (UserMessage(text="x" * 3000), AssistantMessage(text="done"), UserMessage(text="tail"))
    block = summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=2, text="较早请求的摘要"
    )
    policy.projection.apply_summary(
        SummaryBlock(
            message=block, end_message_index=2, first_request_seq=1,
            last_request_seq=1, message_count=2,
            prefix_sha256=canonical_prefix_sha256(canonical[:2]),
        ),
        canonical_length=len(canonical),
    )
    request = ModelRequest(messages=canonical, tools=(), system_prompt="sys", request_context="<ctx-v1/>")
    events = [event async for event in policy.stream(request, CancelToken())]
    assert events[-1].stop_reason == "stop"
    forwarded = inner.requests[0]
    assert forwarded.request_context == "<ctx-v1/>"
    assert forwarded.messages[0].text == block.text
    assert forwarded.messages[1:] == canonical[2:]


async def test_policy_estimate_covers_summary_block():
    """投影后计量包含摘要块: 块+长尾超限时拒绝, 适配器零发送 (01-F10 下游面)."""
    inner = _Recording()
    policy = BudgetPolicy(inner, _config(num_ctx=60))
    canonical = (UserMessage(text="x" * 3000), AssistantMessage(text="done"), UserMessage(text="tail"))
    block = summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=2,
        text="y" * 150,
    )
    policy.projection.apply_summary(
        SummaryBlock(
            message=block, end_message_index=2, first_request_seq=1,
            last_request_seq=1, message_count=2,
            prefix_sha256=canonical_prefix_sha256(canonical[:2]),
        ),
        canonical_length=len(canonical),
    )
    # 有效消息 = 摘要块 (~60) + tail (2) ≈ 62 > 60: 投影后超限
    projected = policy.projection.project(canonical)
    effective_estimate = sum(estimate_message_tokens(m, policy.config) for m in projected)
    assert effective_estimate > 60
    events = [event async for event in policy.stream(
        ModelRequest(messages=canonical, tools=(), system_prompt=""), CancelToken()
    )]
    assert events[-1].stop_reason == "error"
    assert "context budget exceeded" in (events[-1].error or "")
    assert inner.requests == []


async def test_policy_clears_anchor_when_projection_changes():
    """压缩改变投影 → 锚基准 (投影后请求) 失配 → 回退启发式 (D3 保守回退)."""
    inner = _Recording()
    policy = BudgetPolicy(inner, _config())
    request = ModelRequest(messages=(UserMessage(text="q"),), tools=(), system_prompt="sys")
    _ = [event async for event in policy.stream(request, CancelToken())]
    assert policy.anchor is not None
    block = summary_block_message(first_request_seq=1, last_request_seq=1, message_count=1, text="摘要")
    policy.projection.apply_summary(
        SummaryBlock(
            message=block, end_message_index=1, first_request_seq=1,
            last_request_seq=1, message_count=1,
            prefix_sha256=canonical_prefix_sha256(request.messages),
        ),
        canonical_length=1,
    )
    extended = ModelRequest(
        messages=request.messages + (UserMessage(text="q2"),), tools=(), system_prompt="sys"
    )
    _ = [event async for event in policy.stream(extended, CancelToken())]
    assert policy.last_report.source == "estimated", "压缩后投影失配, 不再 anchored"
