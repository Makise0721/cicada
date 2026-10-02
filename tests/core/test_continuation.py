"""T02/T03 内核续写聚焦测试: 显式历史、后缀返回、闭合哨兵与隔离复制."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from cicada.core.agent import Agent
from cicada.core.cancel import CancelToken
from cicada.core.events import (
    AssistantCompleted,
    AssistantTextDelta,
    Event,
    ModelCallCompleted,
    RunFinished,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import (
    ModelRequest,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    ToolContext,
    ToolSpec,
)
from cicada.plugins.fake_model import FakeModel
from cicada.plugins.fake_tools import EchoTool


def make_agent(script, tools=None, max_turns=50):
    model = FakeModel(script)
    tool_list = tools if tools is not None else [EchoTool()]
    tool_map = {t.spec.name: t for t in tool_list}
    return model, Agent(model=model, tools=tool_map, max_turns=max_turns)


class DetailTool:
    """返回带嵌套可变 details 的工具, 用于隔离复制验证."""

    def __init__(self) -> None:
        self.invocations = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="detail",
            description="返回嵌套 details",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        self.invocations += 1
        return ToolResult(
            call_id=ctx.call_id,
            name="detail",
            content="done",
            details={"nested": {"value": [1, 2]}},
        )


class BadIdentityTool:
    """返回 call_id/name 不符的 ToolResult."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="badid",
            description="返回错误身份",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(call_id="other-call", name="badid", content="wrong pairing")


class NonToolResultTool:
    """返回类型不是 ToolResult."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="weird",
            description="返回字符串",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict, ctx: ToolContext) -> object:
        return "not a tool result"


# --- T02: 显式历史与后缀返回 ----------------------------------------------------------------


async def test_history_seeds_model_request_and_system_prompt_stays_independent():
    model, agent = make_agent([[TextDelta("回答二"), StreamDone("stop")]], tools=[])
    history = (
        UserMessage(text="第一个问题"),
        AssistantMessage(text="第一个回答"),
    )
    result = await agent.run("第二个问题", history=history)
    assert result.stop_reason == "stop"
    request = model.requests[0]
    assert [m.text for m in request.messages] == ["第一个问题", "第一个回答", "第二个问题"]
    assert request.system_prompt == ""
    agent_with_prompt = Agent(model=model, tools={}, system_prompt="s")
    await agent_with_prompt.run("x", history=history)
    assert model.requests[1].system_prompt == "s"
    assert not any("s" == m.text for m in model.requests[1].messages)


async def test_runresult_messages_are_suffix_only():
    model, agent = make_agent([[TextDelta("ok"), StreamDone("stop")]], tools=[])
    history = (UserMessage(text="旧问题"), AssistantMessage(text="旧回答"))
    result = await agent.run("新问题", history=history)
    kinds = [type(m) for m in result.messages]
    assert kinds == [UserMessage, AssistantMessage]
    assert result.messages[0].text == "新问题"
    assert result.model_calls[0].run_id == result.run_id


async def test_second_request_carries_full_history_exactly_once():
    model, agent = make_agent(
        [
            [TextDelta("一"), StreamDone("stop")],
            [TextDelta("二"), StreamDone("stop")],
        ],
        tools=[],
    )
    history: tuple = ()
    first = await agent.run("问题一", history=history)
    history = history + first.messages
    second = await agent.run("问题二", history=history)
    # 请求时刻的快照: 第二请求携带一次且仅一次此前完整历史
    second_request_messages = model.requests[1].messages
    texts = [m.text for m in second_request_messages if isinstance(m, UserMessage)]
    assert texts == ["问题一", "问题二"]
    assert [m.text for m in second_request_messages if isinstance(m, AssistantMessage)] == ["一"]
    # 应用层累计一次后的完整历史: 每条消息只出现一次, 没有重复增长
    accumulated = history + second.messages
    accumulated_texts = [m.text for m in accumulated]
    assert accumulated_texts == ["问题一", "一", "问题二", "二"]
    assert len(set(accumulated_texts)) == 4


async def test_multi_turn_tool_loop_history_roundtrip():
    echo = EchoTool()
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("tool_use")],
            [TextDelta("继续"), StreamDone("stop")],
        ],
        tools=[echo],
    )
    history = (
        UserMessage(text="旧任务"),
        AssistantMessage(
            text="旧调用",
            tool_calls=(ToolCall(id="old-1", name="echo", arguments_json='{"text": "old"}'),),
        ),
        ToolResultMessage(result=ToolResult(call_id="old-1", name="echo", content="old")),
    )
    result = await agent.run("新任务", history=history)
    assert result.stop_reason == "stop"
    # 请求时刻快照: 3 历史 + 本次 user; 新 assistant/工具结果随后入会话
    request = model.requests[0]
    assert len(request.messages) == 4
    assert isinstance(request.messages[3], UserMessage)
    # 本轮后缀完整闭合, 后续请求可继续消费
    kinds = [type(m) for m in result.messages]
    assert kinds == [UserMessage, AssistantMessage, ToolResultMessage, AssistantMessage]
    assert result.messages[2].result.call_id == "c1"  # type: ignore[union-attr]


async def test_run_ids_stay_unique_across_history_runs():
    model, agent = make_agent([[StreamDone("stop")], [StreamDone("stop")]], tools=[])
    first = await agent.run("一")
    second = await agent.run("二", history=first.messages)
    assert first.run_id != second.run_id


# --- T03: 执行前拒绝与闭合保证 --------------------------------------------------------------


async def test_dangling_tool_result_rejected_before_any_effect():
    echo = EchoTool()
    model, agent = make_agent([[StreamDone("stop")]], tools=[echo])
    history = (
        UserMessage(text="q"),
        ToolResultMessage(result=ToolResult(call_id="ghost", name="echo", content="x")),
    )
    with pytest.raises(ValueError, match="tool result"):
        await agent.run("x", history=history)
    assert model.requests == []
    assert echo.invocations == []


async def test_unclosed_tool_calls_rejected():
    model, agent = make_agent([[StreamDone("stop")]], tools=[])
    history = (
        UserMessage(text="q"),
        AssistantMessage(
            text="a",
            tool_calls=(ToolCall(id="c1", name="echo", arguments_json="{}"),),
        ),
    )
    with pytest.raises(ValueError, match="unclosed"):
        await agent.run("x", history=history)
    assert model.requests == []


async def test_out_of_order_or_mismatched_results_rejected():
    model, agent = make_agent([[StreamDone("stop")]], tools=[])
    history = (
        UserMessage(text="q"),
        AssistantMessage(
            text="a",
            tool_calls=(ToolCall(id="c1", name="echo", arguments_json="{}"),),
        ),
        ToolResultMessage(result=ToolResult(call_id="c1", name="other", content="x")),
    )
    with pytest.raises(ValueError, match="does not match"):
        await agent.run("x", history=history)
    assert model.requests == []


async def test_duplicate_in_group_id_in_history_rejected():
    model, agent = make_agent([[StreamDone("stop")]], tools=[])
    history = (
        UserMessage(text="q"),
        AssistantMessage(
            text="a",
            tool_calls=(
                ToolCall(id="c1", name="echo", arguments_json="{}"),
                ToolCall(id="c1", name="echo", arguments_json="{}"),
            ),
        ),
        ToolResultMessage(result=ToolResult(call_id="c1", name="echo", content="x")),
        ToolResultMessage(result=ToolResult(call_id="c1", name="echo", content="x")),
    )
    with pytest.raises(ValueError, match="duplicate"):
        await agent.run("x", history=history)
    assert model.requests == []


async def test_unsupported_history_type_and_non_tuple_rejected():
    model, agent = make_agent([[StreamDone("stop")]], tools=[])
    with pytest.raises(ValueError, match="not a supported message type"):
        await agent.run("x", history=(UserMessage(text="q"), "not a message"))  # type: ignore[arg-item]
    with pytest.raises(ValueError, match="tuple"):
        await agent.run("x", history=[UserMessage(text="q")])  # type: ignore[arg-type]
    assert model.requests == []
    # 拒绝不占用活动 run 状态: 随后合法运行正常
    ok = await agent.run("x")
    assert ok.stop_reason == "stop"


async def test_model_error_closes_collected_calls_without_execution():
    echo = EchoTool()
    model, agent = make_agent(
        [[ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("error", "provider boom")]],
        tools=[echo],
    )
    events: list[Event] = []
    agent.subscribe(events.append)
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert result.error == "provider boom"
    assert echo.invocations == []
    assert not any(isinstance(e, ToolStarted) for e in events)
    closes = [e for e in events if isinstance(e, ToolCompleted)]
    assert len(closes) == 1
    assert closes[0].result.content == "model stream failed; tool call not executed"
    assert closes[0].result.is_error
    tail = result.messages[-1]
    assert isinstance(tail, ToolResultMessage)
    assert tail.result.call_id == "c1"


async def test_invalid_call_group_rejected_whole_before_history_and_execution():
    echo = EchoTool()
    model, agent = make_agent(
        [
            [
                ToolCallEvent("", "echo", '{"text": "hi"}'),
                ToolCallEvent("c2", "echo", '{"text": "hi"}'),
                StreamDone("tool_use"),
            ],
            [TextDelta("恢复"), StreamDone("stop")],
        ],
        tools=[echo],
    )
    events: list[Event] = []
    agent.subscribe(events.append)
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert "invalid identity" in (result.error or "")
    assert echo.invocations == []
    assert not any(isinstance(e, ToolStarted) for e in events)
    assistant = next(m for m in result.messages if isinstance(m, AssistantMessage))
    assert assistant.tool_calls == ()
    assert assistant.stop_reason == "error"
    # 已收集文本保留
    assert assistant.text == ""
    # 不改写原身份: 原始调用不进入任何可续写历史
    assert not any(
        isinstance(m, AssistantMessage) and m.tool_calls for m in result.messages
    )


async def test_duplicate_call_id_in_current_group_rejected():
    model, agent = make_agent(
        [
            [
                ToolCallEvent("dup", "echo", '{"text": "a"}'),
                ToolCallEvent("dup", "echo", '{"text": "b"}'),
                StreamDone("tool_use"),
            ],
        ],
        tools=[EchoTool()],
    )
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert "duplicate tool call id" in (result.error or "")


async def test_tool_result_identity_mismatch_closes_with_original_identity():
    model, agent = make_agent(
        [[ToolCallEvent("c1", "badid", "{}"), StreamDone("tool_use")], [StreamDone("stop")]],
        tools=[BadIdentityTool()],
    )
    result = await agent.run("x")
    tool_msg = result.messages[2]
    assert isinstance(tool_msg, ToolResultMessage)
    assert tool_msg.result.call_id == "c1" and tool_msg.result.name == "badid"
    assert tool_msg.result.content == "tool raised: invalid ToolResult identity or type"
    assert tool_msg.result.is_error
    # 已执行协议异常: 保守锁存语义依赖该前缀 (RunPolicy 既有标记覆盖)
    assert tool_msg.result.content.startswith("tool raised:")


async def test_non_toolresult_return_closes_with_original_identity():
    model, agent = make_agent(
        [[ToolCallEvent("c1", "weird", "{}"), StreamDone("tool_use")], [StreamDone("stop")]],
        tools=[NonToolResultTool()],
    )
    result = await agent.run("x")
    tool_msg = result.messages[2]
    assert isinstance(tool_msg, ToolResultMessage)
    assert tool_msg.result.call_id == "c1"
    assert "invalid ToolResult identity or type" in tool_msg.result.content


# --- 隔离复制 -------------------------------------------------------------------------------


async def test_history_seed_is_isolated_from_caller_mutable_data():
    model, agent = make_agent([[TextDelta("ok"), StreamDone("stop")]], tools=[])
    details = {"nested": {"value": [1, 2]}}
    history = (
        UserMessage(text="q"),
        AssistantMessage(
            text="a",
            tool_calls=(ToolCall(id="c1", name="echo", arguments_json="{}"),),
        ),
        ToolResultMessage(result=ToolResult(call_id="c1", name="echo", content="r", details=details)),
    )
    await agent.run("下一问", history=history)
    details["nested"]["value"].append(999)
    details["nested"]["new"] = True
    seeded = model.requests[0].messages[2]
    assert isinstance(seeded, ToolResultMessage)
    assert seeded.result.details == {"nested": {"value": [1, 2]}}  # type: ignore[union-attr]


async def test_subscriber_cannot_mutate_canonical_history_via_event():
    tool = DetailTool()
    model, agent = make_agent(
        [[ToolCallEvent("c1", "detail", "{}"), StreamDone("tool_use")], [StreamDone("stop")]],
        tools=[tool],
    )
    seen: list[Event] = []

    def mutate(event: Event) -> None:
        seen.append(event)
        if isinstance(event, ToolCompleted) and event.result.details is not None:
            event.result.details["nested"]["value"].append(777)
            event.result.details["hacked"] = True

    agent.subscribe(mutate)
    result = await agent.run("x")
    tool_msg = next(m for m in result.messages if isinstance(m, ToolResultMessage))
    assert tool_msg.result.details == {"nested": {"value": [1, 2]}}  # type: ignore[union-attr]


# --- 文本增量事件 ---------------------------------------------------------------------------


async def test_text_deltas_emitted_per_turn_and_completed_is_authoritative():
    model, agent = make_agent(
        [
            [TextDelta("你好"), TextDelta(""), TextDelta("世界"), ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("tool_use")],
            [TextDelta("完成"), StreamDone("stop")],
        ],
        tools=[EchoTool()],
    )
    events: list[Event] = []
    agent.subscribe(events.append)
    result = await agent.run("x")
    assert result.stop_reason == "stop"
    types = [type(e) for e in events]
    # 零字节增量被忽略
    assert types.count(AssistantTextDelta) == 3
    deltas = [e for e in events if isinstance(e, AssistantTextDelta)]
    assert [d.text for d in deltas] == ["你好", "世界", "完成"]
    assert [d.turn_index for d in deltas] == [0, 0, 1]
    # 顺序: RunStarted 开头, RunFinished 结尾; 增量先于同轮 AssistantCompleted
    assert types[0] is RunStarted and types[-1] is RunFinished
    positions = [i for i, e in enumerate(events) if isinstance(e, AssistantTextDelta)]
    assert types[positions[0] - 1] is TurnStarted
    # 同轮最后增量 ("世界") 之后即该轮 AssistantCompleted
    assert events[positions[1]].text == "世界"
    assert isinstance(events[positions[1] + 1], AssistantCompleted)
    # 工具事件先于第二轮增量, 第二轮增量先于其完成事件
    tool_started_pos = next(i for i, e in enumerate(events) if isinstance(e, ToolStarted))
    second_completed_pos = next(
        i for i, e in enumerate(events)
        if isinstance(e, AssistantCompleted) and e.message.text == "完成"
    )
    assert tool_started_pos < positions[2] < second_completed_pos


async def test_cancelled_run_emits_no_partial_deltas_after_abort():
    class AbortMidStreamModel:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
            self.requests.append(request)
            yield TextDelta("可见")
            await cancel.wait()
            yield StreamDone("aborted")

    model = AbortMidStreamModel()
    agent = Agent(model=model, tools={})
    events: list[Event] = []
    agent.subscribe(events.append)
    cancel = CancelToken()
    task = asyncio.create_task(agent.run("x", cancel))
    await asyncio.sleep(0.02)
    cancel.cancel()
    result = await task
    assert result.stop_reason == "aborted"
    deltas = [e for e in events if isinstance(e, AssistantTextDelta)]
    assert [d.text for d in deltas] == ["可见"]
    assert any(isinstance(e, RunFinished) for e in events)
    assert any(isinstance(e, ModelCallCompleted) for e in events)
