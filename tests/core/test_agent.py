import asyncio

import pytest

from cicada.core.agent import Agent
from cicada.core.cancel import CancelToken
from cicada.core.events import (
    AssistantCompleted,
    RunFinished,
    RunStarted,
    TurnStarted,
)
from cicada.core.messages import AssistantMessage, ToolResultMessage, UserMessage
from cicada.core.ports import StreamDone, TextDelta, ToolCallEvent
from cicada.plugins.fake_model import FakeModel
from cicada.plugins.fake_tools import EchoTool, FailTool, SlowTool


def make_agent(script, tools=None, max_turns=50):
    model = FakeModel(script)
    tool_list = tools if tools is not None else [EchoTool()]
    tool_map = {t.spec.name: t for t in tool_list}
    return model, Agent(model=model, tools=tool_map, max_turns=max_turns)


async def test_text_only_run_completes():
    model, agent = make_agent([[TextDelta("你"), TextDelta("好"), StreamDone("stop")]])
    events = []
    agent.subscribe(events.append)
    result = await agent.run("打招呼")
    assert result.stop_reason == "stop"
    assert [type(e) for e in events] == [RunStarted, TurnStarted, AssistantCompleted, RunFinished]
    user, assistant = result.messages
    assert isinstance(user, UserMessage) and user.text == "打招呼"
    assert isinstance(assistant, AssistantMessage) and assistant.text == "你好"
    assert len(model.requests) == 1


async def test_tool_call_executed_and_paired():
    echo = EchoTool()
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("tool_use")],
            [TextDelta("done"), StreamDone("stop")],
        ],
        tools=[echo],
    )
    result = await agent.run("调用工具")
    assert result.stop_reason == "stop"
    assert echo.invocations == [{"text": "hi"}]
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    tool_msg = result.messages[2]
    assert isinstance(tool_msg, ToolResultMessage)
    assert tool_msg.result.call_id == "c1"
    assert tool_msg.result.content == "hi"
    assert not tool_msg.result.is_error
    second_request = model.requests[1]
    assert any(isinstance(m, ToolResultMessage) for m in second_request.messages)


async def test_unknown_tool_returns_error_result_and_continues():
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "nope", "{}"), StreamDone("tool_use")],
            [StreamDone("stop")],
        ]
    )
    result = await agent.run("x")
    tool_msg = result.messages[2]
    assert tool_msg.result.is_error
    assert "unknown tool" in tool_msg.result.content
    assert result.stop_reason == "stop"


async def test_invalid_arguments_json_returns_error_result():
    echo = EchoTool()
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "echo", "{oops"), StreamDone("tool_use")],
            [StreamDone("stop")],
        ],
        tools=[echo],
    )
    result = await agent.run("x")
    assert echo.invocations == []
    assert result.messages[2].result.is_error


async def test_schema_violation_returns_error_result():
    echo = EchoTool()
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "echo", '{"text": 123}'), StreamDone("tool_use")],
            [StreamDone("stop")],
        ],
        tools=[echo],
    )
    result = await agent.run("x")
    assert echo.invocations == []
    assert result.messages[2].result.is_error
    assert "validation" in result.messages[2].result.content


async def test_tool_exception_becomes_error_result():
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "fail", "{}"), StreamDone("tool_use")],
            [TextDelta("recovered"), StreamDone("stop")],
        ],
        tools=[FailTool()],
    )
    result = await agent.run("x")
    assert result.messages[2].result.is_error
    assert "fail tool always raises" in result.messages[2].result.content
    assert result.stop_reason == "stop"


async def test_length_stop_reason_never_executes_tool_calls():
    echo = EchoTool()
    model, agent = make_agent(
        [[ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("length")]],
        tools=[echo],
    )
    result = await agent.run("x")
    assert echo.invocations == []
    assert result.stop_reason == "length"
    tool_msg = result.messages[2]
    assert tool_msg.result.is_error
    assert "truncated" in tool_msg.result.content


async def test_cancel_during_tool_execution():
    slow = SlowTool()
    echo = EchoTool()
    model, agent = make_agent(
        [
            [
                ToolCallEvent("c1", "slow", "{}"),
                ToolCallEvent("c2", "echo", '{"text": "never"}'),
                StreamDone("tool_use"),
            ],
        ],
        tools=[slow, echo],
    )
    cancel = CancelToken()
    task = asyncio.create_task(agent.run("x", cancel))
    await slow.started.wait()
    cancel.cancel()
    result = await task
    assert result.stop_reason == "aborted"
    assert echo.invocations == []
    results = [m.result for m in result.messages if isinstance(m, ToolResultMessage)]
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert all(r.is_error for r in results)


async def test_model_error_event_terminates_run():
    model, agent = make_agent([[StreamDone("error", "provider boom")]])
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert result.error == "provider boom"


async def test_model_stream_raising_becomes_error():
    model, agent = make_agent([RuntimeError("adapter exploded")])
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert "adapter exploded" in (result.error or "")


async def test_stream_without_terminal_event_is_error():
    model, agent = make_agent([[TextDelta("hanging")]])
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert "terminal" in (result.error or "")


async def test_max_turns_exceeded():
    entry = [ToolCallEvent("c1", "echo", '{"text": "x"}'), StreamDone("tool_use")]
    model, agent = make_agent([list(entry) for _ in range(5)], max_turns=3)
    result = await agent.run("x")
    assert result.stop_reason == "error"
    assert "max turns" in (result.error or "")


async def test_subscriber_error_does_not_break_run():
    model, agent = make_agent([[TextDelta("ok"), StreamDone("stop")]])

    def bad_handler(event):
        raise RuntimeError("ui boom")

    agent.subscribe(bad_handler)
    result = await agent.run("x")
    assert result.stop_reason == "stop"
    assert len(agent.subscriber_errors) > 0


async def test_second_run_rejected_while_active():
    slow = SlowTool()
    model, agent = make_agent(
        [
            [ToolCallEvent("c1", "slow", '{"timeout": 0.2}'), StreamDone("tool_use")],
            [StreamDone("stop")],
        ],
        tools=[slow],
    )
    task = asyncio.create_task(agent.run("first"))
    await slow.started.wait()
    with pytest.raises(RuntimeError, match="active run"):
        await agent.run("second")
    result = await task
    assert result.stop_reason == "stop"


async def test_agent_accepts_system_prompt_and_stores_it():
    model = FakeModel([[TextDelta("ok"), StreamDone("stop")]])
    tools = {"echo": EchoTool()}
    default_agent = Agent(model=model, tools=tools)
    assert default_agent._system_prompt == ""
    prompted = Agent(model=model, tools=tools, system_prompt="be terse")
    assert prompted._system_prompt == "be terse"


async def test_run_with_system_prompt_reaches_request():
    model = FakeModel([[TextDelta("ok"), StreamDone("stop")]])
    agent = Agent(model=model, tools={"echo": EchoTool()}, system_prompt="be terse")
    events = []
    agent.subscribe(events.append)
    result = await agent.run("x")
    assert result.stop_reason == "stop"
    assert result.model_calls == ()
    assert [type(e) for e in events] == [RunStarted, TurnStarted, AssistantCompleted, RunFinished]
    assert model.requests[0].system_prompt == "be terse"


async def test_empty_system_prompt_keeps_old_request_shape():
    model, agent = make_agent([[TextDelta("ok"), StreamDone("stop")]])
    result = await agent.run("x")
    assert result.stop_reason == "stop"
    assert model.requests[0].system_prompt == ""
    assert model.requests[0].messages[0].text == "x"


async def test_system_prompt_carried_in_every_turn_request():
    model = FakeModel(
        [
            [ToolCallEvent("c1", "echo", '{"text": "hi"}'), StreamDone("tool_use")],
            [TextDelta("done"), StreamDone("stop")],
        ]
    )
    agent = Agent(model=model, tools={"echo": EchoTool()}, system_prompt="be terse")
    result = await agent.run("调用工具")
    assert result.stop_reason == "stop"
    assert len(model.requests) == 2
    assert all(request.system_prompt == "be terse" for request in model.requests)


async def test_system_prompt_not_in_session_history():
    model = FakeModel([[TextDelta("ok"), StreamDone("stop")]])
    agent = Agent(model=model, tools={"echo": EchoTool()}, system_prompt="be terse")
    result = await agent.run("x")
    for message in result.messages:
        if isinstance(message, (UserMessage, AssistantMessage)):
            assert "be terse" not in message.text


async def test_system_prompt_rejects_non_str():
    model = FakeModel([[StreamDone("stop")]])
    with pytest.raises(TypeError, match="system_prompt"):
        Agent(model=model, tools={}, system_prompt=123)  # type: ignore[arg-type]
