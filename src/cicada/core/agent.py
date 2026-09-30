"""Agent 主循环: stream -> 顺序派发 -> 结果配对 -> 下一轮, 直至终止."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jsonschema

from cicada.core.cancel import CancelToken
from cicada.core.events import (
    AssistantCompleted,
    Event,
    RunFinished,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import (
    ModelPort,
    ModelRequest,
    StreamDone,
    TextDelta,
    ToolCallEvent,
    Tool,
    ToolContext,
)
from cicada.core.session import SessionState


@dataclass(frozen=True)
class RunResult:
    run_id: str
    stop_reason: StopReason
    error: str | None
    messages: tuple[Message, ...]


class Agent:
    def __init__(self, *, model: ModelPort, tools: Mapping[str, Tool], max_turns: int = 50) -> None:
        self._model = model
        self._tools = dict(tools)
        self._max_turns = max_turns
        self._subscribers: list[Callable[[Event], None]] = []
        self.subscriber_errors: list[BaseException] = []
        self._run_ids = itertools.count(1)
        self._active = False

    def subscribe(self, handler: Callable[[Event], None]) -> Callable[[], None]:
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

        return unsubscribe

    async def run(self, prompt: str, cancel: CancelToken | None = None) -> RunResult:
        if self._active:
            raise RuntimeError("agent already has an active run")
        self._active = True
        try:
            return await self._run(prompt, cancel or CancelToken())
        finally:
            self._active = False

    async def _run(self, prompt: str, cancel: CancelToken) -> RunResult:
        run_id = f"run-{next(self._run_ids)}"
        session = SessionState()
        session.append(UserMessage(text=prompt))
        self._emit(RunStarted(run_id))

        stop_reason: StopReason = "stop"
        error: str | None = None

        for turn_index in range(self._max_turns):
            if cancel.cancelled:
                stop_reason = "aborted"
                break
            self._emit(TurnStarted(run_id, turn_index))
            done, text, calls = await self._collect_assistant(session, cancel)

            if cancel.cancelled and done.stop_reason != "error":
                done = StreamDone("aborted")

            assistant = AssistantMessage(
                text=text,
                tool_calls=tuple(calls),
                stop_reason=done.stop_reason,
                error=done.error,
            )
            session.append(assistant)
            self._emit(AssistantCompleted(run_id, assistant))

            if done.stop_reason == "error":
                stop_reason, error = "error", done.error
                break
            if done.stop_reason == "aborted":
                self._cancel_pending_calls(session, run_id, calls)
                stop_reason = "aborted"
                break
            if done.stop_reason == "length":
                # 截断输出的 tool call 一律不执行, 全部回错误结果 (防"半截意图"落地)
                for call in calls:
                    self._finish_call(
                        session,
                        run_id,
                        ToolResult(
                            call_id=call.id,
                            name=call.name,
                            content="model output truncated; tool call not executed",
                            is_error=True,
                        ),
                    )
                stop_reason = "length"
                break
            if not calls:
                if done.stop_reason == "tool_use":
                    stop_reason, error = "error", "model signalled tool_use without any tool call"
                break

            for call in calls:
                result = await self._dispatch(call, run_id, cancel)
                self._finish_call(session, run_id, result)
            if cancel.cancelled:
                stop_reason = "aborted"
                break
        else:
            stop_reason, error = "error", f"max turns ({self._max_turns}) exceeded"

        self._emit(RunFinished(run_id, stop_reason, error))
        return RunResult(run_id, stop_reason, error, session.messages)

    async def _collect_assistant(
        self, session: SessionState, cancel: CancelToken
    ) -> tuple[StreamDone, str, list[ToolCall]]:
        request = ModelRequest(
            messages=session.messages,
            tools=tuple(tool.spec for tool in self._tools.values()),
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        try:
            async for event in self._model.stream(request, cancel):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ToolCallEvent):
                    calls.append(ToolCall(event.id, event.name, event.arguments_json))
                elif isinstance(event, StreamDone):
                    return event, "".join(text_parts), calls
        except asyncio.CancelledError:
            return StreamDone("aborted"), "".join(text_parts), calls
        except Exception as exc:
            return StreamDone("error", f"model stream raised: {exc}"), "".join(text_parts), calls
        return StreamDone("error", "model stream ended without terminal event"), "".join(text_parts), calls

    async def _dispatch(self, call: ToolCall, run_id: str, cancel: CancelToken) -> ToolResult:
        if cancel.cancelled:
            return ToolResult(call_id=call.id, name=call.name, content="cancelled before execution", is_error=True)
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call_id=call.id, name=call.name, content=f"unknown tool: {call.name}", is_error=True)
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError as exc:
            return ToolResult(call_id=call.id, name=call.name, content=f"invalid arguments JSON: {exc}", is_error=True)
        if not isinstance(arguments, dict):
            return ToolResult(call_id=call.id, name=call.name, content="arguments must be a JSON object", is_error=True)
        try:
            jsonschema.validate(arguments, tool.spec.parameters)
        except jsonschema.ValidationError as exc:
            return ToolResult(call_id=call.id, name=call.name, content=f"arguments failed validation: {exc.message}", is_error=True)
        self._emit(ToolStarted(run_id, call.id, call.name))
        try:
            return await tool.execute(arguments, ToolContext(call_id=call.id, cancel=cancel))
        except asyncio.CancelledError:
            return ToolResult(call_id=call.id, name=call.name, content="cancelled during execution", is_error=True)
        except Exception as exc:
            return ToolResult(call_id=call.id, name=call.name, content=f"tool raised: {exc}", is_error=True)

    def _finish_call(self, session: SessionState, run_id: str, result: ToolResult) -> None:
        session.append(ToolResultMessage(result=result))
        self._emit(ToolCompleted(run_id, result))

    def _cancel_pending_calls(self, session: SessionState, run_id: str, calls: list[ToolCall]) -> None:
        for call in calls:
            self._finish_call(
                session,
                run_id,
                ToolResult(call_id=call.id, name=call.name, content="cancelled before execution", is_error=True),
            )

    def _emit(self, event: Event) -> None:
        for handler in list(self._subscribers):
            try:
                handler(event)
            except Exception as exc:  # 订阅者错误隔离: 不破坏主循环
                self.subscriber_errors.append(exc)
