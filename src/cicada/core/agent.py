"""Agent 主循环: stream -> 顺序派发 -> 结果配对 -> 下一轮, 直至终止.

显式续写: run(history=...) 接受已闭合的结构化历史 (执行前校验, 零副作用拒绝);
RunResult.messages 只含本次新增后缀; 系统提示独立于普通消息历史.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jsonschema

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
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    isolate_message,
    isolate_tool_result,
    validate_history_structure,
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
    model_calls: tuple[ModelCallCompleted, ...] = ()


class Agent:
    def __init__(
        self, *, model: ModelPort, tools: Mapping[str, Tool], max_turns: int = 50, system_prompt: str = ""
    ) -> None:
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be str, got {type(system_prompt).__name__}")
        self._model = model
        self._tools = dict(tools)
        self._max_turns = max_turns
        self._system_prompt = system_prompt
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

    async def run(
        self,
        prompt: str,
        cancel: CancelToken | None = None,
        *,
        history: tuple[Message, ...] = (),
    ) -> RunResult:
        if self._active:
            raise RuntimeError("agent already has an active run")
        # 历史校验先于活动标记: 非法历史是调用方编程错误, 不占用也不泄漏 run 状态
        seed = self._validate_history(history)
        self._active = True
        try:
            return await self._run(prompt, cancel or CancelToken(), seed)
        finally:
            self._active = False

    def _validate_history(self, history: tuple[Message, ...]) -> tuple[Message, ...]:
        """执行前校验结构化历史: 类型、组内身份、闭合与次序; 违规 ValueError.

        校验只读不复制; 零模型请求、零工具执行、零会话追加的拒绝在种子建立前完成.
        结构规则与持久化恢复共用 validate_history_structure, 保持同样拒绝内容.
        """
        validate_history_structure(history)
        return history

    async def _run(self, prompt: str, cancel: CancelToken, seed: tuple[Message, ...]) -> RunResult:
        run_id = f"run-{next(self._run_ids)}"
        # 隔离调用者可变数据: 内核私有记录与调用方传入对象不共享嵌套 dict
        session = SessionState(isolate_message(message) for message in seed)
        base_len = len(seed)
        session.append(UserMessage(text=prompt))
        self._emit(RunStarted(run_id))

        stop_reason: StopReason = "stop"
        error: str | None = None
        model_calls: list[ModelCallCompleted] = []

        for turn_index in range(self._max_turns):
            if cancel.cancelled:
                stop_reason = "aborted"
                break
            self._emit(TurnStarted(run_id, turn_index))
            started = time.monotonic()
            done, text, calls = await self._collect_assistant(session, cancel, run_id, turn_index)
            elapsed_s = time.monotonic() - started

            if cancel.cancelled and done.stop_reason != "error":
                # 终结行已被消费后取消仍判 aborted, 但保留已取得的计量
                done = StreamDone("aborted", metrics=done.metrics)

            # 当前调用组身份约束: 非法或重复整组拒绝, 在 assistant 入历史与任何
            # 工具执行之前; 不改写原 ID, 不执行其中任何工具
            group_error = _validate_call_group(calls)

            call_record = ModelCallCompleted(
                run_id=run_id,
                turn_index=turn_index,
                stop_reason=done.stop_reason,
                metrics=done.metrics,
                elapsed_s=elapsed_s,
            )

            if group_error is not None:
                assistant = AssistantMessage(
                    text=text,
                    tool_calls=(),
                    stop_reason="error",
                    error=group_error,
                )
                session.append(assistant)
                self._emit(AssistantCompleted(run_id, assistant))
                model_calls.append(call_record)
                self._emit(call_record)
                stop_reason, error = "error", group_error
                break

            assistant = AssistantMessage(
                text=text,
                tool_calls=tuple(calls),
                stop_reason=done.stop_reason,
                error=done.error,
            )
            session.append(assistant)
            self._emit(AssistantCompleted(run_id, assistant))
            model_calls.append(call_record)
            self._emit(call_record)

            if done.stop_reason == "error":
                # 模型错误前已收集完整调用组: 为尚未执行的调用生成错误结果,
                # 不留下悬空调用; 进程未启动, 属执行前哨兵而非终结未知
                self._finish_pending_calls(
                    session, run_id, calls, "model stream failed; tool call not executed"
                )
                stop_reason, error = "error", done.error
                break
            if done.stop_reason == "aborted":
                self._finish_pending_calls(session, run_id, calls, "cancelled before execution")
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
        return RunResult(
            run_id, stop_reason, error, tuple(session.messages[base_len:]), model_calls=tuple(model_calls)
        )

    async def _collect_assistant(
        self, session: SessionState, cancel: CancelToken, run_id: str, turn_index: int
    ) -> tuple[StreamDone, str, list[ToolCall]]:
        request = ModelRequest(
            messages=session.messages,
            tools=tuple(tool.spec for tool in self._tools.values()),
            system_prompt=self._system_prompt,
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        try:
            async for event in self._model.stream(request, cancel):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    if event.text:
                        # 只读文本增量: 零字节可忽略; 完成事件仍是权威
                        self._emit(AssistantTextDelta(run_id=run_id, turn_index=turn_index, text=event.text))
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
            result = await tool.execute(arguments, ToolContext(call_id=call.id, cancel=cancel))
        except asyncio.CancelledError:
            return ToolResult(call_id=call.id, name=call.name, content="cancelled during execution", is_error=True)
        except Exception as exc:
            return ToolResult(call_id=call.id, name=call.name, content=f"tool raised: {exc}", is_error=True)
        if (
            not isinstance(result, ToolResult)
            or result.call_id != call.id
            or result.name != call.name
        ):
            # 以原调用身份闭合: 不把不可信返回的 details 当作终结证据;
            # "tool raised:" 前缀沿用 RunPolicy 的已执行协议异常保守锁存
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content="tool raised: invalid ToolResult identity or type",
                is_error=True,
            )
        return result

    def _finish_call(self, session: SessionState, run_id: str, result: ToolResult) -> None:
        session.append(ToolResultMessage(result=result))
        # 订阅者拿到隔离副本: 修改 details 不得穿透规范历史
        self._emit(ToolCompleted(run_id=run_id, result=isolate_tool_result(result)))

    def _finish_pending_calls(
        self, session: SessionState, run_id: str, calls: list[ToolCall], content: str
    ) -> None:
        """为尚未执行的调用生成错误结果并发出 ToolCompleted, 保证历史闭合."""
        for call in calls:
            self._finish_call(
                session,
                run_id,
                ToolResult(call_id=call.id, name=call.name, content=content, is_error=True),
            )

    def _emit(self, event: Event) -> None:
        for handler in list(self._subscribers):
            try:
                handler(event)
            except Exception as exc:  # 订阅者错误隔离: 不破坏主循环
                self.subscriber_errors.append(exc)


def _validate_call_group(calls: list[ToolCall]) -> str | None:
    """当前模型响应的调用组身份校验; 返回拒绝原因或 None.

    非法身份/重复 ID 在 assistant 入历史及任何工具执行前整组拒绝;
    非法参数 JSON 仍按既有逐调用错误结果处理, 不在本校验范围.
    """
    ids: set[str] = set()
    for call in calls:
        if (
            not isinstance(call, ToolCall)
            or not isinstance(call.id, str)
            or not call.id
            or not isinstance(call.name, str)
            or not call.name
        ):
            return "model produced a tool call with invalid identity (id and name must be nonempty strings)"
        if not isinstance(call.arguments_json, str):
            return "model produced a tool call with non-string arguments_json"
        if call.id in ids:
            return f"model produced duplicate tool call id {call.id!r} within one turn"
        ids.add(call.id)
    return None
