"""进程活动包装器: 对 powershell 与 check(action=run) 可靠登记开始/终结 (契约 §4).

只服务恢复时的进程判定, 不构成中断请求的模型历史 checkpoint. 包装器保持原工具
声明与调用配对: spec 原样透传, 结果 (含 details/identity) 不改写.

外层 Task.cancel 与 CancelToken 协作取消分别处理: 等待已派发事务时遭取消, 先取消
token, 再继续等待受拥有 completion 的明确结果 (重复取消也不丢弃); 不把
CancelledError 传播给 core._dispatch 替换已取得的结果. 命令零执行返回既有
"cancelled before execution" 哨兵且不锁存 process_uncertain; 写入失败另由应用
锁存 storage_failed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from cicada.core.messages import ToolResult
from cicada.core.ports import Tool, ToolContext
from cicada.persistence.contracts import (
    PersistenceError,
    ProcessTerminalFacts,
    SessionStorePort,
)
from cicada.run_policy import LAUNCH_FAILURE_MARKERS, PRE_EXECUTION_MARKERS
from cicada.persistence.routing import SessionStoreRoute

# powershell 全部登记; check 仅 action=run (status 查询不登记为进程启动, 契约 §4.2);
# read/edit/write/glob/grep 不是进程工具.
ACTIVITY_TOOL_NAMES = frozenset({"powershell", "check"})

_REASON_MAX_CHARS = 200


def should_register_activity(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    """按工具身份与参数判定是否登记进程活动; 未知工具不登记 (内核已先行校验)."""
    if tool_name == "powershell":
        return True
    if tool_name == "check":
        return arguments.get("action") == "run"
    return False


def classify_terminal(result: ToolResult) -> ProcessTerminalFacts:
    """按既有 P4 定义把真实 ToolResult 分类为终结事实 (契约 §4.4).

    只读结构化 details 字段与程序哨兵前缀, 不解析任意输出文字; check 的
    receipt_id 一并记录, 03 用真实账本核对其终结事实.
    """
    content = result.content or ""
    if content.startswith(PRE_EXECUTION_MARKERS) or content.startswith(LAUNCH_FAILURE_MARKERS):
        return ProcessTerminalFacts(kind="not_started", reason=content[:_REASON_MAX_CHARS])
    details = result.details if isinstance(result.details, dict) else None
    timed_out = details is not None and details.get("timed_out") is True
    cancelled = details is not None and details.get("cancelled") is True
    raw_exit = details.get("exit_code") if details is not None else None
    exit_code = raw_exit if isinstance(raw_exit, int) and not isinstance(raw_exit, bool) else None
    raw_complete = details.get("output_complete") if details is not None else None
    output_complete = raw_complete if isinstance(raw_complete, bool) else None
    raw_receipt = details.get("receipt_id") if details is not None else None
    receipt_id = raw_receipt if isinstance(raw_receipt, str) and raw_receipt else None
    if exit_code is not None and output_complete is True and not timed_out and not cancelled:
        return ProcessTerminalFacts(
            kind="proven_terminal", exit_code=exit_code, output_complete=True,
            receipt_id=receipt_id,
        )
    if timed_out:
        reason = "timed out before EOF"
    elif cancelled:
        reason = "cancelled; process termination unproven"
    elif output_complete is False:
        reason = "ended without reaching EOF"
    elif details is None:
        reason = "no structured terminal details"
    else:
        reason = "no trustworthy terminal fact"
    return ProcessTerminalFacts(
        kind="unproven_terminal", exit_code=exit_code, output_complete=output_complete,
        timed_out=timed_out, cancelled=cancelled, receipt_id=receipt_id, reason=reason,
    )


class ActivitySink(Protocol):
    """包装器对应用会话的窄接缝; ApplicationSession 结构化满足."""

    @property
    def active_request_id(self) -> str | None: ...

    def latch_storage_failure(self, detail: str) -> None: ...

    def record_check_receipt(self, receipt_id: str) -> None: ...


class ActivityCoordinator:
    """bootstrap 前创建、会话构造后绑定; 包装器经它取请求身份并锁存存储故障."""

    def __init__(self) -> None:
        self._sink: ActivitySink | None = None

    def bind(self, sink: ActivitySink) -> None:
        self._sink = sink

    def current_request_id(self) -> str | None:
        return self._sink.active_request_id if self._sink is not None else None

    def latch_storage_failure(self, detail: str) -> None:
        if self._sink is not None:
            self._sink.latch_storage_failure(detail)

    def record_check_receipt(self, result: ToolResult) -> None:
        if result.name == "check" and isinstance(result.details, dict):
            receipt_id = result.details.get("receipt_id")
            if isinstance(receipt_id, str) and receipt_id and self._sink is not None:
                self._sink.record_check_receipt(receipt_id)


async def _settle(awaitable) -> Any:
    """等待受拥有 completion 的明确结果; 重复外层取消不丢弃 completion (契约 §4)."""
    task = asyncio.ensure_future(awaitable)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            continue


class ProcessActivityWrapper:
    """Tool 装饰器: 登记前零命令执行, 登记后保留真实结果; 失败锁存交协调器."""

    def __init__(self, inner: Tool, store: SessionStorePort, coordinator: ActivityCoordinator) -> None:
        self._inner = inner
        self._store = store
        self._coordinator = coordinator

    @property
    def spec(self) -> Any:
        return self._inner.spec  # 原声明保持: 模型可见面与指纹不变

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_name = self._inner.spec.name
        if not should_register_activity(tool_name, arguments):
            return await self._inner.execute(arguments, ctx)
        request_id = self._coordinator.current_request_id()
        if request_id is None:
            # 防御: 包装器只在持久装配中出现; 非活动请求内不登记但保持工具可用
            return await self._inner.execute(arguments, ctx)
        store = self._store.capture() if isinstance(self._store, SessionStoreRoute) else self._store
        try:
            activity_id, cancelled = await self._begin(store, request_id, tool_name, ctx, arguments)
        except PersistenceError as exc:
            # 开始写入失败: 命令零执行, 锁存 storage_failed, 返回既有未启动哨兵;
            # 不锁存 process_uncertain (契约 §4.3)
            self._coordinator.latch_storage_failure(f"process activity begin failed: {exc}")
            ctx.cancel.cancel()
            return self._not_started_result(ctx, tool_name)
        if cancelled:
            # 开始写入成功后的外层取消: 写可靠的未启动终结, 不传播 CancelledError
            try:
                await _settle(store.end_process(activity_id, ProcessTerminalFacts(
                    kind="not_started", reason="outer cancellation before execution",
                )))
            except PersistenceError as exc:
                self._coordinator.latch_storage_failure(f"process activity end failed: {exc}")
            return self._not_started_result(ctx, tool_name)
        try:
            result = await self._inner.execute(arguments, ctx)
        except BaseException as exc:
            # 异常/不可得结果: 保留已登记活动为未证终结, 再交还内核协议 (契约 §4.5)
            try:
                await _settle(store.end_process(activity_id, ProcessTerminalFacts(
                    kind="unproven_terminal", reason=f"tool raised {type(exc).__name__}",
                )))
            except PersistenceError as persistence_exc:
                self._coordinator.latch_storage_failure(
                    f"process activity end failed: {persistence_exc}"
                )
            raise
        try:
            self._coordinator.record_check_receipt(result)
            await _settle(store.end_process(activity_id, classify_terminal(result)))
        except PersistenceError as exc:
            # 终结写入失败: 保留真实 ToolResult, 取消 token 停止后继活动;
            # 不得把已执行命令改写成未执行 (契约 §4.5)
            self._coordinator.latch_storage_failure(f"process activity end failed: {exc}")
            ctx.cancel.cancel()
        return result

    async def _begin(
        self, store: SessionStorePort, request_id: str, tool_name: str, ctx: ToolContext, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        task = asyncio.ensure_future(store.begin_process(
            request_id, tool_name=tool_name, call_id=ctx.call_id, arguments=dict(arguments),
        ))
        cancelled = False
        while True:
            try:
                activity_id = await asyncio.shield(task)
                return activity_id, cancelled
            except asyncio.CancelledError:
                # 立即取消 token, 但已派发事务仍须得到明确结果 (契约 §4)
                cancelled = True
                ctx.cancel.cancel()

    @staticmethod
    def _not_started_result(ctx: ToolContext, tool_name: str) -> ToolResult:
        return ToolResult(
            call_id=ctx.call_id, name=tool_name,
            content="cancelled before execution", is_error=True, details=None,
        )
