"""进程活动包装器聚焦测试: 登记/分类/取消硬化/失败锁存 (02 契约 §4).

真实 SQLiteSessionStore + 受控内层工具; 故障经 monkeypatch store 方法注入;
外层 Task.cancel 与 CancelToken 分别驱动, 覆盖重复取消.
"""

from __future__ import annotations

import asyncio

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.persistence.contracts import (
    PersistenceError,
    ProcessTerminalFacts,
)
from cicada.plugins.session_store import open_session_store
from cicada.process_activity import (
    ActivityCoordinator,
    ProcessActivityWrapper,
    classify_terminal,
    should_register_activity,
)


class ScriptedTool:
    """受控内层工具: 可编程结果、异常与执行闸门."""

    def __init__(self, name: str = "powershell", *, result: ToolResult | None = None,
                 error: BaseException | None = None, gate: asyncio.Event | None = None):
        self._name = name
        self._result = result
        self._error = error
        self._gate = gate
        self.calls: list[dict] = []
        self.spec = ToolSpec(name=name, description=f"{name} test tool", parameters={
            "type": "object", "properties": {},
        })

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append({"arguments": dict(arguments), "call_id": ctx.call_id})
        if self._gate is not None:
            await self._gate.wait()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


_PROVEN_RESULT = ToolResult(
    call_id="c1", name="powershell",
    content="[exit_code=0 timed_out=false cancelled=false output_complete=true]",
    is_error=False,
    details={"exit_code": 0, "timed_out": False, "cancelled": False, "output_complete": True},
)


def _store(tmp_path):
    return open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)


async def _begin_request(store, prompt="请求"):
    epoch = await store.start_epoch(_stored_identity())
    request_id = await store.begin_request(epoch, prompt)
    return epoch, request_id


def _stored_identity():
    from cicada.persistence.contracts import LaunchIdentity, StoredIdentity

    return StoredIdentity(
        launch=LaunchIdentity(
            provider="fake", model="fake", base_url=None, think=None, options={},
            max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
        ),
        tool_fingerprint="fp",
    )


def _wrap(tmp_path, inner: ScriptedTool, *, store=None):
    store = store or _store(tmp_path)
    coordinator = ActivityCoordinator()
    wrapper = ProcessActivityWrapper(inner, store, coordinator)
    return store, coordinator, wrapper


def test_should_register_activity_matrix():
    assert should_register_activity("powershell", {})
    assert not should_register_activity("check", {"action": "status"})
    assert should_register_activity("check", {"action": "run", "check_id": "check-1"})
    assert not should_register_activity("read", {"path": "x"})
    assert not should_register_activity("glob", {"pattern": "*"})
    assert not should_register_activity("unknown", {})


def test_classify_terminal_matches_p4_definitions():
    proven = classify_terminal(_PROVEN_RESULT)
    assert proven == ProcessTerminalFacts(kind="proven_terminal", exit_code=0, output_complete=True)

    launch_failed = classify_terminal(ToolResult("c1", "powershell", "pwsh executable not found on PATH", is_error=True))
    assert launch_failed.kind == "not_started"

    # 哨兵形态 (零执行路径的实际产物) 必须归 not_started: 它与 run_policy 的
    # PRE_EXECUTION_MARKERS 同文本, 是「不误锁存 process_uncertain」组合链的本地钉子
    sentinel = classify_terminal(ToolResult("c1", "powershell", "cancelled before execution", is_error=True))
    assert sentinel.kind == "not_started"
    from cicada.run_policy import PRE_EXECUTION_MARKERS

    assert "cancelled before execution" in PRE_EXECUTION_MARKERS

    timed_out = classify_terminal(ToolResult(
        "c1", "powershell", "[exit_code=-1 timed_out=true]", is_error=True,
        details={"exit_code": None, "timed_out": True, "cancelled": False, "output_complete": False},
    ))
    assert timed_out.kind == "unproven_terminal" and timed_out.timed_out

    no_details = classify_terminal(ToolResult("c1", "powershell", "tool raised: boom", is_error=True))
    assert no_details.kind == "unproven_terminal" and no_details.reason is not None

    no_eof = classify_terminal(ToolResult(
        "c1", "powershell", "partial", is_error=True,
        details={"exit_code": 0, "timed_out": False, "cancelled": False, "output_complete": False},
    ))
    assert no_eof.kind == "unproven_terminal" and no_eof.output_complete is False

    cancelled = classify_terminal(ToolResult(
        "c1", "powershell", "cancelled", is_error=True,
        details={"exit_code": None, "timed_out": False, "cancelled": True},
    ))
    assert cancelled.kind == "unproven_terminal" and cancelled.cancelled


async def test_wrapper_registers_begin_and_end_and_keeps_result(tmp_path):
    inner = ScriptedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    token = CancelToken()
    result = await wrapper.execute({"command": "exit 0"}, ToolContext(call_id="c1", cancel=token))
    assert result is _PROVEN_RESULT, "真实结果原样返回 (含 details/identity)"
    assert inner.calls and token.cancelled is False
    activities = store.recovery_snapshot().process_activities
    assert len(activities) == 1
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "proven_terminal"
    assert activities[0].request_id == request_id
    assert activities[0].call_id == "c1"
    await store.aclose()


async def test_non_registered_tools_pass_through_without_activity(tmp_path):
    inner = ScriptedTool(name="read", result=ToolResult("c1", "read", "body", is_error=False))
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    result = await wrapper.execute({"path": "x"}, ToolContext(call_id="c1", cancel=CancelToken()))
    assert result.is_error is False
    assert store.recovery_snapshot().process_activities == ()
    # check action=status 同样不登记
    check = ScriptedTool(name="check", result=ToolResult("c2", "check", "status", is_error=False))
    store2, coordinator2, wrapper2 = _wrap(tmp_path, check)
    epoch2, request2 = await _begin_request(store2)
    coordinator2.bind(_Sink(request2))
    await wrapper2.execute({"action": "status"}, ToolContext(call_id="c2", cancel=CancelToken()))
    assert store2.recovery_snapshot().process_activities == ()
    await store.aclose()
    await store2.aclose()


async def test_begin_failure_zero_execution_latch_and_sentinel(tmp_path, monkeypatch):
    inner = ScriptedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    sink = _Sink(request_id)
    coordinator.bind(sink)

    async def failing_begin(request_id, *, tool_name, call_id, arguments):
        raise PersistenceError("disk gone")

    monkeypatch.setattr(store, "begin_process", failing_begin)
    token = CancelToken()
    result = await wrapper.execute({"command": "x"}, ToolContext(call_id="c1", cancel=token))
    assert result.call_id == "c1" and result.name == "powershell"
    assert result.content == "cancelled before execution"
    assert result.is_error and result.details is None
    assert inner.calls == [], "开始写入失败必须零命令执行"
    assert token.cancelled, "取消 token 停止后继活动"
    assert sink.failures and "begin failed" in sink.failures[0]
    # 恒真防护 (包装器无 uncertain 锁存路径): 「不误锁存」的真实证明是哨兵文本
    # 属 PRE_EXECUTION_MARKERS (见 classify 测试) 且 test_run_policy c5 证明该形态
    # 不进终结扫描; 此断言只防未来错误引入锁存调用
    assert not sink.uncertain, "命令零执行不得误锁存 process_uncertain"
    await store.aclose()


async def test_outer_cancel_during_begin_wait_writes_not_started_and_returns_sentinel(tmp_path):
    """A04: begin 等待时外层取消 -> 事务明确结果 + 未启动终结 + 既有哨兵; 重复取消不丢."""
    inner = ScriptedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    sink = _Sink(request_id)
    coordinator.bind(sink)

    original_begin = store.begin_process
    release = asyncio.Event()

    async def gated_begin(request_id, *, tool_name, call_id, arguments):
        await release.wait()
        return await original_begin(request_id, tool_name=tool_name, call_id=call_id, arguments=arguments)

    store.begin_process = gated_begin  # type: ignore[method-assign]
    token = CancelToken()
    task = asyncio.create_task(wrapper.execute(
        {"command": "x"}, ToolContext(call_id="c1", cancel=token),
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()  # 重复取消: 不丢弃 completion
    await asyncio.sleep(0.05)
    release.set()
    result = await asyncio.wait_for(task, timeout=10)
    assert result.content == "cancelled before execution" and result.details is None
    assert inner.calls == [], "零命令执行"
    assert token.cancelled
    # 同上: 恒真防护, 真实证明在 classify 测试与 test_run_policy 的哨兵形态用例
    assert not sink.uncertain, "取消不等同回滚, 不误锁存 process_uncertain"
    activities = store.recovery_snapshot().process_activities
    assert len(activities) == 1
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "not_started", "开始写入成功后写可靠未启动终结"
    await store.aclose()


async def test_token_cancel_during_begin_wait_inner_sees_cancelled_token(tmp_path):
    """CancelToken 协作取消与外层 Task.cancel 分开: begin 成功后内层观察已取消 token,
    返回真实 (取消形态) 结果 -> 未证终结, 不用哨兵覆盖."""
    cancelled_result = ToolResult(
        call_id="c1", name="powershell", content="cancelled", is_error=True,
        details={"exit_code": None, "timed_out": False, "cancelled": True},
    )

    class TokenObservingTool(ScriptedTool):
        async def execute(self, arguments, ctx):
            self.calls.append({"arguments": dict(arguments), "call_id": ctx.call_id})
            while not ctx.cancel.cancelled:
                await asyncio.sleep(0.01)
            return cancelled_result

    inner = TokenObservingTool(result=cancelled_result)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    token = CancelToken()
    task = asyncio.create_task(wrapper.execute(
        {"command": "x"}, ToolContext(call_id="c1", cancel=token),
    ))
    await asyncio.sleep(0.05)
    token.cancel()  # 协作取消: 内层观察到 token 后返回取消形态结果
    result = await asyncio.wait_for(task, timeout=10)
    assert result is cancelled_result, "内层真实结果原样返回"
    assert len(inner.calls) == 1
    activities = store.recovery_snapshot().process_activities
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "unproven_terminal"
    assert activities[0].terminal.cancelled is True
    await store.aclose()


async def test_outer_cancel_during_inner_execution_records_unproven_and_propagates(tmp_path):
    """A04: 取消打在真实执行等待期 —— CancelledError 原样传播, 活动先登记未证终结.

    内核随后把它归一为 "cancelled during execution" (core/agent.py); 本测试钉住
    包装器在传播前已写 unproven 终结且重复取消不丢登记。
    """
    gate = asyncio.Event()

    class GatedTool(ScriptedTool):
        async def execute(self, arguments, ctx):
            self.calls.append({"arguments": dict(arguments), "call_id": ctx.call_id})
            await gate.wait()
            return _PROVEN_RESULT

    inner = GatedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    token = CancelToken()
    task = asyncio.create_task(wrapper.execute(
        {"command": "x"}, ToolContext(call_id="c1", cancel=token),
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.03)
    task.cancel()  # 重复取消: 不丢活动登记
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
    assert len(inner.calls) == 1, "命令确实进入过执行"
    activities = store.recovery_snapshot().process_activities
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "unproven_terminal"
    assert "CancelledError" in (activities[0].terminal.reason or "")
    await store.aclose()


async def test_end_failure_keeps_real_result_latches_and_cancels_token(tmp_path, monkeypatch):
    inner = ScriptedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    sink = _Sink(request_id)
    coordinator.bind(sink)

    async def failing_end(activity_id, terminal):
        raise PersistenceError("disk gone")

    monkeypatch.setattr(store, "end_process", failing_end)
    token = CancelToken()
    result = await wrapper.execute({"command": "x"}, ToolContext(call_id="c1", cancel=token))
    assert result is _PROVEN_RESULT, "终结写入失败保留真实工具结果, 不得改写成未执行"
    assert token.cancelled, "取消 token 停止后继活动"
    assert sink.failures and "end failed" in sink.failures[0]
    await store.aclose()


async def test_outer_cancel_during_end_wait_returns_real_result(tmp_path):
    """A04: end 等待时外层取消 -> 保留原 output/details/identity, 不传播 CancelledError."""
    inner = ScriptedTool(result=_PROVEN_RESULT)
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    sink = _Sink(request_id)
    coordinator.bind(sink)

    original_end = store.end_process
    release = asyncio.Event()

    async def gated_end(activity_id, terminal):
        await release.wait()
        return await original_end(activity_id, terminal)

    store.end_process = gated_end  # type: ignore[method-assign]
    token = CancelToken()
    task = asyncio.create_task(wrapper.execute(
        {"command": "x"}, ToolContext(call_id="c1", cancel=token),
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    release.set()
    result = await asyncio.wait_for(task, timeout=10)
    assert result is _PROVEN_RESULT, "已取得结果不被取消替换"
    assert store.recovery_snapshot().process_activities[0].terminal is not None
    await store.aclose()


async def test_inner_exception_records_unproven_and_reraises(tmp_path):
    inner = ScriptedTool(error=RuntimeError("boom"))
    store, coordinator, wrapper = _wrap(tmp_path, inner)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    with pytest.raises(RuntimeError, match="boom"):
        await wrapper.execute({"command": "x"}, ToolContext(call_id="c1", cancel=CancelToken()))
    activities = store.recovery_snapshot().process_activities
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "unproven_terminal"
    await store.aclose()


async def test_check_run_registered_but_status_not(tmp_path):
    run = ScriptedTool(name="check", result=ToolResult(
        "c1", "check", "passed", is_error=False,
        details={"receipt_id": "r-1", "exit_code": 0, "output_complete": True},
    ))
    store, coordinator, wrapper = _wrap(tmp_path, run)
    epoch, request_id = await _begin_request(store)
    coordinator.bind(_Sink(request_id))
    await wrapper.execute(
        {"action": "run", "check_id": "check-1"}, ToolContext(call_id="c1", cancel=CancelToken()),
    )
    activities = store.recovery_snapshot().process_activities
    assert len(activities) == 1
    assert activities[0].terminal is not None
    assert activities[0].terminal.kind == "proven_terminal"
    assert activities[0].terminal.receipt_id == "r-1", "check 回执身份留待 03 账本核对"
    await store.aclose()


class _Sink:
    """ActivitySink 测试替身: 记录锁存, 不实现 process_uncertain 锁存."""

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self.failures: list[str] = []
        self.uncertain: list[str] = []

    @property
    def active_request_id(self) -> str | None:
        return self._request_id

    def latch_storage_failure(self, detail: str) -> None:
        self.failures.append(detail)

    def latch_process_uncertain(self, detail: str) -> None:
        self.uncertain.append(detail)
