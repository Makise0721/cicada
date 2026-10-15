"""T02/T03/T07/T08 基础/T09 基础/T11 应用部分: 共享 ApplicationSession 公开执行检查.

真实内核 + 真实插件组装, 模型用确定性 fake; 检查模式用项目内隔离 Git 工作区、
真实 PowerShell 与现有检查/快照/工件实现. 观察公开状态、事件与 RequestOutcome,
不复述内部函数实现. 唯一可控 seam:
- 替换 session._service 为慢速委托代理 (T07 收尾屏障 / T11 监督截止);
- 手工 bootstrap + ApplicationSession 构造最小应用 (T11 关闭监督/清理失败,
  使用公开 bootstrap 与 PluginDefinition.defer).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from cicada.application import (
    ApplicationClosed,
    ApplicationConfig,
    ApplicationOpenError,
    ApplicationSession,
    RequestCompleted,
    RequestPhaseChanged,
    open_application,
)
from cicada.boot import bootstrap
from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
    VerificationView,
)
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.runtime.plugin import PluginDefinition

from tests.support.app import git as _git, prompt_info as _prompt_info, repo as _repo, script as _script

CHECK_COMMAND = "exit 0"


async def test_review_cancel_keeps_lease_until_owned_reader_finishes(tmp_path):
    from cicada.review import ReviewError, ReviewSnapshot

    session = await open_application(_config(_repo(tmp_path), script=[]))
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowReader:
        async def capture(self, review_id, cancel):
            entered.set()
            await cancel.wait()
            await release.wait()  # 模拟收到取消但资源还没落定。
            return ReviewSnapshot(review_id, "fixture", "2026-10-11T00:00:00Z", (), ())

    session._review_reader = SlowReader()
    caller = asyncio.create_task(session.open_review())
    try:
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert session.review_active
        with pytest.raises(ReviewError) as error:
            await session.open_review()
        assert error.value.kind == "busy"
        with pytest.raises(RuntimeError, match="review"):
            await session.execute("cannot bypass the reader lease")
        closing = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await closing
        assert not session.review_active
        assert session.history == ()
    finally:
        release.set()
        await session.aclose()


def _config(root: Path, *, script: list, plan: VerificationPlan | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(script)),
        plan=plan,
        max_turns=6,
        model_label="fake",
    )


def _plan(root: Path, command: str = CHECK_COMMAND) -> VerificationPlan:
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", command, timeout_s=30.0),),
    )


async def test_diagnostic_export_preserves_recorded_receipts_without_revalidation(tmp_path):
    import json
    from cicada.diagnostic_export import ReportError

    root = _repo(tmp_path)
    session = await open_application(_config(root, plan=_plan(root), script=_script(
        {'tool_calls': [{'id': 'w', 'name': 'write', 'arguments': {'path': 'result.txt', 'content': 'private tool body'}}]},
        {'tool_calls': [{'id': 'c', 'name': 'check', 'arguments': {'action': 'run', 'check_id': 'check-1'}}]},
        {'text': 'private answer', 'stop': True})))
    try:
        outcome = await session.execute('private user prompt')
        assert outcome.decision.can_deliver
        before = (session.history, len(session.process_model_calls))
        (root / 'result.txt').write_text('changed after completion', encoding='utf-8')
        destination = session.export_diagnostics(Path('report.json'))
        original = destination.read_bytes()
        report = json.loads(original)
        assert report['latest_request']['snapshot_at_completion'] == outcome.view.snapshot_ref
        assert report['latest_request']['checks_at_completion'][0]['freshness_at_completion'] == 'current'
        assert 'not revalidated' in report['observation']
        assert 'private user prompt' not in original.decode() and 'private tool body' not in original.decode()
        assert 'private answer' not in original.decode()
        assert (session.history, len(session.process_model_calls)) == before
        with pytest.raises(ReportError, match='already exists'):
            session.export_diagnostics(Path('report.json'))
        assert destination.read_bytes() == original
        report['latest_request']['checks_at_completion'].clear()
        assert session.diagnostics_report()['latest_request']['checks_at_completion']
    finally:
        await session.aclose()
    session.export_diagnostics(Path('closed-report.md'))
    assert '导出未重新检查工作区' in (root / 'closed-report.md').read_text(encoding='utf-8')


class Collector:
    """订阅 ApplicationUpdate; 记录顺序并支持等待."""

    def __init__(self, session: ApplicationSession) -> None:
        self.updates: list = []
        self._unsubscribe = session.subscribe(self)

    def __call__(self, update) -> None:
        self.updates.append(update)

    @property
    def phases(self) -> list[tuple[str | None, str]]:
        return [(u.run_id, u.phase) for u in self.updates if isinstance(u, RequestPhaseChanged)]

    @property
    def completed(self):
        return [u.outcome for u in self.updates if isinstance(u, RequestCompleted)]

    @property
    def closed(self) -> ApplicationClosed | None:
        return next((u for u in self.updates if isinstance(u, ApplicationClosed)), None)

    async def wait_for(self, predicate, timeout: float = 10.0) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return False

    def stop(self) -> None:
        self._unsubscribe()


class SlowRefreshService:
    """把真实服务的 refresh 延迟化的可控收尾屏障; 其余全部委托."""

    def __init__(self, inner, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    @property
    def plan(self):
        return self._inner.plan

    def mark_process_uncertain(self, reason: str) -> None:
        self._inner.mark_process_uncertain(reason)

    async def initialize(self, cancel: CancelToken):
        return await self._inner.initialize(cancel)

    async def run_check(self, check_id: str, cancel: CancelToken):
        return await self._inner.run_check(check_id, cancel)

    async def refresh(self, cancel: CancelToken) -> VerificationView:
        await asyncio.sleep(self._delay)
        return await self._inner.refresh(cancel)

    async def finalize(self, cancel: CancelToken):
        return await self._inner.finalize(cancel)

    async def finalize_current(self, cancel):
        await asyncio.sleep(self._delay)
        return await self._inner.finalize_current(cancel)


class IgnoreCancelTool:
    """吞掉取消令牌并持续运行的工具: 关闭监督到点路径的可控资源."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancel_observed = asyncio.Event()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hang",
            description="忽略取消",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict, ctx: ToolContext):
        from cicada.core.messages import ToolResult

        self.started.set()

        async def watch() -> None:
            await ctx.cancel.wait()
            self.cancel_observed.set()
            # 观察到取消但继续运行: 模拟吞取消的外部工作 (第一版不承诺强制终止)

        watcher = asyncio.create_task(watch())
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            watcher.cancel()
            raise


# --- 启动与配置 ------------------------------------------------------------------------------


async def test_config_rejects_invalid_max_turns(tmp_path):
    with pytest.raises(ValueError, match="max_turns"):
        ApplicationConfig(
            workspace=tmp_path, system_prompt="", prompt_info=_prompt_info(tmp_path),
            model_plugin=fake_model_plugin(FakeModel([])), max_turns=0,
        )


async def test_open_failure_maps_boot_error(tmp_path, monkeypatch):
    import cicada.application as app_module
    from cicada.boot import BootError

    async def broken_bootstrap(*args, **kwargs):
        raise BootError("plugin startup failed: boom")

    monkeypatch.setattr(app_module, "bootstrap", broken_bootstrap)
    with pytest.raises(ApplicationOpenError, match="boot failed: plugin startup failed: boom"):
        await open_application(_config(tmp_path, script=[]))


# --- T02: 连续两请求接续结构化历史 -----------------------------------------------------------


async def test_second_request_carries_prior_history_once(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"text": "第一轮回答", "stop": True},
        {"text": "第二轮回答", "stop": True},
    )))
    collector = Collector(session)
    first = await session.execute("第一个问题")
    assert first.exit_code == 0
    second = await session.execute("第二个问题")
    assert second.exit_code == 0
    # 应用累计历史: 每轮后缀追加一次, 无重复增长
    assert [m.text for m in session.history] == [
        "第一个问题", "第一轮回答", "第二个问题", "第二轮回答",
    ]
    # 系统提示独立于普通历史: 以 system_prompt 字段单独携带
    from cicada.plugins.fake_model import FakeModel

    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    assert model.requests[1].system_prompt == "test system prompt"
    assert not any(m.text == "test system prompt" for m in model.requests[1].messages)
    collector.stop()
    await session.aclose()


async def test_phase_events_follow_lifecycle(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script({"text": "回答", "stop": True})))
    collector = Collector(session)
    await session.execute("问题")
    assert collector.phases == [
        (None, "executing"),
        ("run-1", "executing"),
        ("run-1", "finalizing"),
        ("run-1", "idle"),
    ]
    outcome = collector.completed[0]
    assert outcome.result.run_id == "run-1"
    assert outcome.decision is None and outcome.view is None  # 普通模式不产生检查权威
    collector.stop()
    await session.aclose()


# --- T03: 错误路径闭合且可继续追问 ------------------------------------------------------------


async def test_model_error_closes_calls_and_allows_followup(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "error": "provider exploded"},
        {"text": "恢复后回答", "stop": True},
    )))
    collector = Collector(session)
    bad = await session.execute("第一问")
    assert bad.exit_code == 1
    assert bad.result.stop_reason == "error"
    # 错误路径闭合工具调用: 历史以固定哨兵错误结果结尾, 不悬空
    tail = session.history[-1].result
    assert tail.call_id == "c1"
    assert tail.content == "model stream failed; tool call not executed"
    # 结构化历史闭合 -> 允许继续追问
    followup = await session.execute("第二问")
    assert followup.exit_code == 0 and followup.result.stop_reason == "stop"
    collector.stop()
    await session.aclose()


# --- T07: 内核结束后保持 finalizing, 收尾前不接受新请求 ---------------------------------------


async def test_execute_rejected_until_finalization_completes(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "stop": True},
    ), plan=plan))
    real_service = session._service
    assert real_service is not None
    session._service = SlowRefreshService(real_service, delay=0.4)
    collector = Collector(session)

    task = asyncio.create_task(session.execute("运行检查"))
    assert await collector.wait_for(lambda: session.state == "finalizing"), collector.phases
    # 内核已结束 (finalizing), 新请求在收尾完成前被拒绝
    with pytest.raises(RuntimeError, match="finalizing"):
        await session.execute("插队")
    outcome = await asyncio.wait_for(task, timeout=10)
    assert session.state == "idle"
    assert outcome.exit_code == 0
    assert outcome.decision is not None and outcome.decision.can_deliver
    collector.stop()
    await session.aclose()


# --- T08 基础: 多请求共用一次 initialize 的基线和账本 ------------------------------------------


async def test_multi_request_shared_baseline_and_cumulative_artifacts(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "r1", "name": "read", "arguments": {"path": "sample.py"}},
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "第一轮检查完成", "stop": True},
        {"tool_calls": [{"id": "c2", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "tool_use": True},
        {"text": "第二轮检查完成", "stop": True},
    ), plan=plan))
    collector = Collector(session)
    first = await session.execute("改一下再检查")
    assert first.exit_code == 0, first.decision.blocking_reasons if first.decision else None
    assert first.decision is not None and first.decision.can_deliver
    assert first.evidence is not None and first.view is not None
    baseline = first.evidence.baseline_snapshot_ref
    artifact_1 = first.evidence.artifact_path
    assert artifact_1 is not None and artifact_1.exists()
    assert first.evidence.changes, "请求 1 应有累计变更记录"
    assert len(first.view.receipts) == 1

    second = await session.execute("再检查一次")
    assert second.exit_code == 0
    assert second.decision is not None and second.decision.can_deliver
    # 基线/账本跨请求共享: 同一 baseline, 同一 verification_run_id, 回执累计
    assert second.evidence.baseline_snapshot_ref == baseline
    assert second.view.verification_run_id == plan.verification_run_id
    assert len(second.view.receipts) == 2
    # 每次请求独立的累计变更工件: 新文件, 不覆盖
    assert second.evidence.artifact_path is not None
    assert second.evidence.artifact_path != artifact_1
    assert second.evidence.artifact_path.exists() and artifact_1.exists()
    collector.stop()
    await session.aclose()


# --- T09 基础: process_uncertain 拒绝后续编码请求, 清空历史不解锁 -----------------------------


async def test_process_uncertain_blocks_followup_and_survives_clear(tmp_path):
    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "Start-Sleep -Seconds 5", timeout_s=0.5),),
    )
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "stop": True},
        {"text": "汇报超时", "stop": True},
    ), plan=plan))
    collector = Collector(session)
    outcome = await session.execute("跑一个会超时的检查")
    assert outcome.result.stop_reason == "stop"
    assert outcome.view is not None and outcome.view.process_uncertain
    # 后续用户编码请求被拒绝
    with pytest.raises(RuntimeError, match="process_uncertain"):
        await session.execute("继续改代码")
    # 清空历史不解锁: 锁存单调保持
    session.clear_history()
    with pytest.raises(RuntimeError, match="process_uncertain"):
        await session.execute("清空后再试")
    # 关闭仍然可用
    await session.aclose()
    assert session.state == "closed"
    collector.stop()


# --- T11 应用部分: 取消 / 关闭重入 / 迟到完成 / 监督截止 / 清理失败 ---------------------------


async def test_cancel_returns_outcome_and_session_reusable(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "取消后的新请求", "stop": True},
    )))
    collector = Collector(session)
    task = asyncio.create_task(session.execute("会取消的请求"))
    assert await collector.wait_for(lambda: session.state == "executing")
    await asyncio.sleep(0.3)  # 等真实 PowerShell 进程起来
    session.cancel()
    outcome = await asyncio.wait_for(task, timeout=15)
    assert outcome.result.stop_reason == "aborted"
    assert outcome.exit_code == 1
    assert session.state == "idle"
    # 取消后可继续请求
    again = await session.execute("下一个请求")
    assert again.exit_code == 0
    collector.stop()
    await session.aclose()


async def test_close_reentry_shares_one_close_and_rejects_execute(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=[]))
    results = await asyncio.gather(session.aclose(), session.aclose())
    assert results == [None, None]
    assert session.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        await session.execute("after close")


async def test_late_completion_during_close_keeps_closing_state(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    )))
    collector = Collector(session)
    execute_task = asyncio.create_task(session.execute("取消我"))
    assert await collector.wait_for(lambda: session.state == "executing")
    await asyncio.sleep(0.3)
    close_task = asyncio.create_task(session.aclose())
    # 关闭锁存后, 活动请求被取消并完成收尾; RequestCompleted 仍发出一次, 但状态保持 closing
    assert await collector.wait_for(
        lambda: session.state == "closing" and collector.completed, timeout=15
    )
    completed = collector.completed[-1]
    assert completed.result.stop_reason == "aborted"
    assert session.state == "closing"
    await asyncio.wait_for(close_task, timeout=15)
    assert session.state == "closed"
    # ApplicationClosed 是最后一条通知; 正常关闭无清理错误
    assert collector.closed is not None and collector.closed.error is None
    assert collector.updates[-1] is collector.closed
    # execute 正常返回 outcome (迟到完成不能转回 idle 也不悬挂调用方)
    outcome = await asyncio.wait_for(execute_task, timeout=10)
    assert outcome.result.stop_reason == "aborted"
    collector.stop()


async def test_close_supervision_deadline_records_problem(tmp_path, monkeypatch):
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.2)
    root = _repo(tmp_path)

    def hang_setup(ctx):
        ctx.provide("tool.hang", IgnoreCancelTool())

    definitions = [
        PluginDefinition(name="hang", setup=hang_setup, provides=frozenset({"tool.hang"})),
        fake_model_plugin(FakeModel(_script(
            {"tool_calls": [{"id": "c1", "name": "hang", "arguments": {}}], "tool_use": True},
        ))),
    ]
    app = await bootstrap(
        definitions, tool_capabilities=("tool.hang",), system_prompt="t", max_turns=6,
    )
    session = ApplicationSession(app=app, config=_config(root, script=[]), service=None, policy=None)
    hang = app.runtime.capability("tool.hang")
    assert isinstance(hang, IgnoreCancelTool)
    execute_task = asyncio.create_task(session.execute("挂住"))
    await asyncio.wait_for(hang.started.wait(), timeout=5)
    close_task = asyncio.create_task(session.aclose())
    closed = await asyncio.wait_for(close_task, timeout=10)
    assert closed is None  # aclose 正常返回
    assert session.state == "closed"
    # 监督到点: 记录问题且不再回 idle / 不再接收请求
    assert "did not finalize" in (session._close_problems[0] if session._close_problems else "")
    execute_task.cancel()
    try:
        await execute_task
    except asyncio.CancelledError:
        pass


async def test_finalization_supervision_deadline_returns_nondeliverable(tmp_path, monkeypatch):
    """T11 应用部分: 请求收尾监督到点 -> 不可交付 outcome, 任务已终结才回 idle."""
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "REQUEST_FINALIZE_SUPERVISION_S", 0.1)
    root = _repo(tmp_path)
    plan = _plan(root)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    ), plan=plan))
    real_service = session._service
    assert real_service is not None
    session._service = SlowRefreshService(real_service, delay=1.0)
    outcome = await asyncio.wait_for(session.execute("慢收尾"), timeout=5)
    assert outcome.decision is not None and not outcome.decision.can_deliver
    assert "supervision" in outcome.decision.blocking_reasons[0]
    assert outcome.exit_code == 3  # 模型 stop 但收尾超时, 不可交付
    assert outcome.view is not None and outcome.view.snapshot_ref is None
    assert session.state == "idle"  # 收尾任务已终结才回 idle
    assert session._undead_tasks == []  # 协作式取消下无僵尸任务
    await session.aclose()


async def test_late_completion_after_close_deadline_keeps_closed_terminal_state(
    tmp_path, monkeypatch
):
    """02-R-F1 回归: close 监督超时后的迟到完成不得把 closed 终态改回 closing."""
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.2)
    root = _repo(tmp_path)

    def hang_setup(ctx):
        ctx.provide("tool.hang", IgnoreCancelTool())

    definitions = [
        PluginDefinition(name="hang", setup=hang_setup, provides=frozenset({"tool.hang"})),
        fake_model_plugin(FakeModel(_script(
            {"tool_calls": [{"id": "c1", "name": "hang", "arguments": {}}], "tool_use": True},
        ))),
    ]
    app = await bootstrap(
        definitions, tool_capabilities=("tool.hang",), system_prompt="t", max_turns=6,
    )
    session = ApplicationSession(app=app, config=_config(root, script=[]), service=None, policy=None)
    hang = app.runtime.capability("tool.hang")
    assert isinstance(hang, IgnoreCancelTool)
    execute_task = asyncio.create_task(session.execute("挂住"))
    await asyncio.wait_for(hang.started.wait(), timeout=5)
    close_task = asyncio.create_task(session.aclose())
    await asyncio.wait_for(close_task, timeout=10)
    assert session.state == "closed"
    assert session._close_problems, "监督到点必须记录 did not finalize 问题"
    # 迟到完成: 取消 execute 任务后内核捕获并闭合, execute 正常返回 outcome
    execute_task.cancel()
    outcome = await asyncio.wait_for(execute_task, timeout=10)
    assert outcome.result.stop_reason == "aborted"
    assert session.state == "closed"  # 终态不可逆: 不回 closing, 也不回 idle


async def test_close_deadline_bounds_request_finalization(tmp_path, monkeypatch):
    """02-R-F6: closing 锁存后, 请求收尾窗口与 close 截止时间取较早值."""
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.3)
    root = _repo(tmp_path)
    plan = _plan(root)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    ), plan=plan))
    real_service = session._service
    assert real_service is not None
    session._service = SlowRefreshService(real_service, delay=5.0)
    collector = Collector(session)
    task = asyncio.create_task(session.execute("慢收尾"))
    assert await collector.wait_for(lambda: session.state == "finalizing")
    close_task = asyncio.create_task(session.aclose())
    # 5 秒内返回 = 收尾被 close 截止时间 (0.3s) 截断, 而非等待 15s 请求窗口
    outcome = await asyncio.wait_for(task, timeout=5)
    assert outcome.decision is not None and not outcome.decision.can_deliver
    assert "supervision" in outcome.decision.blocking_reasons[0]
    await asyncio.wait_for(close_task, timeout=10)
    assert session.state == "closed"
    collector.stop()


async def test_cleanup_failure_surfaces_in_application_closed(tmp_path):
    root = _repo(tmp_path)

    def leaky_setup(ctx):
        ctx.provide("leaky-cap", object())

        def boom():
            raise RuntimeError("cleanup exploded")

        ctx.defer(boom)

    definitions = [
        PluginDefinition(name="leaky", setup=leaky_setup, provides=frozenset({"leaky-cap"})),
        fake_model_plugin(FakeModel(_script({"text": "ok", "stop": True}))),
    ]
    app = await bootstrap(definitions, tool_capabilities=(), system_prompt="t", max_turns=6)
    session = ApplicationSession(app=app, config=_config(root, script=[]), service=None, policy=None)
    collector = Collector(session)
    outcome = await session.execute("x")
    assert outcome.exit_code == 0
    await session.aclose()
    assert collector.closed is not None
    assert collector.closed.error is not None
    assert "runtime stop failed" in collector.closed.error
    collector.stop()


# --- 隔离 / 清空 / 拒绝 ----------------------------------------------------------------------


async def test_observer_mutation_does_not_touch_canonical_history(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "stop": True},
        {"text": "第二问回答", "stop": True},
    )))
    seen: list = []

    def mutating_observer(update):
        if isinstance(update, RequestCompleted):
            seen.append(update.outcome)
            for message in update.outcome.result.messages:
                result = getattr(message, "result", None)
                if result is not None and result.details:
                    result.details["tampered"] = True

    session.subscribe(mutating_observer)
    await session.execute("调用一次工具")
    # 观察者确实在自己的副本上改到了 details
    assert seen
    tampered_in_copy = any(
        "tampered" in (getattr(m, "result", None).details or {})
        for o in seen for m in o.result.messages
        if getattr(m, "result", None) is not None
    )
    assert tampered_in_copy
    # 规范历史未被展示副本的改动穿透
    for message in session.history:
        result = getattr(message, "result", None)
        if result is not None and result.details:
            assert "tampered" not in result.details
    # 且下一请求的模型输入不受影响: 继续一次带历史的请求并核对请求内 details
    await session.execute("第二个问题")
    from cicada.plugins.fake_model import FakeModel

    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    for message in model.requests[-1].messages:
        result = getattr(message, "result", None)
        if result is not None and result.details:
            assert "tampered" not in result.details
    await session.aclose()


async def test_clear_history_keeps_session_usable_and_verification_state(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root)
    session = await open_application(_config(root, script=_script(
        {"text": "第一轮", "stop": True},
        {"text": "第二轮", "stop": True},
    ), plan=plan))
    await session.execute("第一问")
    assert len(session.history) == 2
    session.clear_history()
    assert session.history == ()
    # 声明的检查从未运行: 模型 stop 但交付判定不通过 (exit 3), 视图仍绑定同一验证运行
    second = await session.execute("第二问")
    assert second.exit_code == 3
    assert second.decision is not None and not second.decision.can_deliver
    assert [m.text for m in session.history] == ["第二问", "第二轮"]
    # 验证状态保留: 收尾仍使用同一 verification_run_id (基线/账本未随历史清空)
    assert second.view is not None
    assert second.view.verification_run_id == plan.verification_run_id
    # 空闲时重复清空合法
    session.clear_history()
    assert session.history == ()
    await session.aclose()


async def test_busy_clear_and_busy_execute_rejected(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    )))
    task = asyncio.create_task(session.execute("忙碌请求"))
    await asyncio.sleep(0.3)
    assert session.state == "executing"
    with pytest.raises(RuntimeError, match="idle"):
        session.clear_history()
    with pytest.raises(RuntimeError, match="finish before the next"):
        await session.execute("插队")
    session.cancel()
    outcome = await asyncio.wait_for(task, timeout=15)
    assert outcome.result.stop_reason == "aborted"
    await session.aclose()


async def test_empty_prompt_rejected_before_state_change(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=[]))
    with pytest.raises(ValueError, match="nonempty"):
        await session.execute("   ")
    assert session.state == "idle"
    await session.aclose()


async def test_subscribe_observer_errors_recorded_not_fatal(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script({"text": "ok", "stop": True})))

    def bad(update):
        raise RuntimeError("observer boom")

    session.subscribe(bad)
    outcome = await session.execute("x")
    assert outcome.exit_code == 0
    assert any(isinstance(e, RuntimeError) for e in session.observer_errors)
    await session.aclose()

@pytest.mark.parametrize("cross_deadline", [False, True])
async def test_active_close_owns_native_file_worker_until_actual_finish(tmp_path, monkeypatch, cross_deadline):
    """文件发布跨关闭监督时直接核对 worker/锁, 不把 close 当成 completion 保存."""
    import threading
    import cicada.application as app_module
    from cicada.core.messages import ToolResultMessage
    from cicada.plugins.coding import file_commit_windows as native
    from cicada.plugins.coding.tool_read import ReadTool
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "write", "name": "write",
                         "arguments": {"path": "created.txt", "content": "created"}}]},
        {"text": "must not run", "stop": True},
    )))
    workspace = session._app.runtime.capability("coding.workspace")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_create = native.create_file
    def blocked_create(*args):
        entered.set()
        assert release.wait(10)
        try:
            return real_create(*args)
        finally:
            finished.set()
    monkeypatch.setattr(native, "create_file", blocked_create)
    if cross_deadline:
        monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.05)
    task = asyncio.create_task(session.execute("active close"))
    close_task = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        close_task = asyncio.create_task(session.aclose())
        if cross_deadline:
            await asyncio.wait_for(close_task, 5)
            assert session._close_problems and session.state == "closed"
            assert not finished.is_set() and not task.done()
            assert any(lock.locked() for lock in workspace._locks.values())
        else:
            await asyncio.sleep(0)
            assert not close_task.done()
    finally:
        release.set()
    outcome = await asyncio.wait_for(task, 10)
    await asyncio.wait_for(close_task, 5)
    assert finished.is_set() and all(not lock.locked() for lock in workspace._locks.values())
    assert outcome.result.stop_reason == "aborted" and len(session._app.agent._model.requests) == 1
    tool_results = [m.result for m in outcome.result.messages if isinstance(m, ToolResultMessage)]
    assert len(tool_results) == 1 and tool_results[0].call_id == "write"
    assert tool_results[0].details["mutation"]["state"] == "committed"
    assert tool_results[0].details["mutation"]["caller_cancelled"]
    assert session.state == "closed"
    result = await ReadTool(workspace).execute(
        {"path": "created.txt"}, ToolContext(call_id="after", cancel=CancelToken()))
    assert not result.is_error and "created" in result.content
