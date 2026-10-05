"""应用层持久协调聚焦测试: 保存->关闭->恢复->续写、失败锁存、清空边界、恢复准入.

真实共享应用 + fake 模型; 存储故障经 port seam 注入 (monkeypatch store 方法),
开库/占用/损坏由 tests/plugins/test_session_store.py 与 CLI 子进程测试覆盖.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cicada.application import (
    ApplicationOpenError,
    ApplicationSession,
    RequestCompleted,
    open_application,
)
from cicada.core.messages import UserMessage
from cicada.persistence.contracts import LaunchIdentity, PersistenceError
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.prompting import PromptInfo

from test_application import _repo, _script  # 复用既有工作区/剧本构造


def _texts(messages) -> list[str]:
    """按消息类型取展示文本; 工具结果以 <tool:{name}> 标记."""
    from cicada.core.messages import ToolResultMessage

    return [
        f"<tool:{m.result.name}>" if isinstance(m, ToolResultMessage) else m.text
        for m in messages
    ]


def _identity() -> LaunchIdentity:
    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None, sources={},
    )


def _prompt_info(root: Path) -> PromptInfo:
    import hashlib

    return PromptInfo(
        version="test", workspace=root, instructions_path=None,
        instructions_sha256=None, prompt_sha256=hashlib.sha256(b"test").hexdigest(),
    )


def _config(root: Path, script: list, *, session_id: str | None):
    from cicada.application import ApplicationConfig, PersistenceConfig

    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(script)),
        max_turns=6,
        model_label="fake",
        persistence=PersistenceConfig(session_id=session_id, identity=_identity()),
    )


def _plain_config(root: Path, script: list):
    from cicada.application import ApplicationConfig

    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(script)),
        max_turns=6,
        model_label="fake",
    )


# --- 默认不落盘 / 保存 / 恢复续写 (A01) -------------------------------------------------------


async def test_default_mode_stays_non_persistent(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_plain_config(root, _script({"text": "答", "stop": True})))
    assert session.persistence is None
    outcome = await session.execute("问")
    assert outcome.persistence is None
    await session.aclose()


async def test_save_then_resume_new_process_continues_without_replay(tmp_path):
    root = _repo(tmp_path)
    script = _script(
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "tool_use": True},
        {"text": "第一轮回答", "stop": True},
    )
    session = await open_application(_config(root, script, session_id=None))
    assert session.persistence is not None and session.persistence.mode == "save"
    session_id = session.persistence.session_id
    first = await session.execute("第一问")
    assert first.persistence is not None and first.persistence.saved
    assert first.exit_code == 0
    assert session.persistence.saved_request_count == 1
    await session.aclose()

    # 新"进程": 全新应用实例按 ID 恢复; 旧工具不重放, 历史不重复
    resumed = await open_application(_config(root, _script({"text": "恢复后的回答", "stop": True}),
                                            session_id=session_id))
    assert resumed.persistence.mode == "resume"
    assert resumed.persistence.restored_request_count == 1
    assert resumed.persistence.restored_model_call_count == 2  # 工具轮 + 收尾轮
    assert _texts(resumed.history) == [
        "第一问", "", "<tool:glob>", "第一轮回答",
    ]

    tool_starts: list = []

    def on_event(update):
        from cicada.application import AgentEventReceived
        from cicada.core.events import ToolStarted

        if isinstance(update, AgentEventReceived) and isinstance(update.event, ToolStarted):
            tool_starts.append(update.event.name)

    resumed.subscribe(on_event)
    completion_at: dict = {}

    def on_update(update):
        if isinstance(update, RequestCompleted):
            completion_at["saved"] = resumed.persistence.saved_request_count
            completion_at["persistence"] = update.outcome.persistence

    resumed.subscribe(on_update)
    second = await resumed.execute("第二问")
    assert tool_starts == [], "恢复后的新请求不得重放旧工具"
    # saved_request_count 只计本进程; RequestCompleted 到达时完成提交必须已成功
    assert completion_at["saved"] == 1
    assert completion_at["persistence"] is not None and completion_at["persistence"].saved
    assert second.exit_code == 0
    # 规范历史一次追加: 恢复前缀 + 新后缀, 无重复
    assert _texts(resumed.history) == [
        "第一问", "", "<tool:glob>", "第一轮回答", "第二问", "恢复后的回答",
    ]
    # 新进程发出的首个 ModelRequest 携带完整恢复历史 + 新 prompt
    model = resumed._app.agent._model
    assert isinstance(model, FakeModel)
    sent = model.requests[0].messages
    assert _texts(sent) == ["第一问", "", "<tool:glob>", "第一轮回答", "第二问"]
    assert isinstance(sent[0], UserMessage) and sent[0].text == "第一问"
    await resumed.aclose()

    # 再恢复一次: 两条请求都在, 计量累计 (2 + 1 次模型调用)
    third = await open_application(_config(root, _script(), session_id=session_id))
    assert third.persistence.restored_request_count == 2
    assert third.persistence.restored_model_call_count == 3
    await third.aclose()


async def test_length_and_error_outcomes_saved_with_real_terminal_facts(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"text": "被截断", "length": True},
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "x"}}],
         "error": "provider down"},
        {"text": "恢复后作答", "stop": True},
    ), session_id=None))
    first = await session.execute("第一问")
    assert first.result.stop_reason == "length" and first.exit_code == 1
    second = await session.execute("第二问")
    assert second.result.stop_reason == "error" and second.exit_code == 1
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_config(root, _script(), session_id=session_id))
    facts = [request.outcome for request in resumed.restored_requests]
    assert [f.stop_reason for f in facts] == ["length", "error"]
    assert facts[0].finalized
    assert resumed.persistence.restored_request_count == 2
    await resumed.aclose()


async def test_aborted_request_still_saved_with_aborted_facts(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "取消后的新请求", "stop": True},
    ), session_id=None))
    task = asyncio.create_task(session.execute("会取消的请求"))
    await asyncio.sleep(0.3)
    session.cancel()
    outcome = await asyncio.wait_for(task, timeout=15)
    assert outcome.result.stop_reason == "aborted"
    assert outcome.persistence is not None and outcome.persistence.saved
    session_id = session.persistence.session_id
    await session.aclose()
    resumed = await open_application(_config(root, _script(), session_id=session_id))
    facts = [request.outcome for request in resumed.restored_requests]
    assert facts[0].stop_reason == "aborted"
    await resumed.aclose()


# --- 保存失败: 零执行 / 锁存 / 退出码 ---------------------------------------------------------


async def test_begin_failure_makes_no_model_call_and_latches_storage_failed(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "不应到达", "stop": True}),
                                            session_id=None))
    store = session._store
    await store.aclose()  # 模拟存储不可用: begin 无法可靠提交
    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    with pytest.raises(RuntimeError, match="session save failed before the request started"):
        await session.execute("问")
    assert model.requests == [], "开始记录失败必须零模型调用"
    assert session.persistence.storage_failed is not None
    with pytest.raises(RuntimeError, match="storage_failed"):
        await session.execute("再问")
    assert model.requests == []
    await session.aclose()


async def test_commit_failure_keeps_facts_latches_and_exit_code_one(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "回答", "stop": True}),
                                            session_id=None))
    store = session._store

    async def failing_commit(request_id, *, messages, outcome):
        raise PersistenceError("disk gone")

    store.commit_request = failing_commit  # type: ignore[method-assign]
    outcome = await session.execute("问")
    assert outcome.persistence is not None and not outcome.persistence.saved
    assert outcome.exit_code == 1, "执行期间保存失败退出码 1"
    assert session.persistence.storage_failed is not None
    assert len(session.history) == 2, "已取得的模型事实仍可显示"
    with pytest.raises(RuntimeError, match="storage_failed"):
        await session.execute("存储恢复也不续跑")
    await session.aclose()


async def test_clear_commit_failure_keeps_memory_unchanged(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    store = session._store

    async def failing_clear(epoch_id):
        raise PersistenceError("disk gone")

    store.commit_clear = failing_clear  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="in-memory history is unchanged"):
        await session.clear_context()
    assert len(session.history) == 2
    await session.aclose()


# --- 清空边界 --------------------------------------------------------------------------------


async def test_persistent_clear_requires_async_entry_and_restores_post_clear_history(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"text": "一答", "stop": True},
        {"text": "二答", "stop": True},
    ), session_id=None))
    await session.execute("一问")
    with pytest.raises(RuntimeError, match="clear_context"):
        session.clear_history()
    assert len(session.history) == 2, "同步入口拒绝且无内存变化"
    await session.clear_context()
    assert session.history == ()
    assert session.persistence.context_generation == 1
    second = await session.execute("二问")
    assert second.exit_code == 0
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_config(root, _script(), session_id=session_id))
    assert [m.text for m in resumed.history] == ["二问", "二答"], "清空边界后的历史不复活旧上下文"
    assert resumed.persistence.restored_request_count == 2
    assert resumed.persistence.context_generation == 1
    await resumed.aclose()


async def test_busy_clear_context_rejected(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    ), session_id=None))
    task = asyncio.create_task(session.execute("忙碌请求"))
    await asyncio.sleep(0.3)
    with pytest.raises(RuntimeError, match="idle"):
        await session.clear_context()
    session.cancel()
    await asyncio.wait_for(task, timeout=15)
    await session.aclose()


# --- 恢复准入: 中断 / 配置差异 / 指纹 (副作用前拒绝) -------------------------------------------


async def test_resume_with_pending_intent_without_unproven_activity_continues(tmp_path):
    """A15: 无未证终结活动的中断请求可以继续; pending intent 标中断并显示."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    store = session._store

    async def failing_commit(request_id, *, messages, outcome):
        raise PersistenceError("commit lost")

    store.commit_request = failing_commit  # type: ignore[method-assign]
    outcome = await session.execute("会留下 pending 的问")
    assert outcome.persistence is not None and not outcome.persistence.saved
    assert session.persistence.storage_failed is not None
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_config(root, _script({"text": "恢复后的回答", "stop": True}),
                                            session_id=session_id))
    assert resumed.view_only_reason is None, "无未证终结活动的中断不阻断续写"
    assert len(resumed.restored_interrupted) == 1
    assert resumed.restored_interrupted[0].status == "interrupted"
    assert resumed.persistence.restored_request_count == 0, "未提交请求不进入可续写历史"
    third = await resumed.execute("中断后的新请求")
    assert third.exit_code == 0
    assert resumed.persistence is not None and resumed.persistence.saved_request_count == 1
    await resumed.aclose()


async def test_resume_with_unproven_process_activity_is_view_only(tmp_path):
    """A15: 未证终结活动使恢复仅可查看; 新编码请求拒绝, 清空不解锁."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    ), session_id=None))
    task = asyncio.create_task(session.execute("会留下未证终结的请求"))
    await asyncio.sleep(0.3)
    session.cancel()
    outcome = await asyncio.wait_for(task, timeout=15)
    assert outcome.persistence is not None and outcome.persistence.saved
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_config(root, _script(), session_id=session_id))
    assert resumed.view_only_reason is not None
    assert "powershell" in resumed.view_only_reason
    assert len(resumed.restored_interrupted) == 0, "已提交请求不是中断"
    model = resumed._app.agent._model
    assert isinstance(model, FakeModel)
    with pytest.raises(RuntimeError, match="view-only"):
        await resumed.execute("新请求被拒绝")
    assert model.requests == [], "view-only 会话零模型调用"
    # 清空不解锁: 清空可以提交, 但查看状态保持
    await resumed.clear_context()
    assert resumed.view_only_reason is not None
    with pytest.raises(RuntimeError, match="view-only"):
        await resumed.execute("清空后仍然拒绝")
    await resumed.aclose()

    # 再次恢复: 未证终结证据持久, 不能绕过
    again = await open_application(_config(root, _script(), session_id=session_id))
    assert again.view_only_reason is not None
    await again.aclose()


async def test_resume_with_config_conflict_names_fields(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    session_id = session.persistence.session_id
    await session.aclose()

    from cicada.application import ApplicationConfig, PersistenceConfig
    from dataclasses import replace as dc_replace

    other_identity = LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=9, prompt_version="test", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None, sources={},
    )
    config = dc_replace(
        _config(root, _script(), session_id=session_id),
        persistence=PersistenceConfig(session_id=session_id, identity=other_identity),
    )
    with pytest.raises(ApplicationOpenError, match="max_turns"):
        await open_application(config)


async def test_resume_with_tool_fingerprint_mismatch_refused_and_lease_released(
    tmp_path, monkeypatch
):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    session_id = session.persistence.session_id
    await session.aclose()

    import cicada.application as app_module

    monkeypatch.setattr(app_module, "_tool_fingerprint", lambda app: "fingerprint-changed")
    with pytest.raises(ApplicationOpenError, match="tool_fingerprint"):
        await open_application(_config(root, _script(), session_id=session_id))
    monkeypatch.undo()
    # 拒绝后 lease 已释放: 重新打开成功
    reopened = await open_application(_config(root, _script(), session_id=session_id))
    await reopened.aclose()


async def test_check_mode_with_persistence_rejected_at_config(tmp_path):
    from cicada.application import ApplicationConfig, PersistenceConfig
    from cicada.plugins.coding.verification import generate_verification_run_id
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan

    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(root),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    with pytest.raises(ValueError, match="check mode"):
        ApplicationConfig(
            workspace=root,
            system_prompt="s",
            prompt_info=_prompt_info(root),
            model_plugin=fake_model_plugin(FakeModel([])),
            plan=plan,
            persistence=PersistenceConfig(session_id=None, identity=_identity()),
        )


# --- 02: 拓扑清理顺序与关闭拥有权 (A14) --------------------------------------------------------


def _recording_model_plugin(order: list, script: list):
    """带清理记录的 fake 模型插件定义; 清理顺序用于验证逆拓扑."""
    from cicada.plugins.fake_model import FakeModel
    from cicada.runtime.plugin import PluginDefinition

    model = FakeModel(script)

    def setup(ctx):
        ctx.provide("model", model)
        ctx.defer(lambda: order.append("model"))

    return PluginDefinition(name="fake-model", setup=setup, provides=frozenset({"model"}))


async def test_persistent_topology_cleans_resources_before_store(tmp_path, monkeypatch):
    """A14: 显式 requires 的真实 open_application 装配 —— 模型先清理, store 最后释放."""
    from dataclasses import replace as dc_replace

    import cicada.application as app_module
    from cicada.application import PersistenceConfig
    from cicada.plugins.session_store import open_session_store

    root = _repo(tmp_path)
    order: list[str] = []
    original_plugin = app_module.session_store_plugin

    def patched_plugin(store):
        original_aclose = store.aclose

        async def recording_aclose():
            order.append("store")
            await original_aclose()

        store.aclose = recording_aclose  # type: ignore[method-assign]
        return original_plugin(store)

    monkeypatch.setattr(app_module, "session_store_plugin", patched_plugin)
    config = dc_replace(
        _plain_config(root, _script({"text": "答", "stop": True})),
        model_plugin=_recording_model_plugin(order, _script({"text": "答", "stop": True})),
        persistence=PersistenceConfig(session_id=None, identity=_identity()),
    )
    session = await open_application(config)
    assert session.persistence is not None
    store = session._store
    await session.execute("问")
    await session.aclose()
    assert order == ["model", "store"], f"store 必须最后清理, 实际 {order}"
    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    await reopened.aclose()


class _GateTool:
    """受控 powershell 形工具: 真实结果形态, 执行等待闸门 (A14 反例的可控工具)."""

    def __init__(self, gate: asyncio.Event) -> None:
        from cicada.core.ports import ToolSpec

        self._gate = gate
        self.spec = ToolSpec(name="powershell", description="gate", parameters={
            "type": "object", "properties": {},
        })
        self.started = asyncio.Event()

    async def execute(self, arguments, ctx):
        from cicada.core.messages import ToolResult

        self.started.set()
        await self._gate.wait()
        return ToolResult(
            call_id=ctx.call_id, name="powershell", content="ok", is_error=False,
            details={"exit_code": 0, "timed_out": False, "cancelled": False,
                     "output_complete": True},
        )


def _identity_stored():
    from cicada.persistence.contracts import StoredIdentity

    return StoredIdentity(launch=_identity(), tool_fingerprint="fp")


async def _manual_persistent_session(tmp_path, script: list, tool, *, store, coordinator):
    """绕过 open_application 手工装配: 注入受控工具与受控事务 (A14 反例需要)."""
    from cicada.application import (
        ApplicationConfig,
        ApplicationSession,
        PersistenceConfig,
        _bind_store_last_cleanup,
    )
    from cicada.boot import bootstrap
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin
    from cicada.plugins.fake_tools import fake_tools_plugin
    from cicada.plugins.session_store import session_store_plugin

    def tool_policy(inner, runtime):
        from cicada.process_activity import ProcessActivityWrapper

        return ProcessActivityWrapper(inner, store, coordinator)

    definitions = [session_store_plugin(store)]
    definitions.extend(_bind_store_last_cleanup([
        fake_model_plugin(FakeModel(script)), fake_tools_plugin([tool]),
    ]))
    app = await bootstrap(definitions, tool_capabilities=("tools",), system_prompt="s",
                          tool_policy=tool_policy)
    config = ApplicationConfig(
        workspace=tmp_path, system_prompt="s", prompt_info=_prompt_info(tmp_path),
        model_plugin=fake_model_plugin(FakeModel([])), max_turns=6, model_label="fake",
        persistence=PersistenceConfig(session_id=None, identity=_identity()),
    )
    epoch_id = await store.start_epoch(_identity_stored())
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None, store=store, epoch_id=epoch_id,
        persistence_mode="save",
    )
    coordinator.bind(session)
    return session


async def test_close_supervision_expiry_keeps_store_until_inflight_end_settles(tmp_path, monkeypatch):
    """A14 反例: 监督到期时 end 事务仍在 worker 在途 —— lease/连接不提前释放."""
    import threading

    import cicada.application as app_module
    from cicada.persistence.contracts import LeaseHeldError
    from cicada.plugins.session_store import open_session_store
    from cicada.process_activity import ActivityCoordinator

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    coordinator = ActivityCoordinator()
    release_end = threading.Event()
    original_immediate = store._immediate
    zone = {"in_end": False}

    def gated_immediate():
        if zone["in_end"]:
            assert release_end.wait(timeout=60), "end 事务闸门未被打开"
        return original_immediate()

    store._immediate = gated_immediate  # type: ignore[method-assign]
    original_end = store.end_process

    async def marked_end(activity_id, terminal):
        zone["in_end"] = True
        try:
            return await original_end(activity_id, terminal)
        finally:
            zone["in_end"] = False

    store.end_process = marked_end  # type: ignore[method-assign]
    tool_gate = asyncio.Event()
    tool_gate.set()  # 工具立即返回: end 派发后挂在事务闸门上
    tool = _GateTool(tool_gate)

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.3)
    session = await _manual_persistent_session(
        tmp_path, _script(
            {"tool_calls": [{"id": "c1", "name": "powershell", "arguments": {}}],
             "tool_use": True},
            {"text": "答", "stop": True},
        ),
        tool, store=store, coordinator=coordinator,
    )
    execute_task = asyncio.create_task(session.execute("问"))
    await asyncio.wait_for(tool.started.wait(), timeout=5)
    close_task = asyncio.create_task(session.aclose())
    await asyncio.sleep(0.8)  # 监督已到期: close 越过生产者等待, 但 end 事务在途
    with pytest.raises(LeaseHeldError):
        open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    release_end.set()
    outcome = await asyncio.wait_for(execute_task, timeout=15)
    await asyncio.wait_for(close_task, timeout=15)
    # end 事务已在释放前排空落库; 但随后的完整提交在封店后被明确拒绝
    assert outcome.persistence is not None and not outcome.persistence.saved
    assert session.persistence.storage_failed is not None
    assert session.state == "closed", "迟到完成不得恢复 idle"
    assert session._close_problems, "监督到期如实进入关闭诊断"
    reopened = open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    snapshot = reopened.recovery_snapshot()
    assert snapshot.process_activities[0].terminal is not None, "迟到的 end 已落库"
    assert not snapshot.requests, "封店后的完整提交被拒绝, 请求未提交"
    assert len(snapshot.interrupted) == 1
    await reopened.aclose()


async def test_sealed_store_gives_late_writes_definitive_results(tmp_path, monkeypatch):
    """A14: 队列暂空即封店释放后, 迟到的 end/commit 得到明确拒绝, 不挂起不丢结果."""
    import cicada.application as app_module
    from cicada.plugins.session_store import open_session_store
    from cicada.process_activity import ActivityCoordinator

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    coordinator = ActivityCoordinator()
    tool_gate = asyncio.Event()
    tool = _GateTool(tool_gate)  # 工具挂住: close 到期时队列确为空

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.2)
    session = await _manual_persistent_session(
        tmp_path, _script(
            {"tool_calls": [{"id": "c1", "name": "powershell", "arguments": {}}],
             "tool_use": True},
            {"text": "答", "stop": True},
        ),
        tool, store=store, coordinator=coordinator,
    )
    execute_task = asyncio.create_task(session.execute("问"))
    await asyncio.wait_for(tool.started.wait(), timeout=5)
    close_task = asyncio.create_task(session.aclose())
    await asyncio.sleep(0.6)  # 监督到期: 队列空 -> store 封住并释放
    tool_gate.set()
    outcome = await asyncio.wait_for(execute_task, timeout=15)
    await asyncio.wait_for(close_task, timeout=15)
    tool_results = [m for m in outcome.result.messages if type(m).__name__ == "ToolResultMessage"]
    assert tool_results and tool_results[0].result.content == "ok", "不得改写成未执行"
    assert outcome.persistence is not None and not outcome.persistence.saved
    assert session.persistence.storage_failed is not None
    assert session.state == "closed"
    reopened = open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    late_activities = reopened.recovery_snapshot().process_activities
    assert len(late_activities) == 1 and late_activities[0].terminal is None, (
        "封店后的迟到 end 未写入; 活动保持未证终结 (恢复为查看状态)"
    )
    await reopened.aclose()


async def test_close_waits_for_inflight_clear_producer(tmp_path):
    """A14: close 与持久清空同时到达 —— 监督对象含清空生产者, 不提前封店."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    store = session._store
    release = asyncio.Event()
    original_clear = store.commit_clear

    async def gated_clear(epoch_id):
        await release.wait()
        return await original_clear(epoch_id)

    store.commit_clear = gated_clear  # type: ignore[method-assign]
    clear_task = asyncio.create_task(session.clear_context())
    await asyncio.sleep(0.1)  # commit_clear 已派发并挂在闸门上
    close_task = asyncio.create_task(session.aclose())
    await asyncio.sleep(0.3)
    assert session.state in ("closing", "closed"), "close 意图已锁存"
    release.set()
    await asyncio.wait_for(clear_task, timeout=15)
    await asyncio.wait_for(close_task, timeout=15)
    assert session.state == "closed"
    assert session._close_problems == [], "清空生产者在监督窗口内完成, 不进关闭诊断"
    # 清空已落库: 重开存档代次前进
    from cicada.plugins.session_store import open_session_store

    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    assert reopened.recovery_snapshot().context_generation == 1
    await reopened.aclose()


async def test_repeated_outer_cancel_during_commit_keeps_completion(tmp_path):
    """A04/A14: 提交等待时重复外层取消 —— completion 持续受拥有, 结果不丢."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    store = session._store
    release = asyncio.Event()
    original_commit = store.commit_request

    async def gated_commit(request_id, *, messages, outcome):
        result = await original_commit(request_id, messages=messages, outcome=outcome)
        await release.wait()
        return result

    store.commit_request = gated_commit  # type: ignore[method-assign]
    task = asyncio.create_task(session.execute("会取消提交等待的问"))
    await asyncio.sleep(0.1)  # 内核已完成, commit 事务已派发, completion 挂在闸门上
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()  # 重复取消: 不得丢弃 completion 或把事务当作未发生
    await asyncio.sleep(0.05)
    release.set()
    outcome = await asyncio.wait_for(task, timeout=15)
    assert outcome.persistence is not None and outcome.persistence.saved, (
        "受拥有 completion 的提交结果照常交付"
    )
    assert session.persistence.saved_request_count == 1
    await session.aclose()


# --- 02 审查处置: begin 等待取消 / 误锁存分类 / 清空等待取消 ------------------------------------


async def test_begin_cancel_waits_for_result_then_closing_without_mislatch(tmp_path):
    """A04: begin 等待外层取消 -> 事务明确结果落库, 应用进 closing, 不误锁存."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    store = session._store
    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    release = asyncio.Event()
    original_begin = store.begin_request

    async def gated_begin(epoch_id, prompt):
        result = await original_begin(epoch_id, prompt)
        await release.wait()
        return result

    store.begin_request = gated_begin  # type: ignore[method-assign]
    task = asyncio.create_task(session.execute("被取消在开始等待的问"))
    await asyncio.sleep(0.1)  # begin 事务已派发并挂起
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=15)
    assert session.state == "closing", "取消属程序性终止, 与执行期异常一样进 closing"
    assert session.persistence.storage_failed is None, "取消不得误锁存 storage_failed"
    assert model.requests == [], "零模型调用"
    with pytest.raises(RuntimeError, match="closing"):
        await session.execute("closing 后拒绝")
    await session.aclose()

    # 事务已落库: pending intent 留在存档 (下次 start_epoch 标中断)
    from cicada.plugins.session_store import open_session_store

    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    snapshot = reopened.recovery_snapshot()
    assert len(snapshot.interrupted) == 1, "取消留下的 pending intent 在快照中可见"
    await reopened.aclose()


async def test_begin_relation_conflict_not_latched_as_storage_failure(tmp_path, monkeypatch):
    """关系冲突 (pending 已存在) 是状态问题不是落盘失败: 不锁存, 文案不误导."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    store = session._store

    async def pending_conflict(epoch_id, prompt):
        raise PersistenceError(
            "a previous request is still pending; only one active request is allowed"
        )

    store.begin_request = pending_conflict  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cannot start a persisted request"):
        await session.execute("问")
    assert session.persistence.storage_failed is None, "关系冲突不得锁存 storage_failed"
    await session.aclose()

    # 对照: 真实写失败仍锁存
    session2 = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                             session_id=None))
    from cicada.persistence.contracts import StorageWriteError

    async def write_failure(epoch_id, prompt):
        raise StorageWriteError("disk gone")

    session2._store.begin_request = write_failure  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="session save failed before the request started"):
        await session2.execute("问")
    assert session2.persistence.storage_failed is not None
    await session2.aclose()


async def test_clear_cancel_syncs_memory_before_propagating(tmp_path):
    """A04: commit_clear 等待外层取消 -> 先等明确结果并同步内存, 再传播取消."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    store = session._store
    release = asyncio.Event()
    original_clear = store.commit_clear

    async def gated_clear(epoch_id):
        result = await original_clear(epoch_id)
        await release.wait()
        return result

    store.commit_clear = gated_clear  # type: ignore[method-assign]
    task = asyncio.create_task(session.clear_context())
    await asyncio.sleep(0.1)  # 清空事务已派发并挂起
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=15)
    assert session.history == (), "事务成功后内存先同步, 不留代次分叉"
    assert session.persistence.context_generation == 1
    assert session.state == "idle", "清空生产者完成, 会话不进 closing"
    await session.aclose()

    from cicada.plugins.session_store import open_session_store

    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    assert reopened.recovery_snapshot().context_generation == 1, "存档与内存一致"
    await reopened.aclose()


