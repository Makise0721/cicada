"""应用层持久协调聚焦测试: 保存->关闭->恢复->续写、失败锁存、清空边界、恢复准入.

真实共享应用 + fake 模型; 存储故障经 port seam 注入 (monkeypatch store 方法),
开库/占用/损坏由 tests/plugins/test_session_store.py 与 CLI 子进程测试覆盖.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from cicada.application import (
    AgentEventReceived,
    ApplicationConfig,
    ApplicationOpenError,
    ApplicationSession,
    OperationBusy,
    PersistenceConfig,
    RequestCompleted,
    _admit_persistent_resume,
    _build_prefix_proof,
    open_application,
)
from cicada.boot import bootstrap
from cicada.core.cancel import CancelToken
from cicada.core.events import ToolStarted
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta, ToolSpec
from cicada.persistence.codec import decode_identity
from cicada.persistence.contracts import (
    LaunchIdentity,
    LeaseHeldError,
    PersistenceError,
    RecoverySnapshot,
    RequestOutcomeFacts,
    RestoredRequest,
    StorageWriteError,
    StoredIdentity,
    StoredReceipt,
)
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.fake_tools import fake_tools_plugin
from cicada.plugins.session_store import open_session_store, session_store_plugin
from cicada.process_activity import ActivityCoordinator, ProcessActivityWrapper
from cicada.run_policy import CONTEXT_CLOSE_TAG, CONTEXT_OPEN_TAG, RunPolicy
from cicada.runtime.plugin import PluginDefinition

from test_application import _repo, _script  # 复用既有工作区/剧本构造
from tests.support.app import identity as _identity, prompt_info as _prompt_info


async def test_session_catalog_owned_during_cancel_close_and_idle_blocks(tmp_path, monkeypatch):
    from cicada.application import SessionManagementError

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(), session_id=None))
    reader = session._catalog_reader
    snapshot = await session.observe_session_catalog(CancelToken())
    assert snapshot.entries[0].session_id == session.persistence.session_id
    session._storage_failed = 'existing failure'
    assert (await session.observe_session_catalog(CancelToken())).complete
    entered, release = threading.Event(), threading.Event()
    original = reader._observe

    def blocked(stop):
        entered.set()
        assert release.wait(5)
        return original(stop)

    monkeypatch.setattr(reader, '_observe', blocked)
    task = asyncio.create_task(session.observe_session_catalog(CancelToken()))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        with pytest.raises(SessionManagementError) as busy:
            session.submit_session_name('busy')
        assert busy.value.code == 'busy'
        with pytest.raises(SessionManagementError):
            await session.observe_session_catalog(CancelToken())
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        close = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        assert not task.done() and not close.done() and session.catalog_active
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await close
    assert session.state == 'closed' and not session.catalog_active
    resumed = open_session_store(directory=session._store.capture().db_path.parent,
                                 session_id=session.persistence.session_id, workspace=root)
    await resumed.aclose()


async def test_session_name_roundtrip_noop_and_metadata_only(tmp_path, monkeypatch):
    from cicada.application import SessionManagementError, SessionNameChanged
    from cicada.plugins.session_store import list_sessions

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "回答", "stop": True}), session_id=None))
    store = session._store.capture()
    updates = []
    session.subscribe(updates.append)
    try:
        # 空档可命名；排序时间仍未知，UUID 保持。
        submission = session.submit_session_name("  中文 任务  ")
        assert session.session_management_active
        with pytest.raises(SessionManagementError) as busy:
            session.submit_session_name("另一项")
        assert busy.value.code == "busy"
        result = await submission.wait()
        assert result == await submission.wait() and result.changed
        empty = list_sessions(directory=store.db_path.parent).entries[0]
        assert empty.user_name == "中文 任务" and empty.updated_at is None
        await session.execute("自动首问")
        before = store.recovery_snapshot()
        history, outcome, epoch = session.history, session._last_outcome, session.persistence.epoch_id
        with sqlite3.connect(store.db_path) as connection:
            tables = {name: connection.execute(f'SELECT * FROM {name}').fetchall() for name in (
                'records', 'request_intents', 'epoch', 'process_activity', 'historical_receipts', 'summaries')}
        assert (await session.submit_session_name("重命名").wait()).changed
        with sqlite3.connect(store.db_path) as connection:
            assert tables == {name: connection.execute(f'SELECT * FROM {name}').fetchall() for name in tables}
        assert store.recovery_snapshot() == dc_replace(before, user_name="重命名")
        assert (session.history, session._last_outcome, session.persistence.epoch_id) == (history, outcome, epoch)
        assert session.session_generation == 0 and session.persistence.saved_request_count == 1
        event_count = len([u for u in updates if isinstance(u, SessionNameChanged)])
        original = store.submit_name
        monkeypatch.setattr(store, 'submit_name', lambda *a, **k: pytest.fail('no-op dispatched SQL'))
        assert not (await session.submit_session_name(" 重命名 ").wait()).changed
        assert len([u for u in updates if isinstance(u, SessionNameChanged)]) == event_count
        monkeypatch.setattr(store, 'submit_name', original)
        assert (await session.submit_session_name(None).wait()).changed
        assert store.recovery_snapshot().short_title == "自动首问"
        monkeypatch.setattr(store, 'submit_name', lambda *a, **k: pytest.fail('clear no-op dispatched SQL'))
        assert not (await session.submit_session_name(None).wait()).changed
        monkeypatch.setattr(store, 'submit_name', original)
        await session.submit_session_name("重启可读").wait()
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    resumed = await open_application(_config(root, _script(), session_id=session_id))
    try:
        assert resumed.persistence.user_name == "重启可读"
        assert resumed.history == history
        assert list_sessions(directory=store.db_path.parent).entries[0].user_name == "重启可读"
    finally:
        await resumed.aclose()


@pytest.mark.parametrize('close', [False, True])
async def test_accepted_name_cancellation_retains_result_and_reservation(tmp_path, monkeypatch, close):
    from cicada.application import SessionManagementError, SessionNameChanged
    from cicada.plugins.session_store import open_session_store

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(), session_id=None))
    store = session._store.capture()
    path, session_id = store.db_path, store.session_id
    entered, release = threading.Event(), threading.Event()
    original = store._immediate
    def slow():
        entered.set()
        assert release.wait(10), 'name transaction release timed out'
        original()
    monkeypatch.setattr(store, '_immediate', slow)
    updates = []
    session.subscribe(updates.append)
    submission = session.submit_session_name("已接受名称")
    waiter = asyncio.create_task(submission.wait())
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        # 取消等待和合作取消均不能取消已经接受的事务。
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        submission.cancel()
        submission.cancel()
        with pytest.raises(RuntimeError, match='session management'):
            await session.execute("不能穿透")
        with pytest.raises(RuntimeError, match='session management'):
            await session.clear_context()
        with pytest.raises(RuntimeError, match='session management'):
            await session.compact_context()
        with pytest.raises(SessionManagementError):
            session.submit_session_name("不能覆盖")
        if close:
            closing = asyncio.create_task(session.aclose())
            await asyncio.sleep(0)
            assert not closing.done()
        release.set()
        assert (await submission.wait()).changed
        assert await submission.wait() == await submission.wait()
        if closing is not None:
            await closing
        assert any(isinstance(u, SessionNameChanged) and u.user_name == "已接受名称" for u in updates)
        assert session.persistence.user_name == "已接受名称"
    finally:
        release.set()
        await session.aclose()
    reopened = open_session_store(directory=path.parent, session_id=session_id, workspace=root)
    try:
        assert reopened.recovery_snapshot().user_name == "已接受名称"
    finally:
        await reopened.aclose()


async def test_name_rejection_cancellation_and_write_failure(tmp_path, monkeypatch):
    from cicada.application import SessionManagementError, SessionNameChanged

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(), session_id=None))
    updates = []
    session.subscribe(updates.append)
    try:
        for name in (' ', 'x\ny', 'x\t', 'x\u200dy', '\ud800', 'x' * 81, 123):
            with pytest.raises(SessionManagementError) as invalid:
                session.submit_session_name(name)
            assert invalid.value.code == 'invalid-input'
        cancelled = session.submit_session_name('取消于接受前')
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled.wait()
        assert not session.session_management_active and session.persistence.user_name is None
        # 真实 SQLite 写拒绝；不是用应用抛异常冒充事务失败。
        with sqlite3.connect(session._store.capture().db_path) as connection:
            connection.execute("CREATE TRIGGER name_failure BEFORE UPDATE OF user_name ON session "
                               "BEGIN SELECT RAISE(ABORT, 'injected name write failure'); END")
        failed = session.submit_session_name('失败名')
        for _ in range(2):
            with pytest.raises(SessionManagementError) as error:
                await failed.wait()
            assert error.value.code == 'write-failed'
        assert session.persistence.storage_failed is not None
        assert session.persistence.user_name is None
        assert not any(isinstance(u, SessionNameChanged) for u in updates)
        assert session.history == () and session._last_outcome is None
        with pytest.raises(RuntimeError, match='storage_failed'):
            await session.execute('拒绝后继')
    finally:
        await session.aclose()
    ephemeral = await open_application(_plain_config(root, _script()))
    try:
        with pytest.raises(SessionManagementError) as error:
            ephemeral.submit_session_name('不可保存')
        assert error.value.code == 'unavailable' and ephemeral.history == ()
    finally:
        await ephemeral.aclose()


def _texts(messages) -> list[str]:
    """按消息类型取展示文本; 工具结果以 <tool:{name}> 标记."""
    return [
        f"<tool:{m.result.name}>" if isinstance(m, ToolResultMessage) else m.text
        for m in messages
    ]


def _config(root: Path, script: list, *, session_id: str | None):
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
    store = session._store.capture()
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
    store = session._store.capture()

    async def failing_commit(request_id, *, messages, outcome, receipts=()):
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
    store = session._store.capture()

    async def failing_commit(request_id, *, messages, outcome, receipts=()):
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


# --- 04: --resume-with-current-config (显式采用当前配置) ---------------------------------------


def _latest_epoch_identity(root: Path, session_id: str):
    """读存档最新 epoch 的身份; 用于核对差异记录通道."""
    db = root / ".cicada" / "sessions" / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    try:
        row = connection.execute(
            "SELECT identity_json FROM epoch ORDER BY opened_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return decode_identity(row[0])


async def test_resume_adopt_current_config_skips_conflict_and_records_epoch(tmp_path):
    """A11: 显式采用当前配置恢复——冲突不拒绝, 差异记入新 epoch, 历史与代次保持."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("第一问")
    session_id = session.persistence.session_id
    assert session.persistence.context_generation == 0
    await session.aclose()

    changed_identity = dc_replace(_identity(), max_turns=9, sources={"num_ctx": "cli"})
    config = dc_replace(
        _config(root, _script({"text": "恢复后的回答", "stop": True}), session_id=session_id),
        persistence=PersistenceConfig(
            session_id=session_id, identity=changed_identity, adopt_current_config=True
        ),
    )
    resumed = await open_application(config)
    assert resumed.persistence is not None and resumed.persistence.mode == "resume"
    assert resumed.persistence.restored_request_count == 1, "采用当前配置不清空恢复历史"
    assert resumed.persistence.context_generation == 0, "换配置不增上下文代次 (与清空耦合)"
    assert len(resumed.history) > 0, "恢复的对话历史保持"
    outcome = await resumed.execute("第二问")
    assert outcome.exit_code == 0 and outcome.persistence.saved
    await resumed.aclose()

    stored = _latest_epoch_identity(root, session_id)
    assert stored.launch.max_turns == 9, "新 epoch 记录采用的当前配置"
    assert stored.launch.sources.get("resume_config") == "current"
    assert stored.launch.sources.get("changed_fields") == "max_turns"
    assert stored.launch.sources.get("num_ctx") == "cli"


async def test_resume_adopt_current_config_without_differences_records_no_change(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    session_id = session.persistence.session_id
    await session.aclose()

    config = dc_replace(
        _config(root, _script(), session_id=session_id),
        persistence=PersistenceConfig(
            session_id=session_id, identity=_identity(), adopt_current_config=True
        ),
    )
    resumed = await open_application(config)
    await resumed.aclose()
    stored = _latest_epoch_identity(root, session_id)
    assert stored.launch.sources.get("resume_config") == "current"
    assert "changed_fields" not in stored.launch.sources, "无差异不记录 changed_fields"


async def test_resume_adopt_current_config_keeps_view_only(tmp_path):
    """A15: 显式配置覆盖不解除未证终结活动的查看状态."""
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

    changed_identity = dc_replace(_identity(), max_turns=9)
    config = dc_replace(
        _config(root, _script(), session_id=session_id),
        persistence=PersistenceConfig(
            session_id=session_id, identity=changed_identity, adopt_current_config=True
        ),
    )
    resumed = await open_application(config)
    assert resumed.view_only_reason is not None, "配置覆盖不解除 view-only"
    model = resumed._app.agent._model
    assert isinstance(model, FakeModel)
    with pytest.raises(RuntimeError, match="view-only"):
        await resumed.execute("新请求被拒绝")
    assert model.requests == [], "view-only 会话零模型调用"
    await resumed.aclose()


async def test_resume_adopt_current_config_adopts_changed_tool_fingerprint(
    tmp_path, monkeypatch
):
    """显式采用当前配置时工具指纹差异不拒绝 (当前启动整体被采用), 新指纹记入 epoch."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True}),
                                            session_id=None))
    await session.execute("问")
    session_id = session.persistence.session_id
    await session.aclose()

    import cicada.application as app_module

    monkeypatch.setattr(app_module, "_tool_fingerprint", lambda app: "fingerprint-changed")
    config = dc_replace(
        _config(root, _script(), session_id=session_id),
        persistence=PersistenceConfig(
            session_id=session_id, identity=_identity(), adopt_current_config=True
        ),
    )
    resumed = await open_application(config)
    assert resumed.persistence is not None and resumed.persistence.mode == "resume"
    await resumed.aclose()
    stored = _latest_epoch_identity(root, session_id)
    assert stored.tool_fingerprint == "fingerprint-changed"


def test_persistence_config_requires_launch_identity():
    """AC5: 无必要身份的持久编程调用拒绝; ephemeral 调用不受影响 (persistence=None)."""
    with pytest.raises(ValueError, match="LaunchIdentity"):
        PersistenceConfig(session_id=None, identity=None)  # type: ignore[arg-type]


# --- 02: 拓扑清理顺序与关闭拥有权 (A14) --------------------------------------------------------


def _recording_model_plugin(order: list, script: list):
    """带清理记录的 fake 模型插件定义; 清理顺序用于验证逆拓扑."""
    model = FakeModel(script)

    def setup(ctx):
        ctx.provide("model", model)
        ctx.defer(lambda: order.append("model"))

    return PluginDefinition(name="fake-model", setup=setup, provides=frozenset({"model"}))


async def test_persistent_topology_cleans_resources_before_store(tmp_path, monkeypatch):
    """A14: 显式 requires 的真实 open_application 装配 —— 模型先清理, store 最后释放."""
    import cicada.application as app_module

    root = _repo(tmp_path)
    order: list[str] = []
    original_plugin = app_module.session_store_plugin

    def patched_plugin(store, catalog=None):
        original_aclose = store.aclose

        async def recording_aclose():
            order.append("store")
            await original_aclose()

        store.aclose = recording_aclose  # type: ignore[method-assign]
        if catalog is not None:
            original_catalog_close = catalog.aclose

            async def recording_catalog_close():
                order.append("catalog")
                await original_catalog_close()

            catalog.aclose = recording_catalog_close
        return original_plugin(store, catalog)

    monkeypatch.setattr(app_module, "session_store_plugin", patched_plugin)
    config = dc_replace(
        _plain_config(root, _script({"text": "答", "stop": True})),
        model_plugin=_recording_model_plugin(order, _script({"text": "答", "stop": True})),
        persistence=PersistenceConfig(session_id=None, identity=_identity()),
    )
    session = await open_application(config)
    assert session.persistence is not None
    store = session._store.capture()
    await session.execute("问")
    await session.aclose()
    assert order == ["model", "catalog", "store"], f"store 必须最后清理, 实际 {order}"
    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    await reopened.aclose()


class _GateTool:
    """受控 powershell 形工具: 真实结果形态, 执行等待闸门 (A14 反例的可控工具)."""

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate
        self.spec = ToolSpec(name="powershell", description="gate", parameters={
            "type": "object", "properties": {},
        })
        self.started = asyncio.Event()

    async def execute(self, arguments, ctx):
        self.started.set()
        await self._gate.wait()
        return ToolResult(
            call_id=ctx.call_id, name="powershell", content="ok", is_error=False,
            details={"exit_code": 0, "timed_out": False, "cancelled": False,
                     "output_complete": True},
        )


def _identity_stored():
    return StoredIdentity(launch=_identity(), tool_fingerprint="fp")


async def _manual_persistent_session(tmp_path, script: list, tool, *, store, coordinator):
    """绕过 open_application 手工装配: 注入受控工具与受控事务 (A14 反例需要)."""
    from cicada.application import _bind_store_last_cleanup

    def tool_policy(inner, runtime):
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
    import cicada.application as app_module

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
    store = session._store.capture()
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
    store = session._store.capture()
    release = asyncio.Event()
    original_commit = store.commit_request

    async def gated_commit(request_id, *, messages, outcome, receipts=()):
        result = await original_commit(request_id, messages=messages, outcome=outcome,
                                       receipts=receipts)
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
    store = session._store.capture()
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
    store = session._store.capture()

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

    async def write_failure(epoch_id, prompt):
        raise StorageWriteError("disk gone")

    session2._store.capture().begin_request = write_failure  # type: ignore[method-assign]
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
    store = session._store.capture()
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

    reopened = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=store.session_id, workspace=root,
    )
    assert reopened.recovery_snapshot().context_generation == 1, "存档与内存一致"
    await reopened.aclose()




# --- 03: A08/A09/A10 检查模式持久化集成 -------------------------------------------------------

_CHECK_RUN = {"tool_calls": [{"id": "k1", "name": "check",
                             "arguments": {"action": "run", "check_id": "check-1"}}],
              "tool_use": True}


def _check_config(root, entries, *, session_id=None, command="exit 0", timeout_s=30.0):
    """检查模式 + 持久会话的应用配置; 同一 root/检查定义保证 checks_signature 一致."""
    from cicada.application import verification_checks_signature

    checks = (CheckDefinition("check-1", command, timeout_s=timeout_s),)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(), root=Path(root), checks=checks,
    )
    return ApplicationConfig(
        workspace=root, system_prompt="test system prompt", prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(_script(*entries))), plan=plan, max_turns=6,
        model_label="fake",
        persistence=PersistenceConfig(
            session_id=session_id, identity=_identity(verification_checks_signature(checks)),
        ),
    )


def _inner_model(session):
    model = session._app.agent._model
    assert isinstance(model, RunPolicy)
    return model.inner


async def test_check_mode_save_resume_old_pass_is_historical_new_ledger_independent(tmp_path):
    """A08: 旧真实 PASS 回执恢复到新验证运行 —— 新运行/基线/账本独立, 旧 PASS 仅历史."""
    root = _repo(tmp_path)
    session = await open_application(_check_config(root, (
        _CHECK_RUN, {"text": "已检查", "stop": True},
    )))
    first = await session.execute("第一问")
    assert first.exit_code == 0, "检查真实跑过且通过"
    assert first.view is not None and first.view.checks[0].status == "passed"
    old_run_id = session.verification_run_id
    old_receipt_id = first.view.receipts[0].receipt_id
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_check_config(root, (
        {"text": "恢复后作答", "stop": True},
    ), session_id=session_id))
    assert resumed.verification_run_id != old_run_id, "新验证运行"
    inner = _inner_model(resumed)
    outcome2 = await resumed.execute("第二问")
    assert outcome2.exit_code == 3, "新账本空: 检查未运行不交付"
    assert outcome2.view is not None
    assert outcome2.view.checks[0].status == "not_run", "旧 PASS 不进当前账本"
    assert outcome2.view.process_uncertain is False, "旧前缀不按新账本锁存"

    sent = inner.requests[0]
    tool_messages = [m for m in sent.messages if isinstance(m, ToolResultMessage)]
    assert tool_messages, "实际 ModelRequest 携带恢复前缀"
    # 缓存 02: 恢复的旧文字不清洗不标注, 执行时正文原样透传 (历史叙述由固定规则说明)
    assert "Cicada freshness" not in tool_messages[0].result.content
    assert "historical result" not in tool_messages[0].result.content
    assert "<cicada-verification-rules>" in sent.system_prompt, "固定规则在 system"
    tail_text = sent.request_context
    assert tail_text.startswith(CONTEXT_OPEN_TAG)
    tail = json.loads(tail_text[len(CONTEXT_OPEN_TAG) + 1:len(tail_text) - len(CONTEXT_CLOSE_TAG) - 1])
    assert tail["verification_run_id"] != old_run_id, "尾部是当前运行的"
    assert tail["checks"][0]["status"] == "not_run", "旧 PASS 不进当前尾部"
    # 原始记录不改: 规范历史无投影标记, details/call 身份保持
    canonical_tool = [m for m in resumed.history if isinstance(m, ToolResultMessage)][0]
    assert "historical result" not in canonical_tool.result.content
    assert canonical_tool.result.details["receipt_id"] == old_receipt_id
    assert canonical_tool.result.call_id == "k1"
    await resumed.aclose()

    # 新检查跑过才可能当前通过; 新回执独立持久化
    third = await open_application(_check_config(root, (
        _CHECK_RUN, {"text": "重新检查后作答", "stop": True},
    ), session_id=session_id))
    outcome3 = await third.execute("第三问")
    assert outcome3.exit_code == 0
    assert outcome3.view is not None
    assert outcome3.view.checks[0].freshness == "current"
    assert outcome3.view.receipts[0].receipt_id != old_receipt_id, "新回执与旧回执身份不同"
    await third.aclose()

    store = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root,
    )
    snapshot = store.recovery_snapshot()
    receipt_ids = {r.receipt_id for r in snapshot.restored_receipts}
    assert {old_receipt_id} < receipt_ids and len(receipt_ids) == 2
    assert not snapshot.recorded_process_uncertain
    await store.aclose()


async def test_new_session_keeps_verifier_and_routes_actual_read_edit_receipts(tmp_path):
    from cicada.application import SessionChanged, SessionManagementError
    from cicada.persistence.routing import SessionStoreRoute

    root = _repo(tmp_path)
    edit = {"tool_calls": [
        {"id": "read-b", "name": "read", "arguments": {"path": "sample.py"}},
        {"id": "edit-b", "name": "edit", "arguments": {"path": "sample.py", "edits": [
            {"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]}},
        {"id": "check-b", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
    ], "tool_use": True}
    session = await open_application(_check_config(root, (
        _CHECK_RUN, {"text": "A", "stop": True}, {"text": "B无检查", "stop": True},
        edit, {"text": "B已修改检查", "stop": True})))
    route = session._store
    assert isinstance(route, SessionStoreRoute)
    updates = []
    session.subscribe(updates.append)
    try:
        a = await session.execute("A检查")
        source = route.capture()
        assert a.persistence.saved and len(source.recovery_snapshot().restored_receipts) == 1
        stable = (session._app, session._service, session._projection, session._context_projection,
                  session.verification_run_id, session._service.baseline_capture)
        submission = session.submit_new_session()
        assert session.session_management_active
        with pytest.raises(SessionManagementError):
            session.submit_new_session()
        with pytest.raises(RuntimeError):
            await session.execute("同步预约拒绝")
        result = await submission.wait()
        assert result == await submission.wait() and result.session_generation == 1
        target = route.capture()
        assert target is not source and target.session_id != source.session_id
        assert stable == (session._app, session._service, session._projection, session._context_projection,
                          session.verification_run_id, session._service.baseline_capture)
        assert session._app.runtime.capability("session.store") is route
        assert session.history == () and session._last_outcome is None
        assert session.persistence.saved_request_count == 0
        assert next(u for u in updates if isinstance(u, SessionChanged)).snapshot.requests == ()
        b = await session.execute("B只回答")
        assert b.persistence.saved and len(b.view.receipts) == 1
        assert target.recovery_snapshot().restored_receipts == (), "共享检查视图不能复制A回执"
        c = await session.execute("B读改检查")
        assert c.exit_code == 0 and "VALUE = 2" in (root / "sample.py").read_text()
        snapshot = target.recovery_snapshot()
        assert len(snapshot.restored_receipts) == 1
        receipt = snapshot.restored_receipts[0]
        assert receipt.request_id == c.persistence.request_id and receipt.epoch_id == session.persistence.epoch_id
        assert receipt.receipt_id != a.view.receipts[0].receipt_id
        assert snapshot.process_activities[0].request_id == c.persistence.request_id
        assert source._closed and route._owned == [target]
        # old lease 已实际释放，原档可重开且没有B的消息/回执。
        reopened = open_session_store(directory=source.db_path.parent, session_id=source.session_id, workspace=root)
        assert len(reopened.recovery_snapshot().requests) == 1
        await reopened.aclose()
        target_id = target.session_id
    finally:
        await session.aclose()
    resumed = await open_application(_check_config(root, ({"text": "重启", "stop": True},), session_id=target_id))
    try:
        assert len(resumed.restored_requests) == 2 and len(resumed.restored_receipts) == 1
        assert (await resumed.execute("续写B")).persistence.saved
    finally:
        await resumed.aclose()


@pytest.mark.parametrize("phase", ["opening", "epoch"])
async def test_new_cancel_and_close_wait_for_owned_prepare_without_late_publish(tmp_path, monkeypatch, phase):
    from cicada.application import SessionChanged

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(), session_id=None))
    source = session._store.capture()
    entered, release = threading.Event(), threading.Event()
    targets = []
    factory = session._new_store_factory

    def gated_factory():
        target = factory()
        targets.append(target)
        if phase == "opening":
            entered.set()
            assert release.wait(5)
        else:
            immediate = target._immediate
            def gated_epoch():
                entered.set()
                assert release.wait(5)
                immediate()
            target._immediate = gated_epoch
        return target

    monkeypatch.setattr(session, "_new_store_factory", gated_factory)
    updates = []
    session.subscribe(updates.append)
    submission = session.submit_new_session()
    assert await asyncio.to_thread(entered.wait, 5)
    waiter = asyncio.create_task(submission.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    submission.cancel()
    submission.cancel()
    close = asyncio.create_task(session.aclose())
    await asyncio.sleep(0)
    assert not close.done() and session.session_management_active
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await submission.wait()
    await close
    assert session._store.capture() is source and session.session_generation == 0
    assert not any(isinstance(u, SessionChanged) for u in updates)
    assert targets[0]._closed and source._closed and session._store._owned == []
    # 空档或epoch提交留在磁盘，取消不假装撤销。
    reopened = open_session_store(directory=source.db_path.parent, session_id=targets[0].session_id, workspace=root)
    assert (reopened.recovery_snapshot().identity is not None) == (phase == "epoch")
    await reopened.aclose()


@pytest.mark.parametrize("published", [False, True])
async def test_new_cleanup_failure_keeps_resource_and_blocks_followups(tmp_path, monkeypatch, published):
    from cicada.application import SessionManagementError

    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(), session_id=None))
    route, source = session._store, session._store.capture()
    factory = session._new_store_factory
    targets = []

    async def failed_close():
        raise StorageWriteError("close injected failure")

    def target_factory():
        target = factory()
        targets.append(target)
        if not published:
            async def failed_epoch(identity):
                raise StorageWriteError("epoch injected failure")
            target.start_epoch = failed_epoch
            target.aclose = failed_close
        return target

    monkeypatch.setattr(session, "_new_store_factory", target_factory)
    source_close = source.aclose
    if published:
        monkeypatch.setattr(source, "aclose", failed_close)
    try:
        with pytest.raises(SessionManagementError) as error:
            await session.submit_new_session().wait()
        assert error.value.code == "cleanup-failed" and error.value.published == published
        assert (route.capture() is targets[0]) == published
        assert len(route._owned) == 2
        with pytest.raises(LeaseHeldError):
            open_session_store(directory=source.db_path.parent,
                               session_id=(source if published else targets[0]).session_id, workspace=root)
        with pytest.raises(SessionManagementError) as blocked:
            session.submit_new_session()
        assert blocked.value.code == "blocked"
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await session.execute("不能绕过")
        assert (await session.observe_session_catalog(CancelToken())).complete
    finally:
        source.aclose = source_close
        if not published:
            # 恢复真实清理；锁存仍持续，运行中没有解锁入口。
            targets[0].aclose = targets[0].__class__.aclose.__get__(targets[0])
        await session.aclose()
    assert route._owned == []


async def test_prefix_tamper_after_resume_refuses_before_sending(tmp_path):
    """A09: 恢复后前缀被改动 -> 发送前拒绝, 内层模型零请求."""
    root = _repo(tmp_path)
    session = await open_application(_check_config(root, (
        {"text": "答", "stop": True}, {"text": "二答", "stop": True},
    )))
    await session.execute("第一问")
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_check_config(root, (
        {"text": "恢复后作答", "stop": True},
    ), session_id=session_id))
    resumed._history[0] = UserMessage(text="被改写的前缀")  # 模拟前缀改动
    outcome = await resumed.execute("第二问")
    assert outcome.result.stop_reason == "error"
    assert "changed since the proof was built" in (outcome.result.error or "")
    assert _inner_model(resumed).requests == [], "发送前拒绝, 零请求"
    await resumed.aclose()


async def test_check_then_clear_drops_context_keeps_display_and_receipts(tmp_path):
    """A10: 真实检查后清空 —— 保存恢复不复活旧上下文, 旧显示与回执保留."""
    root = _repo(tmp_path)
    session = await open_application(_check_config(root, (
        _CHECK_RUN, {"text": "已检查", "stop": True}, {"text": "清空后作答", "stop": True},
    )))
    busy_operations: list[str] = []

    def on_update(update):
        if isinstance(update, OperationBusy):
            busy_operations.append(update.operation)

    session.subscribe(on_update)
    first = await session.execute("第一问")
    assert first.exit_code == 0
    old_receipt_id = first.view.receipts[0].receipt_id
    await session.clear_context()
    assert busy_operations == ["clear"], "operation_busy 可消费"
    assert session.history == ()
    second = await session.execute("第二问")
    assert second.persistence is not None and second.persistence.saved
    # 账本/基线保持: 清空不触碰验证服务, 新请求仍属同一验证运行 (A10)
    assert second.view is not None
    assert second.view.verification_run_id == session.verification_run_id
    assert second.view.checks[0].status == "passed", "旧检查账本仍在, freshness 不被清空重置"
    assert session.restored_receipts == (), "新会话无恢复回执"
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_check_config(root, (
        {"text": "恢复后作答", "stop": True},
    ), session_id=session_id))
    assert [m.text for m in resumed.history if isinstance(m, UserMessage)] == ["第二问"], (
        "清空边界后的模型上下文不复活旧消息"
    )
    assert resumed.persistence.restored_request_count == 2, "原始显示记录保留"
    await resumed.aclose()

    store = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root,
    )
    snapshot = store.recovery_snapshot()
    assert [r.receipt_id for r in snapshot.restored_receipts] == [old_receipt_id], "旧回执保留"
    await store.aclose()


async def test_clear_failure_keeps_memory_and_prefix_proof(tmp_path):
    """A04 清空路径: 提交失败时内存历史与初始恢复前缀证明都不变."""
    root = _repo(tmp_path)
    first = await open_application(_check_config(root, (
        {"text": "答", "stop": True},
    )))
    await first.execute("第一问")
    session_id = first.persistence.session_id
    await first.aclose()

    session = await open_application(_check_config(root, (
        {"text": "二答", "stop": True},
    ), session_id=session_id))
    assert session._projection is not None
    assert session._projection.proof.length == 2, "恢复前缀为初始历史"
    store = session._store.capture()

    async def failing_clear(epoch_id):
        raise PersistenceError("disk gone")

    store.commit_clear = failing_clear  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="in-memory history is unchanged"):
        await session.clear_context()
    assert len(session.history) == 2, "内存历史不变"
    # 前缀证明同样未重置 (清空失败 -> 两者都保持; storage_failed 另按契约锁存)
    assert session._projection.proof.length == 2, "前缀证明未随失败清空重置"
    await session.aclose()


async def test_uncertain_latch_survives_clear_and_restore_is_view_only(tmp_path):
    """A15: 超时检查锁存 process_uncertain —— 清空不解锁, 恢复仅可查看且新服务同步."""
    root = _repo(tmp_path)
    session = await open_application(_check_config(root, (
        _CHECK_RUN, {"text": "超时后作答", "stop": True},
    ), command="Start-Sleep -Seconds 8", timeout_s=1.0))
    first = await session.execute("第一问")
    assert first.view is not None and first.view.process_uncertain
    assert session.has_process_uncertain()
    with pytest.raises(RuntimeError, match="process_uncertain"):
        await session.execute("锁存后拒绝新请求")
    await session.clear_context()
    assert session.has_process_uncertain(), "清空不解除锁存"
    view = await session._service.refresh(CancelToken())
    assert view.process_uncertain, "验证服务锁存与控制段保持"
    session_id = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_check_config(root, (
        {"text": "恢复后作答", "stop": True},
    ), command="Start-Sleep -Seconds 8", timeout_s=1.0, session_id=session_id))
    # 超时检查同时留下未证活动与不可信回执; 此处任一证据都阻断 (收据路径单独单测)
    assert resumed.view_only_reason is not None
    with pytest.raises(RuntimeError, match="view-only"):
        await resumed.execute("恢复后仍拒绝")
    view2 = await resumed._service.refresh(CancelToken())
    assert view2.process_uncertain, "新验证服务同步锁存, 交付判断一致"
    await resumed.aclose()


def test_admission_derives_view_only_from_untrusted_receipts_alone():
    """A15: 无未证活动时, 不可信回执单独构成跨恢复 process_uncertain 证据."""
    identity = _identity()
    snapshot = RecoverySnapshot(
        session_id="s", canonical_workspace="w", created_at=0.0, short_title="",
        identity=StoredIdentity(launch=identity, tool_fingerprint="fp"),
        epoch_count=1, context_generation=0, requests=(), interrupted=(),
        canonical_history=(),
        restored_receipts=(
            StoredReceipt("run-old", "r-bad", "e", "q", "f" * 64, "timed_out", None,
                          True, False, False, "timeout"),
        ),
    )
    reason = _admit_persistent_resume(
        PersistenceConfig(session_id="s", identity=identity), snapshot,
    )
    assert reason is not None
    assert "process_uncertain" in reason
    assert "r-bad" in reason


# --- 03 审查处置: 恢复关系错误的证明构建分支 -----------------------------------------------


def _proof_snapshot(messages, receipts):
    """构造带单一已提交请求的恢复快照 (直接喂 _build_prefix_proof)."""
    identity = _identity()
    return RecoverySnapshot(
        session_id="s", canonical_workspace="w", created_at=0.0, short_title="",
        identity=StoredIdentity(launch=identity, tool_fingerprint="fp"),
        epoch_count=1, context_generation=0,
        requests=(RestoredRequest(
            request_id="q1", epoch_id="e1", seq=2, context_generation=0, prompt="问",
            messages=tuple(messages), outcome=RequestOutcomeFacts(
                run_id="run-1", stop_reason="stop", error=None, exit_code=0, finalized=True,
            ), committed_at=0.0,
        ),),
        interrupted=(), canonical_history=tuple(messages),
        restored_receipts=tuple(receipts),
    )


def _check_messages_pair(receipt_id="r-1", *, details=None):
    if details is None:
        details = {"receipt_id": receipt_id, "exit_code": 0, "timed_out": False,
                   "cancelled": False, "output_complete": True}
    messages = (
        UserMessage(text="问"),
        AssistantMessage(text="", tool_calls=(ToolCall("k1", "check", "{}"),)),
        ToolResultMessage(result=ToolResult("k1", "check", "passed", details=details)),
        AssistantMessage(text="答"),
    )
    return messages


def _stored(receipt_id="r-1", **overrides):
    fields = dict(
        verification_run_id="run-old", receipt_id=receipt_id, epoch_id="e1",
        request_id="q1", definition_fingerprint="f" * 64, execution_status="exited",
        exit_code=0, timed_out=False, cancelled=False, output_complete=True,
        failure_kind=None,
    )
    fields.update(overrides)
    return StoredReceipt(**fields)


def test_prefix_proof_rejects_missing_and_mismatched_receipt_facts():
    """契约 §5 恢复关系错误: 回执缺失 / 事实与消息矛盾 -> 拒绝, 模型零请求."""
    # 回执缺失
    with pytest.raises(ApplicationOpenError, match="not in the archive"):
        _build_prefix_proof(_proof_snapshot(_check_messages_pair("r-missing"), ()))

    # 事实矛盾: 消息声称 exit_code=0/output_complete=True, 回执记录 timed_out
    with pytest.raises(ApplicationOpenError, match="contradicts receipt r-1"):
        _build_prefix_proof(_proof_snapshot(
            _check_messages_pair(), (_stored(timed_out=True, exit_code=None,
                                             output_complete=False),),
        ))

    # 一致事实通过
    projection = _build_prefix_proof(_proof_snapshot(_check_messages_pair(), (_stored(),)))
    assert projection.proof.length == 4
    assert projection.proof.entries[2].receipt_id == "r-1"


def test_prefix_proof_rejects_requests_not_reconstructing_history():
    """requests 拼接与 canonical_history 逐位置不符 -> 关系错误 (显式校验)."""
    snapshot = _proof_snapshot(_check_messages_pair(), (_stored(),))
    # 篡改 canonical_history 使与 requests 分叉 (条数相同内容不同)
    tampered = dc_replace(
        snapshot,
        canonical_history=(UserMessage(text="被改写的问"),) + snapshot.canonical_history[1:],
    )
    with pytest.raises(ApplicationOpenError, match="do not reconstruct"):
        _build_prefix_proof(tampered)


async def test_close_during_request_run_commit_still_saved(tmp_path):
    """阶段收口审查 STG-F2: close 于请求执行在途时启动——提交仍落库、不误锁存.

    钉住"execute 屏障覆盖活动终结及 commit_request"的组合: close 等待请求生产者,
    提交派发先于 store 释放, 不因关闭产生 StorageUnavailableError 误锁
    storage_failed (与摘要路径 PA05 的关闭取消口径一致)。
    """
    release = asyncio.Event()

    class _GatedModel:
        def __init__(self) -> None:
            self.requests: list = []

        async def stream(self, request, cancel):
            self.requests.append(request)
            await release.wait()
            if cancel.cancelled:
                yield StreamDone("aborted")
                return
            yield TextDelta("答")
            yield StreamDone("stop", metrics=ModelMetrics(input_tokens=10, output_tokens=1))

    model = _GatedModel()
    config = dc_replace(
        _config(_repo(tmp_path), [], session_id=None),
        model_plugin=fake_model_plugin(model),
    )
    session = await open_application(config)
    try:
        execute_task = asyncio.create_task(session.execute("问"))
        await asyncio.sleep(0.2)
        assert session.state == "executing", "请求执行在途"
        close_task = asyncio.create_task(session.aclose())
        await asyncio.sleep(0.2)
        assert session.state in ("executing", "cancelling", "closing"), "close 意图已锁存"
        release.set()
        outcome = await asyncio.wait_for(execute_task, timeout=15)
        assert outcome.persistence is not None and outcome.persistence.saved, (
            "close 期间请求提交仍落库"
        )
        await asyncio.wait_for(close_task, timeout=15)
        assert session.state == "closed"
        assert session._close_problems == [], "请求生产者在监督窗口内完成"
    finally:
        if session.state != "closed":
            await session.aclose()

async def test_file_mutation_clear_and_resume_require_current_process_read(tmp_path):
    """A08: 历史与新增 mutation JSON 可恢复, 读取资格只属当前 Workspace."""
    from cicada.core.messages import ToolResultMessage
    root = _repo(tmp_path)
    def call(name, arguments):
        return {"tool_calls": [{"id": name, "name": name, "arguments": arguments}]}
    session = await open_application(_config(root, _script(
        call("read", {"path": "sample.py"}), {"text": "read", "stop": True},
        call("edit", {"path": "sample.py", "edits": [{"old_text": "1", "new_text": "2"}]}),
        {"text": "edited", "stop": True},
    ), session_id=None))
    await session.execute("read")
    await session.clear_context()
    result = await session.execute("edit after clear")
    mutation = next(m.result for m in result.result.messages if isinstance(m, ToolResultMessage))
    assert mutation.details["mutation"]["state"] == "committed"
    assert result.persistence.saved
    sid = session.persistence.session_id
    await session.aclose()

    resumed = await open_application(_config(root, _script(
        call("write", {"path": "sample.py", "content": "VALUE = 3\n"}),
        {"text": "refused", "stop": True},
        call("read", {"path": "sample.py"}),
        call("write", {"path": "sample.py", "content": "VALUE = 3\n"}),
        {"text": "updated", "stop": True},
    ), session_id=sid))
    restored = [m.result for m in resumed.history if isinstance(m, ToolResultMessage)]
    assert restored[-1].details["mutation"] == mutation.details["mutation"]
    refused = await resumed.execute("write without fresh read")
    facts = [m.result for m in refused.result.messages if isinstance(m, ToolResultMessage)]
    assert facts[-1].details["mutation"]["reason"] == "read_required"
    assert (root / "sample.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    updated = await resumed.execute("read then write")
    assert updated.persistence.saved
    assert (root / "sample.py").read_bytes() == b"VALUE = 3\n"
    await resumed.aclose()


async def test_cancelled_native_creation_preserves_application_history_and_saved_fact(tmp_path, monkeypatch):
    """A09: 发布中 Task.cancel 不能泛化已写结果, 核心不再请求后续模型轮次."""
    from cicada.plugins.coding import file_commit_windows as native
    from cicada.core.events import ToolCompleted
    from cicada.ui.tool_view import ToolDocument
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"tool_calls": [{"id": "write", "name": "write",
                         "arguments": {"path": "created.txt", "content": "created"}}]},
        {"text": "must not run", "stop": True},
    ), session_id=None))
    entered, release = threading.Event(), threading.Event()
    real_create = native.create_file
    updates = []
    session.subscribe(updates.append)
    document = ToolDocument()
    session.subscribe(document.apply)

    def blocked_create(*args):
        entered.set()
        assert release.wait(10)
        return real_create(*args)

    monkeypatch.setattr(native, "create_file", blocked_create)
    task = asyncio.create_task(session.execute("create then cancel"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
    finally:
        release.set()
    outcome = await asyncio.wait_for(task, 10)
    assert outcome.result.stop_reason == "aborted" and outcome.persistence.saved
    assert len(session._app.agent._model.requests) == 1
    tool_result = next(m.result for m in session.history if isinstance(m, ToolResultMessage))
    assert tool_result.details["mutation"]["state"] == "committed"
    assert tool_result.details["mutation"]["caller_cancelled"]
    forwarded = [u.event.result for u in updates
                 if isinstance(u, AgentEventReceived) and isinstance(u.event, ToolCompleted)]
    assert forwarded[-1].details == tool_result.details
    assert len(document.entries) == 1
    assert document.entries[0].output_text.startswith("[mutation=committed]")
    assert document.entries[0].status == "success"
    sid = session.persistence.session_id
    workspace = session._app.runtime.capability("coding.workspace")
    assert all(not lock.locked() for lock in workspace._locks.values())
    await session.aclose()
    resumed = await open_application(_config(root, _script(), session_id=sid))
    restored = next(m.result for m in resumed.history if isinstance(m, ToolResultMessage))
    assert restored.details == tool_result.details
    assert resumed.restored_requests[-1].outcome.stop_reason == "aborted"
    await resumed.aclose()
