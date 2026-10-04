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


async def test_resume_with_pending_intent_refused_without_side_effects(tmp_path):
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

    with pytest.raises(ApplicationOpenError, match="interrupted request"):
        await open_application(_config(root, _script(), session_id=session_id))


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
