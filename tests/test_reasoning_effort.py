"""票据 03 关键事务回归: 思考强度切换的成功/失败/接受前后取消与恢复 (契约 C5/C6).

只覆盖票据 Validation 点名而现有检查无法证明的缺口: 事务发布顺序
(存储 epoch 先于同步发布)、身份 sources 的 selection 标注、接受后取消等
真实结果、存储失败保留旧值并锁存。UI badge/键位见 tests/ui/test_effort_badge.py。
"""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from cicada.application import (
    ModelSelectionError, OperationBusy, ReasoningEffortChanged, open_application,
)
from cicada.models import ReasoningEffortState
from cicada.plugins.deepseek import DeepSeekConfig
from cicada.plugins.session_store.store import read_session_identity
from test_application import _repo
from test_deepseek_application import config, response


def transport(bodies):
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=response(None, "答案"))
    return httpx.MockTransport(handle)


def _events(session, kind):
    seen = []
    session.subscribe(lambda u: seen.append(u) if isinstance(u, kind) else None)
    return seen


async def test_transaction_publishes_state_event_identity_and_restores_archive(tmp_path):
    root = _repo(tmp_path)
    bodies = []
    session = await open_application(config(root, transport(bodies)))
    try:
        assert session.reasoning_effort == ReasoningEffortState(("low", "high", "max"), "high", True)
        changes = _events(session, ReasoningEffortChanged)
        old_epoch = session.persistence.epoch_id
        history = session.history
        await session.set_reasoning_effort("max")
        # 事件: previous/current 均为已生效状态; epoch 推进; 零 HTTP、历史不变
        assert len(changes) == 1
        assert changes[0].previous.current == "high" and changes[0].current.current == "max"
        assert changes[0].epoch_id == session.persistence.epoch_id != old_epoch
        assert bodies == [] and session.history == history
        assert session.reasoning_effort.current == "max"
        assert session.config.model_label == "deepseek-flash"
        # 身份: 仅 reasoning_effort 标 selection, 其他来源保留
        sources = session.config.persistence.identity.sources
        assert sources.get("reasoning_effort") == "selection"
        assert session.config.persistence.identity.think is True
        # 预算锚随发布失效 (C6)
        assert session.budget is None or session.budget.anchor is None
    finally:
        await session.aclose()
    # 正常恢复继承归档值: 恢复准入要求启动身份与存档一致, binding 由该启动配置来
    # (CLI 恢复路径即从存档 identity 推导 DeepSeekConfig, 同 test_model_selection)
    saved = read_session_identity(directory=root / ".cicada" / "sessions",
                                  session_id=session.persistence.session_id, workspace=root)
    assert saved.launch.options["reasoning_effort"] == "max"
    initial = config(root, transport(bodies), model_config=DeepSeekConfig(reasoning_effort="max"))
    initial = replace(initial, persistence=replace(initial.persistence, identity=saved.launch))
    restored = await open_application(initial)
    try:
        assert restored.reasoning_effort.current == "max"
    finally:
        await restored.aclose()


async def test_same_level_noop_unknown_level_and_unsupported_provider(tmp_path):
    bodies = []
    session = await open_application(config(_repo(tmp_path), transport(bodies)))
    try:
        changes = _events(session, ReasoningEffortChanged)
        epoch = session.persistence.epoch_id

        def forbidden(*args, **kwargs):
            pytest.fail("same level must not prepare or write")
        session._model_route.binding.model.prepare_reasoning_effort = forbidden  # type: ignore[method-assign]
        await session.set_reasoning_effort("high")  # 同值已启用 no-op
        assert changes == [] and session.persistence.epoch_id == epoch and bodies == []

        with pytest.raises(ModelSelectionError, match="unknown reasoning effort"):
            await session.set_reasoning_effort("ultra")
        # 与 select_model 共用的独占事务准入: 在途/锁定状态拒绝
        session._begin_in_flight = True
        try:
            with pytest.raises(ModelSelectionError, match="progress"):
                await session.set_reasoning_effort("low")
        finally:
            session._begin_in_flight = False
    finally:
        await session.aclose()
    # 无能力 provider: 状态 enabled=False, 应用拒绝
    from tests.ui.test_conversation_view import _config as fake_config, _script
    fake_root = tmp_path / "fake"
    fake_root.mkdir()
    fake_session = await open_application(
        fake_config(fake_root, _script({"text": "答", "stop": True})))
    try:
        assert fake_session.reasoning_effort == ReasoningEffortState((), None, False)
        with pytest.raises(ModelSelectionError, match="unavailable"):
            await fake_session.set_reasoning_effort("low")
    finally:
        await fake_session.aclose()


async def test_off_session_enable_low_marks_think_and_effort_selection(tmp_path):
    bodies = []
    root = _repo(tmp_path)
    session = await open_application(config(root, transport(bodies), model_config=DeepSeekConfig(think=False)))
    try:
        assert session.reasoning_effort == ReasoningEffortState(("low", "high", "max"), None, True)
        changes = _events(session, ReasoningEffortChanged)
        await session.set_reasoning_effort("low")
        assert len(changes) == 1
        assert changes[0].previous.current is None and changes[0].current.current == "low"
        identity = session.config.persistence.identity
        assert identity.think is True
        assert identity.sources.get("think") == "selection"
        assert identity.sources.get("reasoning_effort") == "selection"
    finally:
        await session.aclose()


async def test_thinking_tool_history_rejects_enable_and_keeps_off(tmp_path):
    """关闭思考产生的无 reasoning 工具轮次: 启用被拒, 保持关闭并给有界原因 (C6)."""
    calls = [response(None, "中间", calls=(("c1", "read", {"path": "a.py"}),)), response(None, "答案")]

    def handle(req):
        body = json.loads(req.content)
        if body["thinking"]["type"] == "disabled":
            return httpx.Response(200, content=calls.pop(0))
        return httpx.Response(200, content=response(None, "不应发生"))

    root = _repo(tmp_path)
    session = await open_application(config(root, httpx.MockTransport(handle), model_config=DeepSeekConfig(think=False)))
    try:
        (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
        await session.execute("读取 a.py")
        changes = _events(session, ReasoningEffortChanged)
        with pytest.raises(ModelSelectionError, match="rejected by configuration or history"):
            await session.set_reasoning_effort("low")
        assert changes == [] and session.reasoning_effort.current is None
        assert session._model_in_flight is False
    finally:
        await session.aclose()


async def test_before_acceptance_cancel_zero_commit(tmp_path):
    session = await open_application(config(_repo(tmp_path), transport([])))
    calls = []
    real = session._store.submit_model_change
    def submit(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    task = None
    try:
        def observe(update):
            if isinstance(update, OperationBusy) and update.operation == "model":
                task.cancel()
        session.subscribe(observe)
        task = asyncio.create_task(session.set_reasoning_effort("max"))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [] and session.reasoning_effort.current == "high"
        assert session._model_in_flight is False
        await session.set_reasoning_effort("max")  # 取消不锁存, 可重试
        assert session.reasoning_effort.current == "max"
    finally:
        await session.aclose()


@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_accepted_commit_waits_real_result_then_publishes(tmp_path, action):
    session = await open_application(config(_repo(tmp_path), transport([])))
    real = session._store.submit_model_change
    accepted = asyncio.Event()
    release = asyncio.Event()
    def submit(*args, **kwargs):
        handle = real(*args, **kwargs)
        accepted.set()
        class Delayed:
            async def wait(self):
                epoch = await handle.wait()
                await release.wait()
                return epoch
        return Delayed()
    session._store.submit_model_change = submit
    changes = _events(session, ReasoningEffortChanged)
    task = asyncio.create_task(session.set_reasoning_effort("max"))
    close = None
    try:
        await accepted.wait()
        # 接受后同型事务互斥: 模型选择/强度切换均被挡
        with pytest.raises(ModelSelectionError, match="progress"):
            await session.select_model("deepseek-v4-pro")
        with pytest.raises(ModelSelectionError, match="progress"):
            await session.set_reasoning_effort("low")
        assert session.reasoning_effort.current == "high"
        if action == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # 已接受事务不被调用方取消丢弃
        else:
            close = asyncio.create_task(session.aclose())
            await asyncio.sleep(0)
            assert not close.done()  # close 等待同一拥有任务的真实落定
        release.set()
        if action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert session.reasoning_effort.current == "max" and len(changes) == 1
        snapshot = session._store.recovery_snapshot()
        assert snapshot.identity.launch.options["reasoning_effort"] == "max"
        if close is not None:
            await close
    finally:
        release.set()
        await session.aclose()


async def test_storage_failure_keeps_old_effort_and_latches(tmp_path):
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        await session._store.capture()._dispatch(lambda: session._store.capture()._conn.execute(
            "CREATE TRIGGER reject_model BEFORE INSERT ON epoch BEGIN SELECT RAISE(ABORT, 'injected'); END"
        ))
        old = session.persistence.epoch_id
        with pytest.raises(ModelSelectionError, match="persist"):
            await session.set_reasoning_effort("max")
        assert session.reasoning_effort.current == "high"
        assert session.persistence.epoch_id == old
        assert session._store.recovery_snapshot().identity.launch.options["reasoning_effort"] == "high"
        assert session.persistence.storage_failed is not None
        with pytest.raises(ModelSelectionError, match="blocked"):
            await session.set_reasoning_effort("low")
    finally:
        await session.aclose()
