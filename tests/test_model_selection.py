"""应用选择与真实 SQLite/HTTP 投影; 在事务接受点注入取消和失败."""

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace

import httpx
import pytest

from cicada.application import ModelChanged, ModelSelectionError, OperationBusy, open_application
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest
from cicada.models import ModelRoute
from cicada.persistence.codec import encode_messages
from cicada.persistence.contracts import PersistenceError
from cicada.plugins.deepseek import DeepSeekConfig
from cicada.plugins.session_store.store import read_session_identity
from test_application import _repo
from test_deepseek_application import config, response


def transport(bodies):
    def handle(req):
        body = json.loads(req.content)
        bodies.append(body)
        if body["thinking"]["type"] == "disabled":
            return httpx.Response(200, content=response(None, "完整记录摘要"))
        return httpx.Response(200, content=response(" 原始思考\n ", "答案"))
    return httpx.MockTransport(handle)


async def test_selection_summary_restart_preserves_effective_projection(tmp_path):
    root = _repo(tmp_path)
    bodies = []
    session = await open_application(config(root, transport(bodies)))
    try:
        await session.execute("第一请求")
        await session.execute("保留请求")
        before = session.history
        agent, runtime = session._app.agent, session._app.runtime
        generation = session.persistence.context_generation
        old_epoch = session.persistence.epoch_id
        updates = []
        session.subscribe(updates.append)
        assert session.budget.anchor is not None
        await session.select_model("deepseek-v4-pro")
        assert session.history == before
        assert session._app.agent is agent and session._app.runtime is runtime
        assert session.persistence.context_generation == generation
        assert session.persistence.epoch_id != old_epoch
        assert session.config.persistence.identity.model == "deepseek-v4-pro"
        assert session.budget.anchor is None and session.budget.report is None
        assert isinstance(updates[-1], ModelChanged)
        assert session.busy_operation is None
        await session.compact_context()
        summary = bodies[-1]
        assert summary["model"] == "deepseek-v4-pro"
        assert summary["thinking"] == {"type": "disabled"}
        assert "tools" not in summary and summary["max_tokens"] == 4096
        quoted = summary["messages"][1]["content"].split("\n", 1)[1]
        assert json.loads(quoted) == json.loads(encode_messages(before[:2]))
        assert session.history == before
        record = session.summary_records[-1]
        assert record.call_identity.model == "deepseek-v4-pro" and record.call_identity.think is False
        projection = session._context_projection.project(session.history)
        assert projection[-2:] == before[-2:]
        await session.execute("新模型续写")
        assert bodies[-1]["model"] == "deepseek-v4-pro"
        assert bodies[-1]["messages"][3]["reasoning_content"] == " 原始思考\n "
        sid = session.persistence.session_id
        expected = session._context_projection.project(session.history)
    finally:
        await session.aclose()
    saved = read_session_identity(directory=root / ".cicada" / "sessions", session_id=sid, workspace=root)
    initial = config(root, transport(bodies), sid, DeepSeekConfig(model="deepseek-v4-pro"))
    # 默认配置预读使用同一存档身份, 无覆盖/adopt。
    initial = replace(initial, persistence=replace(initial.persistence, identity=saved.launch))
    restored = await open_application(initial)
    try:
        assert restored.config.model_label == "deepseek-v4-pro"
        assert restored._context_projection.project(restored.history) == expected
        assert restored.summary_records[-1] == record
        await restored.execute("重启后续写")
        assert bodies[-1]["model"] == "deepseek-v4-pro"
    finally:
        await restored.aclose()


@pytest.mark.parametrize("persistent", [True, False])
async def test_same_option_no_io_and_ephemeral_selection(tmp_path, monkeypatch, persistent):
    bodies = []
    initial = config(_repo(tmp_path), transport(bodies))
    if not persistent:
        initial = replace(initial, persistence=None)
    session = await open_application(initial)
    try:
        events = []
        session.subscribe(events.append)
        original = session._model_selection.prepare_model
        def forbidden(*args, **kwargs):
            pytest.fail("same option must not prepare or write")
        monkeypatch.setattr(session._model_selection, "prepare_model", forbidden)
        if persistent:
            monkeypatch.setattr(session._store, "submit_model_change", forbidden)
        await session.select_model("deepseek-flash")
        assert events == [] and bodies == []
        monkeypatch.setattr(session._model_selection, "prepare_model", original)
        if not persistent:
            await session.select_model("deepseek-v4-pro")
            await session.execute("继续")
            assert bodies[-1]["model"] == "deepseek-v4-pro"
    finally:
        await session.aclose()


@pytest.mark.parametrize("lock", ["_process_uncertain", "_storage_failed", "_view_only_reason"])
async def test_session_locks_refuse_selection(tmp_path, lock):
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        setattr(session, lock, True)
        with pytest.raises(ModelSelectionError):
            await session.select_model("deepseek-v4-pro")
        assert session.config.model_label == "deepseek-flash"
        assert getattr(session, lock) is True
    finally:
        setattr(session, lock, False)
        await session.aclose()


@pytest.mark.parametrize("operation", ["request", "clear", "compaction", "model", "session",
                                       "login", "review", "rewind", "catalog"])
async def test_busy_operation_refuses_selection(tmp_path, operation):
    """P0-3: 任一独占事务在途都拒绝模型变更, 不靠入口手查布尔."""
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        session._ownership.claim(operation)
        try:
            with pytest.raises(ModelSelectionError):
                await session.select_model("deepseek-v4-pro")
            assert session.config.model_label == "deepseek-flash"
            assert session.busy_operation == operation
        finally:
            session._ownership.release(operation)
        assert session.busy_operation is None
    finally:
        await session.aclose()


@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_before_acceptance_zero_commit(tmp_path, monkeypatch, action):
    session = await open_application(config(_repo(tmp_path), transport([])))
    calls = []
    real = session._store.submit_model_change
    def submit(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(session._store, "submit_model_change", submit)
    task = None
    close = None
    def observe(update):
        nonlocal close
        if isinstance(update, OperationBusy) and update.operation == "model":
            if action == "cancel":
                task.cancel()
            else:
                close = asyncio.create_task(session.aclose())
    session.subscribe(observe)
    task = asyncio.create_task(session.select_model("deepseek-v4-pro"))
    try:
        with pytest.raises((asyncio.CancelledError, ModelSelectionError)):
            await task
        assert calls == [] and session.config.model_label == "deepseek-flash"
        if close is not None:
            await close
    finally:
        await session.aclose()


@pytest.mark.parametrize("action", ["cancel", "close", "close-timeout", "none"])
async def test_accepted_commit_blocks_operations_until_publication(tmp_path, monkeypatch, action):
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
    monkeypatch.setattr(session._store, "submit_model_change", submit)
    changes = []
    session.subscribe(lambda u: changes.append(u) if isinstance(u, ModelChanged) else None)
    task = asyncio.create_task(session.select_model("deepseek-v4-pro"))
    close = None
    try:
        await accepted.wait()
        for call in (session.execute("blocked"), session.clear_context(), session.compact_context(),
                     session.select_model("deepseek-flash")):
            with pytest.raises(RuntimeError, match="progress"):
                await call
        assert session.config.model_label == "deepseek-flash"
        if action == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # 重复取消也必须等待已接受的事务和发布。
        elif action in ("close", "close-timeout"):
            if action == "close-timeout":
                monkeypatch.setattr("cicada._application.session.CLOSE_SUPERVISION_S", 0.02)
            close = asyncio.create_task(session.aclose())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not close.done()
            if action == "close-timeout":
                await asyncio.sleep(0.05)
                assert not close.done() and not session._store.capture()._closed
                assert session._close_problems and changes == []
        release.set()
        if action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert session.config.model_label == "deepseek-v4-pro" and len(changes) == 1
        assert session._store.recovery_snapshot().identity.launch.model == "deepseek-v4-pro"
        if close:
            await close
    finally:
        release.set()
        await session.aclose()


async def test_sql_failure_rolls_back_and_latches_storage(tmp_path):
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        await session._store.capture()._dispatch(lambda: session._store.capture()._conn.execute(
            "CREATE TRIGGER reject_model BEFORE INSERT ON epoch BEGIN SELECT RAISE(ABORT, 'injected'); END"
        ))
        old = session.persistence.epoch_id
        with pytest.raises(ModelSelectionError, match="persist"):
            await session.select_model("deepseek-v4-pro")
        assert session.persistence.epoch_id == old
        assert session._store.recovery_snapshot().identity.launch.model == "deepseek-flash"
        assert session.persistence.storage_failed is not None
        assert session.config.model_label == "deepseek-flash"
        with pytest.raises(ModelSelectionError, match="blocked"):
            await session.select_model("deepseek-flash")
    finally:
        await session.aclose()


@pytest.mark.parametrize("refusal", ["epoch", "pending", "activity"])
async def test_transaction_refuses_unsafe_state_without_interrupting(tmp_path, refusal):
    session = await open_application(config(_repo(tmp_path), transport([])))
    store = session._store.capture()
    try:
        epoch = session.persistence.epoch_id
        if refusal in ("pending", "activity"):
            rid = await store.begin_request(epoch, "pending")
        if refusal == "activity":
            await store.begin_process(rid, tool_name="powershell", call_id="unsafe", arguments={})
            await store._dispatch(lambda: store._conn.execute(
                "UPDATE request_intents SET status = 'interrupted' WHERE request_id = ?", (rid,)))
        before = store.recovery_snapshot()
        handle = store.submit_model_change(before.identity,
            expected_epoch_id="wrong" if refusal == "epoch" else epoch)
        with pytest.raises(PersistenceError):
            await handle.wait()
        with pytest.raises(PersistenceError):
            await handle.wait()
        assert store.recovery_snapshot() == before
    finally:
        await session.aclose()


async def test_pointer_wins_timestamp_and_reverse_uuid(tmp_path, monkeypatch):
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        from cicada.plugins.session_store import store as module
        ids = iter(["ffffffff-ffff-4fff-8fff-ffffffffffff", "00000000-0000-4000-8000-000000000001"])
        monkeypatch.setattr(module.uuid, "uuid4", lambda: next(ids))
        monkeypatch.setattr(module.time, "time", lambda: 1.0)
        await session.select_model("deepseek-v4-pro")
        await session.select_model("deepseek-flash")
        path = session._store.capture().db_path
        sid = session.persistence.session_id
    finally:
        await session.aclose()
    identity = read_session_identity(directory=path.parent, session_id=sid, workspace=session.config.workspace)
    assert identity.launch.model == "deepseek-flash"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT current_epoch_id FROM session").fetchone()[0] == "00000000-0000-4000-8000-000000000001"


async def test_route_captures_binding_for_entire_stream(tmp_path):
    session = await open_application(config(_repo(tmp_path), transport([])))
    try:
        binding = session._model_route.binding
        gate = asyncio.Event()
        class Stream:
            async def stream(self, request, cancel):
                from cicada.core.ports import TextDelta
                yield TextDelta("first")
                await gate.wait()
                yield TextDelta("original")
        route = ModelRoute(replace(binding, model=Stream()))
        stream = route.stream(ModelRequest(messages=(), tools=(), system_prompt=""), CancelToken())
        assert (await anext(stream)).text == "first"
        route.binding = session._model_selection.prepare_model("deepseek-v4-pro")
        gate.set()
        assert (await anext(stream)).text == "original"
        await stream.aclose()
    finally:
        await session.aclose()


async def test_selection_keeps_current_check_receipt_baseline_and_summary(tmp_path):
    from cicada.application import verification_checks_signature
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
    root = _repo(tmp_path)
    bodies = []
    answers = iter([
        response("写", calls=(("w", "write", {"path": "result.txt", "content": "done"}),)),
        response("检查", calls=(("c", "check", {"action": "run", "check_id": "check-1"}),)),
        response("完成"), response("续写"), response(None, "第一请求摘要"), response("再续写"),
    ])
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))
    initial = config(root, httpx.MockTransport(handle), model_config=DeepSeekConfig(reasoning_effort="low", max_tokens=8192))
    plan = VerificationPlan(verification_run_id="selection-check", root=root,
        checks=(CheckDefinition(check_id="check-1", command="exit 0", timeout_s=10),))
    initial = replace(initial, plan=plan, persistence=replace(initial.persistence,
        identity=replace(initial.persistence.identity, checks_signature=verification_checks_signature(plan.checks))))
    session = await open_application(initial)
    try:
        first = await session.execute("修改并检查")
        assert first.exit_code == 0
        service, policy, agent = session._service, session._policy, session._app.agent
        receipts = first.view.receipts
        await session.execute("第二轮")
        await session.compact_context()
        summaries = session.summary_records
        history = session.history
        await session.select_model("deepseek-v4-pro")
        assert session._service is service and session._policy is policy and session._app.agent is agent
        assert session.summary_records == summaries and session.history == history
        assert session.verification_run_id == "selection-check"
        assert dict(session._model_route.binding.identity.options)["reasoning_effort"] == "low"
        last = await session.execute("沿用已有检查")
        assert last.exit_code == 0
        assert last.view.receipts == receipts
        assert last.evidence.baseline_snapshot_ref == first.evidence.baseline_snapshot_ref
        assert bodies[-1]["model"] == "deepseek-v4-pro" and bodies[-1]["max_tokens"] == 8192
        assert session.budget.report.output_reserve == 8192
        assert session._budget_policy.summary_budget.effective_output_reserve == 4096
    finally:
        await session.aclose()


async def test_compact_does_not_repair_latest_partial_or_discard_archive(tmp_path):
    root = _repo(tmp_path)
    bodies = []
    answers = iter([response("完整"), response("最后残片").replace(b'"stop"', b'"length"'), response(None, "旧请求摘要")])
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))
    session = await open_application(config(root, httpx.MockTransport(handle)))
    try:
        await session.execute("完整请求")
        await session.execute("截断请求")
        history = session.history
        assert history[-1].reasoning.complete is False
        await session.compact_context()
        assert session.history == history
        assert session._context_projection.project(history)[-1] == history[-1]
        before = session._store.recovery_snapshot()
        with pytest.raises(ModelSelectionError, match="history"):
            await session.select_model("deepseek-v4-pro")
        with pytest.raises(ValueError, match="thinking"):
            await session.execute("拒绝")
        assert session._store.recovery_snapshot() == before and len(bodies) == 3
    finally:
        await session.aclose()


async def test_selection_validates_effective_projection_not_covered_partial(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    answers = iter([response(None).replace(b'"stop"', b'"length"'), response(None), response(None, "旧请求摘要")])
    session = await open_application(config(root, httpx.MockTransport(
        lambda _: httpx.Response(200, content=next(answers))), model_config=DeepSeekConfig(think=False)))
    try:
        await session.execute("旧请求")
        await session.execute("保留请求")
        assert session.history[1].stop_reason == "length"
        await session.compact_context()
        before = session.history
        real = session._model_selection.prepare_model
        # 注入严格校验探针, 区分有效投影和规范前缀; 不改变当前模式。
        monkeypatch.setattr(session._model_selection, "prepare_model", lambda option:
            replace(real(option), validate_request=DeepSeekConfig().validate_request))
        await session.select_model("deepseek-v4-pro")
        assert session.history == before and session.config.persistence.identity.think is False
    finally:
        await session.aclose()


async def test_cancelled_handle_wait_keeps_actual_transaction_and_repeat_result(tmp_path, monkeypatch):
    session = await open_application(config(_repo(tmp_path), transport([])))
    store = session._store.capture()
    started, release = threading.Event(), threading.Event()
    immediate = store._immediate
    def gated():
        started.set()
        if not release.wait(5):
            raise RuntimeError("test gate timed out")
        immediate()
    try:
        identity = store.recovery_snapshot().identity
        monkeypatch.setattr(store, "_immediate", gated)
        handle = store.submit_model_change(identity, expected_epoch_id=session.persistence.epoch_id)
        assert await asyncio.to_thread(started.wait, 3)
        waiter = asyncio.create_task(handle.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        epoch = await handle.wait()
        assert await handle.wait() == epoch
        with sqlite3.connect(store.db_path) as conn:
            assert conn.execute("SELECT current_epoch_id FROM session").fetchone() == (epoch,)
    finally:
        release.set()
        await session.aclose()
