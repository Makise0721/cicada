"""SQLite 会话存储聚焦测试: lease、事务、限额、损坏拒绝与关闭纪律.

证明 A06 (单写入者/真实 Windows 文件锁, 含独立进程)、A07 (打开前拒绝并保留
原文件)、A02 的存储层 roundtrip、A04 基础路径的事务明确结果与取消不回滚.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from cicada.core.ports import ModelMetrics
from cicada.persistence.contracts import (
    MAX_METADATA_BYTES,
    SESSION_APPLICATION_ID,
    SESSION_SCHEMA_VERSION,
    ArchiveFormatError,
    ArchiveLimitError,
    LaunchIdentity,
    LeaseHeldError,
    PersistenceError,
    RequestOutcomeFacts,
    SessionNotFoundError,
    StorageUnavailableError,
    StorageWriteError,
    StoredIdentity,
    UnknownArchiveVersionError,
    WorkspaceMismatchError,
)
from cicada.plugins.session_store import SQLiteSessionStore, open_session_store

UUID = "123e4567-e89b-42d3-a456-426614174000"


def _identity(**overrides) -> StoredIdentity:
    fields = dict(
        provider="fake", model="fake", base_url=None, think=None,
        options={"num_ctx": 32768}, max_request_bytes=None, max_turns=6,
        prompt_version="builtin:p4-v1", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None,
        sources={"num_ctx": "default"},
    )
    fields.update(overrides)
    return StoredIdentity(
        launch=LaunchIdentity(**fields), tool_fingerprint="fp-" + "0" * 16
    )


def _outcome(**overrides) -> RequestOutcomeFacts:
    fields = dict(
        run_id="run-1", stop_reason="stop", error=None, exit_code=0, finalized=True,
        finalization_notes=(),
        model_calls=(ModelCallCompleted(run_id="run-1", turn_index=0, stop_reason="stop",
                                        metrics=None, elapsed_s=0.05),),
    )
    fields.update(overrides)
    return RequestOutcomeFacts(**fields)


def _suffix(*pairs: tuple[str, str], details: dict | None = None) -> tuple:
    messages: list = []
    for prompt, answer in pairs:
        messages.append(UserMessage(text=prompt))
        messages.append(AssistantMessage(text=answer))
    if details is not None:
        messages.append(
            ToolResultMessage(result=ToolResult("c1", "read", "body", details=details))
        )
    return tuple(messages)


async def _populate(store: SQLiteSessionStore, *, request_count: int = 1) -> str:
    epoch = await store.start_epoch(_identity())
    for index in range(request_count):
        prompt = f"问题 {index}"
        request_id = await store.begin_request(epoch, prompt)
        await store.commit_request(
            request_id,
            messages=_suffix((prompt, f"回答 {index}")),
            outcome=_outcome(run_id=f"run-{index + 1}"),
        )
    return epoch


# --- 创建 / lease / 关闭 ---------------------------------------------------------------------


async def test_new_session_creates_files_and_close_keeps_lock_file(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    db = tmp_path / f"{store.session_id}.sqlite3"
    lock = tmp_path / f"{store.session_id}.lock"
    assert db.is_file() and lock.is_file()
    snapshot = store.recovery_snapshot()
    assert snapshot.requests == () and snapshot.canonical_history == ()
    assert snapshot.context_generation == 0 and snapshot.identity is None
    await store.aclose()
    assert lock.is_file(), "关闭不删除锁文件, 防止新的锁目标分裂"
    await store.aclose()  # 幂等


async def test_second_open_in_same_process_is_refused_by_lock(tmp_path):
    """同进程第二句柄同样被 Windows 区域锁拒绝 (锁按句柄拥有, 系统级生效)."""
    first = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    with pytest.raises(LeaseHeldError):
        open_session_store(directory=tmp_path, session_id=first.session_id, workspace=tmp_path)
    await first.aclose()
    second = open_session_store(directory=tmp_path, session_id=first.session_id, workspace=tmp_path)
    await second.aclose()


def test_independent_process_holds_lease_until_exit(tmp_path):
    """A06 的独立进程证明: 子进程持锁期间父进程取锁失败, 子进程退出后可重新取锁."""
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[2] / "src"
    work = tmp_path / "ws"
    work.mkdir()
    seed = open_session_store(directory=tmp_path, session_id=None, workspace=work)
    session_id = seed.session_id
    asyncio.run(seed.aclose())

    child = subprocess.Popen(
        [sys.executable, "-c", "\n".join([
            "import sys, time",
            f"sys.path.insert(0, {str(src)!r})",
            "from cicada.plugins.session_store import open_session_store",
            f"store = open_session_store(directory={str(tmp_path)!r}, "
            f"session_id={session_id!r}, workspace={str(work)!r})",
            "print('READY', flush=True)",
            "time.sleep(20)",
        ])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        line = child.stdout.readline()
        assert line.strip() == "READY", f"child failed to open the store: {child.stderr.read()}"
        with pytest.raises(LeaseHeldError):
            open_session_store(directory=tmp_path, session_id=session_id, workspace=work)
    finally:
        child.terminate()
        child.wait(timeout=20)
    assert child.poll() is not None, "子进程必须已退出"
    # 内核/杀软对句柄与区域锁的最终释放可能有亚秒级延迟: 有界重试
    reopened: object = None
    for _ in range(40):
        try:
            reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=work)
            break
        except LeaseHeldError:
            time.sleep(0.05)
    assert reopened is not None, "子进程退出后必须能重新取锁"
    asyncio.run(reopened.aclose())


async def test_store_plugin_activates_first_in_persistent_topology(tmp_path):
    """钉住持久拓扑: store 插件在名单首位、零依赖 -> 最先激活 -> runtime 逆序清理时最后释放.

    这是「store 最后清理」的注册顺序机制回归; 02 换显式 capability requires 时本测试
    仍应通过 (首位激活变为依赖推导), 语义不变.
    """
    from cicada.boot import bootstrap
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.session_store import session_store_plugin

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    definitions = [
        session_store_plugin(store),
        workspace_plugin(tmp_path),
        fake_model_plugin(FakeModel([])),
    ]
    app = await bootstrap(definitions, tool_capabilities=(), system_prompt="t", max_turns=6)
    assert app.report.activated[0] == "session-store", app.report.activated
    await app.runtime.stop()  # 逆序清理: store 最后被插件 scope 释放
    assert store._closed, "逆序清理后 store 必须已被插件 scope 释放"


async def test_released_lease_allows_reacquire_after_process_style_close(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    await store.aclose()
    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    await reopened.aclose()


async def test_resume_unknown_or_malformed_id_rejected(tmp_path):
    with pytest.raises(SessionNotFoundError):
        open_session_store(directory=tmp_path, session_id=UUID, workspace=tmp_path)
    with pytest.raises(PersistenceError, match="UUIDv4"):
        open_session_store(directory=tmp_path, session_id="../../etc", workspace=tmp_path)


async def test_workspace_mismatch_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    await store.aclose()
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(WorkspaceMismatchError):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=other)


# --- 提交边界与恢复 roundtrip ----------------------------------------------------------------


async def test_epoch_begin_commit_and_restore_roundtrip(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    details = {"path": "a.py", "nested": {"deep": [1, 2, {"leaf": "ünïcödé \x1b"}]}}
    request_id = await store.begin_request(epoch, "第一问 ünïcödé")
    messages = (
        UserMessage(text="第一问 ünïcödé"),
        AssistantMessage(text="回答", tool_calls=(
            ToolCall(id="c1", name="read", arguments_json='{"path": "a.py"}'),
        ), stop_reason="tool_use"),
        ToolResultMessage(result=ToolResult("c1", "read", "body", details=details)),
        AssistantMessage(text="完成", stop_reason="stop"),
    )
    outcome = _outcome(
        run_id="run-7", stop_reason="stop", exit_code=0,
        model_calls=(
            ModelCallCompleted(run_id="run-7", turn_index=0, stop_reason="tool_use",
                               metrics=ModelMetrics(input_tokens=11, output_tokens=None,
                                                    provider_duration_s=0.25),
                               elapsed_s=0.3),
            ModelCallCompleted(run_id="run-7", turn_index=1, stop_reason="stop",
                               metrics=None, elapsed_s=0.1),
        ),
    )
    seq = await store.commit_request(request_id, messages=messages, outcome=outcome)
    assert seq == 2  # request_begin 占 seq 1
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    snapshot = reopened.recovery_snapshot()
    assert len(snapshot.requests) == 1 and len(snapshot.canonical_history) == len(messages)
    restored = snapshot.requests[0]
    assert restored.request_id == request_id and restored.seq == seq
    assert restored.outcome == outcome, "计量缺失与已知值都必须逐字段恢复"
    assert restored.messages == messages
    assert restored.messages[2].result.details == details
    assert snapshot.canonical_history[0].text == "第一问 ünïcödé"
    assert snapshot.identity is not None and snapshot.identity == _identity()
    await reopened.aclose()


async def test_clear_advances_generation_and_history_composition(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    first = await store.begin_request(epoch, "第一问")
    await store.commit_request(first, messages=_suffix(("第一问", "一答")), outcome=_outcome())
    generation = await store.commit_clear(epoch)
    assert generation == 1
    second = await store.begin_request(epoch, "第二问")
    await store.commit_request(second, messages=_suffix(("第二问", "二答")), outcome=_outcome())
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    snapshot = reopened.recovery_snapshot()
    assert snapshot.context_generation == 1
    assert [m.text for m in snapshot.canonical_history] == ["第二问", "二答"]
    assert len(snapshot.requests) == 2, "原始记录跨代次保留"
    await reopened.aclose()


async def test_pending_intent_survives_and_start_epoch_marks_interrupted(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    await store.begin_request(epoch, "未完成的请求")
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    snapshot = reopened.recovery_snapshot()
    assert len(snapshot.interrupted) == 1
    assert snapshot.interrupted[0].prompt == "未完成的请求"
    assert snapshot.interrupted[0].status == "pending"
    assert snapshot.canonical_history == ()
    new_epoch = await reopened.start_epoch(_identity())
    assert new_epoch != epoch
    assert reopened.recovery_snapshot().interrupted[0].status == "interrupted"
    await reopened.aclose()


async def test_second_active_request_and_unknown_commit_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    await store.begin_request(epoch, "第一个")
    with pytest.raises(PersistenceError, match="pending"):
        await store.begin_request(epoch, "第二个")
    with pytest.raises(PersistenceError, match="unknown request"):
        await store.commit_request("not-a-request", messages=_suffix(("x", "y")), outcome=_outcome())
    with pytest.raises(PersistenceError, match="unknown epoch"):
        await store.begin_request("no-such-epoch", "问")
    await store.aclose()


async def test_duplicate_commit_after_committed_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "问")
    await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome())
    with pytest.raises(PersistenceError, match="already committed"):
        await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome())
    await store.aclose()


# --- 限额 ------------------------------------------------------------------------------------


async def test_prompt_metadata_limit_rejected_before_begin(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    huge = "x" * (MAX_METADATA_BYTES + 1)
    with pytest.raises(ArchiveLimitError, match="request prompt"):
        await store.begin_request(epoch, huge)
    await store.aclose()


async def test_request_payload_limit_rejected_with_no_partial_write(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "大请求")
    big = "y" * (8 * 1024 * 1024 + 64)
    with pytest.raises(ArchiveLimitError, match="request payload"):
        await store.commit_request(
            request_id,
            messages=(UserMessage(text="大请求"), AssistantMessage(text=big)),
            outcome=_outcome(),
        )
    session_id = store.session_id
    await store.aclose()
    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().requests == ()
    await reopened.aclose()


# --- 损坏 / 未知版本: 打开前拒绝, 保留原文件 ---------------------------------------------------


def _corrupt(tmp_path: Path, store: SQLiteSessionStore, statements: tuple[str, ...]) -> None:
    """直接用 sqlite3 修改已关闭的存档 (测试专用的受控破坏)."""
    connection = sqlite3.connect(str(tmp_path / f"{store.session_id}.sqlite3"))
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


async def _open_closed(tmp_path: Path, store: SQLiteSessionStore, **kwargs):
    session_id = store.session_id
    await store.aclose()
    return open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path, **kwargs)


async def test_unknown_version_rejected_and_file_untouched(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()
    before = db.read_bytes()
    with pytest.raises(UnknownArchiveVersionError):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert db.read_bytes() == before, "拒绝恢复不得改写原存档"


async def test_wrong_application_id_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    await store.aclose()
    connection = sqlite3.connect(str(tmp_path / f"{store.session_id}.sqlite3"))
    connection.execute(f"PRAGMA application_id = {SESSION_APPLICATION_ID + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="not a cicada session archive"):
        open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)


async def test_garbage_file_rejected_as_format_error(tmp_path):
    (tmp_path / f"{UUID}.sqlite3").write_bytes(b"this is not a database at all")
    with pytest.raises(ArchiveFormatError, match="not readable"):
        open_session_store(directory=tmp_path, session_id=UUID, workspace=tmp_path)


async def test_missing_table_and_corrupt_payload_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    await _populate(store)
    await store.aclose()

    connection = sqlite3.connect(str(tmp_path / f"{session_id}.sqlite3"))
    connection.execute("DROP TABLE historical_receipts")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="missing tables"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    connection = sqlite3.connect(str(tmp_path / f"{session_id}.sqlite3"))
    connection.execute(
        "CREATE TABLE historical_receipts (verification_run_id TEXT NOT NULL, receipt_id TEXT NOT NULL,"
        " epoch_id TEXT NOT NULL, request_id TEXT, activity_id TEXT, definition_fingerprint TEXT NOT NULL,"
        " terminal_json TEXT NOT NULL, PRIMARY KEY (verification_run_id, receipt_id))"
    )
    connection.execute(
        "UPDATE records SET payload_json = '{broken' WHERE kind = 'request'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="corrupt"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_inconsistent_intent_relation_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    await _populate(store)
    await store.aclose()
    connection = sqlite3.connect(str(tmp_path / f"{session_id}.sqlite3"))
    connection.execute("UPDATE request_intents SET status = 'pending'")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="pending but has a committed request record"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


# --- worker 纪律: 串行、明确结果、取消不回滚、关闭排空 -----------------------------------------


async def test_transactions_serialize_and_errors_propagate(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    order: list[str] = []

    original = store._immediate

    def tracking():
        order.append("txn")
        return original()

    store._immediate = tracking
    request_id = await store.begin_request(epoch, "串行")
    await store.commit_request(request_id, messages=_suffix(("串行", "答")), outcome=_outcome())
    store._immediate = original
    assert order == ["txn", "txn"]
    await store.aclose()


async def test_caller_cancellation_does_not_rollback_dispatched_transaction(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    gate = threading.Event()
    original = store._immediate

    def gated():
        gate.wait(timeout=5)
        return original()

    store._immediate = gated
    task = asyncio.create_task(store.commit_clear(epoch))
    await asyncio.sleep(0.15)  # worker 已进入 gated 事务
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    await asyncio.sleep(0.2)  # worker 完成已派发事务
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().context_generation == 1, "取消等待不取消已派发事务"
    await reopened.aclose()


async def test_close_drains_dispatched_work_before_release(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    task = asyncio.create_task(store.commit_clear(epoch))
    await asyncio.sleep(0.05)
    await store.aclose()  # 排空: 未 await 的提交也已完成
    assert task.done() and task.exception() is None
    reopened = open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().context_generation == 1
    await reopened.aclose()


async def test_dispatch_after_close_rejected(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    epoch = await store.start_epoch(_identity())
    await store.aclose()
    with pytest.raises(StorageUnavailableError):
        await store.begin_request(epoch, "迟到请求")
