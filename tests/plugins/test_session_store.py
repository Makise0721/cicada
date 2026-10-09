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
from cicada.persistence.codec import dumps_strict
from cicada.persistence.contracts import (
    MAX_METADATA_BYTES,
    SESSION_APPLICATION_ID,
    SESSION_SCHEMA_VERSION,
    ArchiveFormatError,
    ArchiveLimitError,
    LaunchIdentity,
    LeaseHeldError,
    PersistenceError,
    ProcessTerminalFacts,
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


async def _make_session(directory: Path, *, requests: int = 1, title: str | None = None) -> str:
    """经真实 store 建一个带提交请求的会话; 返回 session_id。"""
    store = open_session_store(directory=directory, session_id=None, workspace=directory)
    session_id = store.session_id
    await _populate(store, request_count=requests)
    await store.aclose()
    if title is not None:
        connection = sqlite3.connect(str(directory / f"{session_id}.sqlite3"))
        connection.execute("UPDATE session SET short_title = ?", (title,))
        connection.commit()
        connection.close()
    return session_id


@pytest.fixture
async def store(tmp_path):
    """空白会话存档; 测试结束后幂等关闭 (aclose 幂等由本文件的关闭纪律测试钉住)."""
    handle = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    yield handle
    await handle.aclose()


# --- 创建 / lease / 关闭 ---------------------------------------------------------------------


async def test_new_session_creates_files_and_close_keeps_lock_file(store, tmp_path):
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


async def test_store_plugin_activates_first_in_persistent_topology(store, tmp_path):
    """钉住持久拓扑: store 插件在名单首位、零依赖 -> 最先激活 -> runtime 逆序清理时最后释放.

    这是「store 最后清理」的注册顺序机制回归; 02 换显式 capability requires 时本测试
    仍应通过 (首位激活变为依赖推导), 语义不变.
    """
    from cicada.boot import bootstrap
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.session_store import session_store_plugin

    definitions = [
        session_store_plugin(store),
        workspace_plugin(tmp_path),
        fake_model_plugin(FakeModel([])),
    ]
    app = await bootstrap(definitions, tool_capabilities=(), system_prompt="t", max_turns=6)
    assert app.report.activated[0] == "session-store", app.report.activated
    await app.runtime.stop()  # 逆序清理: store 最后被插件 scope 释放
    assert store._closed, "逆序清理后 store 必须已被插件 scope 释放"


async def test_resume_unknown_or_malformed_id_rejected(tmp_path):
    with pytest.raises(SessionNotFoundError):
        open_session_store(directory=tmp_path, session_id=UUID, workspace=tmp_path)
    with pytest.raises(PersistenceError, match="UUIDv4"):
        open_session_store(directory=tmp_path, session_id="../../etc", workspace=tmp_path)


async def test_workspace_mismatch_rejected(store, tmp_path):
    session_id = store.session_id
    await store.aclose()
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(WorkspaceMismatchError):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=other)


# --- 提交边界与恢复 roundtrip ----------------------------------------------------------------


async def test_epoch_begin_commit_and_restore_roundtrip(store, tmp_path):
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


async def test_clear_advances_generation_and_history_composition(store, tmp_path):
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


async def test_pending_intent_survives_and_start_epoch_marks_interrupted(store, tmp_path):
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


async def test_second_active_request_and_unknown_commit_rejected(store):
    epoch = await store.start_epoch(_identity())
    await store.begin_request(epoch, "第一个")
    with pytest.raises(PersistenceError, match="pending"):
        await store.begin_request(epoch, "第二个")
    with pytest.raises(PersistenceError, match="unknown request"):
        await store.commit_request("not-a-request", messages=_suffix(("x", "y")), outcome=_outcome())
    with pytest.raises(PersistenceError, match="unknown epoch"):
        await store.begin_request("no-such-epoch", "问")


async def test_duplicate_commit_after_committed_rejected(store):
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "问")
    await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome())
    with pytest.raises(PersistenceError, match="already committed"):
        await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome())


# --- 限额 ------------------------------------------------------------------------------------


async def test_prompt_metadata_limit_rejected_before_begin(store):
    epoch = await store.start_epoch(_identity())
    huge = "x" * (MAX_METADATA_BYTES + 1)
    with pytest.raises(ArchiveLimitError, match="request prompt"):
        await store.begin_request(epoch, huge)


async def test_request_payload_limit_rejected_with_no_partial_write(store, tmp_path):
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


async def test_unknown_version_rejected_and_file_untouched(store, tmp_path):
    session_id = store.session_id
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute(f"PRAGMA user_version = {SESSION_SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    before = db.read_bytes()
    with pytest.raises(UnknownArchiveVersionError):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert db.read_bytes() == before, "拒绝恢复不得改写原存档"


async def test_wrong_application_id_rejected(store, tmp_path):
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


async def test_missing_table_and_corrupt_payload_rejected(store, tmp_path):
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
    # 载荷被改动后先撞 sha 校验 (摘要仍是原载荷的); 再把摘要一并伪造成匹配值,
    # JSON 解析损坏才成为拒绝面 —— 两道校验分别钉住
    with pytest.raises(ArchiveFormatError, match="recorded digest"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)

    connection = sqlite3.connect(str(tmp_path / f"{session_id}.sqlite3"))
    row = connection.execute(
        "SELECT seq, payload_json FROM records WHERE kind = 'request'"
    ).fetchone()
    forged = hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest()
    connection.execute(
        "UPDATE records SET payload_sha256 = ? WHERE seq = ?", (forged, row[0])
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="corrupt"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_inconsistent_intent_relation_rejected(store, tmp_path):
    session_id = store.session_id
    await _populate(store)
    await store.aclose()
    connection = sqlite3.connect(str(tmp_path / f"{session_id}.sqlite3"))
    connection.execute("UPDATE request_intents SET status = 'pending'")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="pending but has a committed request record"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


# --- worker 纪律: 明确结果、取消不回滚、关闭排空 -----------------------------------------------


async def test_caller_cancellation_does_not_rollback_dispatched_transaction(store, tmp_path):
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


async def test_close_drains_dispatched_work_before_release(store, tmp_path):
    epoch = await store.start_epoch(_identity())
    task = asyncio.create_task(store.commit_clear(epoch))
    await asyncio.sleep(0.05)
    await store.aclose()  # 排空: 未 await 的提交也已完成
    assert task.done() and task.exception() is None
    reopened = open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().context_generation == 1
    await reopened.aclose()


async def test_dispatch_after_close_rejected(store):
    epoch = await store.start_epoch(_identity())
    await store.aclose()
    with pytest.raises(StorageUnavailableError):
        await store.begin_request(epoch, "迟到请求")


# --- 02: 进程活动 begin/end 与恢复证据 -------------------------------------------------------


_PROVEN = ProcessTerminalFacts(kind="proven_terminal", exit_code=0, output_complete=True)
_UNPROVEN = ProcessTerminalFacts(kind="unproven_terminal", timed_out=True, reason="timed out")


async def test_process_activity_begin_end_roundtrip_and_snapshot_evidence(store, tmp_path):
    """begin 在快照里是未证终结, end 后变已证; 重开存档事实保持, 关系完整."""
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "带命令的请求")
    activity_id = await store.begin_process(
        request_id, tool_name="powershell", call_id="c1",
        arguments={"command": "exit 0", "timeout": 5},
    )
    open_snapshot = store.recovery_snapshot()
    assert len(open_snapshot.process_activities) == 1
    activity = open_snapshot.process_activities[0]
    assert activity.activity_id == activity_id
    assert activity.request_id == request_id
    assert activity.tool_name == "powershell" and activity.call_id == "c1"
    assert activity.terminal is None
    assert activity.description.startswith('{"command"')
    assert open_snapshot.has_unproven_process_activity
    assert open_snapshot.unproven_process_activities() == (activity,)

    await store.end_process(activity_id, _PROVEN)
    closed_snapshot = store.recovery_snapshot()
    assert closed_snapshot.process_activities[0].terminal == _PROVEN
    assert not closed_snapshot.has_unproven_process_activity

    await store.commit_request(
        request_id, messages=_suffix(("带命令的请求", "答")), outcome=_outcome()
    )
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    restored = reopened.recovery_snapshot()
    assert len(restored.process_activities) == 1
    assert restored.process_activities[0].terminal == _PROVEN
    assert restored.process_activities[0].arguments_digest == activity.arguments_digest
    assert not restored.has_unproven_process_activity
    await reopened.aclose()


async def test_begin_process_relation_and_input_validation(store):
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "请求")
    with pytest.raises(PersistenceError, match="unknown request"):
        await store.begin_process(
            "00000000-0000-4000-8000-000000000000", tool_name="powershell",
            call_id="c0", arguments={"command": "x"},
        )
    with pytest.raises(PersistenceError, match="nonempty str"):
        await store.begin_process(request_id, tool_name="", call_id="c1", arguments={})
    with pytest.raises(PersistenceError, match="nonempty str"):
        await store.begin_process(request_id, tool_name="powershell", call_id="", arguments={})
    with pytest.raises(PersistenceError, match="JSON object"):
        await store.begin_process(request_id, tool_name="powershell", call_id="c1", arguments=["x"])
    with pytest.raises(PersistenceError, match="serializable"):
        await store.begin_process(
            request_id, tool_name="powershell", call_id="c1", arguments={"command": object()}
        )
    # 已提交请求不能再登记活动
    await store.commit_request(request_id, messages=_suffix(("请求", "答")), outcome=_outcome())
    with pytest.raises(PersistenceError, match="pending request"):
        await store.begin_process(request_id, tool_name="powershell", call_id="c2", arguments={})


async def test_end_process_rejects_unknown_and_double_terminal(store):
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "请求")
    activity_id = await store.begin_process(
        request_id, tool_name="powershell", call_id="c1", arguments={"command": "x"}
    )
    with pytest.raises(PersistenceError, match="unknown process activity"):
        await store.end_process("not-an-activity-id", _PROVEN)
    await store.end_process(activity_id, _PROVEN)
    with pytest.raises(PersistenceError, match="already has a terminal record"):
        await store.end_process(activity_id, _UNPROVEN)


async def test_commit_request_refuses_open_process_activity(store):
    """完整提交等待本请求全部活动写入: 未闭合活动在 store 侧也被拒绝."""
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "有未闭合活动的请求")
    activity_id = await store.begin_process(
        request_id, tool_name="powershell", call_id="c1", arguments={"command": "x"}
    )
    with pytest.raises(PersistenceError, match="process activity"):
        await store.commit_request(
            request_id, messages=_suffix(("有未闭合活动的请求", "答")), outcome=_outcome()
        )
    await store.end_process(activity_id, _PROVEN)
    seq = await store.commit_request(
        request_id, messages=_suffix(("有未闭合活动的请求", "答")), outcome=_outcome()
    )
    assert seq == 2


def test_orphan_process_activity_rejected_and_file_unchanged(tmp_path):
    """活动引用未知请求属关系损坏: 加载拒绝, 原文件字节不变."""
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    asyncio.run(_populate(store))
    asyncio.run(store.aclose())
    db_path = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db_path))
    connection.execute(
        "INSERT INTO process_activity (activity_id, request_id, tool_name, call_id,"
        " arguments_digest, begin_json, end_json) VALUES ('a-ghost', 'ghost-request',"
        " 'powershell', 'c9', ?, ?, NULL)",
        ("a" * 64, json.dumps({"description": "d", "arguments_digest": "a" * 64, "started_at": 1.0})),
    )
    connection.commit()
    connection.close()
    before = db_path.read_bytes()
    with pytest.raises(ArchiveFormatError, match="unknown request"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert db_path.read_bytes() == before


def test_corrupt_process_activity_records_rejected_on_load(tmp_path):
    """begin/end JSON 损坏与 digest 列不一致都按存档损坏拒绝."""
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    epoch = asyncio.run(store.start_epoch(_identity()))

    async def stage():
        request_id = await store.begin_request(epoch, "损坏面")
        return await store.begin_process(
            request_id, tool_name="powershell", call_id="c1", arguments={"command": "x"}
        )

    asyncio.run(stage())
    asyncio.run(store.aclose())
    db_path = tmp_path / f"{session_id}.sqlite3"

    connection = sqlite3.connect(str(db_path))
    original_begin, original_digest = connection.execute(
        "SELECT begin_json, arguments_digest FROM process_activity"
    ).fetchone()
    connection.close()

    for column, payload, match in (
        ("begin_json", "not-json", "begin is corrupt"),
        ("end_json", '{"kind": "nonsense"}', "terminal record is corrupt"),
    ):
        connection = sqlite3.connect(str(db_path))
        connection.execute(f"UPDATE process_activity SET {column} = ?", (payload,))
        connection.commit()
        connection.close()
        with pytest.raises(ArchiveFormatError, match=match):
            open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
        connection = sqlite3.connect(str(db_path))
        if column == "begin_json":
            connection.execute(
                "UPDATE process_activity SET begin_json = ?", (original_begin,))
        else:
            connection.execute("UPDATE process_activity SET end_json = NULL")
        connection.commit()
        connection.close()

    # digest 列与 begin_json 不一致同样拒绝
    connection = sqlite3.connect(str(db_path))
    connection.execute("UPDATE process_activity SET arguments_digest = ?", ("b" * 64,))
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="digest does not match"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


# --- 02 审查处置: 加载校验篡改面与写读对称 ---------------------------------------------------


async def test_commit_request_rejects_prompt_mismatch_at_write_time(store, tmp_path):
    """写侧与加载侧同一不变量: 首消息文本 != intent prompt 在提交时拒绝."""
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "真实 prompt")
    with pytest.raises(PersistenceError, match="exact intent prompt"):
        await store.commit_request(
            request_id, messages=_suffix(("另一个 prompt", "答")), outcome=_outcome()
        )
    # 拒绝后仍可用正确 prompt 提交, 存档可重开
    seq = await store.commit_request(
        request_id, messages=_suffix(("真实 prompt", "答")), outcome=_outcome()
    )
    assert seq == 2
    session_id = store.session_id
    await store.aclose()
    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert len(reopened.recovery_snapshot().requests) == 1
    await reopened.aclose()


def _seed_tamper_archive(tmp_path):
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    asyncio.run(_populate(store))
    asyncio.run(store.aclose())
    return tmp_path / f"{session_id}.sqlite3", session_id


def test_load_rejects_payload_sha_mismatch(tmp_path):
    """载荷被篡改但摘要未重算 -> 摘要不一致拒绝, 不静默读出改写历史."""
    db, session_id = _seed_tamper_archive(tmp_path)
    connection = sqlite3.connect(str(db))
    payload = connection.execute(
        "SELECT payload_json FROM records WHERE kind='request'"
    ).fetchone()[0]
    tampered = payload.replace("回答 0", "被改写的历史")
    assert tampered != payload
    connection.execute(
        "UPDATE records SET payload_json = ? WHERE kind='request'", (tampered,)
    )
    connection.commit()
    connection.close()
    assert hashlib.sha256(tampered.encode()).hexdigest() != hashlib.sha256(payload.encode()).hexdigest()
    with pytest.raises(ArchiveFormatError, match="recorded digest"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


def test_load_rejects_corrupt_clear_payload(tmp_path):
    db, session_id = _seed_tamper_archive(tmp_path)
    connection = sqlite3.connect(str(db))
    connection.execute("INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
                       " created_at, payload_json, payload_sha256) VALUES (99,"
                       " (SELECT epoch_id FROM records LIMIT 1), 1, 'clear', NULL, 0,"
                       " '{totally not json', NULL)")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="clear record .* corrupt"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


def test_load_rejects_clear_payload_with_content(tmp_path):
    db, session_id = _seed_tamper_archive(tmp_path)
    connection = sqlite3.connect(str(db))
    connection.execute("INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
                       " created_at, payload_json, payload_sha256) VALUES (99,"
                       " (SELECT epoch_id FROM records LIMIT 1), 1, 'clear', NULL, 0,"
                       " '{\"leftover\": 1}', NULL)")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="empty object"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


def test_load_rejects_begin_request_id_mismatch(tmp_path):
    db, session_id = _seed_tamper_archive(tmp_path)
    connection = sqlite3.connect(str(db))
    connection.execute(
        "UPDATE records SET request_id = '00000000-0000-4000-8000-0000000000aa'"
        " WHERE kind = 'request_begin'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="does not match its begin record"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


def test_load_rejects_orphan_request_begin(tmp_path):
    db, session_id = _seed_tamper_archive(tmp_path)
    connection = sqlite3.connect(str(db))
    epoch_id, generation = connection.execute(
        "SELECT epoch_id, context_generation FROM records LIMIT 1"
    ).fetchone()
    connection.execute(
        "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
        " created_at, payload_json, payload_sha256) VALUES (99, ?, ?, 'request_begin',"
        " 'ghost-request', 0, ?, NULL)",
        (epoch_id, generation, json.dumps({"prompt": "孤儿"})),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="orphan record"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_load_rejects_intent_epoch_mismatch(store, tmp_path):
    session_id = store.session_id
    epoch1 = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch1, "问")
    await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome())
    await store.start_epoch(_identity())  # 第二个 epoch: 提供 epoch 交叉校验的对照
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute(
        "UPDATE request_intents SET epoch_id ="
        " (SELECT epoch_id FROM epoch WHERE epoch_id != request_intents.epoch_id LIMIT 1)"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="epoch does not match"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_load_rejects_multiple_open_epochs(store, tmp_path):
    session_id = store.session_id
    await store.start_epoch(_identity())
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    identity_json = connection.execute("SELECT identity_json FROM epoch").fetchone()[0]
    connection.execute(
        "INSERT INTO epoch (epoch_id, opened_at, closed, identity_json, verification_run_id)"
        " VALUES ('ghost-epoch', 999, 0, ?, NULL)",
        (identity_json,),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="more than one open epoch"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_aclose_retry_after_drain_timeout_completes_release(store, tmp_path, monkeypatch):
    """排空超时后重试 aclose: 队列已清空则补完成释放, 不再静默返回."""
    import cicada.plugins.session_store.store as store_module

    epoch = await store.start_epoch(_identity())
    release = threading.Event()
    original_immediate = store._immediate
    zone = {"gated": False}

    def gated_immediate():
        if zone["gated"]:
            assert release.wait(timeout=30), "事务闸门未被打开"
        return original_immediate()

    store._immediate = gated_immediate  # type: ignore[method-assign]
    original_clear = store.commit_clear

    async def marked_clear(epoch_id):
        zone["gated"] = True
        try:
            return await original_clear(epoch_id)
        finally:
            zone["gated"] = False

    store.commit_clear = marked_clear  # type: ignore[method-assign]
    monkeypatch.setattr(store_module, "SESSION_STORE_DRAIN_TIMEOUT_S", 0.2)
    clear_task = asyncio.create_task(store.commit_clear(epoch))
    await asyncio.sleep(0.05)
    with pytest.raises(StorageWriteError, match="drain deadline"):
        await store.aclose()
    # 超时后事务仍在途: 重试 aclose 不得静默返回已关闭
    release.set()
    await asyncio.wait_for(clear_task, timeout=15)
    await store.aclose()  # 队列已排空: 补完成释放
    reopened = open_session_store(directory=tmp_path, session_id=store.session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().context_generation == 1
    await reopened.aclose()


# --- 03: 历史检查回执写入与加载 ---------------------------------------------------------------


def _receipt(request_id: str, epoch_id: str, receipt_id: str = "r-1", **overrides):
    from cicada.persistence.contracts import StoredReceipt

    fields = dict(
        verification_run_id="run-old",
        receipt_id=receipt_id,
        epoch_id=epoch_id,
        request_id=request_id,
        definition_fingerprint="f" * 64,
        execution_status="exited",
        exit_code=0,
        timed_out=False,
        cancelled=False,
        output_complete=True,
        failure_kind=None,
    )
    fields.update(overrides)
    return StoredReceipt(**fields)


async def test_commit_with_receipts_roundtrip_and_activity_link(store, tmp_path):
    """回执行与请求同事务落库; check 活动按 receipt_id 自动关联; 重开事实保持."""
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "带检查的问")
    activity_id = await store.begin_process(
        request_id, tool_name="check", call_id="c1",
        arguments={"action": "run", "check_id": "check-1"},
    )
    await store.end_process(activity_id, ProcessTerminalFacts(
        kind="proven_terminal", exit_code=0, output_complete=True, receipt_id="r-1",
    ))
    messages = (
        UserMessage(text="带检查的问"),
        AssistantMessage(text="", tool_calls=(ToolCall("c1", "check", "{}"),)),
        ToolResultMessage(result=ToolResult(
            "c1", "check", "passed", details={"receipt_id": "r-1", "exit_code": 0,
                                              "output_complete": True},
        )),
        AssistantMessage(text="答"),
    )
    seq = await store.commit_request(
        request_id, messages=messages, outcome=_outcome(),
        receipts=(_receipt(request_id, epoch),),
    )
    assert seq == 2
    snapshot = store.recovery_snapshot()
    assert len(snapshot.restored_receipts) == 1
    receipt = snapshot.restored_receipts[0]
    assert receipt.receipt_id == "r-1" and receipt.activity_id == activity_id
    assert not snapshot.recorded_process_uncertain
    session_id = store.session_id
    await store.aclose()

    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    restored = reopened.recovery_snapshot()
    assert len(restored.restored_receipts) == 1
    assert restored.restored_receipts[0].activity_id == activity_id
    assert not restored.recorded_process_uncertain
    await reopened.aclose()


async def test_commit_receipts_relation_and_duplicate_validation(store):
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "问")
    await store.commit_request(request_id, messages=_suffix(("问", "答")), outcome=_outcome(),
                               receipts=(_receipt(request_id, epoch),))
    other = await store.begin_request(epoch, "第二问")  # 前一请求已提交, 允许新 intent
    messages = _suffix(("第二问", "答"))
    with pytest.raises(PersistenceError, match="not the committing request"):
        await store.commit_request(other, messages=messages, outcome=_outcome(),
                                   receipts=(_receipt(request_id, epoch, receipt_id="r-other"),))
    with pytest.raises(PersistenceError, match="unique within one commit"):
        await store.commit_request(other, messages=messages, outcome=_outcome(),
                                   receipts=(_receipt(other, epoch, receipt_id="r-dup"),
                                             _receipt(other, epoch, receipt_id="r-dup")))
    with pytest.raises(PersistenceError, match="epoch does not match"):
        await store.commit_request(other, messages=messages, outcome=_outcome(),
                                   receipts=(_receipt(other, "ghost-epoch", receipt_id="r-epoch"),))
    await store.commit_request(other, messages=messages, outcome=_outcome())
    # 同一回执不得在后续请求重复关联 (同一 receipt 不逐请求复制)
    request3 = await store.begin_request(epoch, "第三问")
    with pytest.raises(PersistenceError, match="already persisted"):
        await store.commit_request(request3, messages=_suffix(("第三问", "答")),
                                   outcome=_outcome(),
                                   receipts=(_receipt(request_id, epoch),))


async def test_untrustworthy_receipt_records_process_uncertain(store, tmp_path):
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "超时检查")
    await store.commit_request(
        request_id, messages=_suffix(("超时检查", "答")), outcome=_outcome(),
        receipts=(_receipt(request_id, epoch, execution_status="timed_out", exit_code=None,
                           output_complete=False, failure_kind="timeout"),),
    )
    session_id = store.session_id
    await store.aclose()
    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().recorded_process_uncertain
    await reopened.aclose()


def test_load_rejects_receipt_relation_violations(tmp_path):
    """回执引用未知 epoch/未提交请求/损坏 terminal_json 都按关系损坏拒绝."""
    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    session_id = store.session_id

    async def seed_with_receipt():
        epoch = await store.start_epoch(_identity())
        request_id = await store.begin_request(epoch, "问")
        await store.commit_request(
            request_id, messages=_suffix(("问", "答")), outcome=_outcome(),
            receipts=(_receipt(request_id, epoch),),
        )

    asyncio.run(seed_with_receipt())
    asyncio.run(store.aclose())
    db = tmp_path / f"{session_id}.sqlite3"

    connection = sqlite3.connect(str(db))
    connection.execute("UPDATE historical_receipts SET terminal_json = 'not-json'")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="terminal record is corrupt"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)

    # 子用例分开: 非法 execution_status 单独触发 codec 枚举拒绝分支
    # (第一处篡改后的 terminal_json 已是坏值, 这里写回合法 JSON 再改枚举)
    valid = json.dumps({"execution_status": "exited", "exit_code": 0, "timed_out": False,
                        "cancelled": False, "output_complete": True, "failure_kind": None})
    connection = sqlite3.connect(str(db))
    connection.execute("UPDATE historical_receipts SET terminal_json = ?",
                       (valid.replace('"exited"', '"nonsense"'),))
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="must be one of"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)

    connection = sqlite3.connect(str(db))
    connection.execute("UPDATE historical_receipts SET terminal_json = ?",
                       (valid.replace('"exited"', '"timed_out"'),))
    connection.execute("UPDATE historical_receipts SET request_id = 'ghost-request'")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="unknown request"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)

    connection = sqlite3.connect(str(db))
    connection.execute("UPDATE historical_receipts SET request_id ="
                       " (SELECT request_id FROM request_intents LIMIT 1)")
    connection.execute("UPDATE historical_receipts SET epoch_id = 'ghost-epoch'")
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="unknown epoch"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


# --- 03 审查处置: 回执字节配额、id 上界与活动关联对称校验 -------------------------------


async def test_receipt_bytes_counted_into_db_size_checkpoint(store, tmp_path, monkeypatch):
    """回执行 (terminal_json + 标识列) 计入 db-size 检查点: 超限提交被拒, 不写出打不开的存档."""
    import cicada.plugins.session_store.store as store_module
    from cicada.persistence.contracts import StoredReceipt

    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "带大回执的问")
    # 单条回执行经 03 上界后已天然有界; 用多条回执叠加越过紧贴当前库容量的限额
    fat_receipts = tuple(
        StoredReceipt(
            verification_run_id="run-old", receipt_id=f"r-fat-{index}", epoch_id=epoch,
            request_id=request_id, definition_fingerprint="f" * 64,
            execution_status="exited", exit_code=0, timed_out=False, cancelled=False,
            output_complete=True, failure_kind="k" * 512,
        )
        for index in range(64)
    )
    connection = sqlite3.connect(str(tmp_path / f"{store.session_id}.sqlite3"))
    base_bytes = connection.execute("PRAGMA page_count").fetchone()[0] * 4096
    connection.close()
    monkeypatch.setattr(store_module, "MAX_DB_BYTES", base_bytes + 4096)
    with pytest.raises(ArchiveLimitError, match="size limit"):
        await store.commit_request(
            request_id, messages=_suffix(("带大回执的问", "答")), outcome=_outcome(),
            receipts=fat_receipts,
        )
    monkeypatch.undo()
    session_id = store.session_id
    await store.aclose()
    reopened = open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)
    assert reopened.recovery_snapshot().restored_receipts == (), "被拒提交不留部分行"
    await reopened.aclose()


async def test_receipt_id_length_cap_rejected_at_write_and_load(store, tmp_path):
    from cicada.persistence.contracts import RECEIPT_ID_MAX_CHARS, StoredReceipt

    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "问")
    with pytest.raises(ValueError, match="exceeds the limit"):
        StoredReceipt("r" * (RECEIPT_ID_MAX_CHARS + 1), "r-big", epoch, request_id,
                      "f" * 64, "exited", 0, False, False, True, None)
    # 加载侧对称: 直接向库内塞超长 id -> 构造校验拒绝 (inconsistent)
    await store.commit_request(
        request_id, messages=_suffix(("问", "答")), outcome=_outcome(),
        receipts=(_receipt(request_id, epoch),),
    )
    session_id = store.session_id
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute("UPDATE historical_receipts SET receipt_id = ?", ("r" * 300,))
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="inconsistent"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


async def test_load_rejects_receipt_relinked_to_foreign_activity(store, tmp_path):
    """加载侧与写侧对称: 回执关联的活动必须携带同一回执身份."""
    epoch = await store.start_epoch(_identity())
    request_id = await store.begin_request(epoch, "带检查与命令的问")
    check_activity = await store.begin_process(
        request_id, tool_name="check", call_id="c1",
        arguments={"action": "run", "check_id": "check-1"},
    )
    await store.end_process(check_activity, ProcessTerminalFacts(
        kind="proven_terminal", exit_code=0, output_complete=True, receipt_id="r-1",
    ))
    shell_activity = await store.begin_process(
        request_id, tool_name="powershell", call_id="c2", arguments={"command": "x"},
    )
    await store.end_process(shell_activity, _PROVEN)
    await store.commit_request(
        request_id, messages=_suffix(("带检查与命令的问", "答")), outcome=_outcome(),
        receipts=(_receipt(request_id, epoch),),
    )
    session_id = store.session_id
    await store.aclose()
    db = tmp_path / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute(
        "UPDATE historical_receipts SET activity_id = ?", (shell_activity,),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ArchiveFormatError, match="does not carry the same receipt identity"):
        open_session_store(directory=tmp_path, session_id=session_id, workspace=tmp_path)


# --- 05: 工作区会话列表 (A12) -----------------------------------------------------------------


def _inject_activity_and_receipt(
    db: Path,
    *,
    activity_id: str,
    call_id: str,
    end_json: str,
    receipt_id: str,
    terminal_json: str,
    verification_run_id: str = "vrun-listing-test",
) -> None:
    """向已有存档注入一条 check 活动与一条历史回执 (listing 测试的受控组装)."""
    connection = sqlite3.connect(str(db))
    try:
        epoch = connection.execute("SELECT epoch_id FROM epoch LIMIT 1").fetchone()[0]
        request_id = connection.execute(
            "SELECT request_id FROM request_intents LIMIT 1"
        ).fetchone()[0]
        fingerprint = hashlib.sha256(b"check command").hexdigest()
        connection.execute(
            "INSERT INTO process_activity (activity_id, request_id, tool_name, call_id,"
            " arguments_digest, begin_json, end_json) VALUES (?, ?, 'check', ?, 'digest', ?, ?)",
            (activity_id, request_id, call_id,
             dumps_strict({"description": "自检命令", "arguments_digest": "digest",
                           "started_at": 1.0}),
             end_json),
        )
        connection.execute(
            "INSERT INTO historical_receipts (verification_run_id, receipt_id, epoch_id,"
            " request_id, activity_id, definition_fingerprint, terminal_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (verification_run_id, receipt_id, epoch, request_id, activity_id, fingerprint,
             terminal_json),
        )
        connection.commit()
    finally:
        connection.close()


def test_list_sessions_empty_directory(tmp_path):
    from cicada.plugins.session_store import list_sessions

    page = list_sessions(directory=tmp_path)
    assert page.entries == () and page.next_cursor is None


async def test_list_sessions_reports_counts_title_and_order(tmp_path):
    from cicada.plugins.session_store import list_sessions

    first = await _make_session(tmp_path, requests=1)
    second = await _make_session(tmp_path, requests=2, title="修登录页")
    page = list_sessions(directory=tmp_path)
    assert page.next_cursor is None
    assert [entry.session_id for entry in page.entries] == [second, first], "按最近提交时间降序"
    top = page.entries[0]
    assert top.committed_requests == 2 and top.interrupted_requests == 0
    assert top.short_title == "修登录页"
    assert top.updated_at is not None and not top.occupied and not top.blocked
    assert page.entries[1].committed_requests == 1
    assert page.entries[1].short_title != "", "提交后有短标题"


async def test_list_sessions_marks_occupied_bad_and_blockers(tmp_path):
    from cicada.plugins.session_store import list_sessions

    busy = await _make_session(tmp_path, requests=1)
    blocked = await _make_session(tmp_path, requests=1)
    holder = open_session_store(directory=tmp_path, session_id=busy, workspace=tmp_path)
    try:
        # 阻断证据: 未闭合活动 + 不可信回执 (02/03 同源注入方式)
        db = tmp_path / f"{blocked}.sqlite3"
        connection = sqlite3.connect(str(db))
        epoch = connection.execute("SELECT epoch_id FROM epoch LIMIT 1").fetchone()[0]
        seq = connection.execute("SELECT MAX(seq) FROM records").fetchone()[0] + 1
        prompt_payload = dumps_strict({"prompt": "被中断的问"})
        request_id = "11111111-2222-4333-8444-555555555555"
        connection.execute(
            "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
            " created_at, payload_json, payload_sha256) VALUES (?, ?, 0, 'request_begin', ?, 0, ?, NULL)",
            (seq, epoch, request_id, prompt_payload),
        )
        connection.execute(
            "INSERT INTO request_intents (request_id, epoch_id, begin_seq, status)"
            " VALUES (?, ?, ?, 'pending')",
            (request_id, epoch, seq),
        )
        arguments = {"command": "Start-Sleep -Seconds 30"}
        canonical = dumps_strict(arguments)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO process_activity (activity_id, request_id, tool_name, call_id,"
            " arguments_digest, begin_json, end_json) VALUES ('activity-1', ?, 'powershell',"
            " 'c-pending', ?, ?, NULL)",
            (request_id, digest, dumps_strict({
                "description": canonical[:512], "arguments_digest": digest, "started_at": 1.0,
            })),
        )
        connection.commit()
        connection.close()
        # 另一个坏档: 不是合法 SQLite 文件
        (tmp_path / "00000000-1111-4222-8333-444444444444.sqlite3").write_bytes(b"not a database")

        page = list_sessions(directory=tmp_path)
        by_id = {entry.session_id: entry for entry in page.entries}
        assert len(page.entries) == 3, "坏档不消失 (不伪装成空列表/缺失)"
        assert by_id[busy].occupied, "lease 被持有的会话标注占用"
        assert not by_id[blocked].occupied
        assert by_id[blocked].blocked and by_id[blocked].interrupted_requests == 1
        assert by_id[blocked].unproven_activities == 1
        bad = by_id["00000000-1111-4222-8333-444444444444"]
        assert bad.error is not None and "not a database" in bad.error
        assert bad.occupied is False
    finally:
        await holder.aclose()


async def test_list_sessions_pagination_cursor(tmp_path):
    from cicada.plugins.session_store import list_sessions

    for _ in range(3):
        await _make_session(tmp_path, requests=1)
    page_one = list_sessions(directory=tmp_path, limit=2)
    assert len(page_one.entries) == 2 and page_one.next_cursor is not None
    page_two = list_sessions(directory=tmp_path, limit=2, cursor=page_one.next_cursor)
    assert len(page_two.entries) == 1 and page_two.next_cursor is None
    ids = {entry.session_id for entry in page_one.entries + page_two.entries}
    assert len(ids) == 3, "分页不重不漏"


def test_list_sessions_invalid_cursor_rejected(tmp_path):
    from cicada.plugins.session_store import list_sessions

    with pytest.raises(ValueError, match="invalid list cursor"):
        list_sessions(directory=tmp_path, cursor="not-a-number")


async def test_list_sessions_accepts_relative_directory(tmp_path, monkeypatch):
    """相对目录 listing (06): as_uri 只接受绝对路径, 列表必须自行 resolve。"""
    from cicada.plugins.session_store import list_sessions

    sessions = tmp_path / "sessions"
    session_id = await _make_session(sessions, requests=1)
    monkeypatch.chdir(tmp_path)
    page = list_sessions(directory=Path("sessions"))
    assert page.next_cursor is None
    assert [entry.session_id for entry in page.entries] == [session_id]
    assert page.entries[0].error is None, "相对目录不得把有效存档标成坏档"
    assert page.entries[0].committed_requests == 1


async def test_list_sessions_proven_activity_and_trusted_receipt_not_blocked(tmp_path):
    """06 回归: 已证终结活动 + 可信回执不得标阻断; 终结 JSON 须先解析再进 codec."""
    from cicada.plugins.session_store import list_sessions

    session_id = await _make_session(tmp_path, requests=1)
    receipt_id = "chk-" + "a" * 32
    _inject_activity_and_receipt(
        tmp_path / f"{session_id}.sqlite3",
        activity_id="activity-proven",
        call_id="check-1",
        end_json=dumps_strict({
            "kind": "proven_terminal", "exit_code": 0, "output_complete": True,
            "timed_out": False, "cancelled": False, "receipt_id": receipt_id,
            "reason": None,
        }),
        receipt_id=receipt_id,
        terminal_json=dumps_strict({
            "execution_status": "exited", "exit_code": 0, "timed_out": False,
            "cancelled": False, "output_complete": True, "failure_kind": None,
        }),
    )

    page = list_sessions(directory=tmp_path)
    assert len(page.entries) == 1
    entry = page.entries[0]
    assert entry.session_id == session_id and entry.error is None
    assert not entry.blocked, "已证活动与可信回执不是阻断证据"
    assert entry.unproven_activities == 0 and entry.untrusted_receipts == 0


@pytest.mark.parametrize("payload", ["not-json", '{"execution_status": "nonsense"}'])
async def test_list_sessions_corrupt_json_rows_count_conservatively(tmp_path, payload):
    """06 审查 F7: 列表侧坏行按 CodecError 保守计数, 整档不标坏档."""
    from cicada.plugins.session_store import list_sessions

    session_id = await _make_session(tmp_path, requests=1)
    _inject_activity_and_receipt(
        tmp_path / f"{session_id}.sqlite3",
        activity_id="activity-bad",
        call_id="check-bad",
        end_json=payload,
        receipt_id="chk-" + "b" * 32,
        terminal_json=payload,
        verification_run_id="vrun-bad",
    )

    page = list_sessions(directory=tmp_path)
    assert len(page.entries) == 1
    entry = page.entries[0]
    assert entry.session_id == session_id and entry.error is None, "坏行是保守计数, 不作废整档"
    assert entry.unproven_activities == 1 and entry.untrusted_receipts == 1
