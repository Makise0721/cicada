"""schema v2 摘要持久化与原子迁移聚焦测试 (02, D8/PA04/A07).

v1 fixture 由裸 sqlite3 按固定 v1 DDL 现场构造 (不依赖版本常量或 01 工件);
迁移故障在步骤/核对/进程终止三个层面注入, 重开只能是完整 v1 (可重新升级)
或完整 v3; 篡改面逐项拒绝且原文件字节不变。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.core.ports import ModelMetrics
from cicada.persistence.codec import (
    encode_identity,
    encode_messages,
    encode_outcome_facts,
    dumps_strict,
)
from cicada.persistence.contracts import (
    ArchiveFormatError,
    ArchiveLimitError,
    LaunchIdentity,
    PersistenceError,
    RequestOutcomeFacts,
    StoredIdentity,
    SummaryRecord,
)
from cicada.plugins.session_store import open_session_store
import cicada.plugins.session_store.store as store_module

SRC = Path(__file__).resolve().parents[2] / "src"

# --- 固定 v1 DDL (fixture 专用; 与 v1 发布版本逐字对应, 不引用当前常量) -----------------

_V1_APPLICATION_ID = 0x43494331  # "CIC1"

_V1_DDL = (
    """
    CREATE TABLE session (
        schema_version INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        canonical_workspace TEXT NOT NULL,
        created_at REAL NOT NULL,
        short_title TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE epoch (
        epoch_id TEXT PRIMARY KEY,
        opened_at REAL NOT NULL,
        closed INTEGER NOT NULL CHECK (closed IN (0, 1)),
        identity_json TEXT NOT NULL,
        verification_run_id TEXT
    )
    """,
    """
    CREATE TABLE records (
        seq INTEGER PRIMARY KEY,
        epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
        context_generation INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('request_begin', 'request', 'clear')),
        request_id TEXT,
        created_at REAL NOT NULL,
        payload_json TEXT NOT NULL,
        payload_sha256 TEXT,
        CHECK ((kind IN ('request_begin', 'request')) = (request_id IS NOT NULL)),
        CHECK ((kind = 'request') = (payload_sha256 IS NOT NULL))
    )
    """,
    """
    CREATE TABLE request_intents (
        request_id TEXT PRIMARY KEY,
        epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
        begin_seq INTEGER NOT NULL UNIQUE REFERENCES records(seq),
        status TEXT NOT NULL CHECK (status IN ('pending', 'committed', 'interrupted'))
    )
    """,
    """
    CREATE TABLE process_activity (
        activity_id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL REFERENCES request_intents(request_id),
        tool_name TEXT NOT NULL,
        call_id TEXT NOT NULL,
        arguments_digest TEXT NOT NULL,
        begin_json TEXT NOT NULL,
        end_json TEXT
    )
    """,
    """
    CREATE TABLE historical_receipts (
        verification_run_id TEXT NOT NULL,
        receipt_id TEXT NOT NULL,
        epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
        request_id TEXT REFERENCES request_intents(request_id),
        activity_id TEXT REFERENCES process_activity(activity_id),
        definition_fingerprint TEXT NOT NULL,
        terminal_json TEXT NOT NULL,
        PRIMARY KEY (verification_run_id, receipt_id)
    )
    """,
)


def _identity() -> StoredIdentity:
    launch = LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None, sources={},
    )
    return StoredIdentity(launch=launch, tool_fingerprint="fp")


def _outcome() -> RequestOutcomeFacts:
    return RequestOutcomeFacts(
        run_id="run-1", stop_reason="stop", error=None, exit_code=0,
        finalized=True, model_calls=(
            ModelCallCompleted(
                run_id="run-1", turn_index=0, stop_reason="stop",
                metrics=ModelMetrics(input_tokens=11, output_tokens=2), elapsed_s=0.01,
            ),
        ),
    )


def _make_v1_archive(
    directory: Path, session_id: str, canonical_workspace: str, *, prompt: str = "第一问",
) -> None:
    """裸 sqlite3 构造完整 v1 存档: 一个 epoch + 一条已提交完整请求."""
    directory.mkdir(parents=True, exist_ok=True)
    db_path = directory / f"{session_id}.sqlite3"
    (directory / f"{session_id}.lock").touch()
    connection = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        connection.execute(f"PRAGMA application_id = {_V1_APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 1")
        connection.execute("PRAGMA journal_mode = DELETE")
        for statement in _V1_DDL:
            connection.execute(statement)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO session VALUES (1, ?, ?, ?, '')",
            (session_id, canonical_workspace, time.time()),
        )
        epoch_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO epoch VALUES (?, ?, 0, ?, NULL)",
            (epoch_id, time.time(), encode_identity(_identity())),
        )
        request_id = str(uuid.uuid4())
        begin_payload = dumps_strict({"prompt": prompt})
        connection.execute(
            "INSERT INTO records VALUES (1, ?, 0, 'request_begin', ?, ?, ?, NULL)",
            (epoch_id, request_id, time.time(), begin_payload),
        )
        messages = (UserMessage(text=prompt), AssistantMessage(text="回答"))
        payload = dumps_strict({
            "messages": encode_messages(messages),
            "outcome": encode_outcome_facts(_outcome()),
        })
        connection.execute(
            "INSERT INTO records VALUES (2, ?, 0, 'request', ?, ?, ?, ?)",
            (
                epoch_id, request_id, time.time(), payload,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            ),
        )
        connection.execute(
            "INSERT INTO request_intents VALUES (?, ?, 1, 'committed')",
            (request_id, epoch_id),
        )
        connection.execute("COMMIT")
    finally:
        connection.close()


def _raw_version(db_path: Path) -> tuple[int, int, bool]:
    """(user_version, session.schema_version, summaries 表在位)."""
    connection = sqlite3.connect(str(db_path))
    try:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        row = connection.execute("SELECT schema_version FROM session").fetchone()
        tables = {
            str(item[0]) for item in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        return user_version, (int(row[0]) if row else -1), "summaries" in tables
    finally:
        connection.close()


def _make_record(tmp_path: Path, *, seq: int = 2, end: int = 2, count: int = 2) -> SummaryRecord:
    return SummaryRecord(
        summary_id=str(uuid.uuid4()),
        epoch_id="placeholder",  # 由调用侧替换
        context_generation=0,
        first_request_seq=seq,
        last_request_seq=seq,
        message_count=count,
        end_message_index=end,
        prefix_sha256="b" * 64,
        text="较早请求的摘要文本",
        provider="fake",
        model="fake",
        prompt_version="1",
        prompt_sha256="c" * 64,
        metrics=ModelMetrics(input_tokens=7, output_tokens=3),
        reason="manual",
        created_at=time.time(),
    )


class _SessionWithRequests:
    """真实 v2 store + 两条已提交完整请求 (供 seq 引用)."""

    def __init__(self, tmp_path: Path) -> None:
        self.directory = tmp_path / "sessions"
        self.store = open_session_store(
            directory=self.directory, session_id=None, workspace=tmp_path
        )
        self.epoch_id = None  # 由 start_epoch 填充 (async 测试内)

    async def start(self) -> str:
        self.epoch_id = await self.store.start_epoch(_identity())
        return self.epoch_id

    async def commit_two(self) -> None:
        for index, prompt in enumerate(("第一问", "第二问"), start=1):
            request_id = await self.store.begin_request(self.epoch_id, prompt)
            messages = (UserMessage(text=prompt), AssistantMessage(text=f"回答{index}"))
            await self.store.commit_request(
                request_id, messages=messages, outcome=_outcome()
            )


# --- v1 → v2 原子迁移 (PA04/A07) ---------------------------------------------------------


def test_v1_archive_opens_and_upgrades_in_place(tmp_path):
    session_id = str(uuid.uuid4())
    _make_v1_archive(tmp_path / "sessions", session_id, os.path.realpath(tmp_path))
    store = open_session_store(
        directory=tmp_path / "sessions", session_id=session_id, workspace=tmp_path
    )
    try:
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 1, "既有完整请求可恢复"
        assert snapshot.canonical_history[0].text == "第一问"
        assert snapshot.summaries == ()
        user_version, schema_version, has_summaries = _raw_version(
            tmp_path / "sessions" / f"{session_id}.sqlite3"
        )
        assert (user_version, schema_version, has_summaries) == (3, 3, True)
    finally:
        import asyncio

        asyncio.run(store.aclose())
    # 重开幂等: 已是 v3, 请求与身份不变
    store = open_session_store(
        directory=tmp_path / "sessions", session_id=session_id, workspace=tmp_path
    )
    try:
        assert len(store.recovery_snapshot().requests) == 1
    finally:
        import asyncio

        asyncio.run(store.aclose())


def _steps_raising_at(index: int):
    original = store_module._V2_MIGRATION_STEPS

    def _boom(connection) -> None:
        raise RuntimeError(f"injected migration fault at step {index}")

    steps = list(original)
    steps[index] = _boom
    return tuple(steps)


@pytest.mark.parametrize("step_index", [0, 1, 2])
def test_migration_step_fault_rolls_back_to_complete_v1(tmp_path, monkeypatch, step_index):
    """每个迁移步骤注入异常: 原档保持完整 v1, 重开可重新升级 (PA04)."""
    monkeypatch.setattr(store_module, "_V2_MIGRATION_STEPS", _steps_raising_at(step_index))
    session_id = str(uuid.uuid4())
    directory = tmp_path / "sessions"
    _make_v1_archive(directory, session_id, os.path.realpath(tmp_path))
    db_path = directory / f"{session_id}.sqlite3"
    before = db_path.read_bytes()
    with pytest.raises(RuntimeError, match="injected migration fault"):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    # 02-F2: 回滚后字节级不变 (journal 回滚恢复原页), 版本/关系完整
    assert db_path.read_bytes() == before, "迁移失败回滚不改写原档字节"
    assert _raw_version(db_path) == (1, 1, False), "回滚后只能是完整 v1"
    connection = sqlite3.connect(str(db_path))
    try:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 2
        assert connection.execute("SELECT schema_version FROM session").fetchone()[0] == 1
    finally:
        connection.close()
    # 旧完整请求仍可恢复: 卸载注入后正常打开完成升级
    monkeypatch.undo()
    store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    try:
        assert len(store.recovery_snapshot().requests) == 1
    finally:
        import asyncio

        asyncio.run(store.aclose())


def test_migration_verify_fault_rolls_back_to_complete_v1(tmp_path, monkeypatch):
    """v2 结构核对失败同样整体回滚 (提交前最后一道)."""
    def _bad_verify(connection) -> None:
        raise ArchiveFormatError("injected verify fault")

    monkeypatch.setattr(store_module, "_verify_v2_structure", _bad_verify)
    session_id = str(uuid.uuid4())
    directory = tmp_path / "sessions"
    _make_v1_archive(directory, session_id, os.path.realpath(tmp_path))
    db_path = directory / f"{session_id}.sqlite3"
    with pytest.raises(ArchiveFormatError, match="injected verify fault"):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert _raw_version(db_path) == (1, 1, False)


@pytest.mark.parametrize("kill_step", [0, 1, 2])
def test_migration_process_kill_leaves_complete_v1_or_v3(tmp_path, kill_step):
    """独立进程在迁移事务的每个步骤中途终止 (02-F4: 建表/版本标记逐步注入).

    重开只能是完整 v1 (热日志回滚, 可重新升级) 或完整 v3, 旧完整请求可恢复,
    不留混合版本 (PA04)。「提交后终止」无歧义: COMMIT 已落盘即完整 v3。
    """
    session_id = str(uuid.uuid4())
    directory = tmp_path / "sessions"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _make_v1_archive(directory, session_id, os.path.realpath(workspace))
    db_path = directory / f"{session_id}.sqlite3"
    child = (
        "import os, sys\n"
        f"sys.path.insert(0, r'{SRC}')\n"
        f"sys.path.insert(0, r'{Path(__file__).resolve().parents[1]}')\n"
        "import cicada.plugins.session_store.store as store_module\n"
        "def _kill(connection):\n"
        "    os._exit(9)\n"
        f"steps = list(store_module._V2_MIGRATION_STEPS)\n"
        f"steps[{kill_step}] = _kill\n"
        "store_module._V2_MIGRATION_STEPS = tuple(steps)\n"
        "try:\n"
        f"    store_module.open_session_store(directory=r'{directory}',\n"
        f"        session_id='{session_id}', workspace=r'{workspace}')\n"
        "except BaseException:\n"
        "    pass\n"
        "os._exit(0)\n"
    )
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True,
        encoding="utf-8", timeout=60, env=environment,
    )
    assert result.returncode == 9, (result.stdout, result.stderr)
    # 重开: 只能是完整 v1 或完整 v3, 既有请求可恢复, 无混合版本
    store = open_session_store(directory=directory, session_id=session_id, workspace=workspace)
    try:
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 1
        user_version, schema_version, _has_summaries = _raw_version(db_path)
        assert (user_version, schema_version) in {(1, 1), (3, 3)}, "不留混合版本"
    finally:
        import asyncio

        asyncio.run(store.aclose())


# --- commit_summary 与快照摘要视图 ---------------------------------------------------------


async def _store_with_two_requests(tmp_path):
    fixture = _SessionWithRequests(tmp_path)
    await fixture.start()
    await fixture.commit_two()
    return fixture


def _record_for(store, epoch_id: str, **overrides) -> SummaryRecord:
    requests = [r for r in store.recovery_snapshot().requests]
    record = SummaryRecord(
        summary_id=str(uuid.uuid4()),
        epoch_id=epoch_id,
        context_generation=store.recovery_snapshot().context_generation,
        first_request_seq=requests[0].seq,
        last_request_seq=requests[-1].seq,
        message_count=sum(len(r.messages) for r in requests),
        end_message_index=sum(len(r.messages) for r in requests),
        prefix_sha256=hashlib.sha256(encode_messages(tuple(
            message for request in requests for message in request.messages
        )).encode("utf-8")).hexdigest(),
        text="较早请求的摘要文本",
        provider="fake",
        model="fake",
        prompt_version="1",
        prompt_sha256="c" * 64,
        metrics=ModelMetrics(input_tokens=7, output_tokens=3),
        reason="manual",
        created_at=time.time(),
    )
    return record


async def test_commit_summary_persists_and_snapshot_exposes_current_generation(tmp_path):
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    record = _record_for(store, fixture.epoch_id)
    await store.commit_summary(record)
    snapshot = store.recovery_snapshot()
    assert snapshot.summaries == (record,)
    import asyncio

    await store.aclose()
    reopened = open_session_store(
        directory=fixture.directory, session_id=store.session_id, workspace=tmp_path
    )
    try:
        assert reopened.recovery_snapshot().summaries == (record,)
    finally:
        import asyncio

        await reopened.aclose()


async def test_commit_summary_rejects_duplicate_and_relations(tmp_path):
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    record = _record_for(store, fixture.epoch_id)
    await store.commit_summary(record)
    with pytest.raises(Exception, match="already persisted"):
        await store.commit_summary(record)
    wrong_generation = _record_for(store, fixture.epoch_id)
    from dataclasses import replace as dc_replace

    with pytest.raises(Exception, match="does not match the current generation"):
        await store.commit_summary(dc_replace(wrong_generation, context_generation=7))
    with pytest.raises(Exception, match="committed requests of the current generation"):
        await store.commit_summary(
            dc_replace(wrong_generation, first_request_seq=999, last_request_seq=999)
        )
    unknown_epoch = dc_replace(_record_for(store, fixture.epoch_id), epoch_id="no-such-epoch")
    with pytest.raises(Exception, match="unknown epoch"):
        await store.commit_summary(unknown_epoch)
    import asyncio

    await store.aclose()


async def test_commit_summary_enforces_per_generation_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "MAX_SUMMARIES_PER_GENERATION", 1)
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    await store.commit_summary(_record_for(store, fixture.epoch_id))
    second = _record_for(store, fixture.epoch_id)
    with pytest.raises(ArchiveLimitError, match="summaries-per-generation"):
        await store.commit_summary(second)
    import asyncio

    await store.aclose()


async def test_summary_rows_survive_clear_but_leave_current_generation_view(tmp_path):
    """clear 后新代次投影为空; 旧代次摘要记录保留为历史事实 (D8)."""
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    record = _record_for(store, fixture.epoch_id)
    await store.commit_summary(record)
    new_generation = await store.commit_clear(fixture.epoch_id)
    snapshot = store.recovery_snapshot()
    assert snapshot.context_generation == new_generation
    assert snapshot.summaries == (), "当前代次视图不含旧代次摘要"
    import asyncio

    await store.aclose()
    connection = sqlite3.connect(str(store.db_path))
    try:
        count = connection.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    finally:
        connection.close()
    assert count == 1, "旧代次摘要记录仍在档"


# --- 加载校验与篡改面 (A07) ----------------------------------------------------------------


async def _closed_v2_archive_with_summary(tmp_path):
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    record = _record_for(store, fixture.epoch_id)
    await store.commit_summary(record)
    db_path = store.db_path
    session_id = store.session_id
    import asyncio

    await store.aclose()
    return fixture.directory, session_id, db_path


@pytest.mark.parametrize(
    "tamper",
    ["json_text", "summary_id_column", "prefix_column", "end_index_column", "bogus_seq"],
)
async def test_summary_tamper_faces_are_rejected_without_rewrite(tmp_path, tamper):
    built = await _closed_v2_archive_with_summary(tmp_path)
    directory, session_id, db_path = built
    connection = sqlite3.connect(str(db_path))
    try:
        if tamper == "json_text":
            row = connection.execute(
                "SELECT summary_json FROM summaries"
            ).fetchone()
            tampered = row[0].replace("较早请求的摘要文本", "被篡改的文本")
            connection.execute(
                "UPDATE summaries SET summary_json = ?", (tampered,)
            )
        elif tamper == "summary_id_column":
            connection.execute("UPDATE summaries SET summary_id = 'x' || summary_id")
        elif tamper == "prefix_column":
            connection.execute(
                "UPDATE summaries SET prefix_sha256 = ?",
                ("d" * 64,),
            )
        elif tamper == "end_index_column":
            connection.execute("UPDATE summaries SET end_message_index = end_message_index + 1")
        else:
            connection.execute("UPDATE summaries SET first_request_seq = 999")
        connection.commit()
    finally:
        connection.close()
    before = db_path.read_bytes()  # 篡改写之后的基线: 被拒的打开不得再改一字节
    with pytest.raises(ArchiveFormatError):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert db_path.read_bytes() == before, "拒绝恢复不改写原文件"


async def test_summary_codec_roundtrip_and_strictness(tmp_path):
    from cicada.persistence.codec import decode_summary_record, encode_summary_record
    from cicada.persistence.codec import CodecError, strict_load

    record = SummaryRecord(
        summary_id="s-1", epoch_id="e-1", context_generation=0,
        first_request_seq=1, last_request_seq=2, message_count=4,
        end_message_index=4, prefix_sha256="b" * 64, text="正文",
        provider="fake", model="m", prompt_version="1", prompt_sha256="c" * 64,
        metrics=None, reason="auto", created_at=1.0,
        previous_projection_identity="e" * 64,
    )
    decoded = decode_summary_record(strict_load(encode_summary_record(record)))
    assert decoded == record
    payload = strict_load(encode_summary_record(record))
    payload["unexpected"] = 1
    with pytest.raises(CodecError, match="unexpected keys"):
        decode_summary_record(payload)
    fresh = strict_load(encode_summary_record(record))
    fresh["reason"] = "sneaky"
    with pytest.raises(CodecError, match="reason"):
        decode_summary_record(fresh)


# --- 审查处置 (02 独立审查 2026-10-06) --------------------------------------------------


def test_v2_missing_summaries_table_rejected_by_table_set(tmp_path):
    """02-F5: v2 表集合校验含 summaries; 缺表在第一道防线拒绝 (不再依赖查询兜底)."""
    fixture_dir = tmp_path / "s"
    store = open_session_store(directory=fixture_dir, session_id=None, workspace=tmp_path)
    session_id = store.session_id
    import asyncio

    asyncio.run(store.aclose())
    db_path = fixture_dir / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("DROP TABLE summaries")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ArchiveFormatError, match="missing tables"):
        open_session_store(directory=fixture_dir, session_id=session_id, workspace=tmp_path)


async def test_summary_non_integer_column_tamper_is_format_error(tmp_path):
    """02-F7: 标识列被篡改为非整数字符串 -> ArchiveFormatError (非裸 ValueError)."""
    fixture = await _store_with_two_requests(tmp_path)
    store = fixture.store
    record = _record_for(store, fixture.epoch_id)
    await store.commit_summary(record)
    db_path = store.db_path
    session_id = store.session_id
    import asyncio

    await store.aclose()
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            "UPDATE summaries SET context_generation = 'not-an-int'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ArchiveFormatError, match="non-integer identity column"):
        open_session_store(
            directory=fixture.directory, session_id=session_id, workspace=tmp_path
        )


def test_v1_rejections_happen_before_upgrade_and_leave_file_untouched(tmp_path):
    """02-F9: v1 档的工作区不匹配/载荷损坏在迁移提交之前拒绝, 原档字节不变.

    审查证实此前这两类拒绝发生在升级之后 (原档被改写为 v2); 现由加载次序保证。
    """
    for scenario in ("workspace_mismatch", "corrupt_payload"):
        session_id = str(uuid.uuid4())
        directory = tmp_path / f"s-{scenario}"
        stored_workspace = (
            str(tmp_path / "original-ws")
            if scenario == "workspace_mismatch"
            else os.path.realpath(tmp_path)
        )
        _make_v1_archive(directory, session_id, stored_workspace)
        db_path = directory / f"{session_id}.sqlite3"
        if scenario == "corrupt_payload":
            connection = sqlite3.connect(str(db_path))
            try:
                connection.execute(
                    "UPDATE records SET payload_json = '{\"messages\": [], \"outcome\": {}}'"
                    " WHERE seq = 2"
                )
                connection.commit()
            finally:
                connection.close()
        before = db_path.read_bytes()
        with pytest.raises(PersistenceError):
            open_session_store(
                directory=directory, session_id=session_id, workspace=tmp_path
            )
        assert _raw_version(db_path) == (1, 1, False), "拒绝发生在迁移前: 仍是 v1"
        assert db_path.read_bytes() == before, "原档不改写"
