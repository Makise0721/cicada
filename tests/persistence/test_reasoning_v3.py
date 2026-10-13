"""v3 原子升级、旧消息/摘要身份、当前配置指针与安全摘要身份."""

import hashlib
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import replace

import pytest

from cicada.context_policy import canonical_prefix_sha256
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.models import SafeModelIdentity
from cicada.persistence.codec import (
    CodecError, decode_summary_record, dumps_strict, encode_identity, encode_summary_record, strict_load,
)
from cicada.persistence.contracts import ArchiveFormatError
from cicada.plugins.session_store import list_sessions, open_session_store, read_session_identity
import cicada.plugins.session_store.store as store_module
from test_summary import _make_v1_archive, _make_record, _identity


def _legacy(tmp_path, version):
    directory = tmp_path / "sessions"
    session_id = str(uuid.uuid4())
    _make_v1_archive(directory, session_id, os.path.realpath(tmp_path))
    path = directory / f"{session_id}.sqlite3"
    if version == 2:
        with sqlite3.connect(path) as connection:
            # 固定 v2 DDL, 不借当前 schema 常量或迁移步骤制造 fixture。
            connection.execute("""CREATE TABLE summaries (
                summary_id TEXT PRIMARY KEY, epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
                context_generation INTEGER NOT NULL, first_request_seq INTEGER NOT NULL,
                last_request_seq INTEGER NOT NULL, message_count INTEGER NOT NULL,
                end_message_index INTEGER NOT NULL, prefix_sha256 TEXT NOT NULL,
                summary_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL)""")
            connection.execute("UPDATE session SET schema_version = 2")
            connection.execute("PRAGMA user_version = 2")
    return directory, session_id, path


def _versions_and_pointer(path):
    with sqlite3.connect(path) as connection:
        return connection.execute("PRAGMA user_version").fetchone()[0], connection.execute(
            "SELECT schema_version, current_epoch_id FROM session"
        ).fetchone()


def _legacy_name_archive(tmp_path, version):
    directory, session_id, path = _legacy(tmp_path, min(version, 2))
    if version == 3:
        with sqlite3.connect(path) as connection:
            epoch_id = connection.execute('SELECT epoch_id FROM epoch').fetchone()[0]
            connection.execute('ALTER TABLE session ADD COLUMN current_epoch_id TEXT REFERENCES epoch(epoch_id)')
            connection.execute('UPDATE session SET schema_version=3, current_epoch_id=?', (epoch_id,))
            connection.execute('PRAGMA user_version=3')
    return directory, session_id, path


@pytest.mark.parametrize('version', [1, 2, 3])
async def test_v4_migration_failure_rolls_back_all_earlier_steps(tmp_path, monkeypatch, version):
    directory, session_id, path = _legacy_name_archive(tmp_path, version)
    before = path.read_bytes()
    assert read_session_identity(directory=directory, session_id=session_id, workspace=tmp_path)
    assert list_sessions(directory=directory).entries[0].user_name is None
    assert before == path.read_bytes()
    steps = list(store_module._V4_MIGRATION_STEPS)
    def fail(connection):
        raise RuntimeError('injected v4 final step')
    monkeypatch.setattr(store_module, '_V4_MIGRATION_STEPS', (*steps[:2], fail))
    with pytest.raises(RuntimeError, match='v4 final step'):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert before == path.read_bytes(), '不得半升级 v1/v2/v3'
    monkeypatch.setattr(store_module, '_V4_MIGRATION_STEPS', tuple(steps))
    store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    try:
        snapshot = store.recovery_snapshot()
        assert snapshot.user_name is None
        epoch_id = store._state.current_epoch_id
        handle = store.submit_name('中文名称', expected_epoch_id=epoch_id)
        assert await handle.wait() == await handle.wait() == '中文名称'
        assert store.recovery_snapshot() == replace(snapshot, user_name='中文名称')
        assert _versions_and_pointer(path) == (4, (4, epoch_id))
        with pytest.raises(store_module.UnknownArchiveVersionError):
            monkeypatch.setattr(store_module, 'SESSION_SCHEMA_VERSION', 3)
            read_session_identity(directory=directory, session_id=session_id, workspace=tmp_path)
        monkeypatch.setattr(store_module, 'SESSION_SCHEMA_VERSION', 4)
    finally:
        await store.aclose()


async def test_corrupt_v4_names_refused_without_rewriting_archive(tmp_path):
    store = open_session_store(directory=tmp_path / 'sessions', session_id=None, workspace=tmp_path)
    await store.start_epoch(_identity())
    path, session_id = store.db_path, store.session_id
    await store.aclose()
    for name in (' untrimmed ', 'x\u200dy', b'not text', 'x' * 81):
        with sqlite3.connect(path) as connection:
            connection.execute('UPDATE session SET user_name=?', (name,))
        before = path.read_bytes()
        with pytest.raises(ArchiveFormatError, match='session name'):
            open_session_store(directory=path.parent, session_id=session_id, workspace=tmp_path)
        with pytest.raises(ArchiveFormatError, match='session name'):
            read_session_identity(directory=path.parent, session_id=session_id, workspace=tmp_path)
        assert list_sessions(directory=path.parent).entries[0].error
        assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2])
async def test_legacy_migration_preserves_payloads_and_identity(tmp_path, version):
    directory, session_id, path = _legacy(tmp_path, version)
    with sqlite3.connect(path) as connection:
        epoch_id = connection.execute("SELECT epoch_id FROM epoch").fetchone()[0]
        records = connection.execute("SELECT * FROM records ORDER BY seq").fetchall()
        if version == 2:
            record = replace(_make_record(tmp_path), epoch_id=epoch_id,
                             prefix_sha256=canonical_prefix_sha256((UserMessage("第一问"), AssistantMessage("回答"))))
            raw = encode_summary_record(record)
            connection.execute("INSERT INTO summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                record.summary_id, epoch_id, 0, 2, 2, 2, 2, record.prefix_sha256, raw,
                hashlib.sha256(raw.encode()).hexdigest(),
            ))
            summaries = connection.execute("SELECT * FROM summaries").fetchall()
    before = path.read_bytes()
    assert read_session_identity(directory=directory, session_id=session_id, workspace=tmp_path) == _identity()
    assert list_sessions(directory=directory).entries[0].error is None
    assert path.read_bytes() == before, "身份预读与列表不迁移"
    store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    try:
        snapshot = store.recovery_snapshot()
        assert snapshot.canonical_history == (UserMessage("第一问"), AssistantMessage("回答"))
        assert _versions_and_pointer(path) == (4, (4, epoch_id))
        if version == 2:
            assert snapshot.summaries == (record,)
            assert snapshot.summaries[0].prefix_sha256 == canonical_prefix_sha256(snapshot.canonical_history)
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT * FROM records ORDER BY seq").fetchall() == records
            if version == 2:
                assert connection.execute("SELECT * FROM summaries").fetchall() == summaries
    finally:
        await store.aclose()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("step", [0, 2])
def test_v3_step_failure_restores_original_bytes(tmp_path, monkeypatch, version, step):
    """v3 各迁移步骤同分支 (单事务原子回滚): 保留首/尾边界代表."""
    directory, session_id, path = _legacy(tmp_path, version)
    before = path.read_bytes()
    steps = list(store_module._V3_MIGRATION_STEPS)
    def fail(connection):
        raise RuntimeError("injected v3 failure")
    steps[step] = fail
    monkeypatch.setattr(store_module, "_V3_MIGRATION_STEPS", tuple(steps))
    with pytest.raises(RuntimeError, match="injected v3"):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert "current_epoch_id" not in [r[1] for r in connection.execute("PRAGMA table_info(session)")]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("step", [0, 2])
async def test_v3_process_interruption_never_leaves_partial_upgrade(tmp_path, version, step):
    directory, session_id, path = _legacy(tmp_path, version)
    child = f"""
import os
from pathlib import Path
import cicada.plugins.session_store.store as m
def kill(connection):
    os._exit(9)
steps = list(m._V3_MIGRATION_STEPS)
steps[{step}] = kill
m._V3_MIGRATION_STEPS = tuple(steps)
m.open_session_store(directory=Path({str(directory)!r}), session_id={session_id!r}, workspace=Path({str(tmp_path)!r}))
"""
    result = subprocess.run([sys.executable, "-c", child], capture_output=True,
                            env={**os.environ, "PYTHONPATH": str(store_module.Path(__file__).resolve().parents[2] / "src")},
                            timeout=30)
    assert result.returncode == 9, result.stderr
    store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    try:
        assert len(store.recovery_snapshot().requests) == 1
        assert _versions_and_pointer(path)[0] == 4
    finally:
        await store.aclose()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("closed", [False, True])
async def test_old_pre_read_and_upgrade_agree_on_open_or_legacy_order(tmp_path, version, closed):
    directory, session_id, path = _legacy(tmp_path, version)
    with sqlite3.connect(path) as connection:
        original = connection.execute("SELECT epoch_id FROM epoch").fetchone()[0]
        # 当前 open epoch 排在 closed epoch 前; 旧覆盖循环会错误继承后者。
        connection.execute("UPDATE epoch SET opened_at=0, closed=?", (int(closed),))
        other = replace(_identity(), launch=replace(_identity().launch, model="legacy-latest"))
        connection.execute("INSERT INTO epoch VALUES (?, 1, 1, ?, NULL)", ("zzzz", encode_identity(other)))
    expected = other if closed else _identity()
    assert read_session_identity(directory=directory, session_id=session_id, workspace=tmp_path) == expected
    store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    try:
        assert store.recovery_snapshot().identity == expected
        assert _versions_and_pointer(path)[1][1] == ("zzzz" if closed else original)
    finally:
        await store.aclose()


async def test_v3_pointer_wins_same_timestamp_reverse_uuid_and_survives_closed_epoch(tmp_path, monkeypatch):
    store = open_session_store(directory=tmp_path / "sessions", session_id=None, workspace=tmp_path)
    path = store.db_path
    assert _versions_and_pointer(path) == (4, (4, None))
    first_id = "ffffffff-ffff-4fff-bfff-ffffffffffff"
    second_id = "00000000-0000-4000-8000-000000000000"
    ids = iter([first_id, second_id])
    monkeypatch.setattr(store_module.uuid, "uuid4", lambda: next(ids))
    monkeypatch.setattr(store_module.time, "time", lambda: 123.0)
    try:
        await store.start_epoch(_identity())
        selected = replace(_identity(), launch=replace(_identity().launch, model="selected"))
        assert await store.start_epoch(selected) == second_id
    finally:
        await store.aclose()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE epoch SET closed=1")
    assert read_session_identity(directory=path.parent, session_id=store.session_id, workspace=tmp_path) == selected
    reopened = open_session_store(directory=path.parent, session_id=store.session_id, workspace=tmp_path)
    try:
        assert reopened.recovery_snapshot().identity == selected
    finally:
        await reopened.aclose()


@pytest.mark.parametrize("corruption", ["null", "unknown", "other", "multi-open"])
async def test_bad_v3_pointer_or_open_relationship_rejected_by_all_readers(tmp_path, corruption):
    store = open_session_store(directory=tmp_path / "sessions", session_id=None, workspace=tmp_path)
    first = await store.start_epoch(_identity())
    second = await store.start_epoch(_identity())
    path = store.db_path
    await store.aclose()
    with sqlite3.connect(path) as connection:
        if corruption == "multi-open":
            connection.execute("UPDATE epoch SET closed=0")
        else:
            connection.execute("UPDATE session SET current_epoch_id=?", (
                None if corruption == "null" else "missing" if corruption == "unknown" else first,
            ))
    before = path.read_bytes()
    with pytest.raises(ArchiveFormatError):
        read_session_identity(directory=path.parent, session_id=store.session_id, workspace=tmp_path)
    with pytest.raises(ArchiveFormatError):
        open_session_store(directory=path.parent, session_id=store.session_id, workspace=tmp_path)
    assert list_sessions(directory=path.parent).entries[0].error
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2])
def test_multi_open_old_archives_rejected_without_migration(tmp_path, version):
    directory, session_id, path = _legacy(tmp_path, version)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO epoch VALUES ('other', ?, 0, ?, NULL)", (time.time(), encode_identity(_identity())))
    before = path.read_bytes()
    with pytest.raises(ArchiveFormatError, match="more than one open"):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["digest", "source"])
def test_bad_v2_summary_rejected_before_upgrade(tmp_path, corruption):
    directory, session_id, path = _legacy(tmp_path, 2)
    with sqlite3.connect(path) as connection:
        epoch = connection.execute("SELECT epoch_id FROM epoch").fetchone()[0]
        record = replace(_make_record(tmp_path), epoch_id=epoch)
        raw = encode_summary_record(record)
        connection.execute("INSERT INTO summaries VALUES (?, ?, 0, 2, 2, 2, 2, ?, ?, ?)", (
            record.summary_id, epoch, record.prefix_sha256, raw,
            "wrong-digest" if corruption == "digest" else hashlib.sha256(raw.encode()).hexdigest(),
        ))
    before = path.read_bytes()
    with pytest.raises(ArchiveFormatError, match="digest" if corruption == "digest" else "coverage does not match"):
        open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
    assert path.read_bytes() == before


def test_summary_call_identity_optional_and_strict(tmp_path):
    record = _make_record(tmp_path)
    old = encode_summary_record(record)
    assert "call_identity" not in old and decode_summary_record(strict_load(old)) == record
    identity = SafeModelIdentity("deepseek", "test", "https://api.deepseek.com", False,
                                 (("num_ctx", 131072), ("max_tokens", 2048), ("reasoning_effort", "high")), 2097152)
    updated = replace(record, call_identity=identity)
    assert decode_summary_record(strict_load(encode_summary_record(updated))) == updated
    for bad in (None, {}, {**strict_load(encode_summary_record(updated))["call_identity"], "secret": "key"}):
        payload = strict_load(old)
        payload["call_identity"] = bad
        with pytest.raises(CodecError):
            decode_summary_record(payload)
    with pytest.raises(ValueError):
        replace(identity, options=(("api_key", "secret"),))
