"""目录观察的关键预算与取消回收；正文/终结事实仍由恢复读取。"""

import asyncio
import sqlite3
import threading

import pytest

from cicada.core.cancel import CancelToken
from cicada.persistence.contracts import SESSION_SCHEMA_VERSION
from cicada.plugins.session_store import SessionCatalogReader, open_session_store


async def test_combined_catalog_sees_legacy_and_user_sessions(tmp_path):
    from cicada.application import make_session_catalog_reader

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    current = tmp_path / "user-sessions"
    legacy = workspace / ".cicada" / "sessions"
    stores = [
        open_session_store(directory=current, session_id=None, workspace=workspace),
        open_session_store(directory=legacy, session_id=None, workspace=workspace),
    ]
    ids = {store.session_id for store in stores}
    for store in stores:
        await store.aclose()
    reader = make_session_catalog_reader(
        directory=current, legacy_directory=legacy, workspace=str(workspace))
    try:
        snapshot = await reader.observe(CancelToken())
        assert snapshot.complete
        assert {entry.session_id for entry in snapshot.entries} == ids
    finally:
        await reader.aclose()


async def test_catalog_metadata_scope_and_readonly_validation(tmp_path, monkeypatch):
    from cicada.plugins.session_store import catalog

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    path, session_id = store.db_path, store.session_id
    await store.aclose()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE session SET user_name='中文任务', short_title='Straße 首问'")
        # 完整恢复应拒绝这些事实；列表必须既不读取，也不宣称健康。
        db.execute("INSERT INTO epoch VALUES ('bad', 1, 0, 'not JSON', NULL)")
    queries = []
    original = sqlite3.connect

    def connect(*args, **kwargs):
        db = original(*args, **kwargs)
        db.set_trace_callback(queries.append)
        return db

    monkeypatch.setattr(catalog.sqlite3, 'connect', connect)
    reader = SessionCatalogReader(directory=tmp_path, workspace=str(tmp_path))
    try:
        snapshot = await reader.observe(CancelToken())
        assert snapshot.complete
        entry, = snapshot.entries
        assert entry.session_id == session_id and entry.user_name == '中文任务'
        assert entry.updated_at is None and entry.created_at is not None and entry.error is None
        assert not any(word in ' '.join(queries).lower() for word in (
            'identity_json', 'payload_json', 'process_activity', 'historical_receipts', 'request_intents'))
        with original(path) as db:
            db.execute('UPDATE session SET user_name=?', ('x' * 70_000,))
        queries.clear()
        invalid, = (await reader.observe(CancelToken())).entries
        assert invalid.error and 'input limit' in invalid.error
        assert not any('SELECT session_id, canonical_workspace' in sql for sql in queries)
        assert len(invalid.error.encode('utf-8')) <= 512
        # Observe never upgrades the archive or writes source bytes.
        before = path.read_bytes()
        await reader.observe(CancelToken())
        assert path.read_bytes() == before
    finally:
        await reader.aclose()


async def test_catalog_scope_limits_and_top_level_failure(tmp_path, monkeypatch):
    from cicada.plugins.session_store import catalog

    for i in range(3):
        (tmp_path / f'bad-{i}.sqlite3').write_bytes(b'bad')
    reader = SessionCatalogReader(directory=tmp_path, workspace=None)
    try:
        monkeypatch.setattr(catalog, 'CATALOG_MAX_ENTRIES', 2)
        snapshot = await reader.observe(CancelToken())
        assert not snapshot.complete and snapshot.reason == 'candidate-limit' and len(snapshot.entries) == 2
        monkeypatch.setattr(catalog, 'CATALOG_MAX_DIRECTORY_ITEMS', 1)
        snapshot = await reader.observe(CancelToken())
        assert not snapshot.complete and snapshot.reason == 'directory-limit'
        monkeypatch.setattr(catalog, 'CATALOG_MAX_DIRECTORY_ITEMS', 10_000)
        monkeypatch.setattr(catalog, 'CATALOG_MAX_CACHE_BYTES', 1)
        snapshot = await reader.observe(CancelToken())
        assert not snapshot.complete and snapshot.reason == 'cache-limit' and not snapshot.entries
        monkeypatch.setattr(catalog.os, 'scandir', lambda _: (_ for _ in ()).throw(PermissionError('denied')))
        with pytest.raises(PermissionError):
            await reader.observe(CancelToken())
    finally:
        await reader.aclose()


async def test_catalog_numeric_fields_never_transfer_unbounded_blob(tmp_path, monkeypatch):
    from cicada.plugins.session_store import catalog

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    path = store.db_path
    await store.aclose()
    seen = []
    original = catalog._timestamp

    def timestamp(value, **kwargs):
        seen.append(value)
        assert not isinstance(value, (str, bytes)), 'numeric input must be bounded in SQL'
        return original(value, **kwargs)

    monkeypatch.setattr(catalog, '_timestamp', timestamp)
    reader = SessionCatalogReader(directory=tmp_path, workspace=str(tmp_path))
    try:
        for column in ('schema_version', 'created_at'):
            with sqlite3.connect(path) as db:
                db.execute(f'UPDATE session SET {column}=zeroblob(1048576)')
            entry, = (await reader.observe(CancelToken())).entries
            assert entry.error
            with sqlite3.connect(path) as db:
                db.execute('UPDATE session SET schema_version=?, created_at=1', (SESSION_SCHEMA_VERSION,))
        with sqlite3.connect(path) as db:
            # 坏档视图模拟时间列，预算 SQL 不得把聚合 BLOB 搬入 Python。
            db.execute('DROP TABLE records')
            db.execute('CREATE VIEW records AS SELECT zeroblob(1048576) AS created_at')
        entry, = (await reader.observe(CancelToken())).entries
        assert entry.error and 'timestamp type' in entry.error
        assert seen and all(type(v) in (int, float, type(None)) for v in seen)
    finally:
        await reader.aclose()


async def test_catalog_repeated_cancel_and_close_wait_actual_worker(tmp_path, monkeypatch):
    entered, release, reclaimed = threading.Event(), threading.Event(), threading.Event()
    reader = SessionCatalogReader(directory=tmp_path, workspace=None)

    def blocked(stop):
        entered.set()
        try:
            assert release.wait(5)
            assert stop.is_set()
            raise asyncio.CancelledError
        finally:
            reclaimed.set()

    monkeypatch.setattr(reader, '_observe', blocked)
    token = CancelToken()
    task = asyncio.create_task(reader.observe(token))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        close = asyncio.create_task(reader.aclose())
        await asyncio.sleep(0)
        assert not task.done() and not close.done() and not reclaimed.is_set()
        with pytest.raises(RuntimeError):
            await reader.observe(CancelToken())
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await close
    assert reclaimed.is_set() and reader._worker is None


async def test_catalog_sql_deadline_closes_connection(tmp_path, monkeypatch):
    from cicada.plugins.session_store import catalog

    store = open_session_store(directory=tmp_path, session_id=None, workspace=tmp_path)
    path = store.db_path
    await store.aclose()
    entered, release = threading.Event(), threading.Event()
    original = sqlite3.connect
    opened = []

    def connect(*args, **kwargs):
        kwargs['check_same_thread'] = False
        connection = original(*args, **kwargs)
        opened.append(connection)
        # 很小的单档预算，阻塞后仍必须关连接并保留错误行。
        entered.set()
        assert release.wait(5)
        return connection

    monkeypatch.setattr(catalog.sqlite3, 'connect', connect)
    monkeypatch.setattr(catalog, 'CATALOG_FILE_TIMEOUT_S', 0.001)
    reader = SessionCatalogReader(directory=tmp_path, workspace=str(tmp_path))
    task = asyncio.create_task(reader.observe(CancelToken()))
    assert await asyncio.to_thread(entered.wait, 5)
    await asyncio.sleep(0.01)
    release.set()
    snapshot = await task
    assert snapshot.entries[0].error and snapshot.complete
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        opened[0].execute('SELECT 1')
    # 读连接释放，不妨碍原恢复取 lease。
    resumed = open_session_store(directory=tmp_path, session_id=path.stem, workspace=tmp_path)
    await resumed.aclose()
    with original(path) as db:
        db.execute('DROP TABLE records')
        db.execute("CREATE VIEW records AS WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL"
                   " SELECT x+1 FROM n WHERE x<10000000) SELECT x AS created_at FROM n")
    monkeypatch.setattr(catalog, 'CATALOG_FILE_TIMEOUT_S', 0.01)
    entry, = (await reader.observe(CancelToken())).entries
    # 单档预算也覆盖连接前后的 checkpoint，慢宿主可能先到该出口。
    assert entry.error in (
        'OperationalError: interrupted',
        'TimeoutError: session metadata SQL budget exhausted',
    )
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        opened[-1].execute('SELECT 1')
    await reader.aclose()
