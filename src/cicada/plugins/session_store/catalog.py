"""有界基本元数据观察；不读取正文、epoch identity 或终结事实。

一个受拥有线程串行读档，threading.Event 桥接合作取消。返回/关闭之前真实
等待 worker finally 释放 iterator、连接和锁探测句柄；OS 调用没有硬 deadline。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.persistence.contracts import (
    CATALOG_FILE_TIMEOUT_S, CATALOG_MAX_CACHE_BYTES, CATALOG_MAX_DIRECTORY_ITEMS,
    CATALOG_MAX_ENTRIES, CATALOG_MAX_ERROR_BYTES, CATALOG_TIMEOUT_S,
    MAX_DB_BYTES, MAX_METADATA_BYTES, SHORT_TITLE_MAX_CHARS,
    PersistenceError, SessionCatalogEntry, SessionCatalogSnapshot, validate_session_name,
)
from cicada.plugins.session_store.listing import _probe_occupied
from cicada.plugins.session_store.store import _check_identity, validate_session_id


def _bounded_error(value: str) -> str:
    return value.encode("utf-8", "replace")[:CATALOG_MAX_ERROR_BYTES].decode("utf-8", "ignore")


def _timestamp(value, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError("invalid session timestamp")
    return float(value)


class SessionCatalogReader:
    def __init__(self, *, directory: Path, workspace: str | None) -> None:
        self.directory = Path(directory)
        self.workspace = os.path.realpath(workspace) if workspace is not None else None
        self._closed = False
        self._worker: asyncio.Task | None = None
        self._cancel: threading.Event | None = None

    async def observe(self, cancel: CancelToken) -> SessionCatalogSnapshot:
        if self._closed:
            raise RuntimeError("session catalog is closed")
        if self._worker is not None:
            raise RuntimeError("session catalog observation is still owned")
        cancel.throw_if_cancelled()
        stop = threading.Event()
        self._cancel = stop
        worker = asyncio.create_task(asyncio.to_thread(self._observe, stop))
        self._worker = worker

        async def forward_cancel() -> None:
            await cancel.wait()
            stop.set()

        watcher = asyncio.create_task(forward_cancel())
        cancelled = False
        try:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
                    stop.set()
            if cancelled or cancel.cancelled or stop.is_set():
                # Retrieve any late worker failure before propagating cancellation.
                if not worker.cancelled():
                    worker.exception()
                raise asyncio.CancelledError
            return worker.result()
        finally:
            watcher.cancel()
            cleanup = asyncio.gather(watcher, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
                    stop.set()
            self._worker = None
            self._cancel = None
            if cancelled:
                raise asyncio.CancelledError

    async def aclose(self) -> None:
        self._closed = True
        if self._cancel is not None:
            self._cancel.set()
        worker = self._worker
        cancelled = False
        if worker is not None:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Worker 合作取消是关闭的正常结果；外层取消才需传播。
                    cancelled = cancelled or asyncio.current_task().cancelling() > 0
                except Exception:
                    break  # 已完成失败同样经过 worker finally。
            if not worker.cancelled():
                worker.exception()
        if cancelled:
            raise asyncio.CancelledError

    def _observe(self, stop: threading.Event) -> SessionCatalogSnapshot:
        deadline = time.monotonic() + CATALOG_TIMEOUT_S
        entries: list[SessionCatalogEntry] = []
        cache_bytes = len(json.dumps(str(self.directory), ensure_ascii=False).encode("utf-8")) + 128
        reason = None
        try:
            iterator = os.scandir(self.directory)
        except FileNotFoundError:
            iterator = None
        if iterator is not None:
            with iterator:
                for index, item in enumerate(iterator):
                    if stop.is_set():
                        raise asyncio.CancelledError
                    if time.monotonic() >= deadline:
                        reason = "observation-budget"
                        break
                    if index >= CATALOG_MAX_DIRECTORY_ITEMS:
                        reason = "directory-limit"
                        break
                    if not item.name.endswith(".sqlite3"):
                        continue
                    if len(entries) >= CATALOG_MAX_ENTRIES:
                        reason = "candidate-limit"
                        break
                    entry = self._read_entry(Path(item.path), stop, deadline)
                    # 筛选派生文字也计入缓存预算；没有隐藏原文或逐项长路径。
                    size = len(json.dumps(
                        [asdict(entry), entry.search_fields], ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8", "replace"))
                    if cache_bytes + size > CATALOG_MAX_CACHE_BYTES:
                        reason = "cache-limit"
                        break
                    cache_bytes += size
                    entries.append(entry)
        if stop.is_set():
            raise asyncio.CancelledError
        if time.monotonic() >= deadline:
            reason = reason or "observation-budget"
        entries.sort(key=lambda entry: (
            -entry.updated_at if entry.updated_at is not None else float("inf"), entry.session_id,
        ))
        return SessionCatalogSnapshot(uuid.uuid4().hex, str(self.directory), tuple(entries),
                                      reason is None, reason)

    def _read_entry(self, path: Path, stop: threading.Event, deadline: float) -> SessionCatalogEntry:
        session_id = path.stem[:SHORT_TITLE_MAX_CHARS]
        occupied = False
        connection = None
        file_deadline = min(deadline, time.monotonic() + CATALOG_FILE_TIMEOUT_S)

        def checkpoint() -> None:
            if stop.is_set():
                raise asyncio.CancelledError
            if time.monotonic() >= file_deadline:
                raise TimeoutError("session metadata SQL budget exhausted")

        def progress() -> int:
            return int(stop.is_set() or time.monotonic() >= file_deadline)

        try:
            checkpoint()
            occupied = _probe_occupied(path.with_suffix(".lock"))
            session_id = validate_session_id(path.stem)
            if path.stat().st_size > MAX_DB_BYTES:
                raise ValueError("session archive exceeds the size limit")
            checkpoint()
            connection = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True,
                                         timeout=CATALOG_FILE_TIMEOUT_S)
            connection.set_progress_handler(progress, 1_000)
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = 100")
            connection.execute("BEGIN")
            version = _check_identity(connection)
            fields = ["session_id", "canonical_workspace", "short_title"]
            if version >= 4:
                fields.append("user_name")
            # 检查字节上限后才把字段交给 Python；LIMIT 2 检查单行身份。
            lengths = ", ".join(f"length(CAST({field} AS BLOB))" for field in fields)
            rows = connection.execute(
                "SELECT CASE WHEN typeof(schema_version) = 'integer' THEN schema_version END,"
                " CASE WHEN typeof(created_at) IN ('integer', 'real') THEN created_at END,"
                f" {lengths} FROM session LIMIT 2"
            ).fetchall()
            if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] != version:
                raise ValueError("invalid session row or schema version")
            row = rows[0]
            sizes = row[2:]
            if any(size is not None and size > MAX_METADATA_BYTES for size in sizes) or sum(
                size or 0 for size in sizes
            ) > MAX_METADATA_BYTES:
                raise ValueError("session metadata exceeds the input limit")
            values = connection.execute(f"SELECT {', '.join(fields)} FROM session LIMIT 1").fetchone()
            stored_id, workspace, title = values[:3]
            if not isinstance(stored_id, str) or stored_id != session_id or path.stem != session_id:
                raise ValueError("file name does not match the archived session id")
            if not isinstance(workspace, str) or not workspace or "\x00" in workspace:
                raise ValueError("invalid archived workspace")
            if not os.path.isabs(workspace) or os.path.realpath(workspace) != workspace:
                raise ValueError("invalid canonical workspace")
            if self.workspace is not None and workspace != self.workspace:
                raise ValueError("session belongs to a different workspace")
            if not isinstance(title, str) or len(title) > SHORT_TITLE_MAX_CHARS:
                raise ValueError("invalid session short title")
            name = validate_session_name(values[3]) if version >= 4 else None
            created = _timestamp(row[1])
            latest, invalid_time = connection.execute(
                "SELECT MAX(CASE WHEN typeof(created_at) IN ('integer', 'real')"
                " THEN created_at END),"
                " MAX(CASE WHEN typeof(created_at) NOT IN ('integer', 'real') THEN 1 ELSE 0 END)"
                " FROM records"
            ).fetchone()
            if invalid_time:
                raise ValueError("invalid record timestamp type")
            updated = _timestamp(latest, optional=True)
            checkpoint()
            return SessionCatalogEntry(session_id, name, title, created, updated, occupied)
        except (OSError, sqlite3.Error, ValueError, TypeError, PersistenceError) as exc:
            if stop.is_set():
                raise asyncio.CancelledError
            return SessionCatalogEntry(session_id, None, "", None, None, occupied,
                                       _bounded_error(f"{type(exc).__name__}: {exc}"))
        finally:
            if connection is not None:
                connection.close()


class CombinedSessionCatalogReader:
    """新旧目录的只读视图；同 ID 使用用户级副本。"""

    def __init__(self, *, directory: Path, legacy_directory: Path, workspace: str | None) -> None:
        self.current = SessionCatalogReader(directory=directory, workspace=workspace)
        self.legacy = SessionCatalogReader(directory=legacy_directory, workspace=workspace)

    async def observe(self, cancel: CancelToken) -> SessionCatalogSnapshot:
        current = await self.current.observe(cancel)
        legacy = await self.legacy.observe(cancel)
        entries = {entry.session_id: entry for entry in legacy.entries}
        entries.update({entry.session_id: entry for entry in current.entries})
        ordered = sorted(entries.values(), key=lambda entry: (
            -entry.updated_at if entry.updated_at is not None else float("inf"), entry.session_id,
        ))
        complete = current.complete and legacy.complete and len(ordered) <= CATALOG_MAX_ENTRIES
        reason = None if complete else current.reason or legacy.reason or "candidate-limit"
        return SessionCatalogSnapshot(
            uuid.uuid4().hex, current.directory, tuple(ordered[:CATALOG_MAX_ENTRIES]), complete, reason)

    async def aclose(self) -> None:
        try:
            await self.current.aclose()
        finally:
            await self.legacy.aclose()
