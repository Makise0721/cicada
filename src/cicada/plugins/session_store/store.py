"""SQLite 单会话存储: lease、串行 worker、v1 关系、限额与逻辑完整性.

物理设置 (设计契约 §2): journal_mode=DELETE + synchronous=EXTRA + foreign_keys=ON
+ temp_store=MEMORY + busy_timeout=1000ms; application_id/user_version 写后回读.
单写入者 lease 用 msvcrt 非阻塞区域锁锁定 <id>.lock 首字节, 打开生命周期持有;
锁文件存在不是占用依据, 关闭不删除锁文件.

连接与 SQL 只在一个受拥有的串行 worker 线程执行 (check_same_thread=False 由
本纪律背书); 每个派发事务必得明确结果, 调用方取消不取消事务; 关闭先排空
队列, 超时不提前释放连接/lease. 开库与加载在启动路径一次性同步完成 (有界).
"""

from __future__ import annotations

import asyncio
import hashlib
import msvcrt
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from cicada.core.messages import Message, UserMessage, isolate_message, validate_history_structure
from cicada.persistence.codec import (
    CodecError,
    decode_activity_begin,
    decode_identity,
    decode_messages,
    decode_outcome_facts,
    decode_receipt_terminal,
    decode_summary_record,
    decode_terminal_facts,
    encode_activity_begin,
    encode_identity,
    encode_messages,
    encode_outcome_facts,
    encode_receipt_terminal,
    encode_summary_record,
    encode_terminal_facts,
    dumps_strict,
    strict_load,
)
from cicada.persistence.contracts import (
    MAX_COMMITTED_REQUESTS,
    MAX_DB_BYTES,
    MAX_MESSAGES,
    MAX_METADATA_BYTES,
    MAX_REQUEST_PAYLOAD_BYTES,
    MAX_SUMMARIES_PER_GENERATION,
    SESSION_APPLICATION_ID,
    SESSION_SCHEMA_VERSION,
    SHORT_TITLE_MAX_CHARS,
    SUMMARY_TEXT_MAX_BYTES,
    ArchiveFormatError,
    ArchiveLimitError,
    InterruptedRequest,
    LeaseHeldError,
    PersistenceError,
    ProcessActivityFacts,
    ProcessTerminalFacts,
    RecoverySnapshot,
    RequestOutcomeFacts,
    RestoredRequest,
    SessionNotFoundError,
    StorageUnavailableError,
    StorageWriteError,
    StoredIdentity,
    StoredReceipt,
    SummaryRecord,
    UnknownArchiveVersionError,
    WorkspaceMismatchError,
)

SESSION_STORE_DRAIN_TIMEOUT_S = 30.0

# UUIDv4 精确格式; 输入大小写不敏感, 统一归一小写后拼路径
_SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE
)

_SCHEMA_STATEMENTS = (
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
    """
    CREATE TABLE summaries (
        summary_id TEXT PRIMARY KEY,
        epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
        context_generation INTEGER NOT NULL,
        first_request_seq INTEGER NOT NULL,
        last_request_seq INTEGER NOT NULL,
        message_count INTEGER NOT NULL,
        end_message_index INTEGER NOT NULL,
        prefix_sha256 TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL
    )
    """,
)

_EXPECTED_TABLES = frozenset({
    "session", "epoch", "records", "request_intents", "process_activity", "historical_receipts",
})

# v1 存档 (升级前) 的期望表集合; 迁移校验面与既有加载共用同一清单
_V1_EXPECTED_TABLES = _EXPECTED_TABLES

# v1 → v2 原子迁移 (PA04) 的步骤序列; 每步一个连接操作, 全部包在单个显式事务内,
# 任一步失败整体回滚 (重开只能是完整 v1 或完整 v2)。测试经本常量注入故障点。
_V2_MIGRATION_STEPS: tuple[Callable[[sqlite3.Connection], None], ...] = (
    lambda connection: connection.execute(
        """
        CREATE TABLE summaries (
            summary_id TEXT PRIMARY KEY,
            epoch_id TEXT NOT NULL REFERENCES epoch(epoch_id),
            context_generation INTEGER NOT NULL,
            first_request_seq INTEGER NOT NULL,
            last_request_seq INTEGER NOT NULL,
            message_count INTEGER NOT NULL,
            end_message_index INTEGER NOT NULL,
            prefix_sha256 TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        )
        """
    ),
    lambda connection: connection.execute(
        "UPDATE session SET schema_version = ?", (SESSION_SCHEMA_VERSION,)
    ),
    lambda connection: connection.execute(f"PRAGMA user_version = {SESSION_SCHEMA_VERSION}"),
)


def _verify_v2_structure(connection: sqlite3.Connection) -> None:
    """迁移提交前的 v2 结构核对: 双版本标记与新表在位 (PA04, 提交前最后一道)."""
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or int(row[0]) != SESSION_SCHEMA_VERSION:
        raise ArchiveFormatError("v2 migration did not confirm user_version")
    schema_row = connection.execute("SELECT schema_version FROM session").fetchone()
    if schema_row is None or int(schema_row[0]) != SESSION_SCHEMA_VERSION:
        raise ArchiveFormatError("v2 migration did not confirm session.schema_version")
    tables = {
        str(item[0])
        for item in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "summaries" not in tables:
        raise ArchiveFormatError("v2 migration did not create the summaries table")

T = TypeVar("T")


def validate_session_id(session_id: str) -> str:
    """精确校验会话 ID (UUIDv4), 归一小写; 不把任意字符串拼成路径."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise PersistenceError(
            "session id must be a UUIDv4 string like 123e4567-e89b-42d3-a456-426614174000"
        )
    return session_id.lower()


# --- 打开时加载的内存状态 ---------------------------------------------------------------


@dataclass
class _IntentRecord:
    epoch_id: str
    prompt: str
    status: Literal["pending", "committed", "interrupted"]


@dataclass
class _CommittedRequest:
    request_id: str
    epoch_id: str
    seq: int
    context_generation: int
    prompt: str
    messages: tuple[Message, ...]
    outcome: RequestOutcomeFacts
    committed_at: float


@dataclass
class _ActivityRecord:
    activity_id: str
    request_id: str
    tool_name: str
    call_id: str
    arguments_digest: str
    description: str
    started_at: float
    terminal: ProcessTerminalFacts | None


@dataclass
class _LoadedState:
    session_id: str
    canonical_workspace: str
    created_at: float
    short_title: str
    identity: StoredIdentity | None = None
    epoch_count: int = 0
    context_generation: int = 0
    next_seq: int = 1
    committed_count: int = 0
    message_count: int = 0
    current_epoch_id: str | None = None
    requests: list[_CommittedRequest] = field(default_factory=list)
    intents: dict[str, _IntentRecord] = field(default_factory=dict)
    activities: dict[str, _ActivityRecord] = field(default_factory=dict)
    receipts: dict[str, StoredReceipt] = field(default_factory=dict)
    summaries: list[SummaryRecord] = field(default_factory=list)   # 全部代次, 按写入序
    schema_version: int = SESSION_SCHEMA_VERSION                    # 打开时的存档版本


# --- store 实现 -------------------------------------------------------------------------


class SQLiteSessionStore:
    """SessionStorePort 的 SQLite 实现; 单会话单写入者."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        lock_fd: int,
        db_path: Path,
        state: _LoadedState,
    ) -> None:
        self._conn = connection
        self._lock_fd = lock_fd
        self._db_path = db_path
        self._state = state
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._work, name=f"cicada-store-{state.session_id}", daemon=True
        )
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task | None = None  # 并发/重试 aclose 共享同一次关闭
        self._drained = threading.Event()
        # _state 只在 worker 线程持锁修改; recovery_snapshot 在调用线程持锁读取
        self._state_lock = threading.Lock()
        self._worker.start()

    # --- port: 只读 ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._state.session_id

    @property
    def db_path(self) -> Path:
        return self._db_path

    def recovery_snapshot(self) -> RecoverySnapshot:
        """按当前已提交事实重建的隔离快照 (点时视图, 嵌套数据已复制)."""
        with self._state_lock:
            return self._build_snapshot()

    # --- port: 事务 ------------------------------------------------------------------

    async def start_epoch(self, identity: StoredIdentity) -> str:
        identity_json = self._encode_checked(
            encode_identity(identity), MAX_METADATA_BYTES, "epoch identity"
        )

        def job() -> str:
            epoch_id = str(uuid.uuid4())
            now = time.time()
            self._check_db_size(len(identity_json.encode("utf-8")))
            self._immediate()
            try:
                if self._state.current_epoch_id is not None:
                    self._conn.execute(
                        "UPDATE epoch SET closed = 1 WHERE epoch_id = ? AND closed = 0",
                        (self._state.current_epoch_id,),
                    )
                self._conn.execute(
                    "INSERT INTO epoch (epoch_id, opened_at, closed, identity_json,"
                    " verification_run_id) VALUES (?, ?, 0, ?, NULL)",
                    (epoch_id, now, identity_json),
                )
                self._conn.execute(
                    "UPDATE request_intents SET status = 'interrupted' WHERE status = 'pending'"
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.current_epoch_id = epoch_id
            self._state.epoch_count += 1
            self._state.identity = identity
            for request_id, record in self._state.intents.items():
                if record.status == "pending":
                    self._state.intents[request_id] = _IntentRecord(
                        epoch_id=record.epoch_id, prompt=record.prompt, status="interrupted"
                    )
            return epoch_id

        return await self._dispatch(job)

    async def begin_request(self, epoch_id: str, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise PersistenceError("prompt must be a nonempty request")
        begin_payload = self._encode_checked(
            {"prompt": prompt}, MAX_METADATA_BYTES, "request prompt"
        )

        def job() -> str:
            if epoch_id != self._state.current_epoch_id:
                raise PersistenceError(f"unknown epoch for begin_request: {epoch_id}")
            if any(record.status == "pending" for record in self._state.intents.values()):
                raise PersistenceError(
                    "a previous request is still pending; only one active request is allowed"
                )
            self._check_db_size(len(begin_payload.encode("utf-8")))
            request_id = str(uuid.uuid4())
            now = time.time()
            self._immediate()
            try:
                cursor = self._conn.execute(
                    "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
                    " created_at, payload_json, payload_sha256)"
                    " VALUES (?, ?, ?, 'request_begin', ?, ?, ?, NULL)",
                    (
                        self._state.next_seq,
                        epoch_id,
                        self._state.context_generation,
                        request_id,
                        now,
                        begin_payload,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO request_intents (request_id, epoch_id, begin_seq, status)"
                    " VALUES (?, ?, ?, 'pending')",
                    (request_id, epoch_id, cursor.lastrowid),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.next_seq += 1
            self._state.intents[request_id] = _IntentRecord(
                epoch_id=epoch_id, prompt=prompt, status="pending"
            )
            return request_id

        return await self._dispatch(job)

    async def commit_request(
        self, request_id: str, *, messages: tuple[Message, ...], outcome: RequestOutcomeFacts,
        receipts: tuple[StoredReceipt, ...] = (),
    ) -> int:
        validate_history_structure(messages)
        if not messages or not isinstance(messages[0], UserMessage):
            raise PersistenceError("committed request messages must start with the user prompt")
        if not isinstance(receipts, tuple) or any(
            not isinstance(receipt, StoredReceipt) for receipt in receipts
        ):
            raise PersistenceError("receipts must be a tuple of StoredReceipt")
        if len({receipt.receipt_id for receipt in receipts}) != len(receipts):
            raise PersistenceError("receipt ids must be unique within one commit")
        receipt_rows = [
            (
                receipt,
                self._encode_checked(
                    encode_receipt_terminal(receipt), MAX_METADATA_BYTES,
                    "receipt terminal evidence",
                ),
            )
            for receipt in receipts
        ]
        payload = self._encode_checked(
            {
                "messages": encode_messages(messages),
                "outcome": encode_outcome_facts(outcome),
            },
            MAX_REQUEST_PAYLOAD_BYTES,
            "request payload",
        )
        payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        # 回执行 (terminal_json + 标识列) 与请求载荷一并计入 db-size 检查点,
        # 否则回执字节可把库推过限额, 写出本程序自己拒绝打开的存档
        receipt_bytes = sum(
            len(terminal_json.encode("utf-8"))
            + len(receipt.verification_run_id.encode("utf-8"))
            + len(receipt.receipt_id.encode("utf-8"))
            + len(receipt.epoch_id.encode("utf-8"))
            + len(receipt.request_id.encode("utf-8"))
            + len(receipt.definition_fingerprint.encode("utf-8"))
            for receipt, terminal_json in receipt_rows
        )

        def job() -> int:
            record = self._state.intents.get(request_id)
            if record is None:
                raise PersistenceError(f"unknown request for commit: {request_id}")
            if record.status == "committed":
                raise PersistenceError(
                    f"request {request_id} was already committed; repeated commits are "
                    "rejected (the store never appends twice)"
                )
            if record.status != "pending":
                raise PersistenceError(f"request {request_id} is {record.status}; cannot commit")
            if messages[0].text != record.prompt:
                # 与加载侧同一不变量: 写入侧先拒绝, 避免一次坏提交写出打不开的存档
                raise PersistenceError(
                    "committed request messages must start with the exact intent prompt"
                )
            for receipt, _terminal_json in receipt_rows:
                if receipt.receipt_id in self._state.receipts:
                    # 先于请求归属检查: 同一回执改挂到新请求也属重复关联
                    raise PersistenceError(
                        f"receipt {receipt.receipt_id} was already persisted; "
                        "the same receipt is not copied into every request"
                    )
                if receipt.request_id != request_id:
                    raise PersistenceError(
                        f"receipt {receipt.receipt_id} references request {receipt.request_id}, "
                        f"not the committing request {request_id}"
                    )
                if receipt.epoch_id != record.epoch_id:
                    raise PersistenceError(
                        f"receipt {receipt.receipt_id} epoch does not match the request epoch"
                    )
            open_activities = [
                activity_id
                for activity_id, activity in self._state.activities.items()
                if activity.request_id == request_id and activity.terminal is None
            ]
            if open_activities:
                raise PersistenceError(
                    f"request {request_id} still has {len(open_activities)} process activity "
                    "without a terminal record; the full commit waits for every activity write"
                )
            if self._state.committed_count + 1 > MAX_COMMITTED_REQUESTS:
                raise ArchiveLimitError(
                    f"session exceeds the request limit ({MAX_COMMITTED_REQUESTS})"
                )
            if self._state.message_count + len(messages) > MAX_MESSAGES:
                raise ArchiveLimitError(
                    f"session exceeds the message limit ({MAX_MESSAGES})"
                )
            self._check_db_size(len(payload.encode("utf-8")) + receipt_bytes)
            seq = self._state.next_seq
            now = time.time()
            short_title = self._state.short_title or messages[0].text[:SHORT_TITLE_MAX_CHARS]
            self._immediate()
            try:
                self._conn.execute(
                    "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
                    " created_at, payload_json, payload_sha256)"
                    " VALUES (?, ?, ?, 'request', ?, ?, ?, ?)",
                    (
                        seq,
                        record.epoch_id,
                        self._state.context_generation,
                        request_id,
                        now,
                        payload,
                        payload_sha,
                    ),
                )
                self._conn.execute(
                    "UPDATE request_intents SET status = 'committed' WHERE request_id = ?",
                    (request_id,),
                )
                if short_title != self._state.short_title:
                    self._conn.execute(
                        "UPDATE session SET short_title = ? WHERE session_id = ?",
                        (short_title, self._state.session_id),
                    )
                linked_receipts: list[StoredReceipt] = []
                for receipt, terminal_json in receipt_rows:
                    # 关联到本请求内终结事实带同一 receipt_id 的活动 (check-run 包装器记录)
                    receipt_activity = next(
                        (
                            activity_id
                            for activity_id, activity in self._state.activities.items()
                            if activity.request_id == request_id
                            and activity.terminal is not None
                            and activity.terminal.receipt_id == receipt.receipt_id
                        ),
                        None,
                    )
                    self._conn.execute(
                        "INSERT INTO historical_receipts (verification_run_id, receipt_id,"
                        " epoch_id, request_id, activity_id, definition_fingerprint,"
                        " terminal_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            receipt.verification_run_id, receipt.receipt_id, receipt.epoch_id,
                            receipt.request_id, receipt_activity, receipt.definition_fingerprint,
                            terminal_json,
                        ),
                    )
                    linked_receipts.append(
                        replace(receipt, activity_id=receipt_activity)
                        if receipt_activity != receipt.activity_id else receipt
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.next_seq += 1
            self._state.committed_count += 1
            self._state.message_count += len(messages)
            self._state.short_title = short_title
            for receipt in linked_receipts:
                self._state.receipts[receipt.receipt_id] = receipt
            self._state.intents[request_id] = _IntentRecord(
                epoch_id=record.epoch_id, prompt=record.prompt, status="committed"
            )
            self._state.requests.append(_CommittedRequest(
                request_id=request_id,
                epoch_id=record.epoch_id,
                seq=seq,
                context_generation=self._state.context_generation,
                prompt=record.prompt,
                messages=messages,
                outcome=outcome,
                committed_at=now,
            ))
            return seq

        return await self._dispatch(job)

    async def commit_clear(self, epoch_id: str) -> int:
        def job() -> int:
            if epoch_id != self._state.current_epoch_id:
                raise PersistenceError(f"unknown epoch for commit_clear: {epoch_id}")
            new_generation = self._state.context_generation + 1
            self._check_db_size(1024)  # clear 行固定小载荷; 上界含行开销
            self._immediate()
            try:
                self._conn.execute(
                    "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
                    " created_at, payload_json, payload_sha256)"
                    " VALUES (?, ?, ?, 'clear', NULL, ?, '{}', NULL)",
                    (self._state.next_seq, epoch_id, new_generation, time.time()),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.next_seq += 1
            self._state.context_generation = new_generation
            return new_generation

        return await self._dispatch(job)

    async def commit_summary(self, record: SummaryRecord) -> None:
        """可靠提交摘要记录 (02, schema v2); 成功即 store 已接受 (明确结果).

        关系校验: epoch/generation 必须是当前打开值; summary_id 重复拒绝;
        seq 范围必须引用当前代次已提交完整请求; 每代次摘要数限额;
        summary_json 与标识列交叉写入, 摘要字节计入 db-size 检查点。
        """
        if not isinstance(record, SummaryRecord):
            raise PersistenceError("commit_summary requires a SummaryRecord")
        summary_json = self._encode_checked(
            encode_summary_record(record), MAX_METADATA_BYTES, "summary record"
        )
        summary_sha = hashlib.sha256(summary_json.encode("utf-8")).hexdigest()
        row_bytes = (
            len(summary_json.encode("utf-8"))
            + len(record.summary_id.encode("utf-8"))
            + len(record.epoch_id.encode("utf-8"))
            + len(record.prefix_sha256.encode("utf-8"))
            + 64
        )

        def job() -> None:
            if record.epoch_id != self._state.current_epoch_id:
                raise PersistenceError(
                    f"unknown epoch for commit_summary: {record.epoch_id}"
                )
            if record.context_generation != self._state.context_generation:
                raise PersistenceError(
                    f"summary generation {record.context_generation} does not match the "
                    f"current generation {self._state.context_generation}"
                )
            if record.summary_id in {item.summary_id for item in self._state.summaries}:
                raise PersistenceError(
                    f"summary {record.summary_id} was already persisted; repeated commits are rejected"
                )
            seqs = {
                item.seq for item in self._state.requests
                if item.context_generation == self._state.context_generation
            }
            if record.first_request_seq not in seqs or record.last_request_seq not in seqs:
                raise PersistenceError(
                    "summary coverage must reference committed requests of the current generation"
                )
            generation_summaries = sum(
                1 for item in self._state.summaries
                if item.context_generation == self._state.context_generation
            )
            if generation_summaries + 1 > MAX_SUMMARIES_PER_GENERATION:
                raise ArchiveLimitError(
                    "session exceeds the summaries-per-generation limit "
                    f"({MAX_SUMMARIES_PER_GENERATION})"
                )
            self._check_db_size(row_bytes)
            self._immediate()
            try:
                self._conn.execute(
                    "INSERT INTO summaries (summary_id, epoch_id, context_generation,"
                    " first_request_seq, last_request_seq, message_count, end_message_index,"
                    " prefix_sha256, summary_json, payload_sha256)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.summary_id,
                        record.epoch_id,
                        record.context_generation,
                        record.first_request_seq,
                        record.last_request_seq,
                        record.message_count,
                        record.end_message_index,
                        record.prefix_sha256,
                        summary_json,
                        summary_sha,
                    ),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.summaries.append(record)

        return await self._dispatch(job)

    async def begin_process(
        self, request_id: str, *, tool_name: str, call_id: str, arguments: Any
    ) -> str:
        """命令真正执行前可靠登记活动开始; 失败由调用方零执行并锁存存储故障."""
        if not isinstance(tool_name, str) or not tool_name:
            raise PersistenceError("process activity tool_name must be a nonempty str")
        if not isinstance(call_id, str) or not call_id:
            raise PersistenceError("process activity call_id must be a nonempty str")
        if not isinstance(arguments, dict):
            raise PersistenceError("process activity arguments must be a JSON object")
        try:
            canonical = dumps_strict(arguments)
        except CodecError as exc:
            raise PersistenceError(f"process activity arguments are not strictly serializable: {exc}") from exc
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        description = _bounded_description(canonical)

        def job() -> str:
            record = self._state.intents.get(request_id)
            if record is None:
                raise PersistenceError(f"unknown request for process activity: {request_id}")
            if record.status != "pending":
                raise PersistenceError(
                    f"request {request_id} is {record.status}; activities require a pending request"
                )
            started_at = time.time()
            begin_json = self._encode_checked(
                encode_activity_begin(description, digest, started_at),
                MAX_METADATA_BYTES,
                "process activity begin",
            )
            self._check_db_size(len(begin_json.encode("utf-8")))
            activity_id = str(uuid.uuid4())
            self._immediate()
            try:
                self._conn.execute(
                    "INSERT INTO process_activity (activity_id, request_id, tool_name, call_id,"
                    " arguments_digest, begin_json, end_json) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (activity_id, request_id, tool_name, call_id, digest, begin_json),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.activities[activity_id] = _ActivityRecord(
                activity_id=activity_id,
                request_id=request_id,
                tool_name=tool_name,
                call_id=call_id,
                arguments_digest=digest,
                description=description,
                started_at=started_at,
                terminal=None,
            )
            return activity_id

        return await self._dispatch(job)

    async def end_process(self, activity_id: str, terminal: ProcessTerminalFacts) -> None:
        """登记结构化终结事实; 重复终结或未知活动按关系错误拒绝."""
        if not isinstance(terminal, ProcessTerminalFacts):
            raise PersistenceError("end_process requires ProcessTerminalFacts")
        end_json = self._encode_checked(
            encode_terminal_facts(terminal), MAX_METADATA_BYTES, "process terminal evidence"
        )

        def job() -> None:
            record = self._state.activities.get(activity_id)
            if record is None:
                raise PersistenceError(f"unknown process activity: {activity_id}")
            if record.terminal is not None:
                raise PersistenceError(
                    f"process activity {activity_id} already has a terminal record"
                )
            self._check_db_size(len(end_json.encode("utf-8")))
            self._immediate()
            try:
                cursor = self._conn.execute(
                    "UPDATE process_activity SET end_json = ? WHERE activity_id = ? AND end_json IS NULL",
                    (end_json, activity_id),
                )
                if cursor.rowcount != 1:
                    raise PersistenceError(
                        f"process activity {activity_id} changed state concurrently"
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._state.activities[activity_id] = _ActivityRecord(
                activity_id=record.activity_id,
                request_id=record.request_id,
                tool_name=record.tool_name,
                call_id=record.call_id,
                arguments_digest=record.arguments_digest,
                description=record.description,
                started_at=record.started_at,
                terminal=terminal,
            )

        return await self._dispatch(job)

    # --- 关闭 ------------------------------------------------------------------------

    async def aclose(self) -> None:
        """排空已派发事务后释放连接与 lease; 超时不提前释放, 让关闭诊断如实记录.

        并发/重复调用共享同一次关闭任务; 排空超时后再次 aclose() 会重新等待排空
        (队列可能已清空), 给出补完成释放的路径 —— _closing 只封新事务, 不封收尾.
        """
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        if self._close_task is None:
            self._closing = True
            self._queue.put((None, None, None))  # 停机哨兵: 之前入队的事务全部先完成
            self._close_task = loop.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        loop = asyncio.get_running_loop()
        drained = await loop.run_in_executor(
            None, self._drained.wait, SESSION_STORE_DRAIN_TIMEOUT_S
        )
        if not drained:
            # 超时不释放; 放开关闭任务让后续 aclose() 重试排空
            self._close_task = None
            raise StorageWriteError(
                "session store still has unfinished transactions after the drain deadline; "
                "connection and lease are intentionally not released"
            )
        await loop.run_in_executor(None, self._release)
        self._closed = True

    def _release(self) -> None:
        try:
            self._conn.close()
        finally:
            _release_lease(self._lock_fd)

    # --- worker 纪律 -----------------------------------------------------------------

    async def _dispatch(self, job: Callable[[], T]) -> T:
        if self._closing or self._closed:
            raise StorageUnavailableError("session store is closed; no new transactions accepted")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._queue.put((job, loop, future))
        return await future

    def _work(self) -> None:
        while True:
            job, loop, future = self._queue.get()
            if job is None:
                self._drained.set()
                return
            result: Any = None
            error: BaseException | None = None
            try:
                with self._state_lock:  # 事务与状态更新对快照读取原子
                    result = job()
            except sqlite3.Error as exc:
                # 事务执行中的 SQL 失败属于存储故障, 不把裸 sqlite3.Error 泄给调用方
                error = StorageWriteError(f"session store SQL failed: {exc}")
            except BaseException as exc:  # 事务必得明确结果: 异常也送达调用方
                error = exc

            def deliver(
                loop: asyncio.AbstractEventLoop = loop,
                future: asyncio.Future = future,
                result: Any = result,
                error: BaseException | None = error,
            ) -> None:
                if future.cancelled():
                    return  # 调用方放弃等待, 事务结果已落库
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(result)

            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass  # 事件循环已关闭: 事务已完成, 结果无人接收

    # --- SQL 与限额辅助 --------------------------------------------------------------

    def _immediate(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _check_db_size(self, extra_bytes: int) -> None:
        row = self._conn.execute("PRAGMA page_count").fetchone()
        page = self._conn.execute("PRAGMA page_size").fetchone()
        if row is None or page is None:
            raise StorageWriteError("cannot read database size before write")
        if int(row[0]) * int(page[0]) + extra_bytes > MAX_DB_BYTES:
            raise ArchiveLimitError(
                f"session database would exceed the size limit ({MAX_DB_BYTES} bytes)"
            )

    def _encode_checked(self, payload: object, limit: int, what: str) -> str:
        try:
            encoded = payload if isinstance(payload, str) else dumps_strict(payload)
        except CodecError as exc:
            raise PersistenceError(f"{what} is not strictly serializable: {exc}") from exc
        if len(encoded.encode("utf-8")) > limit:
            raise ArchiveLimitError(f"{what} exceeds the limit ({limit} bytes)")
        return encoded

    def _build_snapshot(self) -> RecoverySnapshot:
        current = self._state.context_generation
        requests = tuple(
            RestoredRequest(
                request_id=item.request_id,
                epoch_id=item.epoch_id,
                seq=item.seq,
                context_generation=item.context_generation,
                prompt=item.prompt,
                messages=tuple(isolate_message(m) for m in item.messages),
                outcome=item.outcome,
                committed_at=item.committed_at,
            )
            for item in self._state.requests
        )
        interrupted = tuple(
            InterruptedRequest(
                request_id=request_id,
                epoch_id=record.epoch_id,
                prompt=record.prompt,
                status=record.status,
            )
            for request_id, record in self._state.intents.items()
            if record.status != "committed"
        )
        return RecoverySnapshot(
            session_id=self._state.session_id,
            canonical_workspace=self._state.canonical_workspace,
            created_at=self._state.created_at,
            short_title=self._state.short_title,
            identity=self._state.identity,
            epoch_count=self._state.epoch_count,
            context_generation=current,
            requests=requests,
            interrupted=interrupted,
            canonical_history=tuple(
                isolate_message(message)
                for item in self._state.requests if item.context_generation == current
                for message in item.messages
            ),
            process_activities=tuple(
                ProcessActivityFacts(
                    activity_id=record.activity_id,
                    request_id=record.request_id,
                    tool_name=record.tool_name,
                    call_id=record.call_id,
                    arguments_digest=record.arguments_digest,
                    description=record.description,
                    started_at=record.started_at,
                    terminal=record.terminal,
                )
                for record in self._state.activities.values()
            ),
            restored_receipts=tuple(self._state.receipts.values()),
            summaries=tuple(
                item for item in self._state.summaries
                if item.context_generation == current
            ),
        )


# --- 打开与校验 -------------------------------------------------------------------------


def open_session_store(
    *, directory: Path, session_id: str | None, workspace: Path
) -> SQLiteSessionStore:
    """打开或新建单会话存储; 失败释放已取得资源, 不改写坏文件."""
    canonical_workspace = os.path.realpath(str(workspace))
    if session_id is None:
        return _create_store(Path(directory), canonical_workspace)
    return _resume_store(Path(directory), validate_session_id(session_id), canonical_workspace)


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    try:
        # 每连接设置并立即回读; 不支持的设置不得默默降级
        connection.execute(f"PRAGMA application_id = {SESSION_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SESSION_SCHEMA_VERSION}")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = EXTRA")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA busy_timeout = 1000")
        _read_back(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _read_back(connection: sqlite3.Connection, *, user_version: int | None = SESSION_SCHEMA_VERSION) -> None:
    # application_id 只在新建时写入; 打开既有库时由 _check_identity 校验, 这里仅新建路径到达。
    # user_version 仅在新建路径回读 (resume 的版本路由归 _check_identity, 迁移前的 v1 档不钉当前版本)
    pragmas = [
        ("application_id", SESSION_APPLICATION_ID),
        ("synchronous", 3),  # EXTRA
        ("foreign_keys", 1),
        ("temp_store", 2),   # MEMORY
        ("busy_timeout", 1000),
    ]
    if user_version is not None:
        pragmas.insert(1, ("user_version", user_version))
    for pragma, expected in pragmas:
        row = connection.execute(f"PRAGMA {pragma}").fetchone()
        if row is None or int(row[0]) != expected:
            raise ArchiveFormatError(f"session store {pragma} could not be confirmed")
    mode = connection.execute("PRAGMA journal_mode").fetchone()
    if mode is None or str(mode[0]).lower() != "delete":
        raise ArchiveFormatError("session store requires journal_mode=DELETE")


def _acquire_lease(lock_path: Path) -> int:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        raise StorageUnavailableError(f"cannot open session lock file: {exc}") from exc
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        os.close(fd)
        raise LeaseHeldError(f"session is locked by another process ({lock_path.name})") from exc
    return fd


def _release_lease(lock_fd: int) -> None:
    try:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        os.close(lock_fd)


def _immediate_txn(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")


def _create_store(directory: Path, canonical_workspace: str) -> SQLiteSessionStore:
    session_id = str(uuid.uuid4())
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageUnavailableError(f"cannot create the sessions directory: {exc}") from exc
    db_path = directory / f"{session_id}.sqlite3"
    lock_fd = _acquire_lease(directory / f"{session_id}.lock")
    try:
        try:
            connection = _connect(db_path)
        except sqlite3.Error as exc:
            raise StorageUnavailableError(f"cannot create session database: {exc}") from exc
        try:
            created_at = time.time()
            _immediate_txn(connection)
            try:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO session (schema_version, session_id, canonical_workspace,"
                    " created_at, short_title) VALUES (?, ?, ?, ?, '')",
                    (SESSION_SCHEMA_VERSION, session_id, canonical_workspace, created_at),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        except BaseException:
            connection.close()
            raise
    except BaseException:
        _release_lease(lock_fd)
        raise
    state = _LoadedState(
        session_id=session_id, canonical_workspace=canonical_workspace, created_at=created_at,
        short_title="",
    )
    try:
        return SQLiteSessionStore(connection=connection, lock_fd=lock_fd, db_path=db_path, state=state)
    except BaseException:
        connection.close()
        _release_lease(lock_fd)
        raise


def _resume_store(directory: Path, session_id: str, canonical_workspace: str) -> SQLiteSessionStore:
    db_path = directory / f"{session_id}.sqlite3"
    if not db_path.is_file():
        raise SessionNotFoundError(f"no session archive for id {session_id}")
    try:
        size = db_path.stat().st_size
    except OSError as exc:
        raise StorageUnavailableError(f"cannot stat session archive: {exc}") from exc
    if size > MAX_DB_BYTES:
        raise ArchiveLimitError(
            f"session database exceeds the size limit ({MAX_DB_BYTES} bytes)"
        )
    lock_fd = _acquire_lease(directory / f"{session_id}.lock")
    try:
        try:
            connection = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        except sqlite3.Error as exc:
            raise StorageUnavailableError(f"cannot open session database: {exc}") from exc
        try:
            archive_version = _check_identity(connection)
            _set_session_pragmas(connection)
            _quick_check(connection)
            # 版本感知表集合校验 (PA04): v1 档先按 v1 校验面严格核对再迁移
            expected = _V1_EXPECTED_TABLES if archive_version == 1 else _EXPECTED_TABLES
            _check_schema_tables(connection, expected)
            if archive_version == 1:
                _upgrade_v1_to_v2(connection)
            state = _load_state(connection, session_id, canonical_workspace)
        except sqlite3.OperationalError as exc:
            connection.close()
            text = str(exc).lower()
            if "lock" in text or "busy" in text:
                raise StorageUnavailableError(f"session archive is not readable: {exc}") from exc
            raise ArchiveFormatError(f"session archive is not readable: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise ArchiveFormatError(f"session archive is not readable: {exc}") from exc
        except sqlite3.Error as exc:
            # InterfaceError 等非 DatabaseError 的 sqlite3.Error 同样不得裸逃逸
            connection.close()
            raise ArchiveFormatError(f"session archive is not readable: {exc}") from exc
        except BaseException:
            connection.close()
            raise
    except BaseException:
        _release_lease(lock_fd)
        raise
    try:
        return SQLiteSessionStore(connection=connection, lock_fd=lock_fd, db_path=db_path, state=state)
    except BaseException:
        connection.close()
        _release_lease(lock_fd)
        raise


def _check_identity(connection: sqlite3.Connection) -> int:
    """校验 application_id 并返回存档 user_version; 版本路由迁移/拒绝 (PA04).

    v1 (历史档) 返回 1 由调用方原子升级; 等于当前版本直接打开; 其他值按未知
    版本/坏档拒绝, 不改写原文件。
    """
    row = connection.execute("PRAGMA application_id").fetchone()
    if row is None or int(row[0]) != SESSION_APPLICATION_ID:
        detail = str(row[0]) if row is not None else "missing"
        raise ArchiveFormatError(f"file is not a cicada session archive (application_id={detail})")
    row = connection.execute("PRAGMA user_version").fetchone()
    version = int(row[0]) if row is not None else 0
    if version == SESSION_SCHEMA_VERSION:
        return version
    if version == 1 and SESSION_SCHEMA_VERSION > 1:
        return version  # 历史档: 原子升级路径 (PA04)
    if version > SESSION_SCHEMA_VERSION or version not in (0, 1):
        raise UnknownArchiveVersionError(
            f"session archive version {version} is not supported "
            f"(this build reads v{SESSION_SCHEMA_VERSION})"
        )
    raise ArchiveFormatError(
        f"session archive user_version {version} is not a valid cicada archive version"
    )


def _set_session_pragmas(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA synchronous = EXTRA")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA busy_timeout = 1000")
    # user_version 不在此回读: 版本路由归 _check_identity, 迁移前允许 v1
    _read_back(connection, user_version=None)


def _quick_check(connection: sqlite3.Connection) -> None:
    row = connection.execute("PRAGMA quick_check").fetchone()
    if row is None or str(row[0]) != "ok":
        raise ArchiveFormatError("session archive failed the SQLite quick check")


def _check_schema_tables(connection: sqlite3.Connection, expected: frozenset[str]) -> None:
    tables = {
        str(item[0])
        for item in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing = expected - tables
    if missing:
        raise ArchiveFormatError(f"session archive is missing tables: {sorted(missing)}")


def _upgrade_v1_to_v2(connection: sqlite3.Connection) -> None:
    """v1 → v2 原子增量升级 (PA04/A07): 持 lease 且严格校验 v1 后调用.

    单个显式事务内完成摘要表 DDL、session.schema_version 列与 PRAGMA user_version
    更新及 v2 结构核对; 任一步失败整体回滚, 重开只能是完整 v1 (可重新升级) 或
    完整 v2, 既有完整请求可恢复。故障注入面: monkeypatch 模块级
    _V2_MIGRATION_STEPS / _verify_v2_structure。
    """
    _immediate_txn(connection)
    try:
        for step in _V2_MIGRATION_STEPS:
            step(connection)
        _verify_v2_structure(connection)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _load_state(
    connection: sqlite3.Connection, session_id: str, canonical_workspace: str
) -> _LoadedState:
    rows = connection.execute(
        "SELECT schema_version, session_id, canonical_workspace, created_at, short_title FROM session"
    ).fetchall()
    if len(rows) != 1:
        raise ArchiveFormatError(f"session table must hold exactly one row, found {len(rows)}")
    version, stored_id, stored_workspace, created_at, short_title = rows[0]
    if int(version) != SESSION_SCHEMA_VERSION or str(stored_id) != session_id:
        raise ArchiveFormatError("session row does not match the requested archive")
    if str(stored_workspace) != canonical_workspace:
        raise WorkspaceMismatchError(
            f"session belongs to workspace {stored_workspace!r}, not {canonical_workspace!r}"
        )
    short_title = str(short_title)
    if len(short_title) > SHORT_TITLE_MAX_CHARS:
        raise ArchiveFormatError(
            f"session short title exceeds the limit ({SHORT_TITLE_MAX_CHARS} chars)"
        )
    state = _LoadedState(
        session_id=session_id,
        canonical_workspace=str(stored_workspace),
        created_at=float(created_at),
        short_title=short_title,
    )

    epoch_ids: set[str] = set()
    open_epoch_count = 0
    for epoch_id, _opened_at, closed, identity_json, _run in connection.execute(
        "SELECT epoch_id, opened_at, closed, identity_json, verification_run_id"
        " FROM epoch ORDER BY opened_at, epoch_id"
    ):
        if int(closed) not in (0, 1):
            raise ArchiveFormatError("epoch closed flag must be 0 or 1")
        if int(closed) == 0:
            open_epoch_count += 1
            if open_epoch_count > 1:
                # 写入侧 start_epoch 单事务内只留一个 open epoch; 多个属关系损坏
                raise ArchiveFormatError(
                    "session archive has more than one open epoch (unclosed epoch interleaving)"
                )
        _check_text_size(str(identity_json), MAX_METADATA_BYTES, "epoch identity")
        try:
            identity = decode_identity(str(identity_json))
        except CodecError as exc:
            raise ArchiveFormatError(f"epoch identity is corrupt: {exc}") from exc
        epoch_ids.add(str(epoch_id))
        state.identity = identity
        state.epoch_count += 1
        state.current_epoch_id = str(epoch_id)
    # epoch_count == 0 合法: 新建后 bootstrap 前崩溃的空存档, 恢复准入由应用拒绝

    prompt_by_begin_seq: dict[int, str] = {}
    begin_request_ids: dict[int, str] = {}
    begin_epochs: dict[int, str] = {}
    intents: list[tuple[str, str, int, str]] = []
    for request_id, epoch_id, begin_seq, status in connection.execute(
        "SELECT request_id, epoch_id, begin_seq, status FROM request_intents"
    ):
        if str(epoch_id) not in epoch_ids:
            raise ArchiveFormatError(f"intent {request_id} references an unknown epoch")
        if str(status) not in ("pending", "committed", "interrupted"):
            raise ArchiveFormatError(f"intent {request_id} has unknown status {status!r}")
        intents.append((str(request_id), str(epoch_id), int(begin_seq), str(status)))

    generation = 0
    total_messages = 0
    committed: dict[str, tuple[int, str, int, tuple[Message, ...], RequestOutcomeFacts, float]] = {}
    for seq, epoch_id, record_generation, kind, request_id, created_at, payload_json, _sha in connection.execute(
        "SELECT seq, epoch_id, context_generation, kind, request_id, created_at, payload_json,"
        " payload_sha256 FROM records ORDER BY seq"
    ):
        seq = int(seq)
        kind = str(kind)
        if kind == "clear":
            if request_id is not None:
                raise ArchiveFormatError("clear record must not carry a request id")
            if int(record_generation) != generation + 1:
                raise ArchiveFormatError("clear record must advance the context generation by one")
            generation = int(record_generation)
            _check_text_size(str(payload_json), MAX_METADATA_BYTES, "clear record")
            try:
                clear_payload = _load_payload(str(payload_json))
            except CodecError as exc:
                raise ArchiveFormatError(f"clear record {seq} is corrupt: {exc}") from exc
            if clear_payload != {}:
                raise ArchiveFormatError(f"clear record {seq} payload must be an empty object")
        elif kind in ("request_begin", "request"):
            if request_id is None:
                raise ArchiveFormatError(f"{kind} record must carry a request id")
            if str(epoch_id) not in epoch_ids:
                raise ArchiveFormatError(f"{kind} record {seq} references an unknown epoch")
            if int(record_generation) != generation:
                raise ArchiveFormatError(
                    f"{kind} record {seq} has context generation {record_generation}, expected {generation}"
                )
            if kind == "request_begin":
                _check_text_size(str(payload_json), MAX_METADATA_BYTES, "request_begin record")
                try:
                    payload = _load_payload(str(payload_json))
                except CodecError as exc:
                    raise ArchiveFormatError(f"request_begin record {seq} is corrupt: {exc}") from exc
                if set(payload) != {"prompt"} or not isinstance(payload["prompt"], str):
                    raise ArchiveFormatError("request_begin payload must hold exactly the prompt")
                prompt_by_begin_seq[seq] = payload["prompt"]
                begin_request_ids[seq] = str(request_id)
                begin_epochs[seq] = str(epoch_id)
            else:
                _check_text_size(str(payload_json), MAX_REQUEST_PAYLOAD_BYTES, "request record")
                if _sha is None or not isinstance(_sha, str):
                    raise ArchiveFormatError(f"request record {seq} must carry a payload digest")
                observed_sha = hashlib.sha256(str(payload_json).encode("utf-8")).hexdigest()
                if _sha != observed_sha:
                    # 提交时写入的摘要必须与载荷一致; 不一致即载荷被篡改或损坏
                    raise ArchiveFormatError(
                        f"request record {seq} payload does not match its recorded digest"
                    )
                try:
                    payload = _load_payload(str(payload_json))
                    messages = decode_messages(payload["messages"])
                    outcome = decode_outcome_facts(payload["outcome"])
                except (CodecError, KeyError, TypeError) as exc:
                    raise ArchiveFormatError(f"request record {seq} is corrupt: {exc}") from exc
                try:
                    validate_history_structure(messages)
                except ValueError as exc:
                    raise ArchiveFormatError(
                        f"request record {seq} has an invalid message structure: {exc}"
                    ) from exc
                if not messages or not isinstance(messages[0], UserMessage):
                    raise ArchiveFormatError(f"request record {seq} must start with the user prompt")
                if str(request_id) in committed:
                    raise ArchiveFormatError(f"request {request_id} has more than one request record")
                committed[str(request_id)] = (
                    seq, str(epoch_id), generation, messages, outcome, float(created_at)
                )
                total_messages += len(messages)
        else:
            raise ArchiveFormatError(f"unknown record kind {kind!r}")
        state.next_seq = seq + 1

    if len(committed) > MAX_COMMITTED_REQUESTS or total_messages > MAX_MESSAGES:
        raise ArchiveLimitError("session archive exceeds the request/message limits")

    state.context_generation = generation
    state.committed_count = len(committed)
    state.message_count = total_messages
    intent_ids = {request_id for request_id, _epoch_id, _begin_seq, _status in intents}
    for request_id, entry in committed.items():
        if request_id not in intent_ids:
            raise ArchiveFormatError(
                f"request record {entry[0]} has no matching request intent (orphan record)"
            )
    referenced_begin_seqs = {begin_seq for _request_id, _epoch_id, begin_seq, _status in intents}
    for seq in prompt_by_begin_seq:
        if seq not in referenced_begin_seqs:
            raise ArchiveFormatError(
                f"request_begin record {seq} has no matching request intent (orphan record)"
            )
    for request_id, epoch_id, begin_seq, status in intents:
        prompt = prompt_by_begin_seq.get(begin_seq)
        if prompt is None:
            raise ArchiveFormatError(f"intent {request_id} has no begin record")
        if begin_request_ids.get(begin_seq) != request_id:
            raise ArchiveFormatError(f"intent {request_id} does not match its begin record")
        if begin_epochs.get(begin_seq) != epoch_id:
            raise ArchiveFormatError(f"intent {request_id} epoch does not match its begin record")
        if status == "committed":
            entry = committed.get(request_id)
            if entry is None:
                raise ArchiveFormatError(
                    f"intent {request_id} is committed without a request record"
                )
            seq, record_epoch, record_generation, messages, outcome, committed_at = entry
            if record_epoch != epoch_id:
                raise ArchiveFormatError(f"request record {seq} epoch does not match its intent")
            if messages[0].text != prompt:
                raise ArchiveFormatError(f"request record {seq} does not match its intent prompt")
            state.requests.append(_CommittedRequest(
                request_id=request_id,
                epoch_id=record_epoch,
                seq=seq,
                context_generation=record_generation,
                prompt=prompt,
                messages=messages,
                outcome=outcome,
                committed_at=committed_at,
            ))
            state.intents[request_id] = _IntentRecord(
                epoch_id=epoch_id, prompt=prompt, status="committed"
            )
        else:
            if request_id in committed:
                raise ArchiveFormatError(
                    f"intent {request_id} is {status} but has a committed request record"
                )
            state.intents[request_id] = _IntentRecord(
                epoch_id=epoch_id, prompt=prompt, status=status  # type: ignore[arg-type]
            )

    for activity_id, request_id, tool_name, call_id, digest, begin_json, end_json in connection.execute(
        "SELECT activity_id, request_id, tool_name, call_id, arguments_digest, begin_json, end_json"
        " FROM process_activity ORDER BY rowid"
    ):
        if str(request_id) not in state.intents:
            raise ArchiveFormatError(
                f"process activity {activity_id} references unknown request {request_id}"
            )
        if not isinstance(tool_name, str) or not tool_name:
            raise ArchiveFormatError(f"process activity {activity_id} has an invalid tool name")
        if not isinstance(call_id, str) or not call_id:
            raise ArchiveFormatError(f"process activity {activity_id} has an invalid call id")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ArchiveFormatError(f"process activity {activity_id} has an invalid arguments digest")
        _check_text_size(str(begin_json), MAX_METADATA_BYTES, "process activity begin")
        try:
            description, begin_digest, started_at = decode_activity_begin(
                _load_payload(str(begin_json))
            )
        except CodecError as exc:
            raise ArchiveFormatError(f"process activity {activity_id} begin is corrupt: {exc}") from exc
        if begin_digest != digest:
            raise ArchiveFormatError(
                f"process activity {activity_id} begin digest does not match its column"
            )
        terminal: ProcessTerminalFacts | None = None
        if end_json is not None:
            _check_text_size(str(end_json), MAX_METADATA_BYTES, "process terminal evidence")
            try:
                terminal = decode_terminal_facts(_load_payload(str(end_json)))
            except CodecError as exc:
                raise ArchiveFormatError(
                    f"process activity {activity_id} terminal record is corrupt: {exc}"
                ) from exc
        state.activities[str(activity_id)] = _ActivityRecord(
            activity_id=str(activity_id),
            request_id=str(request_id),
            tool_name=str(tool_name),
            call_id=str(call_id),
            arguments_digest=digest,
            description=description,
            started_at=started_at,
            terminal=terminal,
        )
    for (verification_run_id, receipt_id, epoch_id, request_id, activity_id,
         fingerprint, terminal_json) in connection.execute(
        "SELECT verification_run_id, receipt_id, epoch_id, request_id, activity_id,"
        " definition_fingerprint, terminal_json FROM historical_receipts ORDER BY rowid"
    ):
        if not isinstance(verification_run_id, str) or not verification_run_id:
            raise ArchiveFormatError(f"receipt {receipt_id} has an invalid verification run id")
        if not isinstance(receipt_id, str) or not receipt_id:
            raise ArchiveFormatError("historical receipt must carry a nonempty receipt id")
        if str(epoch_id) not in epoch_ids:
            raise ArchiveFormatError(f"receipt {receipt_id} references an unknown epoch")
        intent = state.intents.get(str(request_id)) if request_id is not None else None
        if intent is None:
            raise ArchiveFormatError(f"receipt {receipt_id} references unknown request {request_id}")
        if intent.status != "committed":
            raise ArchiveFormatError(
                f"receipt {receipt_id} references request {request_id} which is {intent.status}; "
                "receipts are only written with a committed request"
            )
        if intent.epoch_id != str(epoch_id):
            raise ArchiveFormatError(f"receipt {receipt_id} epoch does not match its request")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ArchiveFormatError(f"receipt {receipt_id} has an invalid definition fingerprint")
        if activity_id is not None and str(activity_id) not in state.activities:
            raise ArchiveFormatError(f"receipt {receipt_id} references an unknown process activity")
        if activity_id is not None:
            activity = state.activities[str(activity_id)]
            if activity.request_id != str(request_id):
                raise ArchiveFormatError(f"receipt {receipt_id} activity does not belong to its request")
            # 与写侧对称: 关联活动必须携带同一回执身份 (check-run 包装器记录)
            if activity.terminal is None or activity.terminal.receipt_id != receipt_id:
                raise ArchiveFormatError(
                    f"receipt {receipt_id} activity does not carry the same receipt identity"
                )
        _check_text_size(str(terminal_json), MAX_METADATA_BYTES, "receipt terminal evidence")
        try:
            (execution_status, exit_code, timed_out, cancelled,
             output_complete, failure_kind) = decode_receipt_terminal(_load_payload(str(terminal_json)))
        except CodecError as exc:
            raise ArchiveFormatError(f"receipt {receipt_id} terminal record is corrupt: {exc}") from exc
        try:
            receipt = StoredReceipt(
                verification_run_id=verification_run_id,
                receipt_id=receipt_id,
                epoch_id=str(epoch_id),
                request_id=str(request_id),
                definition_fingerprint=fingerprint,
                execution_status=execution_status,
                exit_code=exit_code,
                timed_out=timed_out,
                cancelled=cancelled,
                output_complete=output_complete,
                failure_kind=failure_kind,
                activity_id=str(activity_id) if activity_id is not None else None,
            )
        except ValueError as exc:
            raise ArchiveFormatError(f"receipt {receipt_id} is inconsistent: {exc}") from exc
        if receipt.receipt_id in state.receipts:
            raise ArchiveFormatError(f"receipt {receipt.receipt_id} appears more than once")
        state.receipts[receipt.receipt_id] = receipt

    # --- 摘要记录 (02, v2): 列与 JSON 交叉核对 + 关系/限额校验 -------------------------
    committed_by_seq = {entry[0]: entry for entry in committed.values()}  # seq -> entry
    per_generation_summaries: dict[int, int] = {}
    for (summary_id, summary_epoch, summary_generation, first_seq, last_seq,
         message_count, end_index, prefix_sha, summary_json, summary_sha) in connection.execute(
        "SELECT summary_id, epoch_id, context_generation, first_request_seq,"
        " last_request_seq, message_count, end_message_index, prefix_sha256, summary_json,"
        " payload_sha256 FROM summaries ORDER BY rowid"
    ):
        if str(summary_epoch) not in epoch_ids:
            raise ArchiveFormatError(f"summary {summary_id} references an unknown epoch")
        _check_text_size(str(summary_json), MAX_METADATA_BYTES, "summary record")
        if not isinstance(summary_sha, str) or (
            summary_sha != hashlib.sha256(str(summary_json).encode("utf-8")).hexdigest()
        ):
            # 与 records 同一不变量: 提交时写入的摘要必须与载荷一致
            raise ArchiveFormatError(
                f"summary {summary_id} payload does not match its recorded digest"
            )
        try:
            record = decode_summary_record(_load_payload(str(summary_json)))
        except CodecError as exc:
            raise ArchiveFormatError(f"summary {summary_id} is corrupt: {exc}") from exc
        # 标识列与 JSON 载荷交叉核对: 任何一列被单独篡改都拒绝 (写读对称)
        if (
            record.summary_id != str(summary_id)
            or record.epoch_id != str(summary_epoch)
            or record.context_generation != int(summary_generation)
            or record.first_request_seq != int(first_seq)
            or record.last_request_seq != int(last_seq)
            or record.message_count != int(message_count)
            or record.end_message_index != int(end_index)
            or record.prefix_sha256 != str(prefix_sha)
        ):
            raise ArchiveFormatError(
                f"summary {summary_id} columns do not match its JSON payload"
            )
        for seq in (record.first_request_seq, record.last_request_seq):
            entry = committed_by_seq.get(seq)
            if entry is None:
                raise ArchiveFormatError(
                    f"summary {summary_id} references request record {seq} which does not exist"
                )
            if entry[2] != record.context_generation:
                raise ArchiveFormatError(
                    f"summary {summary_id} generation does not match its covered requests"
                )
        if record.end_message_index > state.message_count:
            raise ArchiveFormatError(
                f"summary {summary_id} coverage end exceeds the archived message count"
            )
        count = per_generation_summaries.get(record.context_generation, 0) + 1
        if count > MAX_SUMMARIES_PER_GENERATION:
            raise ArchiveLimitError(
                "session archive exceeds the summaries-per-generation limit "
                f"({MAX_SUMMARIES_PER_GENERATION})"
            )
        per_generation_summaries[record.context_generation] = count
        state.summaries.append(record)
    state.requests.sort(key=lambda item: item.seq)
    return state


def _load_payload(raw: str) -> dict[str, Any]:
    parsed = strict_load(raw)
    if not isinstance(parsed, dict):
        raise CodecError("record payload must be a JSON object")
    return parsed


def _check_text_size(text: str, limit: int, what: str) -> None:
    if len(text.encode("utf-8")) > limit:
        raise ArchiveLimitError(f"{what} exceeds the limit ({limit} bytes)")


_DESCRIPTION_MAX_BYTES = 512


def _bounded_description(canonical_arguments: str) -> str:
    """参数派生的有界描述: 规范 JSON 前缀, UTF-8 边界安全; 完整参数以 digest 定位."""
    data = canonical_arguments.encode("utf-8")
    if len(data) <= _DESCRIPTION_MAX_BYTES:
        return canonical_arguments
    return data[:_DESCRIPTION_MAX_BYTES].decode("utf-8", "ignore") + "…"
