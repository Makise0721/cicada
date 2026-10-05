"""工作区会话列表 (票据 05; 契约 §3 list_sessions).

只读轻量读取器: 不取 lease、不做完整格式校验 (完整校验归 open_session_store
的恢复路径); 逐文件独立读取, 读不了的文件是明确错误项而不是被跳过 (契约:
坏文件和活动会话分别标注, 不能伪装成空列表)。占用经 lock 文件非阻塞区域
锁探测 (取得即释放; 探测与写者取锁的窗口竞争由恢复 open 的既有租约语义
兜底)。分页按最近提交记录时间降序, offset 游标; 目录增删是合法并发事实,
游标只承诺"从当前目录状态的该偏移继续"。
"""

from __future__ import annotations

import msvcrt
import os
import sqlite3
from pathlib import Path

from cicada.persistence.codec import (
    CodecError,
    decode_receipt_terminal,
    decode_terminal_facts,
    strict_load,
)
from cicada.persistence.contracts import (
    LIST_SESSIONS_PAGE_SIZE,
    SessionPage,
    SessionSummary,
    StoredReceipt,
    receipt_terminal_trustworthy,
)


def list_sessions(
    *, directory: Path, cursor: str | None = None, limit: int = LIST_SESSIONS_PAGE_SIZE
) -> SessionPage:
    """列出目录下的会话存档; 只读, 无模型/工具/TTY 依赖。

    cursor 为上一页返回的 next_cursor (十进制偏移); limit 有下界保护。
    返回条目按 updated_at 降序 (无记录的排最后, 再按文件名稳定排序)。
    """
    limit = max(1, int(limit))
    directory = Path(directory)
    offset = _parse_cursor(cursor)
    rows: list[tuple[float, str, SessionSummary]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.sqlite3")):
            summary = _summarize_archive(path)
            # 降序按最近提交; 无记录 (含坏档) 排最后, 文件名稳定次序
            updated = summary.updated_at
            key = (-updated if updated is not None else float("inf"), summary.session_id)
            rows.append((key[0], key[1], summary))
    rows.sort(key=lambda item: (item[0], item[1]))
    page = rows[offset : offset + limit]
    next_cursor = str(offset + limit) if offset + limit < len(rows) else None
    return SessionPage(
        directory=str(directory), entries=tuple(item[2] for item in page), next_cursor=next_cursor
    )


def _parse_cursor(cursor: str | None) -> int:
    if cursor is None or cursor == "":
        return 0
    try:
        value = int(cursor)
    except ValueError as exc:
        raise ValueError(f"invalid list cursor: {cursor!r}") from exc
    if value < 0:
        raise ValueError(f"invalid list cursor: {cursor!r}")
    return value


def _summarize_archive(path: Path) -> SessionSummary:
    session_id = path.stem
    occupied = _probe_occupied(path.with_suffix(".lock"))
    try:
        # as_uri 只接受绝对路径; CLI/选择器可传相对目录, URI 前先 resolve
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            return _read_summary(connection, session_id, occupied)
        finally:
            connection.close()
    except (sqlite3.Error, ValueError, TypeError, CodecError) as exc:
        return SessionSummary(
            session_id=session_id, short_title="", created_at=0.0, updated_at=None,
            committed_requests=0, interrupted_requests=0, occupied=occupied,
            unproven_activities=0, untrusted_receipts=0, error=f"{type(exc).__name__}: {exc}",
        )


def _read_summary(connection: sqlite3.Connection, session_id: str, occupied: bool) -> SessionSummary:
    row = connection.execute(
        "SELECT session_id, short_title, created_at FROM session"
    ).fetchall()
    if len(row) != 1:
        raise ValueError(f"session table has {len(row)} rows")
    stored_id, short_title, created_at = row[0]
    if str(stored_id) != session_id:
        raise ValueError("file name does not match the archived session id")
    updated_at = connection.execute("SELECT MAX(created_at) FROM records").fetchone()[0]
    committed = connection.execute(
        "SELECT COUNT(*) FROM request_intents WHERE status = 'committed'"
    ).fetchone()[0]
    interrupted = connection.execute(
        "SELECT COUNT(*) FROM request_intents WHERE status != 'committed'"
    ).fetchone()[0]
    return SessionSummary(
        session_id=str(stored_id),
        short_title=str(short_title),
        created_at=float(created_at),
        updated_at=float(updated_at) if updated_at is not None else None,
        committed_requests=int(committed),
        interrupted_requests=int(interrupted),
        occupied=occupied,
        unproven_activities=_count_unproven_activities(connection),
        untrusted_receipts=_count_untrusted_receipts(connection),
    )


def _load_payload(raw: str) -> dict:
    """与 store 加载同源: 严格 JSON 解析后再进 codec; 坏行按 CodecError 保守计数。"""
    parsed = strict_load(raw)
    if not isinstance(parsed, dict):
        raise CodecError("record payload must be a JSON object")
    return parsed


def _count_unproven_activities(connection: sqlite3.Connection) -> int:
    """未证终结/未闭合活动计数; 终结记录解不了按不可信保守标注。"""
    unproven = 0
    for (end_json,) in connection.execute("SELECT end_json FROM process_activity"):
        if end_json is None:
            unproven += 1
            continue
        try:
            if decode_terminal_facts(_load_payload(str(end_json))).kind == "unproven_terminal":
                unproven += 1
        except CodecError:
            unproven += 1
    return unproven


def _count_untrusted_receipts(connection: sqlite3.Connection) -> int:
    """不可信终结回执计数 (与恢复准入 receipt_terminal_trustworthy 同源); 坏行保守计入。"""
    untrusted = 0
    rows = connection.execute(
        "SELECT verification_run_id, receipt_id, epoch_id, request_id,"
        " definition_fingerprint, terminal_json, activity_id FROM historical_receipts"
    )
    for values in rows:
        try:
            (
                execution_status, exit_code, timed_out, cancelled, output_complete, failure_kind,
            ) = decode_receipt_terminal(_load_payload(str(values[5])))
            receipt = StoredReceipt(
                verification_run_id=str(values[0]), receipt_id=str(values[1]),
                epoch_id=str(values[2]), request_id=str(values[3]),
                definition_fingerprint=str(values[4]), execution_status=execution_status,
                exit_code=exit_code, timed_out=timed_out, cancelled=cancelled,
                output_complete=output_complete, failure_kind=failure_kind,
                activity_id=str(values[6]) if values[6] is not None else None,
            )
        except (CodecError, ValueError, TypeError):
            untrusted += 1
            continue
        if not receipt_terminal_trustworthy(receipt):
            untrusted += 1
    return untrusted


def _probe_occupied(lock_path: Path) -> bool:
    """lock 文件首字节非阻塞锁探测: 取得即释放; 失败/打不开视为占用。"""
    if not lock_path.exists():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return True
    try:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return False
    finally:
        os.close(fd)


__all__ = ["list_sessions"]
