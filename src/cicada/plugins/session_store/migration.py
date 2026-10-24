"""按需复制旧工作区会话到用户级状态目录，旧材料留作只读历史。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path

from cicada.persistence.contracts import StorageUnavailableError
from cicada.plugins.session_store.store import (
    _acquire_lease, _release_lease, read_session_identity, validate_session_id,
)
from cicada.state_paths import session_directory, workspace_state_dir


def migrate_legacy_session(
    workspace: Path, session_id: str, *, runtime_dir: Path | None = None,
) -> None:
    """仅在显式恢复旧 ID 时复制；源 lease 保证不会分叉活动会话。"""
    session_id = validate_session_id(session_id)
    old = Path(workspace) / ".cicada"
    source = old / "sessions" / f"{session_id}.sqlite3"
    destination_dir = runtime_dir / "sessions" if runtime_dir is not None else session_directory(workspace)
    destination = destination_dir / source.name
    if destination.is_file() or not source.is_file():
        return
    lock_path = source.with_suffix(".lock")
    if not lock_path.is_file():
        raise StorageUnavailableError("legacy session has no lock file; migration requires an intact archive")
    lease = _acquire_lease(lock_path)
    temp_dir: Path | None = None
    try:
        if destination.is_file():
            return
        read_session_identity(directory=source.parent, session_id=session_id, workspace=workspace)
        destination_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = destination_dir / f".migrate-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        temp_db = temp_dir / source.name
        with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as reader:
            with closing(sqlite3.connect(temp_db)) as writer:
                reader.backup(writer)
        read_session_identity(directory=temp_dir, session_id=session_id, workspace=workspace)
        destination_root = runtime_dir if runtime_dir is not None else workspace_state_dir(workspace)
        _copy_restore_material(old / "restore", destination_root / "restore", session_id)
        os.replace(temp_db, destination)
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise StorageUnavailableError(f"legacy session migration failed: {exc}") from exc
    finally:
        failure = sys.exception()
        try:
            if temp_dir is not None:
                temp_db = temp_dir / source.name
                temp_db.unlink(missing_ok=True)
                temp_dir.rmdir()
        except OSError as exc:
            if failure is not None:
                failure.add_note(f"legacy session migration cleanup failed: {exc}")
            else:
                raise StorageUnavailableError(f"legacy session migration cleanup failed: {exc}") from exc
        finally:
            _release_lease(lease)


def _copy_restore_material(source: Path, destination: Path, session_id: str) -> None:
    index = source / "points" / f"{session_id}.jsonl"
    if not index.is_file():
        return
    if index.is_symlink():
        return
    rows = index.read_bytes()
    # 索引以追加 JSONL 保存；所需 blob 名称为内容摘要，复制前核对内容。
    hashes: set[str] = set()
    for line in rows.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, TypeError, RecursionError):
            continue  # 原始索引仍复制；还原引擎按既有规则报告损坏。
        if not isinstance(record, dict) or not isinstance(record.get("entries"), list):
            continue
        for entry in record["entries"]:
            if not isinstance(entry, dict):
                continue
            for field in ("before_sha256", "after_sha256"):
                value = entry.get(field)
                if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                    hashes.add(value)
    (destination / "blobs").mkdir(parents=True, exist_ok=True)
    (destination / "points").mkdir(parents=True, exist_ok=True)
    for digest in hashes:
        blob = source / "blobs" / digest
        if not blob.is_file():
            continue  # 已过期材料保持原有缺失语义。
        if blob.is_symlink():
            continue
        target = destination / "blobs" / digest
        if not target.exists():
            staged = target.with_name(f".migrate-{uuid.uuid4().hex}")
            try:
                shutil.copy2(blob, staged)
                with staged.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                        continue
                os.replace(staged, target)
            finally:
                if staged.exists():
                    staged.unlink()
    target_index = destination / "points" / index.name
    # 目标 DB 尚未发布时，旧索引可能来自一次失败的迁移。持有源 lease
    # 重新复制本次索引，让重试使用的 DB 与还原点来自同一轮源档。
    staged_index = target_index.with_name(f".migrate-{uuid.uuid4().hex}")
    try:
        staged_index.write_bytes(rows)
        os.replace(staged_index, target_index)
    finally:
        if staged_index.exists():
            staged_index.unlink()
