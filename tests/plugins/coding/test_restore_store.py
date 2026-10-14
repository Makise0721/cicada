"""还原点捕获与存储 (request-snapshot-revert C2-C4) 的关键行为回归.

只覆盖探针确认过、且修一次会静默退化的高风险: 发布前落盘与 fail-closed 的
原因归属、索引行词表与聚合 (中断来源 / not_written 不算可还原条目)、保留用量与
回收、索引半行与收尾期写失败下的如实标注、无捕获汇时的兼容行为。不重复文件工具
的通用行为 (见 test_tool_write / test_tool_edit)。
"""

import hashlib
import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding import restore_store
from cicada.plugins.coding.restore_store import (
    AVAILABLE, EXPIRED, POINT_LIMIT_PER_SESSION, RestoreStore, restore_plugin,
)
from cicada.plugins.coding.tool_edit import EditTool
from cicada.plugins.coding.tool_read import ReadTool
from cicada.plugins.coding.tool_write import WriteTool
from cicada.plugins.coding.workspace import Workspace


def ctx(call_id="c1"):
    return ToolContext(call_id=call_id, cancel=CancelToken())


async def run(tool, **arguments):
    return await tool.execute(arguments, ctx())


def make(tmp_path, *, sink=True, name="ws"):
    ws = Workspace.create(tmp_path / name)
    store = RestoreStore(ws.root) if sink else None
    if store is not None:
        ws.install_restore_capture(store)
    return ws, store


def rows_of(ws, session_id):
    path = ws.root / ".cicada" / "restore" / "points" / f"{session_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def test_capture_writes_blobs_before_publication_and_settles(tmp_path, monkeypatch):
    """发布前落预行 (entry/unknown, 无 winerror), 发布后就地追加结算行。"""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch-1", "sess-1")
    publish = ws.files._publish_creation
    observed = []

    def verify_before_publish(*args):
        pre = rows_of(ws, "sess-1")
        assert len(pre) == 1 and pre[0]["commit_state"] == "unknown"
        sha = pre[0]["entries"][0]["after_sha256"]
        assert store.read_blob(sha) == b"hello\n"
        observed.append(True)
        return publish(*args)

    monkeypatch.setattr(ws.files, "_publish_creation", verify_before_publish)
    r = await run(WriteTool(ws), path="a.txt", content="hello\n")
    assert not r.is_error, r.content
    assert observed == [True]
    store.note_commit(3)
    store.end_request()

    rows = rows_of(ws, "sess-1")
    assert [row["row"] for row in rows] == ["entry", "settle", "settle"]
    pre, settle, note = rows
    assert pre["commit_state"] == "unknown" and "winerror" not in pre
    assert settle["commit_state"] == "committed" and settle["winerror"] is None
    assert (note["seq"], note["entries"], note["winerror"]) == (3, [], None)

    point = store.read_points("sess-1")[0]
    assert (point.seq, point.commit_state, point.source) == (3, "committed", "committed")
    assert point.availability == AVAILABLE
    entry = point.entries[0]
    assert entry.op == "create" and entry.before_sha256 is None
    blob = store.read_blob(entry.after_sha256)
    assert blob == b"hello\n"
    assert hashlib.sha256(blob).hexdigest() == entry.after_sha256


async def test_same_request_reuses_first_before_and_last_after(tmp_path):
    """同 (请求, 路径) 只留首份 before/op, after 聚合取最后一次。"""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch-1", "sess-1")
    await run(WriteTool(ws), path="a.txt", content="v1\n")
    await run(ReadTool(ws), path="a.txt")
    await run(EditTool(ws), path="a.txt", edits=[{"old_text": "v1", "new_text": "v2"}])
    await run(EditTool(ws), path="a.txt", edits=[{"old_text": "v2", "new_text": "v3"}])
    store.end_request()

    points = store.read_points("sess-1")
    assert len(points) == 1  # 一个请求一个点
    entries = points[0].entries
    assert len(entries) == 1
    assert entries[0].op == "create"  # 首份 op
    assert entries[0].before_sha256 is None  # create 的 before 为空
    assert store.read_blob(entries[0].after_sha256) == b"v3\n"  # after 取最后


async def test_capture_failure_refuses_edit_with_dedicated_reason(tmp_path):
    """blob 落盘失败 -> not_written/restore_capture_failed, 目标未动, details schema 不变。"""
    ws, store = make(tmp_path)
    blocker = ws.root / ".cicada" / "restore"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory\n", encoding="utf-8")
    store.begin_request("req-1", "epoch-1", "sess-1")
    r = await run(WriteTool(ws), path="a.txt", content="data\n")

    mutation = r.details["mutation"]
    assert r.is_error
    assert (mutation["state"], mutation["reason"]) == ("not_written", "restore_capture_failed")
    assert set(mutation) == {
        "version", "state", "reason", "canonical_path", "current_matches",
        "caller_cancelled", "winerror", "retained_paths", "cleanup_warning",
    }
    assert not (ws.root / "a.txt").exists()


async def test_missing_context_inside_request_is_refused(tmp_path):
    """请求内缺失上下文属编程错误: 拒绝并诊断, 不静默降级为无捕获改动。"""
    ws, store = make(tmp_path)
    (ws.root / "a.txt").write_text("keep\n", encoding="utf-8")
    await run(ReadTool(ws), path="a.txt")
    r = await run(EditTool(ws), path="a.txt", edits=[{"old_text": "keep", "new_text": "changed"}])
    assert r.is_error
    assert r.details["mutation"]["reason"] == "restore_capture_failed"
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "keep\n"


async def test_interrupted_request_is_marked_and_still_restorable(tmp_path):
    """裁决 4: 中断收尾的点标 source=interrupted, 已捕获部分仍可还原。"""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch-1", "sess-1")
    await run(WriteTool(ws), path="a.txt", content="partial\n")
    store.end_request(interrupted=True)

    point = store.read_points("sess-1")[0]
    assert point.source == "interrupted"
    assert len(point.entries) == 1
    assert point.entries[0].before_sha256 is None


async def test_not_written_settle_row_drops_its_own_entry_only(tmp_path):
    """not_written 结算行不代表可还原条目; 同路径此前已生效的改动必须保留。"""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch-1", "sess-1")
    applied = store.capture(path="x.txt", op="modify", before=b"a\n", after=b"b\n")
    store.settle(applied, commit_state="committed")
    failed = store.capture(path="x.txt", op="modify", before=b"a\n", after=b"c\n")
    store.settle(failed, commit_state="not_written", winerror=5)
    untouched = store.capture(path="y.txt", op="create", before=None, after=b"y\n")
    store.settle(untouched, commit_state="not_written", winerror=5)
    store.end_request()

    point = store.read_points("sess-1")[0]
    assert [entry.path for entry in point.entries] == ["x.txt"]  # y 未生效 -> 不在
    assert store.read_blob(point.entries[0].after_sha256) == b"b\n"  # 回退到已生效那份


async def test_no_capture_sink_leaves_file_tools_unchanged(tmp_path):
    """非持久/关闭捕获: 无捕获汇时行为与既有逐位一致, 不产生 restore 目录。"""
    ws, _ = make(tmp_path, sink=False)
    r = await run(WriteTool(ws), path="a.txt", content="x\n")
    assert not r.is_error, r.content
    (ws.root / "outside.txt").write_text("o\n", encoding="utf-8")
    refused = await run(EditTool(ws), path="outside.txt",
                        edits=[{"old_text": "o", "new_text": "p"}])
    assert refused.details["mutation"]["reason"] == "read_required"
    await run(ReadTool(ws), path="outside.txt")
    edited = await run(EditTool(ws), path="outside.txt",
                       edits=[{"old_text": "o", "new_text": "p"}])
    assert not edited.is_error, edited.content
    assert not (ws.root / ".cicada" / "restore").exists()


async def test_retention_evicts_oldest_and_marks_expired(tmp_path):
    """C3: 每会话保留最近 100 点, 逐出最旧; 逐出点标 expired 且 blob 被回收。"""
    ws, store = make(tmp_path)
    for index in range(POINT_LIMIT_PER_SESSION + 2):
        store.begin_request(f"req-{index}", "epoch", "sess-1")
        await run(WriteTool(ws), path=f"f{index}.txt", content=f"body-{index}\n")
        store.end_request()

    assert store.maintenance() == ()
    points = store.read_points("sess-1")
    assert len(points) == POINT_LIMIT_PER_SESSION + 2  # 逐出不删索引行
    assert [p.availability for p in points].count(EXPIRED) == 2
    assert points[0].availability == AVAILABLE
    blobs = [p for p in (ws.root / ".cicada" / "restore" / "blobs").glob("*")
             if not p.name.startswith(".tmp")]
    assert len(blobs) <= POINT_LIMIT_PER_SESSION + 1


async def test_store_rejects_overlap_and_escaped_session_id(tmp_path):
    """上下文不可重叠; session id 校验拒绝路径穿越。"""
    _, store = make(tmp_path)
    store.begin_request("req-1", "epoch-1", "sess-1")
    try:
        store.begin_request("req-2", "epoch-1", "sess-1")
    except Exception as exc:
        assert "already open" in str(exc)
    else:  # pragma: no cover - 只在回归时命中
        raise AssertionError("overlapping requests must be refused")
    store.end_request()
    for bad in ("../evil", "a/b", "", "a b"):
        try:
            store.read_points(bad)
        except Exception:
            continue
        raise AssertionError(f"session id {bad!r} must be refused")


async def test_retention_keeps_blobs_when_an_index_is_unreadable(tmp_path):
    """读不出的索引不等于没有引用: 此刻回收 blob 会删掉仍被引用的内容。"""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch", "sess-1")
    await run(WriteTool(ws), path="a.txt", content="abc\n")
    store.end_request()
    blobs = ws.root / ".cicada" / "restore" / "blobs"
    before = sorted(p.name for p in blobs.iterdir())
    assert len(before) == 1

    (ws.root / ".cicada" / "restore" / "points" / "broken.jsonl").mkdir()  # 读不出
    warnings = store.maintenance()

    assert warnings  # 如实告警
    assert sorted(p.name for p in blobs.iterdir()) == before  # 一个都没被回收
    assert store.read_points("sess-1")[0].availability == AVAILABLE


def test_restore_plugin_declares_workspace_dependency(tmp_path):
    """插件契约: 提供 coding.restore, 依赖 coding.workspace。"""
    definition = restore_plugin()
    assert definition.provides == frozenset({"coding.restore"})
    assert definition.requires == frozenset({"coding.workspace"})


@pytest.mark.parametrize("failed_sync", [1, 2])
async def test_capture_fsync_failure_refuses_publication(tmp_path, monkeypatch, failed_sync):
    """blob 与预行各自的 fsync 失败都不能发布目标，保持专属原因。"""
    ws, store = make(tmp_path)
    store._ensure_own_dirs()
    store.begin_request("req", "epoch", "sess")
    fsync = os.fsync
    calls = 0

    def fail(fd):
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError("injected fsync failure")
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", fail)
    result = await run(WriteTool(ws), path="a.txt", content="hello")
    assert result.details["mutation"]["reason"] == "restore_capture_failed"
    assert result.details["mutation"]["state"] == "not_written"
    assert not (ws.root / "a.txt").exists()


async def test_shared_blob_eviction_and_age_expiry_keep_history(tmp_path, monkeypatch):
    ws, store = make(tmp_path)
    monkeypatch.setattr(restore_store, "POINT_LIMIT_PER_SESSION", 2)
    for index in range(3):
        store.begin_request(f"r{index}", "epoch", "sess")
        ticket = store.capture(path=f"{index}.txt", op="create", before=None, after=b"shared")
        store.settle(ticket, commit_state="committed")
        store.note_commit(index + 1)
        store.end_request()
    points = store.read_points("sess")
    assert len(points) == 3 and points[-1].availability == EXPIRED
    assert points[0].availability == AVAILABLE
    # 超期只落状态标记，保留所有点身份；共享 blob 不影响旧点 expired。
    index_path = ws.root / ".cicada/restore/points/sess.jsonl"
    rows = rows_of(ws, "sess")
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    for row in rows:
        row["created_at"] = old
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert store.maintenance() == ()
    assert len(store.read_points("sess")) == 3
    assert all(p.availability == EXPIRED for p in store.read_points("sess"))


def test_incomplete_limit_and_uncommitted_source(tmp_path, monkeypatch):
    _, store = make(tmp_path)
    monkeypatch.setattr(restore_store, "MAX_ENTRIES_PER_POINT", 1)
    store.begin_request("r", "e", "s")
    first = store.capture(path="A.txt", op="create", before=None, after=b"first")
    store.settle(first, commit_state="committed")
    # 文件已写成功不足以证明请求完整；重启也保留 unknown。
    point = RestoreStore(store._root).read_points("s")[0]
    assert point.source == "unknown" and point.seq is None
    with pytest.raises(restore_store.RestoreStoreError, match="entry limit"):
        store.capture(path="other.txt", op="create", before=None, after=b"other")
    again = store.capture(path="a.txt", op="modify", before=b"first", after=b"last")
    store.settle(again, commit_state="committed")
    store.note_commit(1)
    store.end_request()
    point = store.read_points("s")[0]
    assert point.availability == restore_store.INCOMPLETE and len(point.entries) == 1
    assert point.entries[0].before_sha256 is None
    assert store.read_blob(point.entries[0].after_sha256) == b"last"


def test_blob_integrity_and_malformed_index_preserve_materials(tmp_path):
    ws, store = make(tmp_path)
    store.begin_request("r", "e", "s")
    ticket = store.capture(path="a", op="create", before=None, after=b"bytes")
    store.settle(ticket, commit_state="committed")
    store.note_commit(1)
    store.end_request()
    blob = ws.root / ".cicada/restore/blobs" / ticket.after_sha256
    blob.write_bytes(b"wrong")
    assert store.read_points("s")[0].availability == EXPIRED
    index = ws.root / ".cicada/restore/points/s.jsonl"
    with index.open("ab") as handle:
        handle.write(b'{"truncated":\n')
    assert store.maintenance()
    assert blob.read_bytes() == b"wrong"  # 索引不可读时放弃回收


async def test_index_tail_without_newline_does_not_swallow_next_capture(tmp_path):
    """崩溃留下缺尾换行的半行时, 下一次捕获的预行仍必须可读 (C2 的落盘证据)."""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch", "sess-1")
    await run(WriteTool(ws), path="keep.txt", content="keep\n")
    store.end_request()
    index = ws.root / ".cicada" / "restore" / "points" / "sess-1.jsonl"
    index.write_bytes(index.read_bytes().rstrip(b"\n"))  # 模拟半行收尾

    store.begin_request("req-2", "epoch", "sess-1")
    ticket = store.capture(path="new.txt", op="create", before=None, after=b"fresh\n")
    entries = {p.point_id: [e.path for e in p.entries] for p in store.read_points("sess-1")}
    assert entries[ticket.point_id] == ["new.txt"]  # 预行未被半行吞掉
    store.settle(ticket, commit_state="committed")
    store.end_request()
    assert {p.point_id: [e.path for e in p.entries] for p in store.read_points("sess-1")}[
        ticket.point_id] == ["new.txt"]


async def test_interrupted_source_holds_even_when_marker_append_fails(tmp_path):
    """收尾标记写失败也不得把中断点读成 committed (裁决 4 如实标注)."""
    ws, store = make(tmp_path)
    store.begin_request("req-1", "epoch", "sess-1")
    await run(WriteTool(ws), path="a.txt", content="one\n")
    index = ws.root / ".cicada" / "restore" / "points" / "sess-1.jsonl"
    os.chmod(index, stat.S_IREAD)
    try:
        store.end_request(interrupted=True)  # 不得抛出
        point = store.read_points("sess-1")[0]
        assert point.source == "interrupted"
        assert store.read_blob(point.entries[0].after_sha256) == b"one\n"
    finally:
        os.chmod(index, stat.S_IWRITE | stat.S_IREAD)

    store.begin_request("req-2", "epoch", "sess-1")  # 下次成功写入补记挂起标记
    await run(WriteTool(ws), path="b.txt", content="two\n")
    store.end_request()
    assert store._pending_markers == {}
    sources = [p.source for p in store.read_points("sess-1")]
    assert sources.count("interrupted") == 1


async def test_retention_counts_shared_blobs_once(tmp_path):
    """C3 约束 blob 总量: 被多点共享的同一 blob 只计一次, 不因重复计费误逐出。"""
    ws, store = make(tmp_path)
    root = ws.root / ".cicada" / "restore"
    payload = b"shared\n"
    for index in range(4):
        store.begin_request(f"req-{index}", "epoch", "sess-1")
        ticket = store.capture(path="a.txt", op="modify", before=payload, after=payload)
        store.settle(ticket, commit_state="committed")
        store.end_request()
    points = store.read_points("sess-1")
    assert len(points) == 4
    # 索引层每点都算这份 blob, 聚合总量是其倍数——这正是旧实现误逐出的来源。
    assert sum(p.total_bytes for p in points) == 4 * len(payload)

    limit = restore_store.BLOB_TOTAL_LIMIT_BYTES
    restore_store.BLOB_TOTAL_LIMIT_BYTES = 2 * len(payload)  # 高于实际占用
    try:
        assert store.maintenance() == ()
    finally:
        restore_store.BLOB_TOTAL_LIMIT_BYTES = limit
    assert [p.availability for p in store.read_points("sess-1")] == [AVAILABLE] * 4
    assert len(list((root / "blobs").glob("*"))) >= 1  # 共享 blob 未被回收


def test_point_label_agrees_with_its_entries(tmp_path):
    """同点内一成一败时, 点级标签不得与它暴露的条目相反。"""
    store = RestoreStore(tmp_path)
    blobs = tmp_path / ".cicada" / "restore" / "blobs"
    blobs.mkdir(parents=True)
    after = restore_store._sha256(b"a2\n")
    (blobs / after).write_bytes(b"a2\n")
    context = restore_store._RequestContext(
        request_id="req-1", epoch_id="epoch", session_id="sess-1")

    def row(commit_state, path, winerror):
        return restore_store._index_row(
            context, row=restore_store.ROW_SETTLE, point_id="rp-mixed",
            kind=restore_store.KIND_REQUEST, commit_state=commit_state,
            source="committed" if commit_state == "committed" else "unknown",
            entries=(restore_store.StoredEntry(
                path=path, op=restore_store.OP_MODIFY,
                before_sha256=restore_store._sha256(b"a1\n"),
                after_sha256=after, size_bytes=3),),
            seq=None, winerror=winerror)

    points_dir = tmp_path / ".cicada" / "restore" / "points"
    points_dir.mkdir(parents=True)
    index = points_dir / "sess-1.jsonl"
    index.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n"
                for r in (row("committed", "a.txt", None),
                          row("not_written", "b.txt", 183))),
        encoding="utf-8")

    point = store.read_points("sess-1")[0]
    assert [e.path for e in point.entries] == ["a.txt"]
    assert (point.commit_state, point.source, point.winerror) == ("committed", "committed", None)
