import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from cicada.plugins.coding.workspace import (
    PathNotAllowedError,
    Workspace,
    WorkspaceError,
)


def make_workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "ws")


def test_create_canonicalizes_and_creates_output_dir(tmp_path):
    ws = make_workspace(tmp_path)
    assert ws.root == tmp_path / "ws"
    assert ws.output_dir == ws.root / ".cicada" / "outputs"
    assert ws.output_dir.is_dir()


def test_error_hierarchy():
    assert issubclass(PathNotAllowedError, WorkspaceError)
    assert issubclass(WorkspaceError, RuntimeError)


def test_resolve_expands_home_relative_and_normalizes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    ws = make_workspace(tmp_path)
    assert ws.resolve("~\\notes\\a.txt") == Path(os.path.normpath(str(home / "notes" / "a.txt")))
    assert ws.resolve("a\\b.txt") == ws.root / "a" / "b.txt"
    assert ws.resolve("a/b.txt") == ws.root / "a" / "b.txt"
    assert ws.resolve("./x/../y") == ws.root / "y"
    absolute = tmp_path / "elsewhere" / ".." / "z.txt"
    assert ws.resolve(str(absolute)) == Path(os.path.normpath(str(absolute)))


def test_resolve_for_write_allows_inside_root(tmp_path):
    ws = make_workspace(tmp_path)
    (ws.root / "sub").mkdir()
    (ws.root / "sub" / "File.txt").write_text("x", encoding="utf-8")
    assert ws.resolve_for_write("sub\\File.txt") == ws.root / "sub" / "File.txt"
    # 大小写不同但同文件: Windows 语义判为在内
    assert ws.resolve_for_write("SUB\\file.TXT") == ws.root / "sub" / "File.txt"
    # 不存在的新文件, 父目录存在
    assert ws.resolve_for_write("sub\\new.txt") == ws.root / "sub" / "new.txt"


def test_resolve_for_write_rejects_outside(tmp_path):
    ws = make_workspace(tmp_path)
    with pytest.raises(PathNotAllowedError):
        ws.resolve_for_write("..\\outside.txt")
    with pytest.raises(PathNotAllowedError):
        ws.resolve_for_write(str(tmp_path / "elsewhere" / "f.txt"))
    # 父目录不存在且解析后位于 root 外的新路径
    with pytest.raises(PathNotAllowedError):
        ws.resolve_for_write("..\\no_such_dir\\new.txt")


def test_resolve_within_root_is_read_only_and_shares_write_boundary(tmp_path):
    ws = make_workspace(tmp_path)
    (ws.root / "sub").mkdir()
    existing = ws.root / "sub" / "File.txt"
    existing.write_text("unchanged", encoding="utf-8")
    assert ws.resolve_within_root("SUB\\file.TXT") == existing
    missing = ws.resolve_within_root("sub\\missing\\new.txt")
    assert missing == ws.resolve_for_write("sub\\missing\\new.txt")
    assert not missing.parent.exists()
    assert existing.read_text(encoding="utf-8") == "unchanged"
    with pytest.raises(PathNotAllowedError):
        ws.resolve_within_root("..\\outside.txt")


def test_junction_escape_rejected(tmp_path):
    ws = make_workspace(tmp_path)
    target = tmp_path / "real-target"
    target.mkdir()
    link = ws.root / "jump"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
    )
    if result.returncode != 0:
        pytest.skip("junction creation unavailable on this host")
    with pytest.raises(PathNotAllowedError):
        ws.resolve_for_write("jump\\evil.txt")


async def test_mutate_serializes_same_canonical_path(tmp_path):
    ws = make_workspace(tmp_path)
    path = ws.root / "a.txt"
    order = []

    async def op(tag, delay):
        order.append(f"{tag}-enter")
        await asyncio.sleep(delay)
        order.append(f"{tag}-exit")
        return tag

    results = await asyncio.gather(
        ws.mutate(path, lambda: op("one", 0.05)),
        ws.mutate(ws.root / "A.txt", lambda: op("two", 0)),  # 大小写不同 → 同一队列
    )
    assert results == ["one", "two"]
    assert order == ["one-enter", "one-exit", "two-enter", "two-exit"]


async def test_mutate_parallel_across_different_paths(tmp_path):
    ws = make_workspace(tmp_path)
    entered = []
    both = asyncio.Event()

    async def op(tag):
        entered.append(tag)
        if len(entered) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=2)
        return tag

    results = await asyncio.gather(
        ws.mutate(ws.root / "a.txt", lambda: op("a")),
        ws.mutate(ws.root / "b.txt", lambda: op("b")),
    )
    assert sorted(results) == ["a", "b"]


async def test_mutate_op_error_propagates_and_queue_released(tmp_path):
    ws = make_workspace(tmp_path)
    path = ws.root / "a.txt"
    ran_after = []

    async def boom():
        raise ValueError("op boom")

    with pytest.raises(ValueError, match="op boom"):
        await ws.mutate(path, boom)

    async def follow():
        ran_after.append("ok")
        return "ok"

    assert await ws.mutate(path, follow) == "ok"
    assert ran_after == ["ok"]


async def test_mutate_waiter_cancelled_gives_up_place(tmp_path):
    ws = make_workspace(tmp_path)
    path = ws.root / "a.txt"
    release = asyncio.Event()
    ran = []

    async def holder():
        await release.wait()
        ran.append("holder")
        return "holder"

    async def waiter():
        ran.append("waiter")
        return "waiter"

    t1 = asyncio.create_task(ws.mutate(path, holder))
    await asyncio.sleep(0.01)  # t1 持有队列
    t2 = asyncio.create_task(ws.mutate(path, waiter))
    await asyncio.sleep(0.01)  # t2 在队列中等待
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    assert ran == []  # 等待方被取消: 其 op 不执行, 在途 holder 尚未完成
    release.set()
    assert await t1 == "holder"
    assert await ws.mutate(path, waiter) == "waiter"
    assert ran == ["holder", "waiter"]


async def test_mutate_inflight_cancel_does_not_preempt(tmp_path):
    ws = make_workspace(tmp_path)
    path = ws.root / "a.txt"
    started = asyncio.Event()
    done = asyncio.Event()
    order = []

    async def long_op():
        started.set()
        await asyncio.sleep(0.05)
        done.set()
        order.append("op-done")
        return "value"

    t1 = asyncio.create_task(ws.mutate(path, long_op))
    await asyncio.wait_for(started.wait(), timeout=1)
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    # 在途 op 不被抢占, 完成后才释放队列
    await asyncio.wait_for(done.wait(), timeout=1)
    assert order == ["op-done"]

    async def follow():
        order.append("follow-enter")
        return "follow"

    assert await ws.mutate(path, follow) == "follow"
    assert order == ["op-done", "follow-enter"]


def test_new_output_file_unique_and_sanitized(tmp_path):
    ws = make_workspace(tmp_path)
    f1 = ws.new_output_file("run 1: 读文件")
    f2 = ws.new_output_file("run 1: 读文件")
    assert f1.parent == ws.output_dir
    assert f1 != f2
    f3 = ws.new_output_file('a/b\\c:d*e?')
    assert f3.parent == ws.output_dir
    assert not set('<>:"/\\|?*') & set(f3.name)
