"""快照与范围账本的聚焦验证 (04 小步1: 内容身份、单调账本、稳定身份、取消).

用真实 Git/Windows fixture 覆盖公开行为: 不加辅助层, 断言 `capture()` 的可用性、
`Snapshot.entries` 与账本身份。
"""

import asyncio
import subprocess

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import DEFAULT_EXCLUSIONS, GitInventory
from cicada.plugins.coding.snapshot import (
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    PathScopeLedger,
    Snapshotter,
    snapshot_plugin,
)
from cicada.plugins.coding.workspace import Workspace

HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def git(root, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    return root


def make(repo, **limits):
    workspace = Workspace.create(repo)  # canonical root: 与清单同一路径事实
    return Snapshotter(workspace, GitInventory(workspace), **limits)


def entries_of(capture):
    assert capture.available, (capture.failure_kind, capture.error)
    return {entry.relative_path: entry for entry in capture.snapshot.entries}


async def test_snapshot_hashes_content_and_covers_tracked_and_untracked(repo):
    (repo / "tracked.txt").write_text("hello", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    (repo / "untracked.txt").write_text("hello", encoding="utf-8")
    capture = await make(repo).capture(CancelToken())
    entries = entries_of(capture)
    assert set(entries) == {"tracked.txt", "untracked.txt"}
    assert entries["tracked.txt"].sha256 == HELLO_SHA256
    assert entries["untracked.txt"].sha256 == HELLO_SHA256
    assert entries["tracked.txt"].exists and entries["tracked.txt"].size_bytes == 5
    assert capture.snapshot.policy_id and capture.snapshot.exclusions == DEFAULT_EXCLUSIONS


async def test_content_change_moves_snapshot_ref_but_not_scope_identity(repo):
    (repo / "a.txt").write_text("one", encoding="utf-8")
    git(repo, "add", "a.txt")
    first = await make(repo).capture(CancelToken())
    (repo / "a.txt").write_text("two", encoding="utf-8")
    second = await make(repo).capture(CancelToken())
    assert first.available and second.available
    assert first.snapshot.snapshot_ref != second.snapshot.snapshot_ref
    assert first.snapshot.scope_id == second.snapshot.scope_id


async def test_identical_content_yields_identical_encoding_and_identity(repo, tmp_path):
    (repo / "a.txt").write_text("stable", encoding="utf-8")
    git(repo, "add", "a.txt")
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.txt").write_text("stable", encoding="utf-8")
    git(other, "init", "-q")
    git(other, "add", "a.txt")
    first = await make(repo).capture(CancelToken())
    second = await make(other).capture(CancelToken())
    assert first.available and second.available
    assert first.snapshot.snapshot_ref == second.snapshot.snapshot_ref
    assert first.snapshot.entries == second.snapshot.entries


async def test_casefold_order_is_stable_across_independent_captures(repo, tmp_path):
    # Windows 目录名不区分大小写, 大小写事实由文件系统决定; 这里验证排序键稳定,
    # 大小写别名冲突由清单层 (test_inventory) 负责。
    for name in ("Alpha/one.py", "beta/two.py", "b.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    git(repo, "add", ".")
    first = await make(repo).capture(CancelToken())
    second = await make(repo).capture(CancelToken())
    paths = [entry.relative_path for entry in first.snapshot.entries]
    assert paths == sorted(paths, key=lambda path: (path.casefold(), path))
    assert len(paths) == 3
    assert paths == [entry.relative_path for entry in second.snapshot.entries]
    assert first.snapshot.snapshot_ref == second.snapshot.snapshot_ref


async def test_ledger_keeps_previously_observed_paths_after_deletion_and_ignore(repo):
    (repo / "gone.txt").write_text("x", encoding="utf-8")
    (repo / "stays.txt").write_text("y", encoding="utf-8")
    git(repo, "add", ".")
    ledger = PathScopeLedger()
    first = await make(repo, ledger=ledger).capture(CancelToken())
    assert first.available
    (repo / "gone.txt").unlink()
    git(repo, "rm", "--cached", "-q", "gone.txt")
    (repo / ".gitignore").write_text("gone.txt\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    second = await make(repo, ledger=ledger).capture(CancelToken())
    entries = entries_of(second)
    assert entries["gone.txt"].exists is False
    assert entries["gone.txt"].sha256 is None and entries["gone.txt"].size_bytes is None
    # 已见路径留在账本里: 它仍参与比较, 不能靠改 ignore 规则消失。
    assert ledger.paths == (".gitignore", "gone.txt", "stays.txt")
    without_gone = PathScopeLedger()
    without_gone.observe(frozenset({".gitignore", "stays.txt"}))
    assert without_gone.scope_id != second.snapshot.scope_id


async def test_ledger_accumulates_new_paths_and_scope_grows_monotonically(repo):
    (repo / "a.txt").write_text("a", encoding="utf-8")
    git(repo, "add", "a.txt")
    ledger = PathScopeLedger()
    first = await make(repo, ledger=ledger).capture(CancelToken())
    (repo / "new.txt").write_text("new", encoding="utf-8")
    second = await make(repo, ledger=ledger).capture(CancelToken())
    assert first.available and second.available
    assert first.snapshot.scope_id != second.snapshot.scope_id
    assert set(entries_of(second)) == {"a.txt", "new.txt"}
    assert second.snapshot.snapshot_ref != first.snapshot.snapshot_ref


async def test_excluded_tracked_files_stay_out_of_scope(repo):
    for name in (".venv/hidden.py", "src/kept.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    git(repo, "add", ".")
    capture = await make(repo).capture(CancelToken())
    entries = entries_of(capture)
    assert set(entries) == {"src/kept.py"}
    assert ".venv" in capture.snapshot.exclusions


async def test_gitlink_and_reparse_ancestor_make_capture_unavailable(repo):
    oid = git(repo, "hash-object", "-w", "--stdin", input=b"object").strip()
    git(repo, "update-index", "--index-info",
        input=b"160000 " + oid + b" 0\tmodule\n")
    capture = await make(repo).capture(CancelToken())
    assert not capture.available
    assert capture.failure_kind == "submodule"
    assert "module" in capture.error


async def test_reparse_ancestor_in_scope_is_unavailable(repo, tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "outside.py").write_text("x", encoding="utf-8")
    link = repo / "jump"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True)
    if result.returncode:
        pytest.skip("junction creation unavailable on this host")
    oid = git(repo, "hash-object", "-w", "--stdin", input=b"old").strip()
    git(repo, "update-index", "--index-info", input=b"100644 " + oid + b"\tjump/outside.py\n")
    capture = await make(repo).capture(CancelToken())
    assert not capture.available
    assert capture.failure_kind == "reparse"
    assert capture.error.endswith("jump")


@pytest.mark.parametrize(
    "limits,kind",
    [
        ({"max_file_bytes": 4}, "file_limit"),
        ({"max_total_bytes": 6}, "total_limit"),
        ({"max_paths": 1}, "path_limit"),
    ],
)
async def test_limits_make_capture_unavailable(repo, limits, kind):
    for name in ("a.txt", "b.txt"):
        (repo / name).write_text("hello", encoding="utf-8")
    git(repo, "add", ".")
    capture = await make(repo, **limits).capture(CancelToken())
    assert not capture.available
    assert capture.failure_kind == kind


async def test_deadline_and_cancellation_do_not_return_partial_snapshots(repo):
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    git(repo, "add", "a.txt")
    capture = await make(repo, deadline_s=0.001).capture(CancelToken())
    assert not capture.available
    assert capture.failure_kind == "timeout"
    assert capture.snapshot is None

    token = CancelToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await make(repo).capture(token)

    task = asyncio.create_task(make(repo).capture(CancelToken()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_scope_path_replaced_by_a_directory_is_unavailable(repo):
    # Git 仍把 a.txt 记为 tracked, 但磁盘上是目录: 真实读取故障, 不是 helper 假故障。
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    git(repo, "add", "a.txt")
    (repo / "a.txt").unlink()
    (repo / "a.txt").mkdir()
    capture = await make(repo).capture(CancelToken())
    assert not capture.available
    assert capture.failure_kind == "unreadable"
    assert "not a regular file" in capture.error


async def test_ledger_identity_is_stable_and_order_independent():
    first = PathScopeLedger()
    first.observe(frozenset({"b.py", "a.py"}))
    second = PathScopeLedger()
    second.observe(frozenset({"a.py"}))
    second.observe(frozenset({"b.py"}))
    assert first.scope_id == second.scope_id
    assert first.paths == ("a.py", "b.py")
    assert PathScopeLedger(policy_id="other-policy").scope_id != PathScopeLedger().scope_id


async def test_snapshot_plugin_resolves_at_real_bootstrap(repo):
    from cicada.boot import bootstrap
    from cicada.core.ports import StreamDone
    from cicada.plugins.coding.inventory import inventory_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    (repo / "a.txt").write_text("hello", encoding="utf-8")
    git(repo, "add", "a.txt")
    app = await bootstrap(
        [
            workspace_plugin(repo),
            inventory_plugin(),
            snapshot_plugin(),
            fake_model_plugin(FakeModel([[StreamDone("stop")]])),
        ],
        tool_capabilities=(),
    )
    snapshotter = app.runtime.capability("coding.snapshot")
    capture = await snapshotter.capture(CancelToken())
    assert capture.available
    assert [entry.relative_path for entry in capture.snapshot.entries] == ["a.txt"]
    await app.aclose()


def test_snapshot_constants_stay_within_the_published_ceiling():
    assert MAX_FILE_BYTES == 16 * 1024 * 1024
    assert MAX_TOTAL_BYTES == 128 * 1024 * 1024
