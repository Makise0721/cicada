import asyncio
import subprocess
import sys

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory, InventoryError, inventory_plugin
from cicada.plugins.coding.workspace import Workspace


def git(root, *args, input=None):
    return subprocess.run(["git", "-C", str(root), *args], input=input, capture_output=True, check=True).stdout


async def test_git_inventory_lists_tracked_deleted_and_nonignored_files(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "deleted.py").write_text("old", encoding="utf-8")
    (tmp_path / "a file 中文.py").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    (tmp_path / "deleted.py").unlink()
    (tmp_path / "new.py").write_text("new", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("excluded", encoding="utf-8")
    inventory = await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    assert inventory.paths == (".gitignore", "a file 中文.py", "deleted.py", "new.py")
    by_path = {entry.relative_path: entry for entry in inventory.files}
    assert by_path["deleted.py"].exists is False
    assert by_path["a file 中文.py"].git_modes == ("100644",)
    assert by_path["new.py"].git_modes == ()


async def test_inventory_rejects_non_git_and_nested_root(tmp_path, monkeypatch):
    # fixture 位于主仓库内，显式禁止向父目录发现 Git，才是真正的非 Git 情形。
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(InventoryError, match="Git failed"):
        await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    git(tmp_path, "init", "-q")
    sub = tmp_path / "sub"
    sub.mkdir()
    with pytest.raises(InventoryError) as caught:
        await GitInventory(Workspace.create(sub)).list_files(CancelToken())
    assert caught.value.kind == "root_mismatch"


async def test_inventory_excludes_tracked_subtrees_but_not_same_named_file(tmp_path):
    git(tmp_path, "init", "-q")
    for name in ("src/A.py", "src/b.py", ".venv/hidden.py", "src/node_modules/hidden.py", "venv"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    git(tmp_path, "add", ".")
    inventory = await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    assert inventory.paths == ("src/A.py", "src/b.py", "venv")
    assert inventory.excluded_count == 2
    assert ".venv" in inventory.exclusions


async def test_inventory_preserves_gitlink_symlink_modes_and_merge_duplicates(tmp_path):
    git(tmp_path, "init", "-q")
    oid = git(tmp_path, "hash-object", "-w", "--stdin", input=b"object").strip()
    # 只操作本 fixture index，含三个未合并 stage；不 checkout 外部仓库或链接。
    records = b"".join(mode + b" " + oid + b" " + stage + b"\t" + path + b"\n" for mode, stage, path in (
        (b"160000", b"0", b"module"), (b"120000", b"0", b"link"),
        (b"100644", b"1", b"conflict.py"), (b"100644", b"2", b"conflict.py"),
        (b"100755", b"3", b"conflict.py")))
    git(tmp_path, "update-index", "--index-info", input=records)
    inventory = await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    assert inventory.paths == ("conflict.py", "link", "module")
    by_path = {e.relative_path: e for e in inventory.files}
    assert by_path["module"].is_submodule
    assert by_path["link"].is_reparse
    assert by_path["conflict.py"].git_modes == ("100644", "100755")


async def test_inventory_rejects_case_aliases_in_git_index(tmp_path):
    git(tmp_path, "init", "-q")
    oid = git(tmp_path, "hash-object", "-w", "--stdin", input=b"x").strip()
    git(tmp_path, "update-index", "--index-info", input=b"100644 " + oid + b"\tA.py\n100644 " + oid + b"\ta.py\n")
    with pytest.raises(InventoryError) as caught:
        await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    assert caught.value.kind == "alias_conflict"


@pytest.mark.parametrize("limit,kind", [({"max_paths": 1}, "path_limit"), ({"max_output_bytes": 400}, "output_limit")])
async def test_inventory_limits_fail_without_returning_partial_files(tmp_path, limit, kind):
    git(tmp_path, "init", "-q")
    for i in range(12):
        (tmp_path / f"file-{i:02}.py").write_text("x", encoding="utf-8")
    git(tmp_path, "add", ".")
    with pytest.raises(InventoryError) as caught:
        await GitInventory(Workspace.create(tmp_path), **limit).list_files(CancelToken())
    assert caught.value.kind == kind


async def test_inventory_records_junction_ancestor_before_following_it(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    target = tmp_path / "target"
    target.mkdir()
    (target / "outside.py").write_text("x", encoding="utf-8")
    link = root / "jump"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    if result.returncode:
        pytest.skip("junction creation unavailable on this host")
    oid = git(root, "hash-object", "-w", "--stdin", input=b"old").strip()
    git(root, "update-index", "--index-info", input=b"100644 " + oid + b"\tjump/outside.py\n")
    inventory = await GitInventory(Workspace.create(root)).list_files(CancelToken())
    entry = next(e for e in inventory.files if e.relative_path == "jump/outside.py")
    assert entry.reparse_paths == ("jump",)
    assert entry.is_reparse


@pytest.mark.parametrize("stop", ["deadline", "token", "task"])
async def test_inventory_cleans_spawned_process_on_deadline_or_cancellation(tmp_path, monkeypatch, stop):
    # OS adapter seam：真实 Python 子进程模拟不结束的 Git；不模拟清单/快照逻辑。
    create_process = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def launch(*args, **kwargs):
        process = await create_process(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    token = CancelToken()
    task = asyncio.create_task(GitInventory(Workspace.create(tmp_path), timeout_s=0.1 if stop == "deadline" else 5).list_files(token))
    await asyncio.wait_for(started.wait(), timeout=3)
    if stop == "token":
        token.cancel()
    elif stop == "task":
        task.cancel()
    with pytest.raises(InventoryError if stop == "deadline" else asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert processes[0].returncode is not None


async def test_inventory_plugin_is_resolved_at_real_bootstrap(tmp_path):
    from cicada.boot import bootstrap
    from cicada.core.ports import StreamDone
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.runtime.runtime import CapabilityError

    git(tmp_path, "init", "-q")
    (tmp_path / "new.py").write_text("x", encoding="utf-8")
    app = await bootstrap([inventory_plugin(), workspace_plugin(tmp_path), fake_model_plugin(FakeModel([[StreamDone("stop")]]))], tool_capabilities=())
    inventory = await app.runtime.capability("coding.inventory").list_files(CancelToken())
    assert inventory.paths == ("new.py",)
    assert (await app.agent.run("ordinary task")).stop_reason == "stop"
    await app.aclose()
    with pytest.raises(CapabilityError):
        app.runtime.capability("coding.inventory")


async def test_inventory_missing_git_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(InventoryError) as caught:
        await GitInventory(Workspace.create(tmp_path)).list_files(CancelToken())
    assert caught.value.kind == "git_unavailable"
