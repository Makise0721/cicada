"""coding.file_completion 插件: 真实 Git 工作区候选、失败词表与关闭链。

全部离线 (项目内隔离仓库 + fake/monkeypatch), 零 live。
"""

import asyncio
import stat
import sys
from types import SimpleNamespace

import pytest

from cicada.core.cancel import CancelToken
from cicada.file_completion import SCOPE_DIRECTORY, SCOPE_GIT, SCOPE_UNAVAILABLE
from cicada.plugins.coding.file_completion import (
    FILE_COMPLETION_CAPABILITY,
    CodingFileCompletion,
    file_completion_plugin,
)
from cicada.plugins.coding.inventory import GitInventory, InventoryError
from cicada.plugins.coding.workspace import Workspace
from tests.support.plugins import git


def _repo(tmp_path):
    git(tmp_path, "init", "-q")
    for name in ("sample.py", "src/main.py", "src/util/helper.py", "docs/说明 文档.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return tmp_path


async def test_complete_lists_files_and_derived_directories(tmp_path):
    _repo(tmp_path)
    service = CodingFileCompletion(Workspace.create(tmp_path))
    result = await service.complete("", CancelToken())
    assert result.scope == SCOPE_GIT and result.reason is None
    assert result.truncated is False
    assert [c.relative_path for c in result.candidates][:3] == [
        "src/", "docs/", "sample.py",
    ]  # 空 query 基础分 0: 深度、路径长度升序
    paths = {c.relative_path: c.is_directory for c in result.candidates}
    assert paths["src/main.py"] is False
    assert paths["src/util/"] is True
    assert paths["docs/说明 文档.md"] is False  # 中文与空格是候选路径的一部分


async def test_complete_matches_name_prefix_and_directory_scope(tmp_path):
    _repo(tmp_path)
    service = CodingFileCompletion(Workspace.create(tmp_path))
    result = await service.complete("main", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["src/main.py"]
    result = await service.complete("src/mai", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["src/main.py"]
    result = await service.complete("SRC/mai", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["src/main.py"]
    result = await service.complete("./src/mai", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["src/main.py"]
    result = await service.complete("src/", CancelToken())
    # 空 fragment 基础分 0: 同深度按路径长度升序
    assert [c.relative_path for c in result.candidates][:2] == ["src/util/", "src/main.py"]
    assert "src/" not in {c.relative_path for c in result.candidates}  # 目录自身不是其后代
    result = await service.complete("docs/说", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["docs/说明 文档.md"]


async def test_complete_directory_part_must_exist_raw_level_by_level(tmp_path):
    _repo(tmp_path)
    service = CodingFileCompletion(Workspace.create(tmp_path))
    result = await service.complete("missing/x", CancelToken())
    assert result.scope == SCOPE_GIT and result.candidates == ()
    result = await service.complete("src/missing/y", CancelToken())
    assert result.scope == SCOPE_GIT and result.candidates == ()


async def test_complete_filters_deleted_reparse_and_gitlink(tmp_path):
    git(tmp_path, "init", "-q")
    oid = git(tmp_path, "hash-object", "-w", "--stdin", input=b"object").strip()
    records = b"".join(
        mode + b" " + oid + b" " + stage + b"\t" + path + b"\n"
        for mode, stage, path in (
            (b"160000", b"0", b"module"),
            (b"120000", b"0", b"link"),
            (b"100644", b"0", b"gone.py"),
        )
    )
    git(tmp_path, "update-index", "--index-info", input=records)
    (tmp_path / "kept.py").write_text("x", encoding="utf-8")
    service = CodingFileCompletion(Workspace.create(tmp_path))
    result = await service.complete("", CancelToken())
    paths = {c.relative_path for c in result.candidates}
    assert paths == {"kept.py"}  # 已删除/软链/gitlink 均不入候选, 也不派生目录


async def test_subworkspace_filters_uninitialized_gitlink_and_its_ancestors(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    oid = git(tmp_path, "hash-object", "-w", "--stdin", input=b"object").strip()
    workspace = tmp_path / "sub"
    (workspace / "module" / "child").mkdir(parents=True)
    (workspace / "d[1]" / "module").mkdir(parents=True)
    (workspace / "ordinary" / "child").mkdir(parents=True)
    (workspace / "ordinary" / "module").mkdir()
    (workspace / "Ä目录" / "module").mkdir(parents=True)
    (workspace / "ordinary" / "child" / "tracked.py").write_text("x", encoding="utf-8")
    (workspace / "ordinary" / "file.py").write_text("x", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "update-index", "--index-info", input=b"".join(
        b"160000 " + oid + b"\t" + name + b"\n"
        for name in (b"sub/module", b"sub/d[1]/module", b"sub/ordinary/module", "sub/ä目录/module".encode("utf-8"))
    ))
    # 外部 pathspec 环境不能把内部单层模式改成 literal 或递归匹配。
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    service = CodingFileCompletion(Workspace.create(workspace))
    result = await service.complete("", CancelToken())
    assert result.scope == SCOPE_DIRECTORY
    assert {c.relative_path for c in result.candidates} == {"d[1]/", "ordinary/", "Ä目录/"}
    for query in ("module/", "module/child/", "d[1]/", "ORDINARY/MODULE/", "ä目录/"):
        assert (await service.complete(query, CancelToken())).candidates == ()
    result = await service.complete("ordinary/", CancelToken())
    assert {c.relative_path for c in result.candidates} == {"ordinary/child/", "ordinary/file.py"}
    result = await service.complete("ORDINARY/", CancelToken())
    assert {c.relative_path for c in result.candidates} == {"ORDINARY/child/", "ORDINARY/file.py"}


async def test_complete_maps_failures_to_stable_reasons(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inside.py").write_text("x", encoding="utf-8")
    service = CodingFileCompletion(Workspace.create(sub))
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_DIRECTORY, None)
    assert [c.relative_path for c in result.candidates] == ["inside.py"]

    # 明确非 Git 目录进入局部模式，当前 workspace 之外的文件不会加入。
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    service = CodingFileCompletion(Workspace.create(plain))
    result = await service.complete("", CancelToken())
    assert result.scope == SCOPE_DIRECTORY and result.reason is None
    assert result.candidates == ()

    # 清单内部失败词表原样透传为稳定短原因
    class _Fail:
        def __init__(self, workspace, **kwargs) -> None:
            pass

        async def list_files(self, cancel):
            raise InventoryError("path_limit", "Git inventory exceeds paths")

    monkeypatch.setattr("cicada.plugins.coding.file_completion.GitInventory", _Fail)
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "path_limit")


async def test_local_navigation_exclusions_and_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = Workspace.create(tmp_path)
    (tmp_path / "空 目录").mkdir()
    (tmp_path / "空 目录" / "next").mkdir()
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / "venv").write_text("a regular file", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".git").mkdir()  # 空/损坏仓库标记也排除，不能穿入。
    module = tmp_path / "module"
    module.mkdir()
    (module / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    service = CodingFileCompletion(root)
    result = await service.complete("", CancelToken())
    assert result.scope == SCOPE_DIRECTORY
    assert {c.relative_path for c in result.candidates} == {
        "空 目录/", "ignored.txt", ".gitignore", "venv",
    }
    result = await service.complete("空 目录/", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["空 目录/next/"]
    assert (await service.complete("空 目录/next/", CancelToken())).candidates == ()
    for query in ("nested/", "module/", ".venv/", "missing/", "venv/"):
        assert (await service.complete(query, CancelToken())).candidates == ()
    (tmp_path / "fresh.py").write_text("x", encoding="utf-8")
    assert [c.relative_path for c in (await service.complete("fresh", CancelToken())).candidates] == ["fresh.py"]
    (tmp_path / "fresh.py").unlink()
    assert (await service.complete("fresh", CancelToken())).candidates == ()
    for index in range(25):
        (tmp_path / f"f{index:02}.py").write_text("x", encoding="utf-8")
    result = await service.complete("f", CancelToken())
    assert result.truncated and len(result.candidates) == 20
    assert [c.relative_path for c in result.candidates] == [f"f{index:02}.py" for index in range(20)]


async def test_local_reparse_is_not_followed_in_candidate_or_ancestor(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = Workspace.create(tmp_path)
    jump = tmp_path / "jump"
    jump.mkdir()
    (jump / "hidden.py").write_text("x", encoding="utf-8")
    original = type(jump).lstat

    def lstat(path, *args, **kwargs):
        if path == jump:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        if jump in path.parents:
            pytest.fail("reparse ancestor must not be followed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(jump), "lstat", lstat)
    service = CodingFileCompletion(root)
    assert (await service.complete("", CancelToken())).candidates == ()
    assert (await service.complete("jump/", CancelToken())).candidates == ()


async def test_local_iterator_limit_and_actual_release_on_cancel(tmp_path, monkeypatch):
    from cicada.plugins.coding import file_completion as completion

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = Workspace.create(tmp_path)
    (tmp_path / ".venv").mkdir()
    (tmp_path / "late.py").write_text("x", encoding="utf-8")

    class Iterator:
        def __init__(self, count):
            self.count, self.visited, self.closed = count, 0, False

        def __iter__(self):
            return self

        def __next__(self):
            if self.visited == self.count:
                raise StopIteration
            self.visited += 1
            path = tmp_path / (".venv" if self.visited <= 1024 else "late.py")
            return SimpleNamespace(path=str(path))

        def close(self):
            self.closed = True

    service = CodingFileCompletion(root)
    for count in (1024, 1026):
        iterator = Iterator(count)
        monkeypatch.setattr(completion.os, "scandir", lambda path: iterator)
        result = await service.complete("late", CancelToken())
        assert result.scope == SCOPE_DIRECTORY and result.candidates == ()
        assert result.truncated is (count > 1024)
        assert iterator.visited == min(count, 1025) and iterator.closed

    for stop in ("task", "token", "close"):
        iterator = Iterator(1026)
        service = CodingFileCompletion(root)
        token = CancelToken()
        monkeypatch.setattr(completion.os, "scandir", lambda path: iterator)
        original_type = completion._local_type

        def inspect(path):
            if iterator.visited == 32:
                if stop == "task":
                    asyncio.current_task().cancel()
                elif stop == "token":
                    token.cancel()
                else:
                    service.close()
            return original_type(path)

        with monkeypatch.context() as patch:
            patch.setattr(completion, "_local_type", inspect)
            with pytest.raises(asyncio.CancelledError):
                await service.complete("", token)
        await service.wait_settled()
        assert iterator.closed and iterator.visited < 1025

    iterator = Iterator(2)
    service = CodingFileCompletion(root)
    monkeypatch.setattr(completion.os, "scandir", lambda path: iterator)
    def unreadable(path):
        raise PermissionError("cannot inspect candidate")
    monkeypatch.setattr(completion, "_local_type", unreadable)
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "path_unreadable")
    assert iterator.closed


async def test_root_probe_classifies_only_controlled_not_git(tmp_path, monkeypatch):
    from cicada.plugins.coding.git_io import GitIOError, read_git

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    create_process = asyncio.create_subprocess_exec
    diagnostic = "fatal: not a git repository (or any of the parent directories): .git\n"
    cases = [
        (128, diagnostic, "", "not_git"),
        (1, diagnostic, "", "git_failed"),
        (128, "fatal: detected dubious ownership in repository\n", "", "git_failed"),
        (128, "warning: other failure\n" + diagnostic, "", "git_failed"),
        (128, "x" * 4096 + diagnostic, "", "git_failed"),
        (128, diagnostic, "unexpected", "git_failed"),
        (128, "fatal: permission denied\n", "", "git_failed"),
    ]
    for code, stderr, stdout, kind in cases:
        async def launch(*args, **kwargs):
            assert kwargs["env"]["LC_ALL"] == kwargs["env"]["LANGUAGE"] == "C"
            script = f"import sys; sys.stderr.write({stderr!r}); sys.stdout.write({stdout!r}); sys.exit({code})"
            return await create_process(sys.executable, "-c", script, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        with pytest.raises(GitIOError) as caught:
            await read_git(tmp_path, ("rev-parse", "--show-toplevel"), CancelToken(), 8192, classify_not_git=True)
        assert caught.value.kind == kind


async def test_corrupt_repository_and_git_failures_do_not_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    service = CodingFileCompletion(Workspace.create(tmp_path))
    (tmp_path / "would-leak.py").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()  # Git 给出普通 not_git 诊断，但元信息存在。
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "git_failed")

    class Fail:
        def __init__(self, workspace, **kwargs):
            pass

        async def list_files(self, token):
            raise InventoryError(kind, "private diagnostic")

    monkeypatch.setattr("cicada.plugins.coding.file_completion.GitInventory", Fail)
    for kind in ("git_unavailable", "git_failed", "path_unreadable", "invalid_inventory", "path_limit", "output_limit", "timeout"):
        result = await service.complete("", CancelToken())
        assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, kind)


async def test_directory_fallback_shares_probe_deadline(tmp_path, monkeypatch):
    from cicada.plugins.coding import file_completion as completion

    root = Workspace.create(tmp_path)
    reached = asyncio.Event()
    entered = asyncio.Event()
    class Probe:
        def __init__(self, workspace, **kwargs):
            pass

        async def list_files(self, token):
            await asyncio.sleep(0.12)
            reached.set()
            raise InventoryError("not_git", "controlled")

    async def directory(self, normalized, token, deadline, blocked):
        assert reached.is_set()
        entered.set()
        # 只有重置 deadline 才能完整等待；共享 deadline 应在此取消。
        await asyncio.sleep(0.12)
        pytest.fail("deadline was reset on fallback")

    monkeypatch.setattr(completion, "QUERY_DEADLINE_S", 0.2)
    monkeypatch.setattr(completion, "GitInventory", Probe)
    monkeypatch.setattr(completion.CodingFileCompletion, "_directory", directory)
    result = await CodingFileCompletion(root).complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "timeout")
    assert entered.is_set()


async def test_complete_rejects_unsupported_query_forms(tmp_path):
    service = CodingFileCompletion(Workspace.create(_repo(tmp_path)))
    for query in ("../x", "C:/x", "~user", "a\tb"):
        result = await service.complete(query, CancelToken())
        assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "unsupported_path"), query


async def test_complete_propagates_cancellation(tmp_path):
    service = CodingFileCompletion(Workspace.create(_repo(tmp_path)))
    token = CancelToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await service.complete("", token)


async def test_repeated_task_cancel_releases_completion_registration(tmp_path, monkeypatch):
    service = CodingFileCompletion(Workspace.create(tmp_path))
    started = asyncio.Event()

    async def blocked(query, token):
        started.set()
        await token.wait()
        token.throw_if_cancelled()

    monkeypatch.setattr(service, "_complete", blocked)
    task = asyncio.create_task(service.complete("", CancelToken()))
    await started.wait()
    task.cancel()
    asyncio.get_running_loop().call_soon(task.cancel)
    with pytest.raises(asyncio.CancelledError):
        await task
    service.close()
    await asyncio.wait_for(service.wait_settled(), 1)


async def test_inventory_cleanup_timeout_defaults_and_completion_uses_none(tmp_path, monkeypatch):
    workspace = Workspace.create(_repo(tmp_path))
    assert GitInventory(workspace).cleanup_timeout_s == 1.0  # 原调用默认行为不变
    with pytest.raises(ValueError):
        GitInventory(workspace, cleanup_timeout_s=0)
    with pytest.raises(ValueError):
        GitInventory(workspace, cleanup_timeout_s=-1.0)

    captured = {}

    real_read_git = __import__(
        "cicada.plugins.coding.git_io", fromlist=["read_git"]
    ).read_git

    async def spy_read_git(root, arguments, cancel, budget, **kwargs):
        captured["cleanup_timeout_s"] = kwargs.get("cleanup_timeout_s")
        return await real_read_git(root, arguments, cancel, budget, **kwargs)

    monkeypatch.setattr(
        "cicada.plugins.coding.inventory.read_git", spy_read_git
    )
    service = CodingFileCompletion(workspace)
    result = await service.complete("main", CancelToken())
    assert [c.relative_path for c in result.candidates] == ["src/main.py"]
    assert captured["cleanup_timeout_s"] is None  # 补全内部实例走实际等待回收分支


async def test_close_refuses_entry_cancels_active_and_waits_settled(tmp_path, monkeypatch):
    workspace = Workspace.create(_repo(tmp_path))
    started = asyncio.Event()

    class SlowInventory:
        def __init__(self, workspace, **kwargs) -> None:
            assert kwargs.get("cleanup_timeout_s") is None

        async def list_files(self, cancel):
            started.set()
            await cancel.wait()
            raise asyncio.CancelledError

    monkeypatch.setattr(
        "cicada.plugins.coding.file_completion.GitInventory", SlowInventory
    )
    service = CodingFileCompletion(workspace)
    task = asyncio.create_task(service.complete("", CancelToken()))
    await asyncio.wait_for(started.wait(), timeout=5)
    service.close()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    await asyncio.wait_for(service.wait_settled(), timeout=5)  # 实际落定后返回
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "session_closed")


def test_plugin_definition_declares_only_workspace_dependency():
    definition = file_completion_plugin()
    assert definition.provides == frozenset({FILE_COMPLETION_CAPABILITY})
    assert definition.requires == frozenset({"coding.workspace"})
