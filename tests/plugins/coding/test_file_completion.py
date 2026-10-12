"""coding.file_completion 插件: 真实 Git 工作区候选、失败词表与关闭链。

全部离线 (项目内隔离仓库 + fake/monkeypatch), 零 live。
"""

import asyncio

import pytest

from cicada.core.cancel import CancelToken
from cicada.file_completion import SCOPE_GIT, SCOPE_UNAVAILABLE
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


async def test_complete_maps_failures_to_stable_reasons(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    sub = tmp_path / "sub"
    sub.mkdir()
    service = CodingFileCompletion(Workspace.create(sub))
    result = await service.complete("", CancelToken())
    assert (result.scope, result.reason) == (SCOPE_UNAVAILABLE, "root_mismatch")
    assert result.candidates == ()

    # 非 Git 目录: 探针失败原样返回短原因, 不做局部回退 (票 03 才有受控 not_git 分类)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    service = CodingFileCompletion(Workspace.create(plain))
    result = await service.complete("", CancelToken())
    assert result.scope == SCOPE_UNAVAILABLE
    assert result.reason in ("git_failed", "git_unavailable")
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
