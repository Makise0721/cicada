"""coding.search 引擎验证: 限定 glob 语法、Git 清单范围、字面量搜索与有界渲染."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import (
    MAX_CONTENT_BYTES,
    NO_MATCHES_TEXT,
    QUERY_TIMEOUT_S,
    SearchError,
    classify_candidate,
    glob_matches,
    parse_glob,
    resolve_scope,
    run_glob,
    run_grep,
    scope_files,
)
from cicada.plugins.coding.workspace import Workspace


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """真实 Git 工作区: 清单范围内的 tracked / untracked / 忽略 / 二进制 / 超大文件."""
    # fixture 在主仓库内, 显式禁止向父目录发现 Git, 保证 root 就是本 fixture
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = tmp_path / "repo"
    (root / "src" / "deep").mkdir(parents=True)
    (root / "src" / "alpha.py").write_text(
        "import os\nneedle here\nother\nneedle again\n", encoding="utf-8"
    )
    (root / "src" / "beta.txt").write_text("NEEDLE case\n", encoding="utf-8")
    (root / "src" / "deep" / "gamma.py").write_text("needle deep\n", encoding="utf-8")
    (root / "top.py").write_text("needle at top\n", encoding="utf-8")
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (root / "ignored.txt").write_text("needle ignored\n", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01needle")
    git(root, "init", "-q")
    git(root, "add", ".")
    (root / "untracked.py").write_text("needle untracked\n", encoding="utf-8")
    workspace = Workspace.create(root)
    return workspace, GitInventory(workspace)


def make_scope(workspace: Workspace, inventory: GitInventory, raw_path: str):
    record = asyncio.run(inventory.list_files(CancelToken()))
    return resolve_scope(workspace, record, raw_path), record


# --- 限定 glob 语法 ---


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("**/*.py", {"top.py", "src/alpha.py", "src/deep/gamma.py", "untracked.py"}),
        ("*.py", {"top.py", "untracked.py"}),
        ("src/*.py", {"src/alpha.py"}),
        ("src/**", {"src/alpha.py", "src/beta.txt", "src/deep/gamma.py"}),
        ("**", {"top.py", "untracked.py", "src/alpha.py", "src/beta.txt", "src/deep/gamma.py"}),
        ("**/deep/*.py", {"src/deep/gamma.py"}),
        ("src/**/*.py", {"src/alpha.py", "src/deep/gamma.py"}),
        ("src/**/**/*.py", {"src/deep/gamma.py"}),
        ("?op.py", {"top.py"}),
        ("*", {"top.py", "untracked.py"}),
    ],
)
def test_glob_syntax_matches_declared_scope(pattern, expected):
    matchers = parse_glob(pattern)
    universe = {
        "top.py",
        "untracked.py",
        "src/alpha.py",
        "src/beta.txt",
        "src/deep/gamma.py",
    }
    assert {path for path in universe if glob_matches(matchers, path)} == expected


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        "/abs/*.py",
        "a//b.py",
        "src/",
        "/src",
        "a**b",
        "***",
        "**a",
        "./a.py",
        "../a.py",
        "a/[bc].py",
        "a/{b,c}.py",
        "\n",
        "x" * 600,
    ],
)
def test_glob_rejects_invalid_patterns(pattern):
    with pytest.raises(SearchError) as caught:
        parse_glob(pattern)
    assert caught.value.kind == "invalid_pattern"


def test_glob_normalizes_windows_separators_and_casefold():
    matchers = parse_glob(r"SRC\*.PY")
    assert glob_matches(matchers, "src/alpha.py")
    assert not glob_matches(matchers, "src/deep/alpha.py")


# --- 范围解析 ---


def test_scope_directory_and_single_file(repo):
    workspace, inventory = repo
    scope, _ = make_scope(workspace, inventory, ".")
    assert scope.root_prefix == ""
    assert set(path for _, path in scope_files(scope)) == {
        ".gitignore",
        "binary.bin",
        "src/alpha.py",
        "src/beta.txt",
        "src/deep/gamma.py",
        "top.py",
        "untracked.py",
    }
    sub, _ = make_scope(workspace, inventory, "src")
    assert sub.root_prefix == "src/"
    # scope_files 第二项是 root 相对路径 (供文件系统与展示), 第一项是匹配路径
    assert set(path for _, path in scope_files(sub)) == {
        "src/alpha.py",
        "src/beta.txt",
        "src/deep/gamma.py",
    }
    assert set(match for match, _ in scope_files(sub)) == {
        "alpha.py",
        "beta.txt",
        "deep/gamma.py",
    }
    single, _ = make_scope(workspace, inventory, "src/alpha.py")
    assert single.root_prefix == "src/"
    assert list(scope_files(single)) == [("alpha.py", "src/alpha.py")]
    root_file, _ = make_scope(workspace, inventory, "top.py")
    assert list(scope_files(root_file)) == [("top.py", "top.py")]


@pytest.mark.parametrize(
    "raw_path,kind",
    [
        ("../outside", "invalid_path"),
        ("does-not-exist", "not_found"),
        ("ignored.txt", "not_in_inventory"),
    ],
)
def test_scope_errors_are_explicit(repo, raw_path, kind):
    workspace, inventory = repo
    with pytest.raises(SearchError) as caught:
        make_scope(workspace, inventory, raw_path)
    assert caught.value.kind == kind


# --- glob 查询 ---


async def test_glob_returns_readable_absolute_paths(repo):
    workspace, inventory = repo
    rendered, _ = await run_glob(
        workspace, inventory, pattern="**/*.py", raw_path=".", limit=200,
        cancel=CancelToken(),
    )
    assert rendered.truncated is False
    assert rendered.shown == 4
    body = [line for line in rendered.content.splitlines() if line.startswith('"')]
    paths = [json.loads(line) for line in body]
    assert set(paths) == {
        str(workspace.root / name)
        for name in ("top.py", "untracked.py", "src/alpha.py", "src/deep/gamma.py")
    }
    for path in paths:
        assert Path(path).is_file()
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES


async def test_glob_zero_matches_is_normal_result(repo):
    workspace, inventory = repo
    rendered, _ = await run_glob(
        workspace, inventory, pattern="*.rs", raw_path=".", limit=200, cancel=CancelToken()
    )
    assert rendered.shown == 0
    assert rendered.truncated is False
    assert NO_MATCHES_TEXT in rendered.content
    assert "[shown=0 truncated=false complete=true]" in rendered.content


async def test_glob_limit_marks_truncation_not_completeness(repo):
    workspace, inventory = repo
    rendered, _ = await run_glob(
        workspace, inventory, pattern="**/*", raw_path=".", limit=2, cancel=CancelToken()
    )
    assert rendered.shown == 2
    assert rendered.truncated is True
    assert rendered.reason == "limit"
    assert "complete=false" in rendered.content


async def test_glob_excludes_ignored_and_reports_scope(repo):
    workspace, inventory = repo
    rendered, _ = await run_glob(
        workspace, inventory, pattern="**/*", raw_path=".", limit=1000, cancel=CancelToken()
    )
    assert "ignored.txt" not in rendered.content
    assert "policy=cicada-git-files-v1" in rendered.content


# --- grep 查询 ---


async def test_grep_literal_lines_and_absolute_paths(repo):
    workspace, inventory = repo
    rendered, skipped = await run_grep(
        workspace, inventory, pattern="needle", raw_path=".", include="**/*",
        ignore_case=False, context=0, limit=100, cancel=CancelToken(),
    )
    assert rendered.truncated is False
    rows = [
        json.loads(line)
        for line in rendered.content.splitlines()
        if line.startswith("{")
    ]
    alpha = [row for row in rows if row["path"].endswith("alpha.py")]
    assert [(row["line"], row["text"]) for row in alpha] == [
        (2, "needle here"),
        (4, "needle again"),
    ]
    assert all(Path(row["path"]).is_file() for row in rows)
    assert skipped["non_text"] == 1  # binary.bin


async def test_grep_ignore_case_and_include(repo):
    workspace, inventory = repo
    rendered, _ = await run_grep(
        workspace, inventory, pattern="needle", raw_path=".", include="**/*.txt",
        ignore_case=True, context=0, limit=100, cancel=CancelToken(),
    )
    rows = [json.loads(line) for line in rendered.content.splitlines() if line.startswith("{")]
    assert [Path(row["path"]).name for row in rows] == ["beta.txt"]
    assert rows[0]["text"] == "NEEDLE case"


async def test_grep_context_merges_windows_and_counts_matched_lines(repo):
    workspace, inventory = repo
    rendered, _ = await run_grep(
        workspace, inventory, pattern="needle", raw_path="src/alpha.py", include="**/*",
        ignore_case=False, context=2, limit=100, cancel=CancelToken(),
    )
    blocks = [json.loads(line) for line in rendered.content.splitlines() if line.startswith("{")]
    assert len(blocks) == 1
    assert blocks[0]["lines"] == "1-4"
    assert blocks[0]["text"][0] == " 1:import os"
    assert blocks[0]["text"][1] == ">2:needle here"
    assert rendered.shown == 2  # shown 数匹配行, 不是上下文行


async def test_grep_zero_matches_is_normal_result(repo):
    workspace, inventory = repo
    rendered, _ = await run_grep(
        workspace, inventory, pattern="zzz-not-present", raw_path=".", include="**/*",
        ignore_case=False, context=0, limit=100, cancel=CancelToken(),
    )
    assert rendered.shown == 0
    assert rendered.truncated is False
    assert NO_MATCHES_TEXT in rendered.content


async def test_grep_limit_truncates(repo):
    workspace, inventory = repo
    rendered, _ = await run_grep(
        workspace, inventory, pattern="needle", raw_path=".", include="**/*",
        ignore_case=False, context=0, limit=2, cancel=CancelToken(),
    )
    assert rendered.shown == 2
    assert rendered.truncated is True
    assert "complete=false" in rendered.content


@pytest.mark.parametrize(
    "pattern,kind",
    [("", "invalid_pattern"), ("a\nb", "invalid_pattern"), ("x" * 1500, "invalid_pattern")],
)
async def test_grep_rejects_invalid_patterns(repo, pattern, kind):
    workspace, inventory = repo
    with pytest.raises(SearchError) as caught:
        await run_grep(
            workspace, inventory, pattern=pattern, raw_path=".", include="**/*",
            ignore_case=False, context=0, limit=10, cancel=CancelToken(),
        )
    assert caught.value.kind == kind


async def test_grep_include_pattern_is_validated(repo):
    workspace, inventory = repo
    with pytest.raises(SearchError) as caught:
        await run_grep(
            workspace, inventory, pattern="needle", raw_path=".", include="a**b",
            ignore_case=False, context=0, limit=10, cancel=CancelToken(),
        )
    assert caught.value.kind == "invalid_pattern"


# --- 有界输出 ---


async def test_grep_long_single_line_is_bounded_and_partial(repo):
    workspace, inventory = repo
    (workspace.root / "long.txt").write_text(f"needle {'x' * 20000}\n", encoding="utf-8")
    git(workspace.root, "add", "long.txt")
    rendered, _ = await run_grep(
        workspace, inventory, pattern="needle", raw_path=".", include="*.txt",
        ignore_case=False, context=0, limit=100, cancel=CancelToken(),
    )
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    rows = [json.loads(line) for line in rendered.content.splitlines() if line.startswith("{")]
    assert len(rows) == 1
    assert rows[0]["partial_text"] is True
    assert rows[0]["path"].endswith("long.txt")
    assert rows[0]["line"] == 1
    assert len(rows[0]["text"].encode("utf-8")) <= 1024


async def test_grep_many_matches_stay_within_budget(repo):
    workspace, inventory = repo
    (workspace.root / "many.txt").write_text(
        "".join(f"needle line {index} {'y' * 60}\n" for index in range(2000)), encoding="utf-8"
    )
    git(workspace.root, "add", "many.txt")
    rendered, _ = await run_grep(
        workspace, inventory, pattern="needle", raw_path=".", include="many.txt",
        ignore_case=False, context=0, limit=500, cancel=CancelToken(),
    )
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert rendered.truncated is True
    assert rendered.shown < 500 or rendered.reason == "bytes"


async def test_glob_many_paths_stay_within_budget(repo):
    workspace, inventory = repo
    for index in range(60):
        (workspace.root / f"generated-{index:03}-{'n' * 40}.py").write_text(
            "x\n", encoding="utf-8"
        )
    git(workspace.root, "add", ".")
    rendered, _ = await run_glob(
        workspace, inventory, pattern="generated-*.py", raw_path=".", limit=1000,
        cancel=CancelToken(),
    )
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert rendered.truncated is True
    assert rendered.reason == "bytes"


# --- 文件事实与失败路径 ---


def test_non_git_workspace_reports_git_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "plain.txt").write_text("needle\n", encoding="utf-8")
    workspace = Workspace.create(tmp_path)
    inventory = GitInventory(workspace)
    with pytest.raises(SearchError) as caught:
        asyncio.run(
            run_grep(
                workspace, inventory, pattern="needle", raw_path=".", include="**/*",
                ignore_case=False, context=0, limit=10, cancel=CancelToken(),
            )
        )
    assert caught.value.kind in {"git_failed", "git_unavailable"}


def test_classify_candidate_detects_junction_before_dereference(repo):
    workspace, inventory = repo
    outside = workspace.root.parent / "outside"
    outside.mkdir()
    (outside / "leak.py").write_text("needle leaked\n", encoding="utf-8")
    link = workspace.root / "jump"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True
    )
    if result.returncode:
        pytest.skip("junction creation unavailable on this host")
    assert classify_candidate(workspace.root, "jump") == "reparse"
    assert classify_candidate(workspace.root, "jump/leak.py") == "reparse"


def test_junction_candidate_is_skipped_not_followed(repo):
    workspace, inventory = repo
    outside = workspace.root.parent / "outside2"
    outside.mkdir()
    (outside / "leak.py").write_text("needle leaked\n", encoding="utf-8")
    link = workspace.root / "jump2"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True
    )
    if result.returncode:
        pytest.skip("junction creation unavailable on this host")
    git(workspace.root, "add", "-f", "jump2/leak.py")
    rendered, skipped = asyncio.run(
        run_grep(
            workspace, inventory, pattern="needle", raw_path=".", include="**/*",
            ignore_case=False, context=0, limit=100, cancel=CancelToken(),
        )
    )
    assert "leak" not in rendered.content
    assert skipped["reparse"] >= 1


def test_oversize_file_is_skipped_and_reported(repo):
    workspace, inventory = repo
    (workspace.root / "huge.txt").write_text(
        "needle\n" + "x" * (1024 * 1024 + 16), encoding="utf-8"
    )
    git(workspace.root, "add", "huge.txt")
    rendered, skipped = asyncio.run(
        run_grep(
            workspace, inventory, pattern="needle", raw_path=".", include="huge.txt",
            ignore_case=False, context=0, limit=100, cancel=CancelToken(),
        )
    )
    assert skipped["too_large"] == 1
    assert "skipped: too_large=1" in rendered.content


def test_reparse_ancestor_path_is_rejected(repo):
    workspace, inventory = repo
    outside = workspace.root.parent / "outside3"
    outside.mkdir()
    (outside / "inner").mkdir()
    link = workspace.root / "jump3"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True
    )
    if result.returncode:
        pytest.skip("junction creation unavailable on this host")
    # canonical 边界解析会因 junction 落在 root 之外而拒绝, 这不晚于 reparse 拒绝
    with pytest.raises(SearchError) as caught:
        make_scope(workspace, inventory, "jump3/inner")
    assert caught.value.kind in {"reparse_path", "invalid_path"}


@pytest.mark.parametrize("stop", ["token", "task"])
async def test_search_stops_on_cancellation(repo, stop):
    workspace, inventory = repo
    token = CancelToken()
    task = asyncio.create_task(
        run_grep(
            workspace, inventory, pattern="needle", raw_path=".", include="**/*",
            ignore_case=False, context=0, limit=500, cancel=token,
        )
    )
    if stop == "token":
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
