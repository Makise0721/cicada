"""coding.search 引擎验证: 限定 glob 语法、Git 清单范围、字面量搜索与有界渲染."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest

import cicada.plugins.coding.search as search_module
from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import (
    MAX_CONTENT_BYTES,
    MAX_FILE_BYTES,
    NO_MATCHES_TEXT,
    QUERY_TIMEOUT_S,
    SearchError,
    classify_candidate,
    glob_matches,
    parse_glob,
    read_text_file_chunked,
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
        # 每个独立 ** 都支持零段: 组合后与单个 ** 的可匹配集合一致 (F01)
        ("src/**/**/*.py", {"src/alpha.py", "src/deep/gamma.py"}),
        ("src/**/**/**/*.py", {"src/alpha.py", "src/deep/gamma.py"}),
        ("**/**/*.py", {"top.py", "src/alpha.py", "src/deep/gamma.py", "untracked.py"}),
        ("src/**/**", {"src/alpha.py", "src/beta.txt", "src/deep/gamma.py"}),
        ("**/**", {"top.py", "untracked.py", "src/alpha.py", "src/beta.txt", "src/deep/gamma.py"}),
        ("src/**/**/*", {"src/alpha.py", "src/beta.txt", "src/deep/gamma.py"}),
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


def test_glob_star_never_crosses_directory_boundary():
    matchers = parse_glob("src/*/*.py")
    assert glob_matches(matchers, "src/deep/gamma.py")
    assert not glob_matches(matchers, "src/gamma.py")
    assert not glob_matches(matchers, "src/a/b/c.py")


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


async def test_glob_zero_segment_combination_matches_via_public_tool(repo):
    """F01: 组合 `src/**/**/*.py` 的零段路径 `src/alpha.py` 必须被匹配."""
    workspace, inventory = repo
    rendered, _ = await run_glob(
        workspace, inventory, pattern="src/**/**/*.py", raw_path=".", limit=200,
        cancel=CancelToken(),
    )
    paths = {
        json.loads(line)
        for line in rendered.content.splitlines()
        if line.startswith('"')
    }
    assert str(workspace.root / "src" / "alpha.py") in paths
    assert str(workspace.root / "src" / "deep" / "gamma.py") in paths
    assert rendered.truncated is False


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


def render(header: str, rows: list[str], **overrides):
    arguments = dict(
        header=header,
        rows=rows,
        shown=len(rows),
        truncated=False,
        reason=None,
        skipped={},
        no_match_text=NO_MATCHES_TEXT,
    )
    arguments.update(overrides)
    return search_module._render(**arguments)


def test_render_reserves_the_truncated_footer_before_rows():
    """F08: 预算内放不下时 footer/收窄提示完整保留, 放不下的行整条丢弃.

    header 长度按"旧实现填满行后正好剩 0 字节给 footer"选取, 旧实现必然裁断 footer。
    """
    row = json.dumps({"path": "C:/x/" + "n" * 200 + ".py", "line": 1, "text": "needle"})
    row_size = len(row.encode("utf-8")) + 1
    header_size = MAX_CONTENT_BYTES - 1 - 32 * row_size
    header = "[grep " + "h" * (header_size - len("[grep ]")) + "]"
    assert len(header.encode("utf-8")) == header_size
    rendered = render(header, [row] * 40)
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert rendered.truncated is True
    assert rendered.shown == 31
    lines = rendered.content.splitlines()
    assert lines[-2] == "[shown=31 truncated=true complete=false reason=bytes]"
    assert lines[-1] == search_module._TRUNCATED_NOTE
    assert json.loads(lines[-3])["line"] == 1  # 保留的行仍是完整合法 JSON


def test_render_never_claims_zero_matches_when_rows_do_not_fit():
    header = "[grep " + "h" * 7950 + "]"
    row = json.dumps({"path": "C:/x/" + "n" * 200 + ".py", "line": 1, "text": "needle"})
    rendered = render(header, [row])
    assert len(rendered.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert rendered.truncated is True
    assert rendered.shown == 0
    assert NO_MATCHES_TEXT not in rendered.content
    assert search_module._TRUNCATED_NOTE in rendered.content


def test_render_reports_bounded_error_when_metadata_does_not_fit():
    with pytest.raises(SearchError) as caught:
        render("[" + "h" * (MAX_CONTENT_BYTES - 8) + "]", [])
    assert caught.value.kind == "metadata_too_large"


# --- F09: 内容读取的字节上限与截止时间 ---


class _GrowingFile:
    """预 stat 之后仍持续有数据的文件替身; 记录实际被读取的字节数."""

    def __init__(self, payload=b"needle\n", chunks=20, on_read=None):
        self.payload = payload
        self.chunks = chunks
        self.on_read = on_read
        self.calls = 0
        self.served = 0

    def read(self, size: int) -> bytes:
        if self.on_read is not None:
            self.on_read(self.calls)
        if self.calls >= self.chunks:
            return b""
        self.calls += 1
        self.served += size
        return self.payload + b"x" * max(size - len(self.payload), 0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def patch_open(monkeypatch, handle):
    monkeypatch.setattr(search_module, "open", lambda *args, **kwargs: handle, raising=False)
    return handle


def test_reader_caps_actual_bytes_for_growing_file(repo, monkeypatch):
    workspace, inventory = repo
    handle = patch_open(monkeypatch, _GrowingFile())
    rendered, skipped = asyncio.run(
        run_grep(
            workspace, inventory, pattern="needle", raw_path="src", include="alpha.py",
            ignore_case=False, context=0, limit=100, cancel=CancelToken(),
        )
    )
    assert handle.served <= MAX_FILE_BYTES + 1
    assert skipped["too_large"] == 1
    assert rendered.shown == 0
    assert "too_large=1" in rendered.content


def test_reader_stops_on_cancellation_per_chunk(repo, monkeypatch):
    """F09: 单文件读取途中取消必须由 reader 自己的分块检查发现."""
    workspace, inventory = repo
    token = CancelToken()
    patch_open(monkeypatch, _GrowingFile(on_read=lambda calls: token.cancel()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_grep(
                workspace, inventory, pattern="needle", raw_path="src/alpha.py",
                include="**/*", ignore_case=False, context=0, limit=100, cancel=token,
            )
        )


async def test_reader_stops_on_query_deadline_per_chunk(repo):
    workspace, _ = repo
    with pytest.raises(SearchError) as caught:
        await read_text_file_chunked(
            workspace.root / "src" / "alpha.py",
            cancel=CancelToken(),
            started=time.monotonic() - QUERY_TIMEOUT_S - 1,
        )
    assert caught.value.kind == "timeout"


async def test_reader_reads_a_file_at_the_byte_limit(repo):
    workspace, _ = repo
    exact = workspace.root / "exact.txt"
    exact.write_bytes(b"n" * MAX_FILE_BYTES)
    assert await read_text_file_chunked(
        exact, cancel=CancelToken(), started=time.monotonic()
    ) == "n" * MAX_FILE_BYTES
    over = workspace.root / "over.txt"
    over.write_bytes(b"n" * (MAX_FILE_BYTES + 1))
    with pytest.raises(search_module.FileTooLarge):
        await read_text_file_chunked(over, cancel=CancelToken(), started=time.monotonic())


async def test_search_deadline_is_rechecked_before_rendering(repo, monkeypatch):
    """F09: 单文件扫描期间越过 deadline 时不得把部分扫描渲染成完整结果."""
    workspace, inventory = repo
    real_read = search_module.read_text_file_chunked
    calls: list[int] = []

    async def slow_read(path, **kwargs):
        calls.append(1)
        time.sleep(1.2)
        return await real_read(path, **kwargs)

    monkeypatch.setattr(search_module, "read_text_file_chunked", slow_read)
    monkeypatch.setattr(search_module, "QUERY_TIMEOUT_S", 1.0)
    with pytest.raises(SearchError) as caught:
        await run_grep(
            workspace, inventory, pattern="needle", raw_path="src/alpha.py",
            include="**/*", ignore_case=False, context=0, limit=100, cancel=CancelToken(),
        )
    # scope 只有这一个文件且已经读过: 超时只能来自读取之后的渲染前检查
    assert calls == [1]
    assert caught.value.kind == "timeout"


async def test_queued_cancellation_runs_between_files(tmp_path, monkeypatch):
    """Spec R2: 清单返回时排队的同 loop 取消必须在扫描中真正执行并传播."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    for name in ("a.txt", "b.txt"):
        (root / name).write_text(
            ("ordinary line\n" * 5000) + "needle\n", encoding="utf-8"
        )
    git(root, "add", "a.txt", "b.txt")
    workspace = Workspace.create(root)

    class InventoryWithQueuedCancel(GitInventory):
        def __init__(self, ws):
            super().__init__(ws)
            self.queued = False

        async def list_files(self, cancel):
            record = await super().list_files(cancel)
            self.queued = True
            asyncio.get_running_loop().call_soon(cancel.cancel)
            return record

    inventory = InventoryWithQueuedCancel(workspace)
    token = CancelToken()
    with pytest.raises(asyncio.CancelledError):
        await run_grep(
            workspace, inventory, pattern="needle", raw_path=".", include="**/*",
            ignore_case=False, context=0, limit=100, cancel=token,
        )
    assert inventory.queued is True
    assert token.cancelled is True  # 排队的取消回调确实得到了执行机会


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
