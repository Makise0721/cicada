"""grep 工具验证: schema、字面量/大小写/context 语义、有界 content 与 bootstrap 多轮读取."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import MAX_CONTENT_BYTES
from cicada.plugins.coding.tool_grep import GrepTool, grep_plugin
from cicada.plugins.coding.workspace import Workspace


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "alpha.py").write_text(
        "def target_function():\n    return 'needle value'\n\n\nother = 1\n",
        encoding="utf-8",
    )
    (root / "src" / "beta.txt").write_text("NEEDLE upper\n", encoding="utf-8")
    (root / "notes.md").write_text("no hits here\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    return Workspace.create(root)


def tool_for(workspace: Workspace) -> GrepTool:
    return GrepTool(workspace, GitInventory(workspace))


async def run(tool: GrepTool, cancel=None, **arguments):
    return await tool.execute(
        dict(arguments), ToolContext(call_id="c1", cancel=cancel or CancelToken())
    )


def rows_of(result) -> list[dict]:
    return [
        json.loads(line)
        for line in result.content.splitlines()
        if line.startswith("{")
    ]


def test_spec_schema_matches_declared_parameters(workspace):
    spec = tool_for(workspace).spec
    assert spec.name == "grep"
    assert spec.parameters["required"] == ["pattern"]
    assert spec.parameters["additionalProperties"] is False
    assert spec.parameters["properties"]["context"]["maximum"] == 2
    assert spec.parameters["properties"]["limit"]["maximum"] == 500
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"pattern": "x", "offset": 1}, spec.parameters)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"pattern": "x", "context": 3}, spec.parameters)
    jsonschema.validate({"pattern": "x"}, spec.parameters)
    jsonschema.validate(
        {"pattern": "x", "path": "src", "include": "**/*.py", "ignore_case": True,
         "context": 2, "limit": 10},
        spec.parameters,
    )


async def test_grep_literal_path_and_line(workspace):
    result = await run(tool_for(workspace), pattern="needle")
    assert result.is_error is False
    assert result.details["shown"] == 1
    row = rows_of(result)[0]
    assert Path(row["path"]).is_file()
    assert row["path"].endswith("alpha.py")
    assert row["line"] == 2
    assert row["text"] == "    return 'needle value'"


async def test_grep_casefold_and_include(workspace):
    result = await run(
        tool_for(workspace), pattern="needle", include="**/*.txt", ignore_case=True
    )
    rows = rows_of(result)
    assert [Path(row["path"]).name for row in rows] == ["beta.txt"]
    assert rows[0]["text"] == "NEEDLE upper"
    # 同一 pattern 区分大小写时不命中
    strict = await run(tool_for(workspace), pattern="needle", include="**/*.txt")
    assert strict.details["shown"] == 0


async def test_grep_single_file_path_and_context_merge(workspace):
    result = await run(
        tool_for(workspace), pattern="needle", path="src/alpha.py", context=2
    )
    blocks = rows_of(result)
    assert len(blocks) == 1
    assert blocks[0]["lines"] == "1-4"
    assert blocks[0]["text"][1] == ">2:    return 'needle value'"
    assert blocks[0]["text"][3] == " 4:"
    assert result.details["shown"] == 1


async def test_grep_zero_matches_is_not_an_error(workspace):
    result = await run(tool_for(workspace), pattern="absent-token")
    assert result.is_error is False
    assert result.details["shown"] == 0
    assert result.details["complete"] is True
    assert "No matches found" in result.content


async def test_grep_truncation_reports_incomplete(workspace):
    (workspace.root / "many.txt").write_text(
        "".join(f"needle {index}\n" for index in range(50)), encoding="utf-8"
    )
    git(workspace.root, "add", "many.txt")
    result = await run(tool_for(workspace), pattern="needle", include="many.txt", limit=5)
    assert result.is_error is False
    assert result.details["shown"] == 5
    assert result.details["truncated"] is True
    assert result.details["complete"] is False
    assert "complete=false" in result.content


async def test_grep_binary_and_encoding_skips_are_disclosed(workspace):
    (workspace.root / "binary.bin").write_bytes(b"\x00needle\x01")
    (workspace.root / "latin.txt").write_bytes(b"needle \xe9\xe8\n")
    git(workspace.root, "add", ".")
    result = await run(tool_for(workspace), pattern="needle")
    assert result.details["skipped"]["non_text"] == 2
    assert "skipped: non_text=2" in result.content


async def test_grep_long_line_is_bounded_and_partial(workspace):
    (workspace.root / "long.txt").write_text(f"needle {'z' * 30000}\n", encoding="utf-8")
    git(workspace.root, "add", "long.txt")
    result = await run(tool_for(workspace), pattern="needle", include="long.txt")
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    row = rows_of(result)[0]
    assert row["partial_text"] is True
    assert row["line"] == 1
    assert len(row["text"].encode("utf-8")) <= 1024


async def test_grep_many_matches_stay_within_budget(workspace):
    (workspace.root / "many.txt").write_text(
        "".join(f"needle {index} {'q' * 80}\n" for index in range(400)), encoding="utf-8"
    )
    git(workspace.root, "add", "many.txt")
    result = await run(tool_for(workspace), pattern="needle", include="many.txt", limit=500)
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert result.details["truncated"] is True


async def test_grep_long_filenames_keep_footer_and_narrowing_hint(workspace):
    """F08: 报告反例 (60 个长文件名) 的公共 GrepTool 结果不再裁断 footer 字段.

    文件名长度取到"旧实现填满行后剩余空间小于截断 footer": 旧实现必然把 footer 裁断。
    """
    count, padding = 60, 80
    for _attempt in range(400):
        for existing in workspace.root.glob("f*-*.txt"):
            existing.unlink()
        for index in range(count):
            (workspace.root / f"f{index:03d}-{'x' * padding}.txt").write_text(
                "needle\n", encoding="utf-8"
            )
        git(workspace.root, "add", "-A", ".")
        record = await GitInventory(workspace).list_files(CancelToken())
        header_size = len(
            (
                '[grep pattern="needle" path="." include="**/*" ignore_case=false '
                f"context=0 limit=100 files={len(record.paths)} "
                'policy=cicada-git-files-v1 excluded=0]'
            ).encode("utf-8")
        )
        sample = str(workspace.root / f"f000-{'x' * padding}.txt")
        row_size = (
            len(json.dumps({"path": sample, "line": 1, "text": "needle"}).encode("utf-8")) + 1
        )
        if (8192 - header_size - 1) % row_size < 100:
            break
        padding += 1
    result = await run(tool_for(workspace), pattern="needle")
    content = result.content
    assert len(content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert result.details["shown"] > 0
    assert result.details["truncated"] is True
    assert result.details["complete"] is False
    lines = content.splitlines()
    assert re.fullmatch(
        r"\[shown=\d+ truncated=true complete=false reason=(bytes|limit)\]", lines[-2]
    )
    assert lines[-1] == "[results truncated; narrow path, include or pattern and query again]"
    for line in lines:
        if line.startswith("{"):
            json.loads(line)  # 整条合法 JSON: 字段与转义不被按字节裁断


@pytest.mark.parametrize(
    "arguments",
    [
        {"pattern": "a\nb"},
        {"pattern": "x" * 1200},
        {"pattern": "needle", "include": "a**b"},
        {"pattern": "needle", "path": "../../outside"},
        {"pattern": "needle", "path": "missing"},
    ],
)
async def test_grep_errors_are_explicit(workspace, arguments):
    result = await run(tool_for(workspace), **arguments)
    assert result.is_error is True
    assert result.details is None


async def test_grep_then_read_through_bootstrap_multiturn(tmp_path, monkeypatch):
    """最高公开 seam: 真实 Git fixture + bootstrap + FakeModel 多轮 search → read."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    from cicada.boot import bootstrap
    from cicada.core.ports import StreamDone, ToolCallEvent
    from cicada.plugins.coding.inventory import inventory_plugin
    from cicada.plugins.coding.tool_grep import grep_plugin
    from cicada.plugins.coding.tool_read import read_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "alpha.py").write_text(
        "line one\nneedle here\nline three\n", encoding="utf-8"
    )
    git(root, "init", "-q")
    git(root, "add", ".")

    model = FakeModel(
        [
            [
                ToolCallEvent("call_1", "grep", '{"pattern": "needle"}'),
                StreamDone("tool_use"),
            ],
            [
                ToolCallEvent(
                    "call_2", "read", json.dumps({"path": "src/alpha.py", "offset": 2, "limit": 1})
                ),
                StreamDone("tool_use"),
            ],
            [StreamDone("stop")],
        ]
    )
    app = await bootstrap(
        [
            workspace_plugin(root),
            inventory_plugin(),
            grep_plugin(),
            read_plugin(),
            fake_model_plugin(model),
        ],
        tool_capabilities=("tool.grep", "tool.read"),
    )
    try:
        outcome = await app.agent.run("locate the needle and read it")
        assert outcome.stop_reason == "stop"
        results = [
            message.result
            for message in outcome.messages
            if getattr(message, "result", None) is not None
        ]
        assert [result.name for result in results] == ["grep", "read"]
        grep_row = json.loads(
            next(
                line
                for line in results[0].content.splitlines()
                if line.startswith("{")
            )
        )
        assert grep_row["line"] == 2
        assert Path(grep_row["path"]).is_file()
        # grep 给出的路径+行号能直接续读同一行
        assert "needle here" in results[1].content
        assert "2\tneedle here" in results[1].content
        assert results[1].is_error is False
    finally:
        await app.aclose()
