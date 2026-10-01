"""glob 工具验证: schema、参数拒绝、无分页承诺与公开 bootstrap seam."""

from __future__ import annotations

import subprocess
from pathlib import Path

import jsonschema
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.search import MAX_CONTENT_BYTES
from cicada.plugins.coding.tool_glob import GlobTool, glob_plugin
from cicada.plugins.coding.workspace import Workspace


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (root / "src" / "beta.txt").write_text("beta\n", encoding="utf-8")
    (root / "notes.md").write_text("# notes\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    return Workspace.create(root)


def tool_for(workspace: Workspace) -> GlobTool:
    return GlobTool(workspace, GitInventory(workspace))


async def run(tool: GlobTool, cancel=None, **arguments):
    return await tool.execute(
        dict(arguments), ToolContext(call_id="c1", cancel=cancel or CancelToken())
    )


def test_spec_schema_forbids_extra_arguments(workspace):
    spec = tool_for(workspace).spec
    assert spec.name == "glob"
    assert spec.parameters["required"] == ["pattern"]
    assert spec.parameters["additionalProperties"] is False
    assert spec.parameters["properties"]["limit"]["maximum"] == 1000
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"pattern": "*.py", "offset": 10}, spec.parameters)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"pattern": "*.py", "limit": 0}, spec.parameters)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"pattern": "*.py", "limit": 1001}, spec.parameters)
    jsonschema.validate({"pattern": "*.py"}, spec.parameters)
    jsonschema.validate({"pattern": "*.py", "path": "src", "limit": 5}, spec.parameters)


async def test_glob_returns_paths_that_read_accepts(workspace):
    result = await run(tool_for(workspace), pattern="**/*.py")
    assert result.is_error is False
    assert result.details["shown"] == 1
    assert result.details["truncated"] is False
    assert result.details["complete"] is True
    assert "alpha.py" in result.content
    # 结果路径是绝对路径, 可直接作为 read 的输入
    path = next(
        line.strip('"') for line in result.content.splitlines() if line.startswith('"')
    )
    assert Path(path).is_file()


async def test_glob_zero_matches_is_not_an_error(workspace):
    result = await run(tool_for(workspace), pattern="*.rs")
    assert result.is_error is False
    assert result.details["shown"] == 0
    assert "No matches found" in result.content


async def test_glob_truncation_is_reported(workspace):
    result = await run(tool_for(workspace), pattern="**/*", limit=1)
    assert result.is_error is False
    assert result.details["truncated"] is True
    assert result.details["complete"] is False
    assert "complete=false" in result.content


@pytest.mark.parametrize(
    "arguments",
    [
        {"pattern": "a**b"},
        {"pattern": "*/../*.py"},
        {"pattern": "*.py", "path": "../../etc"},
        {"pattern": "*.py", "path": "missing-dir"},
    ],
)
async def test_glob_errors_are_explicit(workspace, arguments):
    result = await run(tool_for(workspace), **arguments)
    assert result.is_error is True
    assert result.details is None


async def test_glob_content_is_bounded(workspace):
    for index in range(80):
        (workspace.root / f"many-{index:03}-{'n' * 40}.py").write_text("x\n", encoding="utf-8")
    git(workspace.root, "add", ".")
    result = await run(tool_for(workspace), pattern="many-*.py", limit=1000)
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert result.details["truncated"] is True
    assert result.details["truncation_reason"] == "bytes"


async def test_glob_plugin_resolves_at_bootstrap_and_is_used_by_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    from cicada.boot import bootstrap
    from cicada.core.ports import StreamDone, ToolCallEvent
    from cicada.plugins.coding.inventory import inventory_plugin
    from cicada.plugins.coding.workspace import workspace_plugin
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "target.py").write_text("target\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    model = FakeModel(
        [
            [
                ToolCallEvent("call_1", "glob", '{"pattern": "**/*.py"}'),
                StreamDone("tool_use"),
            ],
            [StreamDone("stop")],
        ]
    )
    app = await bootstrap(
        [
            workspace_plugin(root),
            inventory_plugin(),
            glob_plugin(),
            fake_model_plugin(model),
        ],
        tool_capabilities=("tool.glob",),
    )
    try:
        run_result = await app.agent.run("find the target file")
        assert run_result.stop_reason == "stop"
        results = [
            message.result
            for message in run_result.messages
            if getattr(message, "result", None) is not None
        ]
        assert len(results) == 1
        assert results[0].is_error is False
        assert "target.py" in results[0].content
        assert results[0].details["shown"] == 1
    finally:
        await app.aclose()
