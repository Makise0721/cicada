"""跨层 e2e: fake model 剧本经 boot 驱动真实 coding 工具; 另覆盖 CLI 入口."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cicada.boot import bootstrap
from cicada.core.messages import ToolResultMessage
from cicada.core.ports import StreamDone, TextDelta, ToolCallEvent
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.runtime.runtime import CapabilityError

TOOL_CAPABILITIES = ("tool.read", "tool.edit", "tool.write", "tool.powershell")

SRC = Path(__file__).resolve().parents[1] / "src"


def coding_definitions(workspace: Path, model: FakeModel):
    return [
        workspace_plugin(workspace),
        process_plugin(),
        read_plugin(),
        edit_plugin(),
        write_plugin(),
        powershell_plugin(),
        fake_model_plugin(model),
    ]


async def test_e2e_read_edit_write_powershell(tmp_path):
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    model = FakeModel(
        [
            [ToolCallEvent("c1", "read", '{"path": "hello.txt"}'), StreamDone("tool_use")],
            [
                ToolCallEvent(
                    "c2",
                    "edit",
                    json.dumps(
                        {"path": "hello.txt", "edits": [{"old_text": "line2", "new_text": "LINE2"}]}
                    ),
                ),
                StreamDone("tool_use"),
            ],
            [
                ToolCallEvent(
                    "c3", "write", json.dumps({"path": "notes/result.txt", "content": "完成"})
                ),
                StreamDone("tool_use"),
            ],
            [
                ToolCallEvent("c4", "powershell", json.dumps({"command": "Get-Content hello.txt"})),
                StreamDone("tool_use"),
            ],
            [TextDelta("完成"), StreamDone("stop")],
        ]
    )
    app = await bootstrap(coding_definitions(tmp_path, model), tool_capabilities=TOOL_CAPABILITIES)
    result = await app.agent.run("改文件并验证")
    assert result.stop_reason == "stop"

    tool_msgs = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert [m.result.call_id for m in tool_msgs] == ["c1", "c2", "c3", "c4"]
    assert all(not m.result.is_error for m in tool_msgs)
    assert "1\tline1" in tool_msgs[0].result.content
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8").splitlines()[1] == "LINE2"
    assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "完成"
    assert "LINE2" in tool_msgs[3].result.content
    assert tool_msgs[3].result.details["exit_code"] == 0

    await app.aclose()
    with pytest.raises(CapabilityError):
        app.runtime.capability("tool.read")


async def test_e2e_write_outside_workspace_rejected(tmp_path):
    model = FakeModel(
        [
            [
                ToolCallEvent("c1", "write", json.dumps({"path": "../evil.txt", "content": "x"})),
                StreamDone("tool_use"),
            ],
            [TextDelta("ok"), StreamDone("stop")],
        ]
    )
    app = await bootstrap(coding_definitions(tmp_path, model), tool_capabilities=TOOL_CAPABILITIES)
    result = await app.agent.run("越界写入")
    tool_msgs = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert tool_msgs[0].result.is_error
    assert not (tmp_path.parent / "evil.txt").exists()
    await app.aclose()


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [sys.executable, "-m", "cicada", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
    )


def test_cli_with_script_runs_tools(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            [
                {"tool_calls": [{"id": "c1", "name": "read", "arguments": {"path": "a.txt"}}]},
                {"text": "读完了", "stop": True},
            ]
        ),
        encoding="utf-8",
    )
    proc = run_cli("--workspace", str(tmp_path), "--script", str(script), "读取文件")
    assert proc.returncode == 0, proc.stderr
    assert ">> read (c1)" in proc.stdout
    assert "1\talpha" in proc.stdout
    assert "读完了" in proc.stdout
    assert "finished: stop" in proc.stdout


def test_cli_without_script_reports_real_model_pending(tmp_path):
    proc = run_cli("--workspace", str(tmp_path), "测试")
    assert proc.returncode == 2
    assert "Ollama" in proc.stderr
