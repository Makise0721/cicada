"""公开观察 launcher 的确定性留证验收，不调用真实模型。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from p4_observer import Observer


PROJECT = Path(__file__).resolve().parents[2]
OBSERVER = Path(__file__).with_name("p4_observer.py")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.name=observer-test",
         "-c", "user.email=observer@test", *args],
        cwd=root, capture_output=True, check=True,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("x = 0\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "建立观察测试基线")
    return root


def _script(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _call(id_: str, name: str, **arguments) -> dict:
    return {"tool_calls": [{"id": id_, "name": name, "arguments": arguments}]}


def _launch(evidence: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(OBSERVER), "--evidence-dir", str(evidence), "--", *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src"),
             "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_observer_preserves_real_cli_and_records_full_delivery(tmp_path):
    root = _repo(tmp_path)
    script = _script(tmp_path, [
        _call("baseline", "check", action="run", check_id="check-1"),
        _call("read", "read", path="app.py"),
        _call("edit", "edit", path="app.py", edits=[{"old_text": "x = 0", "new_text": "x = 1"}]),
        _call("verified", "check", action="run", check_id="check-1"),
        {"text": "完成", "stop": True},
    ])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    command = "if ((Get-Content app.py -Raw).Contains('x = 1')) { exit 0 } else { exit 1 }"
    proc = _launch(evidence, "--workspace", str(root), "--script", str(script),
                   "--check-command", command, "修改 x 并检查")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[delivery can_deliver=true model_stopped=true]" in proc.stdout
    result = _json(evidence / "run-result.json")
    assert result["type"] == "RunResult" and result["stop_reason"] == "stop"
    assert len(result["model_calls"]) == 5
    tools = [m["result"] for m in result["messages"] if m["type"] == "ToolResultMessage"]
    assert [r["call_id"] for r in tools] == ["baseline", "read", "edit", "verified"]
    assert tools[0]["is_error"] is False and tools[-1]["is_error"] is False
    events = _jsonl(evidence / "events.jsonl")
    assert events[0]["type"] == "RunStarted" and events[-1]["type"] == "RunFinished"
    assert [e["result"] for e in events if e["type"] == "ToolCompleted"] == tools
    verification = _jsonl(evidence / "verification.jsonl")
    plan = next(v["value"] for v in verification if v["operation"] == "plan")
    assert plan["root"] == str(root.resolve()) and plan["checks"][0]["command"] == command
    baseline = next(v["value"] for v in verification if v["operation"] == "initialize")
    assert baseline["snapshot"]["entries"][0]["relative_path"] == "app.py"
    receipts = [v["value"] for v in verification if v["operation"] == "run_check"]
    assert [r["verification_status"] for r in receipts] == ["failed", "passed"]
    delivery = _json(evidence / "delivery.json")
    assert delivery["result"] == result and delivery["view"]["receipts"] == receipts
    assert delivery["decision"]["can_deliver"] is True
    assert delivery["evidence"]["snapshot_ref"] == delivery["view"]["snapshot_ref"]
    assert delivery["evidence"]["baseline_snapshot_ref"] == baseline["snapshot"]["snapshot_ref"]
    assert delivery["evidence"]["changes"][0]["relative_path"] == "app.py"
    snapshots = _jsonl(evidence / "snapshots.jsonl")
    final = snapshots[-1]["value"]["snapshot"]
    assert final["snapshot_ref"] == delivery["evidence"]["snapshot_ref"]
    assert final["entries"][0]["relative_path"] == "app.py" and final["entries"][0]["exists"] is True
    observer = _json(evidence / "observer-result.json")
    assert observer["main_exit_code"] == proc.returncode and observer["evidence_complete"] is True
    assert observer["recording_errors"] == [] and observer["wall_time_s"] > 0
    assert observer["sys_executable"] == sys.executable
    assert observer["module_paths"]["cicada.__main__"] == str(PROJECT / "src/cicada/__main__.py")


@pytest.mark.parametrize(("entries", "extra", "exit_code"), [
    ([{"error": "model failed"}], [], 1),
    ([{"stop": True}], ["--check-timeout", "0"], 2),
    ([{"stop": True}], [], 3),
])
def test_observer_keeps_real_cli_failure_exit_codes(tmp_path, entries, extra, exit_code):
    root = _repo(tmp_path)
    script = _script(tmp_path, entries)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    proc = _launch(evidence, "--workspace", str(root), "--script", str(script),
                   "--check-command", "exit 0", *extra, "任务")
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    observer = _json(evidence / "observer-result.json")
    assert observer["main_exit_code"] == exit_code and observer["recording_errors"] == []
    if exit_code != 2:
        assert _json(evidence / "delivery.json")["decision"]["can_deliver"] is False


def test_observer_reports_write_failure_without_changing_cli_success(tmp_path):
    root = _repo(tmp_path)
    script = _script(tmp_path, [{"text": "完成", "stop": True}])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    # 真正文件系统故障：单值工件路径被目录占用，不能写入 RunResult。
    (evidence / "run-result.json").mkdir()
    proc = _launch(evidence, "--workspace", str(root), "--script", str(script), "任务")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "=== finished: stop" in proc.stdout and "[run_summary" in proc.stdout
    observer = _json(evidence / "observer-result.json")
    assert observer["main_exit_code"] == 0 and observer["evidence_complete"] is False
    assert observer["recording_errors"][0]["operation"] == "run-result.json"
    assert observer["recording_errors"][0]["type"] in {"PermissionError", "IsADirectoryError", "FileExistsError"}


@pytest.mark.asyncio
async def test_async_observation_preserves_return_identity_and_exceptions(tmp_path):
    observer = Observer(tmp_path)
    value = {"nested": (Path("app.py"), {"value": None})}

    async def succeeds():
        return value

    result = await observer.observe_async(succeeds, "sample", "sample.json", append=False)()
    assert result is value and _json(tmp_path / "sample.json") == {"nested": ["app.py", {"value": None}]}
    failure = RuntimeError("original exception")

    async def fails():
        raise failure

    with pytest.raises(RuntimeError) as raised:
        await observer.observe_async(fails, "sample.failure", "unused.json")()
    assert raised.value is failure
    assert _jsonl(tmp_path / "observed-exceptions.jsonl") == [{
        "operation": "sample.failure", "exception": {"type": "RuntimeError", "message": "original exception"},
    }]
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.asyncio
async def test_recording_error_keeps_original_return_and_original_cancellation(tmp_path):
    observer = Observer(tmp_path)
    value = object()  # 不支持的序列化对象：返回身份仍必须原样保留。

    async def succeeds():
        return value

    assert await observer.observe_async(succeeds, "sample", "sample.json", append=False)() is value
    assert observer.recording_errors[0]["type"] == "TypeError"
    import asyncio
    cancelled = asyncio.CancelledError("cancelled by caller")

    async def fails():
        raise cancelled

    with pytest.raises(asyncio.CancelledError) as raised:
        await observer.observe_async(fails, "cancelled", "unused.json")()
    assert raised.value is cancelled
