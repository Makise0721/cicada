"""离线 P4 私有 fixture、行为 checker 与观察证据核对。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"
VENV_PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"


# 固定三件套; 跨尝试不变, hash 记入证据。
CHECKER_NAME = "check_p4.py"
CHECKER_SOURCE = r'''from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cicada.core.agent import RunResult
from cicada.core.messages import ToolResult, ToolResultMessage
from cicada.reporting import build_run_summary
cases = [("empty", [], 0), ("all_success", [False, False], 2),
         ("mixed", [False, True, False], 2), ("all_error", [True, True], 0)]
failures = []
for name, errors, expected_ok in cases:
    messages = tuple(ToolResultMessage(ToolResult(str(i), "read", "probe", is_error=e))
                     for i, e in enumerate(errors))
    summary = build_run_summary(RunResult("probe", "stop", None, messages))
    first = summary.splitlines()[0]
    required = {"tools_ok": expected_ok, "tools": len(errors), "tool_errors": sum(errors)}
    missing = [f"{key}={value}" for key, value in required.items()
               if not re.search(rf"(?: |\[){key}={value}(?: |\])", first)]
    if missing:
        failures.append((name, missing, first))
    if len(summary.splitlines()) != 3 or "input_tokens_known=unknown" not in summary:
        failures.append((name, "existing summary fields/shape changed", summary))
if failures:
    print("CHECK_FAIL", failures)
    raise SystemExit(1)
print("CHECK_PASS: 4 behavior cases; existing counters, shape and unknown metrics preserved")
'''
# 检查命令只引用相对路径; python 解释器由 harness 的 PATH 提供 (launcher 事实)。
CHECK_COMMAND = "& python check_p4.py"
TASK_PROMPT = (
    "这个工作区是 Cicada 项目的一份源码副本。请完成一个小改动: 给 build_run_summary 函数"
    "生成的摘要第一行增加 tools_ok=N, 表示 is_error=False 的工具结果数。"
    "保留原有计数、三行结构和未知计量语义。"
    "请先用 grep 工具搜索 \"def build_run_summary\" 定位实现文件, 再用 read 读取相关源码, "
    "先用 check 工具运行指定检查 check-1 确认初始失败, 再完成最小修改并重检通过。"
    "不要修改检查器, 最后汇报实际 diff 和检查退出状态。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.email=p4@local", "-c", "user.name=p4", *args],
        cwd=repo, capture_output=True, text=True, encoding="utf-8",
    )


def manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    }


def build_fixture(dest: Path) -> dict:
    """独立 Git fixture: 复制实施后 src/ + 固定 checker, 并完成 baseline 反证.

    返回 fixture 事实 (commit/manifest/checker 反证/导入优先探针), 全部进入证据。
    """
    dest.mkdir(parents=True, exist_ok=False)
    copied = 0
    for source in sorted(SRC.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        target = dest / "src" / source.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    checker_path = dest / CHECKER_NAME
    checker_path.write_text(CHECKER_SOURCE, encoding="utf-8", newline="\n")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()
    commit = _git(dest, "init", "-q")
    assert commit.returncode == 0, commit.stderr
    assert _git(dest, "add", "-A").returncode == 0
    assert _git(dest, "commit", "-q", "-m", "P4 验收副本基线").returncode == 0
    fixture_commit = _git(dest, "rev-parse", "HEAD").stdout.strip()
    baseline_manifest = manifest(dest)

    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    # baseline 反证 1: checker 对原始 fixture exit 1 且无 traceback -> 唯一失败分支是 tools_ok
    baseline_run = subprocess.run(
        [str(VENV_PYTHON), CHECKER_NAME], cwd=dest, capture_output=True, text=True,
        encoding="utf-8", env=env, timeout=60,
    )
    assert baseline_run.returncode == 1, (
        f"checker must fail on baseline, got {baseline_run.returncode}: {baseline_run.stderr}"
    )
    assert baseline_run.stderr == "", f"baseline failure must not be an import/traceback: {baseline_run.stderr}"
    assert baseline_run.stdout.startswith("CHECK_FAIL "), baseline_run.stdout
    failures = ast.literal_eval(baseline_run.stdout.removeprefix("CHECK_FAIL ").strip())
    expected_missing = {
        "empty": ["tools_ok=0"], "all_success": ["tools_ok=2"],
        "mixed": ["tools_ok=2"], "all_error": ["tools_ok=0"],
    }
    assert len(failures) == 4 and {item[0]: item[1] for item in failures} == expected_missing, (
        f"baseline must fail only for missing tools_ok counts: {failures}"
    )
    # baseline 反证 2: 导入优先绑定 fixture/src (相同 PYTHONPATH 环境下)
    probe = subprocess.run(
        [str(VENV_PYTHON), "-c",
         f"import sys; sys.path.insert(0, r'{(dest / 'src').resolve()}'); import cicada; print(cicada.__file__)"],
        cwd=dest, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    expected_init = (dest.resolve() / "src" / "cicada" / "__init__.py")
    assert probe.stdout.strip().lower() == str(expected_init).lower(), probe.stdout

    return {
        "fixture_root": str(dest),
        "fixture_commit": fixture_commit,
        "copied_py_files": copied,
        "checker_sha256": sha256_text(CHECKER_SOURCE),
        "checker_baseline_exit": baseline_run.returncode,
        "checker_baseline_stderr_empty": baseline_run.stderr == "",
        "checker_baseline_stdout": baseline_run.stdout,
        "import_probe": probe.stdout.strip(),
        "main_head": head,
        "baseline_manifest": baseline_manifest,
    }


def validate_attempt(result: dict) -> None:
    """同时核对真实行为、观察留证、请求profile与交付身份; 结果落盘后才返回."""
    evidence_dir = Path(result["evidence_dir"])
    try:
        _assert_attempt(result, evidence_dir)
    except BaseException as exc:
        _write_json(evidence_dir / "acceptance.json", {
            "accepted": False, "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    _write_json(evidence_dir / "acceptance.json", {"accepted": True})


def _assert_attempt(result: dict, evidence_dir: Path) -> None:
    assert not result["timed_out"] and result["episode_within_budget"], "600s episode budget exceeded"
    assert not result["proxy_errors"], result["proxy_errors"]
    assert result["returncode"] == 0, f"CLI exit {result['returncode']}"
    observer = json.loads((evidence_dir / "observer-result.json").read_text(encoding="utf-8"))
    assert observer["evidence_complete"] and not observer["recording_errors"], observer
    assert observer["main_exit_code"] == result["returncode"]
    run = json.loads((evidence_dir / "run-result.json").read_text(encoding="utf-8"))
    assert run["stop_reason"] == "stop" and run["error"] is None
    delivery = json.loads((evidence_dir / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["result"] == run
    assert delivery["decision"]["can_deliver"] and delivery["decision"]["model_stopped"]
    view, changes = delivery["view"], delivery["evidence"]
    assert not view["process_uncertain"] and not view["blocking_reasons"]
    assert changes["complete"] and changes["snapshot_ref"] == view["snapshot_ref"]
    assert len(view["checks"]) == 1
    state = view["checks"][0]
    assert state["check_id"] == "check-1" and state["freshness"] == "current"
    assert state["receipt"]["verification_status"] == "passed" and state["receipt"]["exit_code"] == 0
    assert state["receipt"]["snapshot_after"] == view["snapshot_ref"]
    # 初始红、修改后绿都保留真实回执; 不能只看模型汇报.
    assert any(r["verification_status"] == "failed" and r["exit_code"] == 1 for r in view["receipts"])
    assert view["receipts"][-1] == state["receipt"]
    for receipt in view["receipts"]:
        if receipt["output_artifact_path"] is not None:
            saved = evidence_dir / "artifacts" / Path(receipt["output_artifact_path"]).name
            assert sha256_file(saved) == receipt["output_artifact_sha256"]
    assert sha256_file(evidence_dir / "artifacts" / Path(changes["artifact_path"]).name) == changes["artifact_sha256"]
    assert [c["relative_path"] for c in changes["changes"]] == ["src/cicada/reporting.py"]
    snapshots = [json.loads(line)["value"] for line in (evidence_dir / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    snapshot = snapshots[-1]["snapshot"]
    assert snapshot["snapshot_ref"] == view["snapshot_ref"]
    entry = next(e for e in snapshot["entries"] if e["relative_path"] == "src/cicada/reporting.py")
    assert entry["sha256"] == changes["changes"][0]["after_sha256"]
    events = [json.loads(line) for line in (evidence_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    names = [event["name"] for event in events if event["type"] == "ToolStarted"]
    search = next(i for i, name in enumerate(names) if name in ("grep", "glob"))
    read = names.index("read", search + 1)
    edit = next(i for i in range(read + 1, len(names)) if names[i] in ("edit", "write"))
    assert "check" in names[edit + 1:], names
    assert result["checker_exit"] == 0 and result["checker_unchanged"]
    changed = [line for line in result["changed_git_lines"] if line != "?? .cicada/"]
    assert changed == [" M src/cicada/reporting.py"], changed
    assert "tools_ok" in result["git_diff"]
    assert "[delivery can_deliver=true model_stopped=true]" in result["stdout"]
