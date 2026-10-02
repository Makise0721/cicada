"""P4 06 跨层 e2e: 公开 CLI + 真实 Git 工作树 + 真实 PowerShell 指定检查 + FakeModel 剧本.

最高 seam 验收 (spec Testing Decisions): 不测 helper, 走 `python -m cicada` 子进程,
核对退出码、独立 delivery section 与真实文件变更。场景覆盖:
0/1/2/3 退出码、多检查、最近失败覆盖旧 PASS、变更/新增/删除使回执失效、脏基线、
工件缺失/篡改、process_uncertain 锁存、终局再次变化、fake 漏报/虚报完成。
普通模式默认注册 glob/grep 也在此验证 (A01/A07/A08/A11 的 CLI 集成面)。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

APP_NO_TOOLS_OK = 'def build_run_summary(result):\n    return "summary"\n'
APP_WITH_TOOLS_OK = 'def build_run_summary(result, tools_ok=False):\n    return "summary"\n'

CHECK_TOOLS_OK = (
    "if (Select-String -Path app.py -Pattern 'tools_ok' -Quiet) "
    "{ exit 0 } else { Write-Output 'app.py missing tools_ok'; exit 1 }"
)


def run_cli(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
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
        timeout=timeout,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, capture_output=True, check=True,
    )


def make_repo(tmp_path: Path, *, app_content: str = APP_NO_TOOLS_OK, extra: dict[str, str] | None = None) -> Path:
    """独立 Git 工作树: tracked app.py (+可选额外 tracked 文件), 单个初始 commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(app_content, encoding="utf-8", newline="\n")
    for name, content in (extra or {}).items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")
    return repo


def write_script(repo: Path, entries: list[dict]) -> Path:
    script = repo / "script.json"
    script.write_text(json.dumps(entries), encoding="utf-8", newline="\n")
    return script


def call(id_: str, name: str, arguments: dict) -> dict:
    return {"tool_calls": [{"id": id_, "name": name, "arguments": arguments}]}


def stop(text: str = "任务完成。") -> dict:
    return {"text": text, "stop": True}


def edit_app(script_id: str, old: str, new: str) -> dict:
    return call(script_id, "edit", {"path": "app.py", "edits": [{"old_text": old, "new_text": new}]})


def run_check(script_id: str, check_id: str = "check-1") -> dict:
    return call(script_id, "check", {"action": "run", "check_id": check_id})


def checked_cli(repo: Path, script: Path, *extra: str, prompt: str = "给 app.py 增加 tools_ok 参数") -> subprocess.CompletedProcess:
    return run_cli(
        "--workspace", str(repo), "--script", str(script),
        "--check-command", CHECK_TOOLS_OK, *extra, prompt,
    )


def delivery(stdout: str) -> str:
    start = stdout.index("[delivery ")
    return stdout[start:]


# --- 退出码 0: 完整交付 ---------------------------------------------------------------


def test_full_delivery_flow_exits_zero(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        call("r1", "read", {"path": "app.py"}),
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        run_check("c1"),
        stop("我已完成修改并通过检查。"),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    section = delivery(proc.stdout)
    assert "[delivery can_deliver=true model_stopped=true]" in section
    assert "- check-1 status=passed freshness=current" in section
    assert "[changed files: 1]" in section
    assert "- modified app.py" in section
    assert "[change artifact: " in section and " sha256=" in section
    assert "[blocking_reasons]\n- none" in section
    # P3 三行摘要仍按原样输出, 不被误改为验收逻辑
    assert "[run_summary run_id=run-1 stop_reason=stop model_calls=4" in proc.stdout
    assert "[scope policy_id=" in section  # 范围政策披露


def test_dirty_baseline_and_multi_change_scope(tmp_path):
    repo = make_repo(tmp_path, extra={"old.txt": "stale content\n"})
    # 脏基线: 任务开始前已有未提交修改; 基线以启动时真实内容为准
    (repo / "old.txt").write_text("dirty before run\n", encoding="utf-8", newline="\n")
    script = write_script(repo, [
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        call("w1", "write", {"path": "new_file.txt", "content": "created\n"}),
        call("p1", "powershell", {"command": "Remove-Item old.txt"}),
        run_check("c1"),
        stop(),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    section = delivery(proc.stdout)
    # run 中的真实变更: 修改 + 新增 + 删除; 删除的 before hash 是脏基线字节而非 commit 内容
    assert "[changed files: 3]" in section
    assert "- modified app.py" in section
    assert "- added new_file.txt" in section
    dirty_sha = hashlib.sha256(b"dirty before run\n").hexdigest()
    assert f"- deleted old.txt before_sha256={dirty_sha} after_sha256=None" in section


# --- 退出码 3: 模型 stop 但条件不满足 ---------------------------------------------------


def test_latest_failure_overrides_old_pass(tmp_path):
    repo = make_repo(tmp_path, app_content=APP_WITH_TOOLS_OK)
    script = write_script(repo, [
        run_check("c1"),
        edit_app("e1", "def build_run_summary(result, tools_ok=False):", "def build_run_summary(result):"),
        run_check("c2"),
        stop("检查已通过。"),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert "- check-1 status=failed" in section  # 最近一次失败取代旧 PASS
    assert "nonzero_exit" in section
    assert any("check check-1 is failed" in line for line in section.splitlines())
    assert "[delivery can_deliver=false model_stopped=true]" in section


def test_change_after_pass_leaves_receipt_stale(tmp_path):
    repo = make_repo(tmp_path, app_content=APP_WITH_TOOLS_OK)
    script = write_script(repo, [
        run_check("c1"),
        edit_app("e1", '    return "summary"', '    return "summary"  # touched'),
        stop("检查已通过, 无进一步改动。"),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert "- check-1 status=passed freshness=stale" in section
    assert any("receipt is stale" in line for line in section.splitlines())


def test_fake_underreporting_does_not_hide_real_changes(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        {"text": "我没有修改任何文件, 任务无法完成。", "stop": True},
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 3  # 检查从未运行
    section = delivery(proc.stdout)
    assert "- modified app.py" in section  # 独立变更证据揭穿漏报
    assert any("check check-1 has not been run" in line for line in section.splitlines())
    # 虚构的"没改过"不进入判定, 但变更清单来自真实字节
    assert (repo / "app.py").read_text(encoding="utf-8").find("tools_ok") >= 0


def test_multiple_checks_require_all_passed(tmp_path):
    repo = make_repo(tmp_path, extra={"notes.md": "notes\n"})
    script = write_script(repo, [
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        run_check("c1"),
        call("c2", "check", {"action": "run", "check_id": "check-2"}),
        stop(),
    ])
    proc = run_cli(
        "--workspace", str(repo), "--script", str(script),
        "--check-command", CHECK_TOOLS_OK,
        "--check-command", "Write-Output 'second check'; exit 2",
        "任务",
    )
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert "- check-1 status=passed freshness=current" in section
    assert "- check-2 status=failed" in section
    assert any("check check-2 is failed" in line for line in section.splitlines())


def test_check_timeout_latches_process_uncertain(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        run_check("c1"),
        stop(),
    ])
    proc = run_cli(
        "--workspace", str(repo), "--script", str(script),
        "--check-command", "Start-Sleep -Seconds 8",
        "--check-timeout", "2",
        "任务",
    )
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert "- check-1 status=blocked" in section
    assert "[process_uncertain=true]" in section
    assert any("process_uncertain" in line for line in section.splitlines())


def test_powershell_timeout_latch_survives_later_check_pass(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        call("p1", "powershell", {"command": "Start-Sleep -Seconds 8", "timeout": 2}),
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        run_check("c1"),
        stop("全部检查已通过。"),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 3  # 新 PASS 不能清除本 run 的不确定锁存
    section = delivery(proc.stdout)
    assert "- check-1 status=passed freshness=current" in section
    assert "[process_uncertain=true]" in section


def test_check_output_artifact_tampered_blocks_delivery(tmp_path):
    repo = make_repo(tmp_path, app_content=APP_WITH_TOOLS_OK)
    # 检查通过且产生输出, 才有可被篡改的输出工件
    check_with_output = (
        "Write-Output 'running tools_ok check'; " + CHECK_TOOLS_OK
    )
    script = write_script(repo, [
        run_check("c1"),
        call(
            "p1", "powershell",
            {"command": "Get-ChildItem .cicada/outputs -Filter 'powershell-output-*' | "
                        "ForEach-Object { Add-Content -Path $_.FullName -Value 'tampered' }"},
        ),
        stop(),
    ])
    proc = run_cli(
        "--workspace", str(repo), "--script", str(script),
        "--check-command", check_with_output, "任务",
    )
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert "- check-1 status=passed freshness=current" in section
    assert any("output artifact no longer matches" in line for line in section.splitlines())


def test_baseline_artifact_deleted_blocks_evidence(tmp_path):
    repo = make_repo(tmp_path, app_content=APP_WITH_TOOLS_OK)
    script = write_script(repo, [
        run_check("c1"),
        call("p1", "powershell", {"command": "Remove-Item .cicada/outputs/baseline-*.txt"}),
        stop(),
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 3
    section = delivery(proc.stdout)
    assert any("the change evidence is incomplete (baseline_corrupt)" in line
               for line in section.splitlines())
    assert "sha256=unverified" in section  # 基线工件被删后不能再声称完整证据


# --- 退出码 1: 模型非 stop 终结 ---------------------------------------------------------


def test_model_error_exits_one_in_check_mode(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        edit_app("e1", "def build_run_summary(result):", "def build_run_summary(result, tools_ok=False):"),
        run_check("c1"),
        {"error": "model exploded"},
    ])
    proc = checked_cli(repo, script)
    assert proc.returncode == 1
    section = delivery(proc.stdout)
    assert "[delivery can_deliver=false model_stopped=false]" in section
    assert any("stop_reason=error" in line for line in section.splitlines())
    assert "=== finished: error (model exploded)" in proc.stdout


# --- 退出码 2: 输入/启动错误 -------------------------------------------------------------


def test_non_git_workspace_rejected(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    script = write_script(plain, [stop()])
    proc = run_cli(
        "--workspace", str(plain), "--script", str(script),
        "--check-command", "exit 0", "任务",
    )
    assert proc.returncode == 2
    assert "verification initialize failed" in proc.stderr


def test_workspace_subdirectory_rejected(tmp_path):
    repo = make_repo(tmp_path)
    sub = repo / "pkg"
    sub.mkdir()
    (repo / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    script = write_script(repo, [stop()])
    proc = run_cli(
        "--workspace", str(sub), "--script", str(script),
        "--check-command", "exit 0", "任务",
    )
    assert proc.returncode == 2
    assert "verification initialize failed" in proc.stderr
    assert "Git worktree root" in proc.stderr


# --- 普通模式: glob/grep 默认注册, 语义不变 ----------------------------------------------


def test_normal_mode_registers_glob_and_grep(tmp_path):
    repo = make_repo(tmp_path, app_content=APP_NO_TOOLS_OK)
    script = write_script(repo, [
        call("g1", "glob", {"pattern": "**/*.py"}),
        call("s1", "grep", {"pattern": "build_run_summary", "include": "**/*.py"}),
        stop("已定位实现。"),
    ])
    proc = run_cli("--workspace", str(repo), "--script", str(script), "定位 build_run_summary")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "app.py" in proc.stdout  # glob 结果含真实相对路径
    assert "No matches" not in proc.stdout  # grep 命中 build_run_summary
    assert "[delivery" not in proc.stdout  # 普通模式没有 delivery section
    assert "unknown tool: check" not in proc.stdout  # 未注册 check, 剧本也不调用


def test_normal_mode_check_tool_not_registered(tmp_path):
    repo = make_repo(tmp_path)
    script = write_script(repo, [
        call("c1", "check", {"action": "status"}),
        stop(),
    ])
    proc = run_cli("--workspace", str(repo), "--script", str(script), "任务")
    assert proc.returncode == 0  # 工具错误反馈给模型, 模型继续 stop
    assert "unknown tool: check" in proc.stdout
