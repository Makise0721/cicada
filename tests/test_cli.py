"""CLI 配置与接线聚焦测试: num_ctx 解析/互斥/校验时序、启动显示、摘要输出.

经子进程核对 `_run` 接线 (P3 §9: 不能只测 helper). preflight 失败路径用死端口,
确定性零网络依赖; 指令文件错误必须先于 preflight。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


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


def _dead_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _script(tmp_path: Path) -> Path:
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "done", "stop": True}]), encoding="utf-8")
    return script


def test_num_ctx_non_integer_rejected(tmp_path):
    proc = run_cli("--workspace", str(tmp_path), "--num-ctx", "abc", "任务")
    assert proc.returncode == 2


def test_num_ctx_non_positive_rejected_before_preflight(tmp_path):
    # 死端口: 若先 preflight 会报 Ollama 不可达; 先报 num_ctx 证明校验时序
    dead = f"http://127.0.0.1:{_dead_port()}"
    for value in ("0", "-5"):
        proc = run_cli("--workspace", str(tmp_path), "--ollama-url", dead, "--num-ctx", value, "任务")
        assert proc.returncode == 2, value
        assert "--num-ctx 必须是正整数" in proc.stderr
        assert "Ollama 不可达" not in proc.stderr


def test_script_and_num_ctx_are_mutually_exclusive(tmp_path):
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(_script(tmp_path)), "--num-ctx", "8192", "任务"
    )
    assert proc.returncode == 2
    assert "互斥" in proc.stderr


def test_real_mode_displays_default_and_cli_num_ctx(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli("--workspace", str(tmp_path), "--ollama-url", dead, "任务")
    assert proc.returncode == 2
    assert "context_requested=32768 source=default" in proc.stdout
    assert "Ollama 不可达" in proc.stderr

    proc = run_cli(
        "--workspace", str(tmp_path), "--ollama-url", dead, "--num-ctx", "8192", "任务"
    )
    assert proc.returncode == 2
    assert "context_requested=8192 source=cli" in proc.stdout


def test_instructions_file_error_precedes_preflight(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli(
        "--workspace", str(tmp_path), "--ollama-url", dead,
        "--instructions-file", "missing.md", "任务",
    )
    assert proc.returncode == 2
    assert "invalid instructions file" in proc.stderr
    assert "Ollama 不可达" not in proc.stderr


def test_instructions_file_in_real_mode_displayed(tmp_path):
    (tmp_path / "RULES.md").write_text("先跑测试再汇报。", encoding="utf-8")
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli(
        "--workspace", str(tmp_path), "--ollama-url", dead,
        "--instructions-file", "RULES.md", "任务",
    )
    assert proc.returncode == 2  # preflight 失败
    assert "[prompt version=builtin:p4-v1" in proc.stdout
    assert "instructions=" in proc.stdout and "RULES.md" in proc.stdout
    assert "instructions_sha256=" in proc.stdout and "prompt_sha256=" in proc.stdout
    assert "先跑测试再汇报" not in proc.stdout  # 不打印提示全文


def test_script_mode_uses_builtin_prompt_and_prints_summary(tmp_path):
    (tmp_path / "NOTE.md").write_text("项目约定", encoding="utf-8")
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
        "--instructions-file", "NOTE.md", "任务",
    )
    assert proc.returncode == 0, proc.stderr
    assert "[prompt version=builtin:p4-v1" in proc.stdout
    assert "instructions=" in proc.stdout
    assert "=== finished: stop" in proc.stdout
    assert "[run_summary run_id=run-1 stop_reason=stop model_calls=1 tools=0 tool_errors=0]" in proc.stdout
    # fake model 无计量: unknown 而非 0
    assert "input_tokens_known=unknown input_reported_calls=0/1" in proc.stdout
    assert "model_time_s=" in proc.stdout and "model_time_s=unknown" not in proc.stdout


# --- P4 §5 检查模式输入边界: 全部 exit 2 -------------------------------------------------


def test_check_timeout_without_check_command_rejected(tmp_path):
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
        "--check-timeout", "30", "任务",
    )
    assert proc.returncode == 2
    assert "--check-timeout 只能与 --check-command 一起使用" in proc.stderr


def test_check_command_count_limit_rejected(tmp_path):
    commands = [part for i in range(9) for part in ("--check-command", f"exit {i}")]
    proc = run_cli("--workspace", str(tmp_path), "--script", str(_script(tmp_path)), *commands, "任务")
    assert proc.returncode == 2
    assert "最多 8 条" in proc.stderr


def test_check_command_empty_rejected(tmp_path):
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
        "--check-command", "   ", "任务",
    )
    assert proc.returncode == 2
    assert "空命令" in proc.stderr


def test_check_command_byte_limit_rejected(tmp_path):
    oversized = "Write-Output '" + "x" * 4200 + "'"
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
        "--check-command", oversized, "任务",
    )
    assert proc.returncode == 2
    assert "超过 4096 UTF-8 bytes" in proc.stderr


def test_check_timeout_range_rejected(tmp_path):
    for value in ("0", "-1", "300.5"):
        proc = run_cli(
            "--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
            "--check-command", "exit 0", "--check-timeout", value, "任务",
        )
        assert proc.returncode == 2, value
        assert "--check-timeout 必须在" in proc.stderr


def test_check_mode_small_num_ctx_rejected_before_preflight(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli(
        "--workspace", str(tmp_path), "--ollama-url", dead,
        "--check-command", "exit 0", "--num-ctx", "8192", "任务",
    )
    assert proc.returncode == 2
    assert "检查模式要求 --num-ctx >= 32768" in proc.stderr
    assert "Ollama 不可达" not in proc.stderr  # 输入校验先于 preflight


def test_check_mode_real_model_profile_displayed(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli(
        "--workspace", str(tmp_path), "--ollama-url", dead,
        "--check-command", "exit 0", "任务",
    )
    assert proc.returncode == 2  # preflight 失败, 但先显示检查模式 profile
    assert "num_predict=2048" in proc.stdout
    assert "request_bytes_limit=65536" in proc.stdout
    assert "[check_mode checks=1 ids=check-1" in proc.stdout
    assert "max_turns=25" in proc.stdout


def test_normal_mode_has_no_request_limit(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli("--workspace", str(tmp_path), "--ollama-url", dead, "任务")
    assert proc.returncode == 2
    assert "request_bytes_limit" not in proc.stdout  # 普通模式保持原语义
