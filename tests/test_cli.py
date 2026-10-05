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


# --- T14: --tui 接线 (非 TTY / 缺 UI 依赖 / prompt 语义 / 零 UI 导入) ---------------------------


def test_tui_non_tty_rejected_with_clear_exit(tmp_path):
    # 子进程 stdin/stdout 均为管道: --tui 必须给出明确退出码 2
    proc = run_cli("--tui", "--workspace", str(tmp_path))
    assert proc.returncode == 2
    assert "交互式终端" in proc.stderr


def test_tui_non_tty_rejected_even_with_prompt(tmp_path):
    proc = run_cli("--tui", "--workspace", str(tmp_path), "首问")
    assert proc.returncode == 2
    assert "交互式终端" in proc.stderr


def test_single_mode_requires_prompt(tmp_path):
    proc = run_cli("--workspace", str(tmp_path))
    assert proc.returncode == 2
    assert "缺少 prompt" in proc.stderr


def test_tui_missing_ui_dependency_exits_cleanly(tmp_path):
    # 用一个 raise ImportError 的假 prompt_toolkit 模块模拟未安装 UI 可选组
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "prompt_toolkit.py").write_text(
        "raise ImportError('simulated missing prompt_toolkit')", encoding="utf-8"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocker}{os.pathsep}{SRC}",
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "cicada", "--tui", "--workspace", str(tmp_path), "首问"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "TUI 依赖未安装" in proc.stderr
    assert "simulated missing prompt_toolkit" in proc.stderr


def test_plain_mode_has_zero_ui_imports(tmp_path):
    # T14: 普通单次模式全程不导入 UI 可选依赖 (prompt_toolkit / rich)。
    # 运行完整 main() (fake 剧本, 零网络) 而非仅模块导入, 覆盖 _run 共享段。
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "hi", "stop": True}]), encoding="utf-8")
    code = (
        "import sys; "
        f"sys.argv = ['cicada', '--workspace', r'{tmp_path}', '--script', r'{script}', '任务']; "
        "import cicada.__main__ as m; "
        "rc = m.main(); "
        "assert rc == 0, f'exit {rc}'; "
        "assert 'prompt_toolkit' not in sys.modules, 'UI 依赖被普通模式导入'; "
        "assert 'rich' not in sys.modules, 'rich 被普通模式导入'"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_tui_startup_is_quiet_no_prompt_or_config_lines(tmp_path, monkeypatch, capsys):
    """P12: --tui 启动不打印 prompt/config 诊断行 (身份进 /status); 单次模式保持打印。"""
    import asyncio

    import cicada.__main__ as m

    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "答", "stop": True}]), encoding="utf-8")

    class FakeTTY:
        def isatty(self) -> bool:
            return True

        def reconfigure(self, **kwargs) -> None:  # main() 入口的编码设置
            return None

    monkeypatch.setattr(sys, "stdin", FakeTTY())
    monkeypatch.setattr(sys, "stdout", FakeTTY())

    captured: dict = {}

    async def fake_run_tui_session(session, **kwargs):
        captured["has_diagnostics"] = kwargs.get("diagnostics") is not None
        return 0

    monkeypatch.setattr("cicada.ui.run_tui_session", fake_run_tui_session)
    monkeypatch.setattr(
        sys, "argv",
        ["cicada", "--tui", "--workspace", str(tmp_path), "--script", str(script)],
    )
    code = m.main()
    out = capsys.readouterr().out
    assert code == 0
    assert captured["has_diagnostics"], "CLI 必须注入诊断"
    assert "prompt version=" not in out, "TUI 启动不打印 prompt 身份行"
    assert "[config model=" not in out, "TUI 启动不打印 config 行"
    assert "[check_mode" not in out


# --- 会话保存恢复 v1 (01): --save-session / --resume 全子进程路径 -----------------------------


def test_save_and_resume_roundtrip_via_separate_processes(tmp_path):
    """A01 的 CLI 级证明: 保存->关闭->新进程恢复->新请求, 旧工具不重放。"""
    work = tmp_path / "ws"
    work.mkdir()
    first_script = tmp_path / "first.json"
    first_script.write_text(json.dumps([
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "tool_use": True},
        {"text": "第一轮回答", "stop": True},
    ]), encoding="utf-8")
    second_script = tmp_path / "second.json"
    second_script.write_text(json.dumps([{"text": "恢复后的回答", "stop": True}]), encoding="utf-8")

    first = run_cli("--workspace", str(work), "--script", str(first_script),
                    "--save-session", "第一问")
    assert first.returncode == 0, first.stderr
    assert "[session id=" in first.stdout and "mode=save restored_requests=0" in first.stdout
    assert "[session saved request_id=" in first.stdout and "seq=" in first.stdout
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]
    assert len(session_id) == 36

    second = run_cli("--workspace", str(work), "--script", str(second_script),
                     "--resume", session_id, "第二问")
    assert second.returncode == 0, second.stderr
    assert f"[session id={session_id} mode=resume restored_requests=1]" in second.stdout
    assert "[session saved request_id=" in second.stdout
    # 旧工具只在第一次进程执行: 恢复进程不重放
    assert first.stdout.count(">> glob") == 1
    assert second.stdout.count(">> glob") == 0
    assert "<< glob" in first.stdout and "<< glob" not in second.stdout


def test_resume_unknown_id_reports_not_found_before_preflight(tmp_path):
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli("--workspace", str(tmp_path), "--ollama-url", dead,
                   "--resume", "123e4567-e89b-42d3-a456-426614174000", "任务")
    assert proc.returncode == 2
    assert "session store open failed" in proc.stderr
    assert "no session archive" in proc.stderr
    assert "Ollama 不可达" not in proc.stderr, "可达性探测不得先于存档预检"


def test_resume_malformed_id_rejected(tmp_path):
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(tmp_path), "--script", str(script),
                   "--resume", "not-a-uuid", "任务")
    assert proc.returncode == 2
    assert "UUIDv4" in proc.stderr


def test_save_session_and_resume_are_mutually_exclusive(tmp_path):
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(tmp_path), "--script", str(script),
                   "--save-session", "--resume", "123e4567-e89b-42d3-a456-426614174000", "任务")
    assert proc.returncode == 2
    assert "互斥" in proc.stderr


def test_resume_with_current_config_requires_resume(tmp_path):
    proc = run_cli("--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
                   "--resume-with-current-config", "任务")
    assert proc.returncode == 2
    assert "--resume-with-current-config 只能与 --resume" in proc.stderr


def _latest_epoch_launch(work: Path, session_id: str):
    """读存档最新 epoch 的身份 (04 差异记录证据)."""
    import sqlite3

    from cicada.persistence.codec import decode_identity

    db = work / ".cicada" / "sessions" / f"{session_id}.sqlite3"
    connection = sqlite3.connect(str(db))
    try:
        row = connection.execute(
            "SELECT identity_json FROM epoch ORDER BY opened_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return decode_identity(row[0])


def test_resume_with_current_config_adopts_check_mode_and_records_epoch(tmp_path):
    """04: 显式采用当前配置恢复——检查模式差异不拒绝, 新 epoch 记录采用与差异."""
    work = _git_worktree(tmp_path)
    script = _script(tmp_path)
    first = run_cli("--workspace", str(work), "--script", str(script), "--save-session", "第一问")
    assert first.returncode == 0, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]

    # 默认恢复时切换检查模式是冲突; 显式采用当前配置则放行并记录
    adopted = run_cli("--workspace", str(work), "--script", str(script),
                      "--resume", session_id, "--resume-with-current-config",
                      "--check-command", "exit 0", "第二问")
    assert adopted.returncode == 3, (adopted.stdout, adopted.stderr)  # 检查模式: stop 但检查未运行
    assert f"[session id={session_id} mode=resume" in adopted.stdout
    assert "[session saved request_id=" in adopted.stdout

    stored = _latest_epoch_launch(work, session_id)
    assert stored.launch.checks_signature is not None, "新 epoch 记录采用的检查模式"
    assert stored.launch.sources.get("resume_config") == "current"
    assert "checks_signature" in stored.launch.sources.get("changed_fields", "")


def test_resume_with_current_config_does_not_lift_view_only(tmp_path):
    """A15: 未证终结活动的恢复不因显式配置覆盖解除; CLI 新 prompt 零执行退出码 3."""
    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)), "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]
    _seed_archive(work, session_id, open_activity=True)
    proc = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                   "--resume", session_id, "--resume-with-current-config", "不应执行的问")
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert "[session view-only:" in proc.stderr
    assert ">> " not in proc.stdout, "view-only 会话零工具执行"
    assert "[session saved" not in proc.stdout


def _git_worktree(tmp_path):
    """检查模式需要 Git 工作树根 (验证基线); 建最小仓库."""
    import subprocess as _sp

    work = tmp_path / "ws"
    work.mkdir()
    _sp.run(["git", "init", "-q"], cwd=work, check=True, capture_output=True)
    (work / "sample.py").write_text("VALUE = 1" + chr(10), encoding="utf-8")
    _sp.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    _sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    return work


def test_save_session_with_check_mode_persists(tmp_path):
    """03: 检查模式 + 持久保存放开 —— 真实子进程保存成功."""
    work = _git_worktree(tmp_path)
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(work), "--script", str(script),
                   "--save-session", "--check-command", "exit 0", "任务")
    # 检查模式语义: 模型 stop 但检查未运行 -> 3; 保存照常完成
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert "[session id=" in proc.stdout
    assert "[session saved request_id=" in proc.stdout


def test_resume_check_mode_same_checks_continues_and_different_conflicts(tmp_path):
    work = _git_worktree(tmp_path)
    script = _script(tmp_path)
    first = run_cli("--workspace", str(work), "--script", str(script),
                    "--save-session", "--check-command", "exit 0", "第一问")
    assert first.returncode == 3, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]

    same = run_cli("--workspace", str(work), "--script", str(script),
                   "--resume", session_id, "--check-command", "exit 0", "第二问")
    assert same.returncode == 3, (same.stdout, same.stderr)
    assert f"[session id={session_id} mode=resume restored_requests=1]" in same.stdout
    assert "[session saved request_id=" in same.stdout

    conflict = run_cli("--workspace", str(work), "--script", str(script),
                       "--resume", session_id, "--check-command", "exit 1", "第三问")
    assert conflict.returncode == 2
    assert "checks_signature" in conflict.stderr


def test_persistence_with_tui_rejected(tmp_path):
    script = _script(tmp_path)
    proc = run_cli("--tui", "--workspace", str(tmp_path), "--script", str(script),
                   "--save-session", "任务")
    assert proc.returncode == 2
    assert "阶段 05" in proc.stderr


def test_resume_config_conflict_names_field(tmp_path):
    """同配置恢复之外的变化在副作用前拒绝, 且不先做模型可达性探测。

    存档由 fake 会话写入 (provider=fake); 恢复时改用真实模型旗标 ->
    provider/base_url 冲突先于 Ollama preflight 报出。
    """
    work = tmp_path / "ws2"
    work.mkdir()
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"text": "答", "stop": True}]), encoding="utf-8")
    first = run_cli("--workspace", str(work), "--script", str(script),
                    "--save-session", "问")
    assert first.returncode == 0, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]
    dead = f"http://127.0.0.1:{_dead_port()}"
    proc = run_cli("--workspace", str(work), "--ollama-url", dead,
                   "--resume", session_id, "任务")
    assert proc.returncode == 2
    assert "config conflicts" in proc.stderr
    assert "provider" in proc.stderr
    assert "Ollama 不可达" not in proc.stderr


def test_resume_after_workspace_move_rejected(tmp_path):
    """存档绑定解析后的真实路径: 整个工作区 (含 .cicada) 移动后拒绝恢复。"""
    import shutil

    work = tmp_path / "ws3"
    work.mkdir()
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"text": "答", "stop": True}]), encoding="utf-8")
    first = run_cli("--workspace", str(work), "--script", str(script), "--save-session", "问")
    assert first.returncode == 0, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]
    moved = tmp_path / "moved-workspace"
    shutil.move(str(work), str(moved))
    moved_script = moved.parent / "s2.json"
    moved_script.write_text(json.dumps([{"text": "答", "stop": True}]), encoding="utf-8")
    proc = run_cli("--workspace", str(moved), "--script", str(moved_script),
                   "--resume", session_id, "问")
    assert proc.returncode == 2
    assert "session store open failed" in proc.stderr
    assert "workspace" in proc.stderr


class _CaptureStream:
    """main() 会调用 reconfigure; 捕获文本供断言."""

    def __init__(self) -> None:
        self.value = ""

    def write(self, text: str) -> int:
        self.value += text
        return len(text)

    def flush(self) -> None:
        return None

    def reconfigure(self, **kwargs) -> None:  # main() 入口的编码设置
        return None


def _run_main_with_storage_failure(tmp_path, monkeypatch, method: str):
    """经 in-process main() 钉住「执行期间保存失败退出码 1」的 CLI 接线 (契约 §6)."""
    import cicada.__main__ as m
    from cicada.persistence.contracts import StorageWriteError
    from cicada.plugins.session_store.store import SQLiteSessionStore

    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"text": "答", "stop": True}]), encoding="utf-8")

    async def failing(self, *args, **kwargs):
        # 真实落盘故障类; 裸 PersistenceError 在 02 后按关系冲突分类, 不再锁存
        raise StorageWriteError("disk gone")

    monkeypatch.setattr(SQLiteSessionStore, method, failing)
    out, err = _CaptureStream(), _CaptureStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(
        sys, "argv",
        ["cicada", "--workspace", str(tmp_path), "--script", str(script),
         "--save-session", "任务"],
    )
    code = m.main()
    return code, err.value


def test_begin_save_failure_exits_one(tmp_path, monkeypatch):
    code, err = _run_main_with_storage_failure(tmp_path, monkeypatch, "begin_request")
    assert code == 1, err
    assert "session save failed before the request started" in err


def test_commit_save_failure_exits_one(tmp_path, monkeypatch):
    code, err = _run_main_with_storage_failure(tmp_path, monkeypatch, "commit_request")
    assert code == 1, err
    assert "[session save failed:" in err


# --- 02: view-only 恢复退出码与中断显示 --------------------------------------------------


def _cli_identity(work):
    """与 CLI fake 剧本路径一致的启动身份; 直建存档时用于准入匹配."""
    from cicada.persistence.contracts import LaunchIdentity
    from cicada.prompting import DEFAULT_TOOLS, compose_system_prompt

    _prompt, info = compose_system_prompt(work, None, tools=DEFAULT_TOOLS, checks=())
    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_request_bytes=None, max_turns=50,
        prompt_version=info.version, prompt_sha256=info.prompt_sha256,
        instructions_sha256=info.instructions_sha256, checks_signature=None, sources={},
    )


def _seed_archive(work: Path, session_id: str, *, open_activity: bool) -> None:
    """向真实 CLI 保存的存档注入 pending intent (+可选未闭合活动), 继承合法身份/指纹."""
    import hashlib
    import sqlite3

    from cicada.persistence.codec import dumps_strict

    db = work / ".cicada" / "sessions" / f"{session_id}.sqlite3"
    prompt_payload = dumps_strict({"prompt": "被中断的问"})
    request_id = "11111111-2222-4333-8444-555555555555"
    arguments = {"command": "Start-Sleep -Seconds 30"}
    canonical = dumps_strict(arguments)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    begin_json = dumps_strict({
        "description": canonical[:512], "arguments_digest": digest, "started_at": 1.0,
    })
    connection = sqlite3.connect(str(db))
    epoch = connection.execute(
        "SELECT epoch_id FROM epoch ORDER BY opened_at DESC LIMIT 1"
    ).fetchone()[0]
    generation = connection.execute(
        "SELECT context_generation FROM records ORDER BY seq DESC LIMIT 1"
    ).fetchone()[0]
    seq = connection.execute("SELECT MAX(seq) FROM records").fetchone()[0] + 1
    connection.execute(
        "INSERT INTO records (seq, epoch_id, context_generation, kind, request_id,"
        " created_at, payload_json, payload_sha256) VALUES (?, ?, ?, 'request_begin', ?, 0, ?, NULL)",
        (seq, epoch, generation, request_id, prompt_payload),
    )
    connection.execute(
        "INSERT INTO request_intents (request_id, epoch_id, begin_seq, status)"
        " VALUES (?, ?, ?, 'pending')",
        (request_id, epoch, seq),
    )
    if open_activity:
        connection.execute(
            "INSERT INTO process_activity (activity_id, request_id, tool_name, call_id,"
            " arguments_digest, begin_json, end_json) VALUES ('activity-1', ?, 'powershell',"
            " 'c-pending', ?, ?, NULL)",
            (request_id, digest, begin_json),
        )
    connection.commit()
    connection.close()


def test_resume_view_only_exits_three_without_execution(tmp_path):
    """A15: 未证终结活动的恢复, CLI 新 prompt 零执行并退出码 3."""
    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)), "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]
    _seed_archive(work, session_id, open_activity=True)
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(work), "--script", str(script),
                   "--resume", session_id, "不应执行的问")
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert f"[session id={session_id} mode=resume restored_requests=1]" in proc.stdout
    assert "[session interrupted_requests=1]" in proc.stdout
    assert "[session view-only:" in proc.stderr
    assert "powershell" in proc.stderr
    assert ">> " not in proc.stdout, "view-only 会话零工具执行"
    assert "[session saved" not in proc.stdout


def test_resume_with_pending_intent_prints_interrupted_and_continues(tmp_path):
    """A15: 无未证终结活动的中断可继续; 中断计数进入显示."""
    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)), "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]
    _seed_archive(work, session_id, open_activity=False)
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(work), "--script", str(script),
                   "--resume", session_id, "中断后的新问题")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "[session interrupted_requests=1]" in proc.stdout
    assert "mode=resume restored_requests=1" in proc.stdout
    assert "[session saved request_id=" in proc.stdout
