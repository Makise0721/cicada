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
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# 04: 与应用级测试共享存档身份读取 helper (避免同段 SQL 复制)
from test_persistence_application import _latest_epoch_identity as _latest_epoch_launch


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


class _OllamaStub:
    """最小本地 Ollama 替身 (A11 真实模型组合的确定性证据):

    GET /api/version -> 200 (preflight); POST /api/chat -> 单条 NDJSON stop 终结流.
    requests 记录收到的请求行, 用于「参数来源/冲突字段/零意外调用」断言。
    """

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self
        self.requests: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def _reply(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                outer.requests.append(f"GET {self.path}")
                self._reply(b'{"version": "stub"}', "application/json")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests.append(f"POST {self.path}")
                self._reply(
                    b'{"message": {"role": "assistant", "content": "stub answer"},'
                    b' "done": true, "done_reason": "stop"}\n',
                    "application/x-ndjson",
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


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


def test_resume_zero_epoch_archive_reports_missing_identity(tmp_path):
    """零 epoch 存档 (如保存进程在 bootstrap 前崩溃遗留) 恢复时点名拒绝, 不静默退出。"""
    import asyncio

    from cicada.plugins.session_store import open_session_store

    work = tmp_path / "ws"
    work.mkdir()

    async def _seed() -> str:
        store = open_session_store(
            directory=work / ".cicada" / "sessions", session_id=None, workspace=work
        )
        session_id = store.session_id
        await store.aclose()
        return session_id

    session_id = asyncio.run(_seed())
    proc = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                   "--resume", session_id, "问")
    assert proc.returncode == 2
    assert "no saved launch identity" in proc.stderr


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
    assert f"[session id={session_id} mode=resume restored_requests=1]" in adopted.stdout
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


def test_resume_without_id_rejected_outside_tui(tmp_path):
    """05: 无 ID 的 --resume 仅 TUI 可用; 单次模式明确拒绝 (退出码 2)."""
    proc = run_cli("--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
                   "任务", "--resume")
    assert proc.returncode == 2
    assert "仅在 --tui 下可用" in proc.stderr
    # nargs="?" 吃词: 紧随 --resume 的词按 UUIDv4 提前校验, 不延迟到开库
    proc = run_cli("--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
                   "--resume", "任务")
    assert proc.returncode == 2
    assert "UUIDv4" in proc.stderr


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


# --- 04: 默认恢复的配置继承 (A11; 真实模型组合 + 本地 ollama 替身) ------------------------------


def test_real_mode_resume_inherits_omitted_config_and_names_conflicts(tmp_path):
    """省略参数继承存档 (来源 archive), 显式相同可用, 显式不同点名冲突且零模型探测."""
    stub = _OllamaStub()
    try:
        work = tmp_path / "ws"
        work.mkdir()
        first = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                        "--model", "stub-model", "--think", "--num-ctx", "8192",
                        "--save-session", "第一问")
        assert first.returncode == 0, (first.stdout, first.stderr)
        assert "model=stub-model context_requested=8192 source=cli think=true" in first.stdout
        assert stub.requests == ["GET /api/version", "POST /api/chat"]
        session_id = first.stdout.split("[session id=")[1].split(" ")[0]
        stub.requests.clear()

        inherited = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                            "--resume", session_id, "第二问")
        assert inherited.returncode == 0, (inherited.stdout, inherited.stderr)
        # 省略 model/think/num-ctx 全部继承存档, 来源标记 archive
        assert "model=stub-model context_requested=8192 source=archive think=true" in inherited.stdout
        assert stub.requests == ["GET /api/version", "POST /api/chat"], "恰一次探测与一次调用"
        stub.requests.clear()

        same = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                       "--resume", session_id,
                       "--model", "stub-model", "--think", "--num-ctx", "8192", "第三问")
        assert same.returncode == 0, (same.stdout, same.stderr)
        assert "model=stub-model context_requested=8192 source=cli think=true" in same.stdout
        stub.requests.clear()

        inherited_url = run_cli("--workspace", str(work), "--resume", session_id, "第四问")
        assert inherited_url.returncode == 0, (inherited_url.stdout, inherited_url.stderr)
        # 省略 --ollama-url: 服务地址继承存档 (仍指向本替身)
        assert stub.requests == ["GET /api/version", "POST /api/chat"], "base_url 继承存档"
        stub.requests.clear()

        conflict = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                           "--resume", session_id, "--num-ctx", "4096", "第五问")
        assert conflict.returncode == 2
        assert "config conflicts" in conflict.stderr
        assert "options" in conflict.stderr
        assert stub.requests == [], "配置冲突先于可达性探测 (零模型请求)"
    finally:
        stub.stop()


def test_resume_unknown_saved_option_key_rejected(tmp_path):
    """契约 §6: 未知配置键不执行——继承前明确拒绝且零模型探测."""
    import sqlite3

    stub = _OllamaStub()
    try:
        work = tmp_path / "ws"
        work.mkdir()
        first = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                        "--num-ctx", "8192", "--save-session", "问")
        assert first.returncode == 0, first.stderr
        session_id = first.stdout.split("[session id=")[1].split(" ")[0]
        db = work / ".cicada" / "sessions" / f"{session_id}.sqlite3"
        connection = sqlite3.connect(str(db))
        try:
            row = connection.execute(
                "SELECT identity_json FROM epoch ORDER BY opened_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
            payload = json.loads(row[0])
            payload["options"]["bogus_threads"] = 4
            connection.execute(
                "UPDATE epoch SET identity_json = ? WHERE identity_json = ?",
                (json.dumps(payload), row[0]),
            )
            connection.commit()
        finally:
            connection.close()
        stub.requests.clear()
        proc = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                       "--resume", session_id, "问")
        assert proc.returncode == 2
        assert "未知配置键" in proc.stderr and "bogus_threads" in proc.stderr
        assert stub.requests == []
    finally:
        stub.stop()


def test_resume_omitted_checks_on_check_mode_archive_rejected(tmp_path):
    """检查定义只存签名: 检查模式存档省略 --check-command 恢复时给出明确指引."""
    work = _git_worktree(tmp_path)
    script = _script(tmp_path)
    first = run_cli("--workspace", str(work), "--script", str(script),
                    "--save-session", "--check-command", "exit 0", "第一问")
    assert first.returncode == 3, first.stderr  # 检查模式: stop 但检查未运行
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]
    proc = run_cli("--workspace", str(work), "--script", str(script),
                   "--resume", session_id, "第二问")
    assert proc.returncode == 2
    assert "--check-command" in proc.stderr
    assert "--resume-with-current-config" in proc.stderr


def test_real_mode_resume_with_current_config_overrides_inheritance(tmp_path):
    """A11: 旗标下省略参数取当前默认 (而非存档), 差异记入新 epoch."""
    stub = _OllamaStub()
    try:
        work = tmp_path / "ws"
        work.mkdir()
        first = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                        "--num-ctx", "8192", "--save-session", "第一问")
        assert first.returncode == 0, first.stderr
        session_id = first.stdout.split("[session id=")[1].split(" ")[0]
        stub.requests.clear()

        adopted = run_cli("--workspace", str(work), "--ollama-url", stub.url,
                          "--resume", session_id, "--resume-with-current-config", "第二问")
        assert adopted.returncode == 0, (adopted.stdout, adopted.stderr)
        # 省略 --num-ctx 取当前默认 32768 (非存档 8192), 来源 default
        assert "context_requested=32768 source=default" in adopted.stdout
        assert "[session saved request_id=" in adopted.stdout

        stored = _latest_epoch_launch(work, session_id)
        assert stored.launch.options.get("num_ctx") == 32768
        assert stored.launch.sources.get("resume_config") == "current"
        assert "options" in stored.launch.sources.get("changed_fields", "")
    finally:
        stub.stop()


def test_fake_resume_records_new_script_source_per_epoch(tmp_path):
    """AC5 复核: fake 恢复显式提供后继剧本, 新来源记入新 epoch; 旧剧本不重放、
    剧本内容不参与身份比对 (01 宽进点维持, 来源可追溯)."""
    work = tmp_path / "ws"
    work.mkdir()
    first_script = tmp_path / "first.json"
    first_script.write_text(json.dumps([
        {"tool_calls": [{"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "tool_use": True},
        {"text": "第一轮", "stop": True},
    ]), encoding="utf-8")
    second_script = tmp_path / "second.json"
    second_script.write_text(json.dumps([{"text": "恢复后回答", "stop": True}]), encoding="utf-8")

    first = run_cli("--workspace", str(work), "--script", str(first_script),
                    "--save-session", "第一问")
    assert first.returncode == 0, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]

    second = run_cli("--workspace", str(work), "--script", str(second_script),
                     "--resume", session_id, "第二问")
    assert second.returncode == 0, second.stderr
    assert ">> glob" not in second.stdout, "后继剧本只服务新调用, 旧剧本不重放"

    stored = _latest_epoch_launch(work, session_id)
    assert stored.launch.sources.get("script") == str(second_script), "新剧本来源记入新 epoch"


def test_resume_normal_archive_with_checks_conflicts_on_signature(tmp_path):
    """普通存档 + 显式 --check-command 恢复: checks_signature 冲突点名拒绝 (AC2 组合钉)."""
    work = tmp_path / "ws"
    work.mkdir()
    script = _script(tmp_path)
    first = run_cli("--workspace", str(work), "--script", str(script), "--save-session", "第一问")
    assert first.returncode == 0, first.stderr
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]
    proc = run_cli("--workspace", str(work), "--script", str(script),
                   "--resume", session_id, "--check-command", "exit 0", "第二问")
    assert proc.returncode == 2
    assert "config conflicts" in proc.stderr
    assert "checks_signature" in proc.stderr


# --- 05: --list-sessions 只读分页列表 (A12) ----------------------------------------------------


def test_list_sessions_reports_saved_sessions_and_rejects_execution_flags(tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    script = _script(tmp_path)
    first = run_cli("--workspace", str(work), "--script", str(script),
                    "--save-session", "修复登录页的问题")
    assert first.returncode == 0, first.stderr
    id_one = first.stdout.split("[session id=")[1].split(" ")[0]

    proc = run_cli("--workspace", str(work), "--list-sessions")
    assert proc.returncode == 0, proc.stderr
    assert "会话列表" in proc.stdout
    assert id_one in proc.stdout and "修复登录页的问题" in proc.stdout
    assert "请求=1" in proc.stdout

    for extra in (["任务"], ["--save-session"], ["--resume", id_one],
                  ["--resume-with-current-config"], ["--tui"]):
        combo = run_cli("--workspace", str(work), "--list-sessions", *extra)
        assert combo.returncode == 2, extra


def test_list_sessions_marks_bad_archive_and_occupied(tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    script = _script(tmp_path)
    proc = run_cli("--workspace", str(work), "--script", str(script), "--save-session", "问")
    assert proc.returncode == 0, proc.stderr
    session_id = proc.stdout.split("[session id=")[1].split(" ")[0]
    (work / ".cicada" / "sessions" / "00000000-1111-4222-8333-444444444444.sqlite3").write_bytes(
        b"not a database"
    )
    # 同进程持有 lease: 子进程列表看到占用标注
    from cicada.plugins.session_store import open_session_store

    holder = open_session_store(directory=work / ".cicada" / "sessions",
                                session_id=session_id, workspace=work)
    try:
        listing = run_cli("--workspace", str(work), "--list-sessions")
        assert listing.returncode == 0, listing.stderr
        assert session_id in listing.stdout and "占用" in listing.stdout
        assert "00000000-1111-4222-8333-444444444444" in listing.stdout
        assert "坏档" in listing.stdout, "坏文件是明确错误项, 不是缺失"
    finally:
        import asyncio as _asyncio

        _asyncio.run(holder.aclose())


def test_render_session_page_bounded_and_paged():
    """翻页提示与有界渲染的纯函数单测 (子进程只覆盖扁平列表)."""
    import cicada.__main__ as m
    from cicada.persistence.contracts import SessionPage, SessionSummary

    entries = tuple(
        SessionSummary(
            session_id=f"id-{index}", short_title=f"标题 {index}", created_at=1.0,
            updated_at=100.0 + index, committed_requests=index, interrupted_requests=0,
            occupied=index == 1, unproven_activities=2 if index == 1 else 0,
            untrusted_receipts=0,
        )
        for index in range(2)
    )
    lines = m.render_session_page(SessionPage(directory="d", entries=entries, next_cursor="25"))
    text = "\n".join(lines)
    assert "id-0" in text and "标题 1" in text and "请求=1" in text
    assert "占用" in text and "未证终结活动×2" in text
    assert "--list-cursor 25" in text
    empty = m.render_session_page(SessionPage(directory="d", entries=(), next_cursor=None))
    assert any("暂无会话" in line for line in empty)


# --- 05: TUI 恢复入口 (按 ID / 选择器 / view-only 退出码) --------------------------------------


class _CaptureTTY:
    """进程内 TUI 驱动: TTY 语义 + 输出捕获 (失败时 stderr 可断言)。"""

    def __init__(self) -> None:
        self.value = ""

    def isatty(self) -> bool:
        return True

    def reconfigure(self, **kwargs) -> None:
        return None

    def write(self, data: str) -> int:
        self.value += data
        return len(data)

    def flush(self) -> None:
        return None


def _run_main(monkeypatch, argv, *, tui_code: int = 0):
    """进程内 main(): TTY 替身 + run_tui_session 替身; 返回 (退出码, stderr 捕获)。"""
    import cicada.__main__ as m

    async def fake_run_tui_session(session, **kwargs):
        return tui_code

    monkeypatch.setattr("cicada.ui.run_tui_session", fake_run_tui_session)
    err = _CaptureTTY()
    monkeypatch.setattr(sys, "stdin", _CaptureTTY())
    monkeypatch.setattr(sys, "stdout", _CaptureTTY())
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["cicada", *argv])
    return m.main(), err


def test_tui_persistent_save_allowed_and_default(tmp_path, monkeypatch):
    """05: --tui 默认持久; --save-session 允许组合 (不改变行为); 每次启动自动保存新会话."""
    for extra in ([], ["--save-session"]):
        code, err = _run_main(monkeypatch, ["--tui", "--workspace", str(tmp_path),
                                            "--script", str(_script(tmp_path)), *extra])
        assert code == 0, err.value
    archives = list((tmp_path / ".cicada" / "sessions").glob("*.sqlite3"))
    assert len(archives) == 2, "两次 TUI 启动各自动保存一个新会话"


def test_tui_resume_by_id_exit_zero(tmp_path, monkeypatch):
    """05: TUI 按 ID 恢复打开; 正常关闭退出码 0。"""
    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                   "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]

    code, err = _run_main(monkeypatch, ["--tui", "--workspace", str(work),
                                        "--script", str(_script(tmp_path)), "--resume", session_id])
    assert code == 0, err.value


def test_tui_resume_selector_cancel_exits_zero(tmp_path, monkeypatch):
    """05: --tui --resume 无 ID 启动选择器; Esc 取消退出码 0; 未打开任何会话。"""
    import cicada.__main__ as m

    async def fake_selector(workspace: Path):
        return None

    monkeypatch.setattr(m, "_launch_session_selector", fake_selector)
    code, err = _run_main(monkeypatch, ["--tui", "--workspace", str(tmp_path),
                                        "--script", str(_script(tmp_path)), "--resume"])
    assert code == 0, err.value


def test_tui_resume_selector_pick_opens_session(tmp_path, monkeypatch):
    """05: 选择器选定 ID 后走共享恢复路径打开该会话。"""
    import cicada.__main__ as m

    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                   "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]

    async def fake_selector(workspace: Path):
        assert (workspace / ".cicada" / "sessions").is_dir()
        return session_id

    monkeypatch.setattr(m, "_launch_session_selector", fake_selector)
    code, err = _run_main(monkeypatch, ["--tui", "--workspace", str(work),
                                        "--script", str(_script(tmp_path)), "--resume"])
    assert code == 0, err.value


def test_tui_view_only_restore_exits_three(tmp_path, monkeypatch):
    """05/A15: view-only 恢复的 TUI 可查看; 无新请求关闭后退出码 3。"""
    work = tmp_path / "ws"
    work.mkdir()
    save = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                   "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]
    _seed_archive(work, session_id, open_activity=True)

    code, err = _run_main(monkeypatch, ["--tui", "--workspace", str(work),
                                        "--script", str(_script(tmp_path)), "--resume", session_id])
    assert code == 3, err.value


def test_tui_resume_with_initial_prompt_executes_once(tmp_path, monkeypatch):
    """审查 05-M2: 恢复 + 选择后初始 prompt——新请求恰执行一次, 历史不重发."""
    import asyncio

    import cicada.__main__ as m

    work = tmp_path / "ws"
    work.mkdir()
    first_script = tmp_path / "first.json"
    first_script.write_text(json.dumps([{"text": "第一轮回答", "stop": True}]), encoding="utf-8")
    save = run_cli("--workspace", str(work), "--script", str(first_script), "--save-session", "第一问")
    assert save.returncode == 0, save.stderr
    session_id = save.stdout.split("[session id=")[1].split(" ")[0]

    second_script = tmp_path / "second.json"
    second_script.write_text(json.dumps([{"text": "恢复后回答", "stop": True}]), encoding="utf-8")

    captured: dict = {}

    async def fake_run_tui_session(session, *, initial_prompt=None, **kwargs):
        captured["initial_prompt"] = initial_prompt
        model = session._app.agent._model
        if initial_prompt:
            await session.execute(initial_prompt)
        captured["requests"] = list(model.requests)
        return 0

    monkeypatch.setattr("cicada.ui.run_tui_session", fake_run_tui_session)
    err = _CaptureTTY()
    monkeypatch.setattr(sys, "stdin", _CaptureTTY())
    monkeypatch.setattr(sys, "stdout", _CaptureTTY())
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(
        sys, "argv",
        ["cicada", "--tui", "--workspace", str(work), "--script", str(second_script),
         "--resume", session_id, "恢复后首问"],
    )
    code = m.main()
    assert code == 0, err.value
    assert captured["initial_prompt"] == "恢复后首问"
    assert len(captured["requests"]) == 1, "初始 prompt 只执行一次"
    messages = captured["requests"][0].messages
    assert "第一问" in "".join(getattr(msg, "text", "") for msg in messages), "恢复历史进入请求"
    from cicada.core.messages import UserMessage

    user_texts = [msg.text for msg in messages if isinstance(msg, UserMessage)]
    assert user_texts[-1] == "恢复后首问", "新 prompt 是最后一条用户消息"


def test_list_sessions_no_model_probes_and_no_ui_imports(tmp_path):
    """审查 05-F10: 列表入口零模型探测 (A12) 且零 UI 依赖导入."""
    stub = _OllamaStub()
    try:
        (tmp_path / ".cicada" / "sessions").mkdir(parents=True)
        proc = run_cli("--workspace", str(tmp_path), "--ollama-url", stub.url, "--list-sessions")
        assert proc.returncode == 0, proc.stderr
        assert stub.requests == [], "列表入口零模型探测"
    finally:
        stub.stop()

    code = (
        "import sys; "
        f"sys.argv = ['cicada', '--workspace', r'{tmp_path}', '--list-sessions']; "
        "import cicada.__main__ as m; "
        "rc = m.main(); "
        "assert rc == 0, f'exit {rc}'; "
        "assert 'prompt_toolkit' not in sys.modules, 'UI 依赖被列表入口导入'; "
        "assert 'rich' not in sys.modules, 'rich 被列表入口导入'"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_resume_error_history_new_request_exits_zero(tmp_path):
    """审查 05-F11: 历史失败不当本进程最后请求——error 结局存档恢复后新请求 rc 0."""
    work = tmp_path / "ws"
    work.mkdir()
    error_script = tmp_path / "error.json"
    error_script.write_text(json.dumps([{"error": "boom"}]), encoding="utf-8")
    first = run_cli("--workspace", str(work), "--script", str(error_script), "--save-session", "第一问")
    assert first.returncode == 1, (first.stdout, first.stderr)  # error 结局
    assert "[session saved request_id=" in first.stdout, "错误结局仍保存真实终结 (A03)"
    session_id = first.stdout.split("[session id=")[1].split(" ")[0]

    second = run_cli("--workspace", str(work), "--script", str(_script(tmp_path)),
                     "--resume", session_id, "第二问")
    assert second.returncode == 0, (second.stdout, second.stderr)
    assert "[session saved request_id=" in second.stdout


def test_list_cursor_without_list_sessions_rejected(tmp_path):
    """审查 05-F6: 游标只属于列表入口; 误用拒绝而非静默执行请求."""
    proc = run_cli("--workspace", str(tmp_path), "--script", str(_script(tmp_path)),
                   "--list-cursor", "3", "任务")
    assert proc.returncode == 2
    assert "--list-cursor 只能与 --list-sessions" in proc.stderr
