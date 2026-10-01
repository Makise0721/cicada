import asyncio
import subprocess
import time
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.process import BoundedText
from cicada.plugins.coding.tool_powershell import MAX_CONTENT_BYTES, PowerShellTool
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, PowerShellTool(ws)


async def run(tool, cancel=None, **arguments):
    return await tool.execute(
        arguments, ToolContext(call_id="c1", cancel=cancel or CancelToken())
    )


def pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
    ).stdout
    return str(pid) in out


async def wait_for_file(path, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if path.exists():
            return True
        await asyncio.sleep(0.1)
    return path.exists()


async def wait_pid_gone(pid: int, attempts: int = 30) -> bool:
    for _ in range(attempts):
        if not pid_alive(pid):
            return True
        await asyncio.sleep(0.1)
    return not pid_alive(pid)


def visible_tail(content: str) -> list[str]:
    """去掉状态行与 footer (以 '[' 开头的行), 只留可见输出正文."""
    return [line for line in content.split("\n") if line and not line.startswith("[")]


async def test_utf8_output_intact(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command='Write-Output "中文测试"')
    assert not r.is_error
    assert r.details["exit_code"] == 0
    assert "中文测试" in r.content
    assert "\ufffd" not in r.content


async def test_multibyte_across_chunks_not_corrupted(tmp_path):
    ws, tool = make(tmp_path)
    s = "中🎉ok" * 40
    command = f"1..400 | ForEach-Object {{ Write-Output '{s}' }}"
    r = await run(tool, command=command)
    assert not r.is_error
    assert r.details["truncated"] is True
    full_path = Path(r.details["full_output_path"])
    assert full_path.parent == ws.output_dir
    full_lines = full_path.read_text(encoding="utf-8").splitlines()
    assert len(full_lines) == 400
    assert all(line == s for line in full_lines)  # 多字节跨 chunk 无乱码、无丢字
    tail_lines = [l for l in r.content.split("\n") if l and not l.startswith("[")]
    assert 0 < len(tail_lines) < 400
    assert all(line == s for line in tail_lines)
    assert str(ws.output_dir) in r.content


async def test_nonzero_exit(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Write-Output hi; exit 3")
    assert r.is_error
    assert r.details["exit_code"] == 3
    assert "hi" in r.content


async def test_stderr_merged_in_arrival_order(tmp_path):
    ws, tool = make(tmp_path)
    command = (
        "Write-Output o1; Start-Sleep -Milliseconds 150; "
        "Write-Error e1; Start-Sleep -Milliseconds 150; Write-Output o2"
    )
    r = await run(tool, command=command)
    assert not r.is_error
    assert r.content.index("o1") < r.content.index("e1") < r.content.index("o2")


async def test_timeout_kills_process_tree(tmp_path):
    ws, tool = make(tmp_path)
    # 先落盘子进程 pid 再长睡: 1s 超时会在 pwsh 启动延迟内到达, 顺序反过来会拿不到证据
    command = (
        "$p = Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 60' -NoNewWindow -PassThru; "
        "$p.Id | Out-File -FilePath watch.pid -Encoding utf8; Start-Sleep -Seconds 60"
    )
    t0 = time.monotonic()
    r = await run(tool, command=command, timeout=2.0)
    elapsed = time.monotonic() - t0
    assert r.is_error
    assert r.details["timed_out"] is True
    assert r.details["cancelled"] is False
    assert r.details["exit_code"] is None
    assert r.details["output_complete"] is False
    assert elapsed < 10
    pidfile = ws.root / "watch.pid"
    if not await wait_for_file(pidfile):
        pytest.skip("pwsh startup slower than the timeout window on this host")
    pid = int(pidfile.read_text(encoding="utf-8-sig").strip())
    assert await wait_pid_gone(pid)


async def test_cancel_kills_tree_and_next_run_clean(tmp_path):
    ws, tool = make(tmp_path)
    cancel = CancelToken()
    task = asyncio.create_task(
        run(tool, cancel=cancel, command="Write-Output started; Start-Sleep -Seconds 60")
    )
    await asyncio.sleep(1.5)
    cancel.cancel()
    r = await task
    assert r.is_error
    assert r.details["cancelled"] is True
    assert r.details["timed_out"] is False
    assert r.details["exit_code"] is None
    r2 = await run(tool, command="Write-Output fresh")
    assert not r2.is_error
    assert r2.content.startswith("fresh")
    assert "started" not in r2.content


async def test_grandchild_holding_pipe_returns_after_grace(tmp_path):
    ws, tool = make(tmp_path)
    # 孙进程工作目录指向 basetemp 之外, 避免其存续期间锁住 pytest 的 tmp 目录清理
    repo_root = Path(__file__).resolve().parents[2]
    command = (
        "Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        f"'Start-Sleep -Seconds 8' -NoNewWindow -WorkingDirectory '{repo_root}' "
        "-PassThru | Out-Null; Write-Output parent-exit"
    )
    t0 = time.monotonic()
    r = await run(tool, command=command)
    elapsed = time.monotonic() - t0
    assert r.details["exit_code"] == 0
    assert "parent-exit" in r.content
    # 父进程已退出但孙进程仍持管道: 有限宽限内没有 EOF, 采集不完整且收尾超时;
    # 退出码 0 不自动解释为命令完整终结
    assert r.details["output_complete"] is False
    assert r.details["timed_out"] is True
    assert elapsed < 6  # 孙进程睡 8s: 有限宽限保证不挂死


async def test_truncation_writes_full_output(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="1..1200 | ForEach-Object { Write-Output ('x' * 100) }")
    assert not r.is_error
    assert r.details["truncated"] is True
    full_path = Path(r.details["full_output_path"])
    assert full_path.parent == ws.output_dir
    full_lines = full_path.read_text(encoding="utf-8").splitlines()
    assert len(full_lines) == 1200
    assert all(line == "x" * 100 for line in full_lines)
    tail_lines = visible_tail(r.content)
    assert 0 < len(tail_lines) < 1200
    assert all(line == "x" * 100 for line in tail_lines)
    assert str(full_path) in r.content


async def test_trailing_newline_not_overcounted(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Write-Output done")
    assert not r.is_error
    assert r.content == (
        "done\n"
        "[exit_code=0 timed_out=False cancelled=False truncated=False "
        "output_complete=True artifact_truncated=False]\n"
        f"[full output: {r.details['full_output_path']}]"
    )


async def test_missing_cmdlet_diagnosable(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Get-WidgetDoesNotExist")
    assert r.is_error
    assert r.details["exit_code"] != 0
    assert "Get-WidgetDoesNotExist" in r.content


# --- 03: 严格 50 KiB content 与工具层完整性字段 ---


async def test_single_60kb_line_stays_within_50kib(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Write-Output ('x' * 60000)")
    assert not r.is_error
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert r.details["truncated"] is True
    assert r.details["output_complete"] is True
    assert r.details["full_output_bytes"] > 60000
    full = Path(r.details["full_output_path"]).read_text(encoding="utf-8")
    assert full.strip() == "x" * 60000
    assert "raw total" in r.content


async def test_multiline_unicode_and_escapes_stay_within_cap(tmp_path):
    ws, tool = make(tmp_path)
    # 中文/emoji 是多字节, 反斜杠与引号是 JSON 转义膨胀来源
    line = '中文🙂 "quoted" \\backslash\\ ' * 8
    command = f"1..900 | ForEach-Object {{ Write-Output '{line}' }}"
    r = await run(tool, command=command)
    assert not r.is_error
    content = r.content.encode("utf-8")
    assert len(content) <= MAX_CONTENT_BYTES
    assert content.decode("utf-8") == r.content  # 边界裁剪不产生半个字符
    assert "\ufffd" not in r.content
    assert "exit_code=0" in r.content  # 状态行没有被裁掉
    full_lines = Path(r.details["full_output_path"]).read_text(encoding="utf-8").splitlines()
    assert len(full_lines) == 900
    assert all(item == line for item in full_lines)


async def test_empty_output_reports_status_only(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="")
    assert not r.is_error
    assert r.details["exit_code"] == 0
    assert r.details["output_complete"] is True
    assert r.details["truncated"] is False
    assert r.details["artifact_truncated"] is False
    assert r.content.startswith("[exit_code=0 ")


async def test_artifact_write_failure_is_reported_not_silent(tmp_path):
    ws, tool = make(tmp_path)
    # 真实存储故障: 用同名文件占住工件目录路径, mkdir 必然失败
    ws.output_dir.rmdir()
    ws.output_dir.write_text("not a directory", encoding="utf-8")
    r = await run(tool, command="1..50 | ForEach-Object { Write-Output ('line' * 200) }")
    # 工件写失败不是命令执行失败: exit_code 与终结事实照旧, 但完整性必须如实报告
    assert r.details["exit_code"] == 0
    assert r.details["full_output_path"] is None
    assert "cannot open output artifact" in r.details["artifact_error"]
    assert r.details["output_complete"] is True
    assert r.details["artifact_truncated"] is False
    assert "cannot open output artifact" in r.content
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES


# --- 03: 工具层裁剪单元覆盖 (不启动真实进程) ---


def _bounded(text="", **kwargs):
    payload = {
        "text": text,
        "truncated": False,
        "total_bytes": 0,
        "total_lines": 0,
        "full_output_path": None,
    }
    payload.update(kwargs)
    return BoundedText(**payload)


def test_bounded_text_frozen_defaults_are_backward_compatible():
    bounded = _bounded("hello")
    assert bounded.artifact_truncated is False
    assert bounded.artifact_error is None


def test_tool_adds_drop_note_when_body_budget_trims_visible_tail():
    bounded = _bounded(
        "line\n" * 20000, truncated=True, total_bytes=200000, full_output_path=None
    )
    content = PowerShellTool._assemble(
        [
            "[exit_code=1 timed_out=False cancelled=False truncated=True "
            "output_complete=True artifact_truncated=False]"
        ],
        bounded,
        None,
        True,
        [],
    )
    assert len(content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert content.startswith("[tail omitted to fit the 50 KiB content cap]")
    assert "truncated=true reasons:" in content


def test_tool_drops_entire_tail_when_status_block_consumes_budget():
    bounded = _bounded("x" * (MAX_CONTENT_BYTES + 100), truncated=True, total_bytes=999999)
    content = PowerShellTool._assemble(
        ["[exit_code=0 timed_out=False cancelled=False truncated=True "
         "output_complete=True artifact_truncated=False]"],
        bounded,
        "C:\\" + "very-long-path\\" * 400 + "artifact.txt",
        True,
        [],
    )
    assert len(content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert "artifact_truncated" in content or "truncated" in content
