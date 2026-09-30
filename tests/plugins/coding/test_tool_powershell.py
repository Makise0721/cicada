import asyncio
import subprocess
import time
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.tool_powershell import PowerShellTool
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


async def wait_pid_gone(pid: int, attempts: int = 30) -> bool:
    for _ in range(attempts):
        if not pid_alive(pid):
            return True
        await asyncio.sleep(0.1)
    return not pid_alive(pid)


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
    command = (
        "$p = Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 60' -NoNewWindow -PassThru; "
        "$p.Id | Out-File -FilePath watch.pid -Encoding utf8; Start-Sleep -Seconds 60"
    )
    t0 = time.monotonic()
    r = await run(tool, command=command, timeout=1.0)
    elapsed = time.monotonic() - t0
    assert r.is_error
    assert r.details["timed_out"] is True
    assert r.details["cancelled"] is False
    assert r.details["exit_code"] is None
    assert elapsed < 10
    pid = int((ws.root / "watch.pid").read_text(encoding="utf-8-sig").strip())
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
    command = (
        "Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 8' -NoNewWindow -PassThru | Out-Null; "
        "Write-Output parent-exit"
    )
    t0 = time.monotonic()
    r = await run(tool, command=command)
    elapsed = time.monotonic() - t0
    assert not r.is_error
    assert r.details["exit_code"] == 0
    assert "parent-exit" in r.content
    assert elapsed < 6  # 孙进程睡 8s: grace 保证不挂死


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
    tail_lines = [l for l in r.content.split("\n") if l and not l.startswith("[")]
    assert 0 < len(tail_lines) < 1200
    assert all(line == "x" * 100 for line in tail_lines)
    assert str(full_path) in r.content


async def test_trailing_newline_not_overcounted(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Write-Output done")
    assert not r.is_error
    assert r.content == "done\n[exit_code=0 timed_out=False cancelled=False truncated=False]"


async def test_missing_cmdlet_diagnosable(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, command="Get-WidgetDoesNotExist")
    assert r.is_error
    assert r.details["exit_code"] != 0
    assert "Get-WidgetDoesNotExist" in r.content
