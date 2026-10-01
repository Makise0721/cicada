"""coding.process 运行器与有界采集的聚焦验证 (03: 严格有界、流式工件、收尾与取消)."""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.process import (
    ARTIFACT_MAX_BYTES,
    READ_CHUNK_BYTES,
    RESOURCE_CLEANUP_SECONDS,
    PowerShellRunner,
    _OutputCollector,
)
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    return Workspace.create(tmp_path / "ws")


async def run_runner(runner, ws, cancel=None, **arguments):
    return await runner.run(
        cwd=ws.root,
        cancel=cancel or CancelToken(),
        output_dir=ws.output_dir,
        **arguments,
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


# --- 采集器单元行为 (不启动真实进程) ---


def collect(chunks, output_dir) -> _OutputCollector:
    collector = _OutputCollector(output_dir)
    for chunk in chunks:
        collector.feed(chunk)
    collector.finish()
    return collector


# 大负载在用例内构造: pytest 会把参数值编码进环境变量 (Windows 上限 32767 字符)
COLLECTOR_PAYLOADS = {
    "long_single_line": lambda: b"x" * 60000 + b"\n",
    "multibyte_multiline": lambda: ("中文🙂ok\n" * 9000).encode(),
    "no_newline": lambda: b"w" * 60000,
    "multibyte_single_line": lambda: ("中文🙂" * 20000 + "\n").encode(),
    "empty": lambda: b"",
    "short": lambda: b"hi\n",
}


@pytest.mark.parametrize("label", sorted(COLLECTOR_PAYLOADS))
def test_collector_tail_stays_bounded_and_raw_total_is_exact(tmp_path, label):
    payload = COLLECTOR_PAYLOADS[label]()
    collector = collect([payload], tmp_path)
    assert len(collector.text.encode("utf-8")) <= 50 * 1024
    assert collector.total_bytes == len(payload)
    if payload:
        assert collector.artifact_bytes == len(payload)
        assert collector.artifact_path.read_bytes() == payload
    else:
        assert collector.artifact_path is None


def test_collector_marks_omission_for_oversized_single_line(tmp_path):
    collector = collect([b"z" * 60000 + b"\n"], tmp_path)
    assert collector.truncated is True
    assert "bytes omitted" in collector.text
    assert collector.omitted_bytes > 0
    # 完整内容仍在工件里, 可见文本只是有界视图
    assert collector.artifact_path.read_text(encoding="utf-8").strip() == "z" * 60000


def test_collector_decodes_multibyte_across_chunk_boundary(tmp_path):
    payload = ("中文🎉ok" * 4000 + "\n").encode("utf-8")
    chunks = [payload[index : index + 7] for index in range(0, len(payload), 7)]
    collector = collect(chunks, tmp_path)
    assert "\ufffd" not in collector.text
    assert collector.artifact_path.read_bytes() == payload


# --- 有界 tail / 工件上限 (真实进程) ---


async def test_tail_and_pending_are_bounded_for_long_single_line(tmp_path):
    ws = make(tmp_path)
    result = await run_runner(
        PowerShellRunner(), ws, command="Write-Output ('z' * 60000)", timeout=60.0
    )
    assert result.exit_code == 0
    assert result.output_complete is True
    assert result.output.truncated is True
    assert len(result.output.text.encode("utf-8")) <= 50 * 1024
    assert result.output.total_bytes >= 60000
    assert "bytes omitted" in result.output.text
    assert result.output.full_output_path.read_text(encoding="utf-8").strip() == "z" * 60000


async def test_artifact_cap_truncates_prefix_and_keeps_raw_total(tmp_path):
    ws = make(tmp_path)
    # 约 17 MiB 输出: 工件只保留 16 MiB 前缀, 原始 total_bytes 仍如实上报
    result = await run_runner(
        PowerShellRunner(),
        ws,
        command="1..600 | ForEach-Object { Write-Output ('y' * 30000) }",
        timeout=120.0,
    )
    assert result.exit_code == 0
    assert result.output_complete is True
    assert result.output.artifact_truncated is True
    assert result.output.artifact_error is None
    assert result.output.truncated is True
    assert result.output.total_bytes > ARTIFACT_MAX_BYTES
    stored = result.output.full_output_path.stat().st_size
    assert stored <= ARTIFACT_MAX_BYTES
    assert stored > ARTIFACT_MAX_BYTES - 65536
    assert result.output.full_output_path.read_bytes().decode("utf-8", "replace")


async def test_read_chunk_is_fixed_and_independent_of_total_output(tmp_path, monkeypatch):
    ws = make(tmp_path)
    sizes: list[int] = []
    original = _OutputCollector.feed

    def counting_feed(self, data):
        sizes.append(len(data))
        original(self, data)

    monkeypatch.setattr(_OutputCollector, "feed", counting_feed)
    result = await run_runner(
        PowerShellRunner(),
        ws,
        command="1..600 | ForEach-Object { Write-Output ('y' * 30000) }",
        timeout=120.0,
    )
    assert result.exit_code == 0
    assert sizes and max(sizes) <= READ_CHUNK_BYTES


# --- deadline / EOF / 取消收尾 ---


async def test_grandchild_holding_pipe_is_bounded_and_incomplete(tmp_path):
    ws = make(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]
    command = (
        "Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        f"'Start-Sleep -Seconds 8' -NoNewWindow -WorkingDirectory '{repo_root}' "
        "-PassThru | Out-Null; Write-Output parent-exit"
    )
    t0 = time.monotonic()
    result = await run_runner(PowerShellRunner(), ws, command=command, timeout=60.0)
    elapsed = time.monotonic() - t0
    assert result.exit_code == 0
    assert result.output.text.strip() == "parent-exit"
    # 父进程已退出但孙进程仍持管道: 有限宽限内没有 EOF, 采集不完整且收尾超时
    assert result.output_complete is False
    assert result.timed_out is True
    assert elapsed < 6


async def test_grandchild_writing_forever_cannot_extend_deadline(tmp_path):
    ws = make(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]
    command = (
        "Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'while ($true) { Write-Output ticking }' -NoNewWindow "
        f"-WorkingDirectory '{repo_root}' -PassThru | Out-Null; Write-Output parent-exit"
    )
    t0 = time.monotonic()
    result = await run_runner(
        PowerShellRunner(), ws, command=command, timeout=3.0
    )
    elapsed = time.monotonic() - t0
    assert "parent-exit" in result.output.text
    assert result.output_complete is False
    assert result.timed_out is True
    assert elapsed < 6  # 3s deadline + 有限宽限 + 有界清理


async def test_continuous_output_does_not_reset_deadline(tmp_path):
    ws = make(tmp_path)
    t0 = time.monotonic()
    result = await run_runner(
        PowerShellRunner(),
        ws,
        command="while ($true) { Write-Output ticking; Start-Sleep -Milliseconds 50 }",
        timeout=2.0,
    )
    elapsed = time.monotonic() - t0
    assert result.timed_out is True
    assert result.exit_code is None
    assert result.output_complete is False
    assert result.output.total_bytes > 0
    assert elapsed < 8


async def test_cancel_token_is_bounded_and_marks_incomplete(tmp_path):
    ws = make(tmp_path)
    cancel = CancelToken()
    task = asyncio.create_task(
        run_runner(
            PowerShellRunner(),
            ws,
            cancel=cancel,
            command="Write-Output started; Start-Sleep -Seconds 60",
            timeout=60.0,
        )
    )
    await asyncio.sleep(1.5)
    t0 = time.monotonic()
    cancel.cancel()
    result = await task
    assert time.monotonic() - t0 < RESOURCE_CLEANUP_SECONDS
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.exit_code is None
    assert result.output_complete is False
    assert "started" in result.output.text


async def test_task_cancel_without_token_cleans_up_and_next_run_is_clean(tmp_path):
    ws = make(tmp_path)
    runner = PowerShellRunner()
    task = asyncio.create_task(
        run_runner(
            runner,
            ws,
            command="Write-Output started; Start-Sleep -Seconds 60",
            timeout=60.0,
        )
    )
    await asyncio.sleep(0.5)  # 等 pwsh 真正启动
    t0 = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - t0 < RESOURCE_CLEANUP_SECONDS
    # CancelToken 未被置位: 取消路径仍必须自行完成有界清理, 不留下干扰下一次运行的资源
    result = await run_runner(runner, ws, command="Write-Output fresh", timeout=30.0)
    assert result.exit_code == 0
    assert result.output_complete is True
    assert result.output.text.strip() == "fresh"


async def test_timeout_kills_process_tree_and_reports_incomplete(tmp_path):
    ws = make(tmp_path)
    # 先落盘子进程 pid 再长睡: 短超时会在 pwsh 启动延迟内到达, 顺序反过来会拿不到证据
    command = (
        "$p = Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 60' -NoNewWindow -PassThru; "
        "$p.Id | Out-File -FilePath watch.pid -Encoding utf8; Start-Sleep -Seconds 60"
    )
    t0 = time.monotonic()
    result = await run_runner(PowerShellRunner(), ws, command=command, timeout=2.0)
    elapsed = time.monotonic() - t0
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.exit_code is None
    assert result.output_complete is False
    assert elapsed < 10
    pidfile = ws.root / "watch.pid"
    if not await _wait_for_file(pidfile):
        pytest.skip("pwsh startup slower than the timeout window on this host")
    pid = int(pidfile.read_text(encoding="utf-8-sig").strip())
    assert await wait_pid_gone(pid)


async def _wait_for_file(path: Path, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if path.exists():
            return True
        await asyncio.sleep(0.1)
    return path.exists()


async def test_nonzero_exit_code_is_preserved(tmp_path):
    ws = make(tmp_path)
    result = await run_runner(
        PowerShellRunner(), ws, command="Write-Output hi; exit 3", timeout=60.0
    )
    assert result.exit_code == 3
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.output_complete is True
    assert result.output.text.strip() == "hi"
