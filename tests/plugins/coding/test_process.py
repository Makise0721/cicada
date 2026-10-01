"""coding.process 运行器与有界采集的聚焦验证 (03: 严格有界、流式工件、收尾与取消)."""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

import cicada.plugins.coding.process as process_module
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


async def wait_until(predicate, attempts: int = 200, interval: float = 0.05) -> bool:
    """等条件成立, 避免与 pwsh 启动/首段输出竞态 (负载高时启动会明显变慢)."""
    for _ in range(attempts):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


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


# --- 03 修复回归: reader 完成事实 / 清理总预算 / 取消时工件句柄 ---
#
# 受控替身只替换 runner 的子进程与进程句柄 seam, 生产代码路径保持不变;
# 真实 Windows runner 的用例仍走上面的真实 pwsh。


class _FakeStream:
    """受控管道替身: 可先给数据再抛错, 或永远不返回 (不结束的 read)."""

    def __init__(self, chunks=(), error=None, hang=False):
        self._chunks = list(chunks)
        self._error = error
        self._hang = hang

    async def read(self, size):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error is not None:
            raise self._error
        if self._hang:
            await asyncio.Event().wait()
        return b""


class _FakeTransport:
    def get_pipe_transport(self, fd):
        return None


class _FakeProcess:
    def __init__(self, stdout, stderr, returncode, pid=999999):
        self.pid = pid
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self._transport = _FakeTransport()

    async def wait(self):
        return self.returncode


class _FakeWatcher:
    def __init__(self, pid, proc):
        self._proc = proc
        self.closed = False

    async def wait_once(self, seconds):
        await asyncio.sleep(min(seconds, 0.02))
        return self._proc.returncode is not None

    @property
    def exit_code(self):
        return self._proc.returncode

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_os(monkeypatch):
    """把 runner 的子进程/进程句柄 seam 换成受控替身, 由用例填入 proc."""
    state: dict = {}

    async def create(*args, **kwargs):
        return state["proc"]

    monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(process_module, "_ProcessWatcher", _FakeWatcher)
    return state


async def test_reader_read_error_never_reports_output_complete(tmp_path, fake_os):
    """S2: 管道读失败必须归为采集不完整, 命令 exit0 不能覆盖它."""
    ws = make(tmp_path)
    fake_os["proc"] = _FakeProcess(
        stdout=_FakeStream([b"before-error\n"], error=OSError("simulated pipe read failure")),
        stderr=_FakeStream(),
        returncode=0,
    )
    result = await run_runner(PowerShellRunner(), ws, command="ignored", timeout=5.0)
    assert result.exit_code == 0
    assert result.output_complete is False
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.output.text == "before-error"  # 已到达输出仍然保留


async def test_reader_without_eof_is_not_output_complete(tmp_path, fake_os):
    """S2: 读者没有读到 EOF (随后被收尾取消) 同样不是完整采集."""
    ws = make(tmp_path)
    fake_os["proc"] = _FakeProcess(
        stdout=_FakeStream([b"partial\n"], hang=True),
        stderr=_FakeStream(),
        returncode=0,
    )
    result = await run_runner(
        PowerShellRunner(eof_grace_s=0.05), ws, command="ignored", timeout=5.0
    )
    assert result.exit_code == 0
    assert result.output_complete is False
    assert result.timed_out is True
    assert result.output.text == "partial"


async def test_cleanup_budget_bounds_hanging_kill_and_readers(tmp_path, monkeypatch):
    """S1/F06: 不返回的 kill 与不结束的 read 都不能突破清理总预算."""
    ws = make(tmp_path)
    runner = PowerShellRunner(cleanup_budget_s=0.05)

    async def hanging_kill(pid, stop):
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_kill_tree", hanging_kill)
    proc = _FakeProcess(_FakeStream(hang=True), _FakeStream(hang=True), returncode=None)
    collector = _OutputCollector(ws.output_dir)
    readers = [
        asyncio.ensure_future(runner._pump(proc.stdout, collector)),
        asyncio.ensure_future(runner._pump(proc.stderr, collector)),
    ]
    await asyncio.sleep(0)
    watcher = _FakeWatcher(proc.pid, proc)
    started = time.monotonic()
    await runner._finish_resource_cleanup(proc, readers, collector, watcher, None)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert all(task.done() for task in readers)
    assert collector.eof is False
    assert watcher.closed is True


async def test_kill_tree_abandons_taskkill_that_overruns_the_budget(monkeypatch):
    """S1/F06: taskkill 自身的等待也受预算约束, 超预算时终止它而不是无限等待."""
    killed: list[bool] = []

    class _Killer:
        async def wait(self):
            await asyncio.Event().wait()

        def kill(self):
            killed.append(True)

    async def create(*args, **kwargs):
        return _Killer()

    monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", create)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await PowerShellRunner._kill_tree(1234, started + 0.05)
    assert loop.time() - started < 1.0
    assert killed == [True]


async def test_task_cancel_flushes_and_closes_the_artifact_handle(tmp_path, monkeypatch):
    """S4: Task.cancel 后 finally 必须 flush/close 已创建的流式工件句柄."""
    ws = make(tmp_path)
    captured: list[_OutputCollector] = []
    base_collector = _OutputCollector

    class Capturing(base_collector):
        def __init__(self, output_dir):
            super().__init__(output_dir)
            captured.append(self)

    monkeypatch.setattr(process_module, "_OutputCollector", Capturing)
    runner = PowerShellRunner()
    task = asyncio.create_task(
        run_runner(
            runner,
            ws,
            command="Write-Output started; Start-Sleep -Seconds 60",
            timeout=60.0,
        )
    )
    # 等 pwsh 真正产出首段输出再取消: 否则只测到"尚未创建工件", 不是取消收尾
    fed = await wait_until(lambda: bool(captured) and captured[-1]._artifact_bytes > 0)
    assert fed, "pwsh did not produce collectable output in time"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    collector = captured[-1]
    assert collector._artifact_file is None  # 句柄已关闭, 不是只丢引用
    assert collector.artifact_error is None
    assert collector.artifact_path is not None
    assert b"started" in collector.artifact_path.read_bytes()  # flush 后内容可读


async def test_timeout_cannot_be_extended_by_a_hanging_taskkill(tmp_path, monkeypatch):
    """S1/F06: 公开 run() 在 taskkill 不返回时仍按预算返回并保守终结."""
    ws = make(tmp_path)
    runner = PowerShellRunner(cleanup_budget_s=0.1)
    attempted: list[int] = []

    async def hanging_kill(pid, stop):
        attempted.append(pid)
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_kill_tree", hanging_kill)
    started = time.monotonic()
    try:
        result = await run_runner(runner, ws, command="Start-Sleep -Seconds 20", timeout=0.8)
    finally:
        # kill 被替换为不返回: 用真实 taskkill 清掉遗留 pwsh, 不留孤儿进程
        for pid in set(attempted):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True
            )
    elapsed = time.monotonic() - started
    assert attempted, "超时路径必须尝试终止已知进程树"
    assert result.timed_out is True
    assert result.output_complete is False
    assert elapsed < 3.0
    for pid in set(attempted):
        assert await wait_pid_gone(pid)


async def test_eof_grace_cannot_extend_past_the_original_deadline(tmp_path):
    """F06: 父进程退出后的 EOF 宽限不超过原命令 deadline."""
    ws = make(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]
    command = (
        "Start-Process pwsh -ArgumentList '-NoProfile','-Command',"
        "'Start-Sleep -Seconds 8' -NoNewWindow "
        f"-WorkingDirectory '{repo_root}' -PassThru | Out-Null; Write-Output parent-exit"
    )
    started = time.monotonic()
    result = await run_runner(
        PowerShellRunner(eof_grace_s=5.0), ws, command=command, timeout=3.0
    )
    elapsed = time.monotonic() - started
    if result.exit_code is None:
        pytest.skip("pwsh startup slower than the 3s deadline on this host")
    assert result.exit_code == 0
    assert result.output_complete is False
    assert result.timed_out is True
    # 5s 的 EOF 宽限不得把收尾拖过原 deadline (修复前该路径实测约 7s)
    assert elapsed < 5.0
