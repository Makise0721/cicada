"""进程运行器: PowerShell 执行、有界输出、进程树终止."""

from __future__ import annotations

import asyncio
import codecs
import ctypes
import os
import shutil
import sys
import time
import uuid
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.runtime.plugin import PluginContext, PluginDefinition

TAIL_MAX_LINES = 2000
TAIL_MAX_BYTES = 50 * 1024
GRACE_SECONDS = 0.1

UTF8_PREFIX = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "$OutputEncoding = [System.Text.Encoding]::UTF8;"
)


def resolve_pwsh() -> str:
    """定位 pwsh: PATH 优先, 回退到 PowerShell 7 标准安装位置 (PATH 不含它的开发环境)."""
    found = shutil.which("pwsh")
    if found:
        return found
    candidates = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "PowerShell" / "7" / "pwsh.exe",
        Path(os.environ.get("LocalAppData", "")) / "Programs" / "PowerShell" / "7" / "pwsh.exe"
        if os.environ.get("LocalAppData")
        else None,
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return "pwsh"  # 保留原名, 让 create_subprocess_exec 报 FileNotFoundError


@dataclass(frozen=True)
class BoundedText:
    """有界输出视图: tail 窗口 + 统计 + 完整输出位置."""

    text: str
    truncated: bool
    total_bytes: int
    total_lines: int
    full_output_path: Path | None


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    output: BoundedText


class _OutputCollector:
    """流式 UTF-8 解码 + tail 窗口维护 + 全量留存."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""
        self._parts: list[str] = []
        self._lines: deque[str] = deque()
        self._tail_bytes = 0
        self.total_bytes = 0
        self.total_lines = 0
        self.truncated = False

    def feed(self, data: bytes) -> None:
        self.total_bytes += len(data)
        text = self._decoder.decode(data)
        if not text:
            return
        self._parts.append(text)
        self._pending += text
        self._flush_complete_lines()

    def finalize(self) -> tuple[str, str]:
        """返回 (tail 文本, 全量文本); 末尾无换行的部分行计为一行, 空则不计."""
        rest = self._decoder.decode(b"", True)
        if rest:
            self._parts.append(rest)
            self._pending += rest
            self._flush_complete_lines()
        if self._pending:
            self._add_line(self._pending)
            self._pending = ""
        return "\n".join(self._lines), "".join(self._parts)

    def _flush_complete_lines(self) -> None:
        while True:
            nl = self._pending.find("\n")
            if nl < 0:
                return
            line = self._pending[:nl]
            self._pending = self._pending[nl + 1 :]
            self._add_line(line)

    def _add_line(self, line: str) -> None:
        if line.endswith("\r"):  # Windows 管道换行统一为 \n 视图
            line = line[:-1]
        self.total_lines += 1
        self._lines.append(line)
        self._tail_bytes += len(line.encode("utf-8")) + 1
        while len(self._lines) > TAIL_MAX_LINES or (
            self._tail_bytes > TAIL_MAX_BYTES and len(self._lines) > 1
        ):
            dropped = self._lines.popleft()
            self._tail_bytes -= len(dropped.encode("utf-8")) + 1
            self.truncated = True


if sys.platform == "win32":
    _SYNCHRONIZE = 0x00100000
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    # WaitForSingleObject 需要 SYNCHRONIZE; GetExitCodeProcess 需要 QUERY_LIMITED
    _WATCH_RIGHTS = _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION
    _WAIT_TIMEOUT = 0x102
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class _ProcessWatcher:
    """进程真实退出检测与退出码获取.

    Windows: SYNCHRONIZE 句柄 + WaitForSingleObject (executor 线程内等待).
    其他平台 (本项目第一版不支持, 仅保持可导入): 回退到 proc.poll().
    """

    def __init__(self, pid: int, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._code: int | None = None
        self._handle: int | None = None
        if sys.platform == "win32":
            self._handle = _kernel32.OpenProcess(_WATCH_RIGHTS, False, pid) or None

    async def wait_once(self, seconds: float) -> bool:
        """等待至多 seconds 秒; 返回进程是否已退出."""
        if self._handle is None:
            # 无句柄 (非 Windows 或 OpenProcess 失败): 退回 transport 的 returncode
            # 轮询, 避免把正常退出误判为 exit_code=None
            await asyncio.sleep(seconds)
            if self._proc.returncode is not None:
                self._code = self._proc.returncode
            return self._proc.returncode is not None
        loop = asyncio.get_running_loop()
        rc = await loop.run_in_executor(
            None, _kernel32.WaitForSingleObject, self._handle, max(int(seconds * 1000), 1)
        )
        if rc != 0:
            return False
        code = wintypes.DWORD(0xFFFFFFFF)
        if _kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            self._code = code.value
        return True

    @property
    def exit_code(self) -> int | None:
        return self._code

    async def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _kernel32.CloseHandle, handle)


class PowerShellRunner:
    """argv = [pwsh, -NoProfile, -NonInteractive, -ExecutionPolicy, Bypass, -Command, <utf8 prefix + command>]."""

    async def run(
        self,
        *,
        command: str,
        cwd: Path,
        timeout: float,
        cancel: CancelToken,
        output_dir: Path,
    ) -> ProcessResult:
        argv = [
            resolve_pwsh(),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            UTF8_PREFIX + command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        collector = _OutputCollector()
        readers = [
            asyncio.ensure_future(self._pump(proc.stdout, collector)),
            asyncio.ensure_future(self._pump(proc.stderr, collector)),
        ]

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        # Windows proactor 的 proc.wait() 要等管道 EOF 才返回, 孙进程持有继承管道时会挂死;
        # 真实退出检测改用进程句柄等待, 退出码经 GetExitCodeProcess 获取.
        watcher = _ProcessWatcher(proc.pid, proc)
        timed_out = False
        cancelled = False
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                if cancel.cancelled:
                    cancelled = True
                    break
                if await watcher.wait_once(0.1):
                    break
        finally:
            await watcher.close()

        if timed_out or cancelled:
            await self._kill_tree(proc.pid)

        exit_code = None if (timed_out or cancelled) else watcher.exit_code
        await self._drain_readers(proc, readers, collector)

        text, full_text = collector.finalize()
        full_output_path = (
            _write_full_output(output_dir, full_text) if collector.truncated else None
        )
        return ProcessResult(
            exit_code=exit_code,
            timed_out=timed_out,
            cancelled=cancelled,
            output=BoundedText(
                text=text,
                truncated=collector.truncated,
                total_bytes=collector.total_bytes,
                total_lines=collector.total_lines,
                full_output_path=full_output_path,
            ),
        )

    async def _pump(self, stream: asyncio.StreamReader, collector: _OutputCollector) -> None:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            collector.feed(chunk)

    async def _drain_readers(
        self,
        proc: asyncio.subprocess.Process,
        readers: list[asyncio.Task],
        collector: _OutputCollector,
    ) -> None:
        """进程退出后收集输出; 孙进程持有继承管道时, 输出空闲超过 grace 即放弃等待 close."""
        marker = collector.total_bytes
        while True:
            pending = {task for task in readers if not task.done()}
            if not pending:
                return
            _, still = await asyncio.wait(pending, timeout=GRACE_SECONDS)
            if not still:
                return
            if collector.total_bytes > marker:
                marker = collector.total_bytes  # 新输出重置空闲计时
                continue
            for task in still:
                task.cancel()
            await asyncio.gather(*still, return_exceptions=True)
            # 放弃等待 close 后显式关闭管道传输, 避免悬挂的 overlapped 读在 GC 时告警;
            # get_pipe_transport 是 asyncio 私有接口, 变动时放弃显式关闭, 依赖 GC 兜底
            for fd in (1, 2):
                try:
                    pipe = proc._transport.get_pipe_transport(fd)
                except AttributeError:
                    continue
                if pipe is not None:
                    pipe.close()
            return

    @staticmethod
    async def _kill_tree(pid: int) -> None:
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        except OSError:
            pass  # 进程已退出时 taskkill 报错可忽略


def _write_full_output(output_dir: Path, text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"powershell-output-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.txt"
    )
    path.write_text(text, encoding="utf-8", newline="")
    return path


def process_plugin() -> PluginDefinition:
    """coding-process 插件: 提供 coding.process 能力."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide("coding.process", PowerShellRunner())

    return PluginDefinition(
        name="coding-process", setup=setup, provides=frozenset({"coding.process"})
    )
