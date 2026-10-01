"""进程运行器: PowerShell 执行、有界输出、流式工件、进程树终止与有界收尾."""

from __future__ import annotations

import asyncio
import codecs
import ctypes
import os
import shutil
import sys
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.runtime.plugin import PluginContext, PluginDefinition

TAIL_MAX_LINES = 2000
TAIL_MAX_BYTES = 50 * 1024
# 可见 tail 预算留出状态行/footer/工件路径的余量, 使工具层正常不需要二次裁剪
TAIL_BUDGET_BYTES = TAIL_MAX_BYTES - 1024
# 未换行内容只保留有界前缀, 保证无换行的巨量输出也不扩张内存
PENDING_MAX_BYTES = 8 * 1024
READ_CHUNK_BYTES = 65536
ARTIFACT_MAX_BYTES = 16 * 1024 * 1024  # 工件实际存储的 UTF-8 上限
GRACE_SECONDS = 0.1  # 管道空闲 grace
EOF_GRACE_SECONDS = 0.2  # 父进程退出后等待管道 EOF 的有限宽限
RESOURCE_CLEANUP_SECONDS = 5.0  # 取消/异常后额外资源清理总预算

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
    """有界输出视图: tail 窗口 + 统计 + 完整输出工件的完整性事实.

    artifact_truncated 与 artifact_error 是独立事实, 默认值兼容旧构造:
    没有工件时两者都表示"未触及上限/未发生存储失败"。
    """

    text: str
    truncated: bool
    total_bytes: int
    total_lines: int
    full_output_path: Path | None
    artifact_truncated: bool = False
    artifact_error: str | None = None


@dataclass(frozen=True)
class ProcessResult:
    """exit_code/timed_out/cancelled 是执行事实; output_complete 是采集完整性事实."""

    exit_code: int | None
    timed_out: bool
    cancelled: bool
    output: BoundedText
    output_complete: bool = True


class _OutputCollector:
    """流式 UTF-8 解码 + 有界 tail 窗口 + 持续写工件.

    内存只保留 fixed chunk、有限 tail 行窗口和有限未完成行前端; 全量输出进入工件,
    工件达到 ARTIFACT_MAX_BYTES 后停写但继续计数并丢弃, 不扩张内存。
    """

    def __init__(self, output_dir: Path) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""
        self._pending_omitted = 0
        self._lines: deque[str] = deque()
        self._tail_bytes = 0
        self._omitted_bytes = 0
        self.artifact_path: Path | None = None
        self.artifact_truncated = False
        self.artifact_error: str | None = None
        self._artifact_bytes = 0
        self._artifact_file = None
        self._finished = False
        self.total_bytes = 0
        self.total_lines = 0
        self.truncated = False
        self.eof = True
        self.output_dir = output_dir

    def feed(self, data: bytes) -> None:
        self.total_bytes += len(data)
        self._append(self._decoder.decode(data))

    def finish(self) -> None:
        """输入结束: 刷出解码器余量与未完成行, 关闭工件.

        幂等: 正常返回路径已收尾时, 取消/异常的 finally 兜底不会二次提交,
        但确实没有收尾过的句柄一定在这里 flush/close, 不会仅丢引用。
        """
        if self._finished:
            return
        self._finished = True
        self._append(self._decoder.decode(b"", True))
        if self._pending:
            self._add_line(self._pending)
            self._pending = ""
        self._close_artifact()

    @property
    def text(self) -> str:
        return "\n".join(self._lines)

    @property
    def tail_bytes(self) -> int:
        return self._tail_bytes

    @property
    def omitted_bytes(self) -> int:
        return self._omitted_bytes

    @property
    def artifact_bytes(self) -> int:
        return self._artifact_bytes

    def _append(self, text: str) -> None:
        if not text:
            return
        self._write_artifact(text)
        if self._pending:
            text = self._pending + text
            self._pending = ""
        start = 0
        while True:
            newline = text.find("\n", start)
            if newline < 0:
                line = text[start:]
                if len(line.encode("utf-8")) <= PENDING_MAX_BYTES:
                    self._pending = line
                else:
                    # 未换行内容本身超界: 只留有限前缀, 其余计入省略 (完整内容仍在工件里)
                    head = _utf8_prefix(line, PENDING_MAX_BYTES)
                    self._pending = head
                    self._pending_omitted += len(line.encode("utf-8")) - len(
                        head.encode("utf-8")
                    )
                    self.truncated = True
                return
            self._add_line(text[start:newline])
            start = newline + 1

    def _add_line(self, line: str) -> None:
        if self._pending_omitted:
            line = f"…[{self._pending_omitted} bytes omitted]…{line}"
            self._omitted_bytes += self._pending_omitted
            self._pending_omitted = 0
        if line.endswith("\r"):  # Windows 管道换行统一为 \n 视图
            line = line[:-1]
        self.total_lines += 1
        self._lines.append(line)
        self._tail_bytes += len(line.encode("utf-8")) + 1
        while len(self._lines) > TAIL_MAX_LINES or (
            self._tail_bytes > TAIL_BUDGET_BYTES and len(self._lines) > 1
        ):
            dropped = self._lines.popleft()
            size = len(dropped.encode("utf-8")) + 1
            self._tail_bytes -= size
            self._omitted_bytes += size
            self.truncated = True
        if self._tail_bytes > TAIL_BUDGET_BYTES:
            # 单行自身超预算 (多字节字符也按字节算): 只保留有界前缀并说明省略。
            # 先按字符估算、再用精确字节上限收口, 因此这里只执行一次, 不会反复裁剪。
            room = TAIL_BUDGET_BYTES - 1  # 行末换行计在 tail 预算内
            line = self._lines[0]
            prefix = _utf8_prefix(line, max(room - 64, 1))[: max(room - 64, 1)]
            omitted = len(line.encode("utf-8")) - len(prefix.encode("utf-8"))
            marker = _omission_marker(omitted)
            while len(marker.encode("utf-8")) + len(prefix.encode("utf-8")) > room and prefix:
                prefix = prefix[: len(prefix) - 1]
            self._lines[0] = f"{marker}{prefix}"
            self._omitted_bytes += omitted
            self.truncated = True
            self._recount_tail()

    def _recount_tail(self) -> None:
        self._tail_bytes = sum(len(line.encode("utf-8")) + 1 for line in self._lines)

    def _write_artifact(self, text: str) -> None:
        """流式写工件; 达到 16MiB 后停写并标记, 不中断管道 drain.

        写入按实际存储的 UTF-8 字节计; 单次写入超出剩余空间时按字符边界切开,
        因此工件内容始终是合法 UTF-8 前缀, 不产生半个多字节字符。
        """
        if self.artifact_truncated or self.artifact_error is not None:
            return
        if not self._ensure_artifact():
            return
        room = ARTIFACT_MAX_BYTES - self._artifact_bytes
        try:
            if len(text.encode("utf-8")) > room:
                piece = _utf8_prefix(text, room)
                if piece:
                    data = piece.encode("utf-8")
                    self._artifact_file.write(data)
                    self._artifact_bytes += len(data)
                self.artifact_truncated = True
                return
            data = text.encode("utf-8")
            self._artifact_file.write(data)
            self._artifact_bytes += len(data)
        except OSError as exc:
            self.artifact_error = f"cannot write output artifact: {exc}"

    def _ensure_artifact(self) -> bool:
        if self._artifact_file is not None:
            return True
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.artifact_path = self.output_dir / (
                f"powershell-output-{time.strftime('%Y%m%d-%H%M%S')}"
                f"-{os.getpid()}-{time.monotonic_ns() % 1_000_000_000:09d}.txt"
            )
            self._artifact_file = open(self.artifact_path, "wb")
        except OSError as exc:
            self.artifact_error = f"cannot open output artifact: {exc}"
            return False
        return True

    def _close_artifact(self) -> None:
        if self._artifact_file is None:
            return
        try:
            self._artifact_file.flush()
            self._artifact_file.close()
        except OSError as exc:
            self.artifact_error = self.artifact_error or f"cannot finalize output artifact: {exc}"
        finally:
            self._artifact_file = None


def _utf8_prefix(text: str, max_bytes: int) -> str:
    """不超过 max_bytes 的最长字符前缀; 只切在字符边界上."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(text[:middle].encode("utf-8")) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _omission_marker(omitted_bytes: int) -> str:
    return f"…[{omitted_bytes} bytes omitted]…"


def _reader_incomplete(task: asyncio.Task) -> bool:
    """reader 任务是否没有正常读到 EOF.

    正常 EOF、读取异常、被取消是三种不同事实: 只有正常 EOF 才算采集完整,
    异常与取消都必须取回 task 结果识别, 不能只看"任务已 done"。
    """
    if not task.done() or task.cancelled():
        return True
    return task.exception() is not None


def _consume_task_result(task: asyncio.Task) -> None:
    """取回被放弃任务的异常, 避免 "exception was never retrieved" 噪声."""
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:  # pragma: no cover - 只在任务状态异常时兜底
        pass


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
    """argv = [pwsh, -NoProfile, -NonInteractive, -ExecutionPolicy, Bypass, -Command, <utf8 prefix + command>].

    原 deadline 覆盖启动后的等待与 drain: 父进程退出后管道未 EOF 时只给有限宽限,
    不会被子进程输出或持管道无限延长。取消/异常后的额外资源清理同样有界。
    """

    def __init__(
        self,
        *,
        eof_grace_s: float = EOF_GRACE_SECONDS,
        cleanup_budget_s: float = RESOURCE_CLEANUP_SECONDS,
    ) -> None:
        self.eof_grace_s = eof_grace_s
        self.cleanup_budget_s = cleanup_budget_s

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
        collector = _OutputCollector(output_dir)
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
        exited = False
        # 超时/取消先花掉的清理预算与 finally 的收尾共用同一个绝对截止时间
        cleanup_deadline: float | None = None
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                if cancel.cancelled:
                    cancelled = True
                    break
                if await watcher.wait_once(min(0.1, remaining)):
                    exited = True
                    break
            # 已确证真实退出就保留真实退出码, 不因同轮 deadline 到点丢掉它
            exit_code = watcher.exit_code if exited else None
            if timed_out or cancelled:
                cleanup_deadline = loop.time() + self.cleanup_budget_s
                await self._bounded(
                    self._kill_tree(proc.pid, cleanup_deadline), cleanup_deadline
                )
            # 注意: 本体被外部 cancel 时这里以上都不会执行, 由 finally 兜底清理。
            drained = await self._drain_readers(proc, readers, exited, deadline)
            collector.finish()
            if exited and not drained:
                # 命令已退出但输出管道没采到 EOF: 属于未能在 deadline 内收尾
                timed_out = True
            return ProcessResult(
                exit_code=exit_code,
                timed_out=timed_out,
                cancelled=cancelled,
                output=BoundedText(
                    text=collector.text,
                    truncated=collector.truncated or bool(collector.artifact_truncated),
                    total_bytes=collector.total_bytes,
                    total_lines=collector.total_lines,
                    full_output_path=collector.artifact_path,
                    artifact_truncated=collector.artifact_truncated,
                    artifact_error=collector.artifact_error,
                ),
                output_complete=(
                    exited and drained and collector.eof and not cancelled and not timed_out
                ),
            )
        finally:
            await self._finish_resource_cleanup(
                proc, readers, collector, watcher, cleanup_deadline
            )

    async def _pump(self, stream: asyncio.StreamReader, collector: _OutputCollector) -> None:
        """固定大小读取; 每次 read 后把数据交给有界 collector, 不保留整段输出."""
        while True:
            chunk = await stream.read(READ_CHUNK_BYTES)
            if not chunk:
                return
            collector.feed(chunk)

    async def _drain_readers(
        self,
        proc: asyncio.subprocess.Process,
        readers: list[asyncio.Task],
        exited: bool,
        deadline: float,
    ) -> bool:
        """收集已到达输出; 返回两个读者是否都正常读到 EOF。

        父进程已确证退出时只再给有限宽限, 且**不超过原命令 deadline**; 未退出
        (超时/取消)时按原 deadline 立即收口。reader 异常或被取消都不是 EOF:
        必须取回 task 结果并归一为不完整, 退出码 0 不能覆盖采集失败。
        """
        loop = asyncio.get_running_loop()
        drain_deadline = min(deadline, loop.time() + self.eof_grace_s) if exited else deadline
        while True:
            pending = [task for task in readers if not task.done()]
            if not pending:
                break
            if any(_reader_incomplete(task) for task in readers if task.done()):
                break  # 已有读者失败: 不必再等另一个
            remaining = drain_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.wait(pending, timeout=min(GRACE_SECONDS, remaining))
        for task in readers:
            if not task.done():
                task.cancel()
        # 取回全部 task 结果: 只有两个正常 EOF 才算采集完整
        await asyncio.gather(*readers, return_exceptions=True)
        complete = not any(_reader_incomplete(task) for task in readers)
        if not complete:
            self._close_pipes(proc)
        return complete

    @staticmethod
    def _close_pipes(proc: asyncio.subprocess.Process) -> None:
        # 放弃等待 close 后显式关闭管道传输, 避免悬挂的 overlapped 读在 GC 时告警;
        # get_pipe_transport 是 asyncio 私有接口, 变动时放弃显式关闭, 依赖 GC 兜底
        for fd in (1, 2):
            try:
                pipe = proc._transport.get_pipe_transport(fd)
            except AttributeError:
                continue
            if pipe is not None:
                pipe.close()

    async def _finish_resource_cleanup(
        self,
        proc: asyncio.subprocess.Process,
        readers: list[asyncio.Task],
        collector: _OutputCollector,
        watcher: _ProcessWatcher,
        deadline: float | None = None,
    ) -> None:
        """有界收尾: 正常路径是廉价 no-op; 取消/异常路径尝试终止并关闭资源后不吞掉异常.

        一个绝对清理总预算覆盖 watcher/readers/pipe/工件关闭以及 taskkill 的启动与等待
        和 proc.wait; 故意不返回的 kill/read 不能突破它。本地管道与工件句柄在任何 await
        之前同步关闭, 因此预算耗尽或被再次取消时也不会留下未释放的句柄。
        不把 best-effort taskkill 当整个进程树已退出的证明, 只保证自身资源被关闭。
        """
        loop = asyncio.get_running_loop()
        stop = deadline if deadline is not None else loop.time() + self.cleanup_budget_s
        if any(not task.done() for task in readers):
            # 任何未完成的读者的取消/放弃都意味着没有采到 EOF, 先锁存该事实再清理
            collector.eof = False
        for task in readers:
            if not task.done():
                task.cancel()
        # 先同步关闭本地资源 (读者管道 + 工件句柄) 再等待: 即使预算用尽、或本协程
        # 再次被取消, 自身已持资源也已释放, 工件句柄不会只丢引用而不 flush/close。
        self._close_pipes(proc)
        collector.finish()
        await self._wait_readers(readers, stop)
        await self._bounded(watcher.close(), stop)
        if proc.returncode is not None:
            return
        # 父进程已退出时对 PID 的 taskkill 可能无对象, 属 best-effort
        await self._bounded(self._kill_tree(proc.pid, stop), stop)
        await self._bounded(proc.wait(), stop)

    @staticmethod
    async def _wait_readers(readers: list[asyncio.Task], stop: float) -> None:
        """在预算内等读者收尾; 预算用尽就放弃等待, 但仍取回其异常."""
        loop = asyncio.get_running_loop()
        pending = [task for task in readers if not task.done()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=max(stop - loop.time(), 0.0))
        for task in readers:
            if not task.done():
                task.add_done_callback(_consume_task_result)

    @staticmethod
    async def _bounded(awaitable, stop: float) -> None:
        """在清理总预算内等待一个协程; 预算用尽就放弃等待, 由调用方保守处理终结事实.

        这是清理路径上每个 await 的硬上限: 未返回的 taskkill / proc.wait / 句柄关闭
        都不能把取消或异常拖过预算; 本地管道与工件句柄已在调用它之前同步关闭。
        """
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(awaitable, max(stop - loop.time(), 0.001))
        except (TimeoutError, OSError):
            pass

    @staticmethod
    async def _kill_tree(pid: int, stop: float) -> None:
        """best-effort 终止已知进程树; 启动与等待都在清理总预算内, 不无限等待 taskkill."""
        loop = asyncio.get_running_loop()
        remaining = stop - loop.time()
        if remaining <= 0:
            return
        killer: asyncio.subprocess.Process | None = None
        try:
            async with asyncio.timeout(remaining):
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
        except TimeoutError:
            # 预算内没返回: 放弃等待并终止 taskkill 自身, 由调用方保守处理终结事实
            if killer is not None:
                killer.kill()
        except OSError:
            pass  # 进程已退出时 taskkill 报错可忽略


def process_plugin() -> PluginDefinition:
    """coding-process 插件: 提供 coding.process 能力."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide("coding.process", PowerShellRunner())

    return PluginDefinition(
        name="coding-process", setup=setup, provides=frozenset({"coding.process"})
    )
