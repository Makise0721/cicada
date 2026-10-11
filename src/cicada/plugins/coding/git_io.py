"""inventory 与审阅共用的私有有界 Git IO；不使用 shell。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from cicada.core.cancel import CancelToken


class GitIOError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


async def read_git(
    root: Path, arguments: tuple[str, ...], cancel: CancelToken, budget: int,
    *, input_data: bytes | None = None, allowed_codes: tuple[int, ...] = (0,),
    preview_bytes: int | None = None,
) -> tuple[bytes, int, bool]:
    """累计 stdout 有硬上限；预览可截断但继续有界排空，取消等实际回收。"""
    cancel.throw_if_cancelled()
    environment = {k: v for k, v in os.environ.items() if k.upper() not in {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS",
        "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS",
    } and not k.upper().startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "--no-pager", "--literal-pathspecs", "-c", "core.fsmonitor=false",
            "-c", "diff.external=", "-c", "core.quotepath=false", "-C", str(root), *arguments,
            stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=environment,
        )
    except FileNotFoundError as exc:
        raise GitIOError("git_unavailable", "Git executable is unavailable") from exc
    except OSError as exc:
        raise GitIOError("git_launch_failed", "cannot launch Git") from exc

    async def read_stdout() -> tuple[bytes, bool]:
        output = bytearray()
        total = 0
        keep = budget if preview_bytes is None else min(budget, preview_bytes)
        while chunk := await process.stdout.read(4096):
            total += len(chunk)
            if total > budget:
                raise GitIOError("output_limit", "Git stdout exceeds the operation limit")
            output.extend(chunk[:max(0, keep - len(output))])
        return bytes(output), total > keep

    async def read_stderr() -> bytes:
        tail = b""
        while chunk := await process.stderr.read(4096):
            tail = (tail + chunk)[-4096:]
        return tail

    async def write_stdin() -> None:
        if input_data is not None:
            try:
                process.stdin.write(input_data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

    readers = [asyncio.create_task(read_stdout()), asyncio.create_task(read_stderr()),
               asyncio.create_task(process.wait()), asyncio.create_task(write_stdin())]
    operation = asyncio.gather(*readers)
    cancellation = asyncio.create_task(cancel.wait())
    try:
        done, _ = await asyncio.wait((operation, cancellation), return_when=asyncio.FIRST_COMPLETED)
        if cancellation in done:
            cancel.throw_if_cancelled()
        (output, truncated), stderr, returncode, _ = await operation
        if returncode not in allowed_codes:
            # stderr 可含仓库配置中的凭据/命令，不暴露给显示 DTO。
            raise GitIOError("git_failed", f"Git failed ({returncode})")
        return output, returncode, truncated
    finally:
        cancellation.cancel()
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        operation.cancel()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(operation, cancellation, *readers, return_exceptions=True)
        # 强引用保持到回收。调用方的监督可以隔离此任务，不能把超时当作 released。
        await process.wait()
