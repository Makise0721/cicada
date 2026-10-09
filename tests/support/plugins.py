"""D5 (plugins) 域共享小工具: 真实 git 调用与 Windows 进程存活探针."""

from __future__ import annotations

import asyncio
import subprocess


def git(root, *args, input=None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True, check=True
    ).stdout


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
