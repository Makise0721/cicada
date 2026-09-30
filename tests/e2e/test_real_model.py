"""跨层 live e2e: 真实 Ollama 模型经 CLI 子进程驱动真实 coding 工具完成文件任务.

Gate: Ollama 可达且默认模型在 /api/tags 中, 否则 skip (收集期一次性判定).
失败归因顺序: 先排查适配器, 再归因模型指令遵循能力 (9B 本地模型能力不在验收范围,
模型持续无法闭环时按 P2 计划 I2 的裁决出口交主 Agent).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from cicada.plugins.ollama import OllamaConfig

CONFIG = OllamaConfig()
SRC = Path(__file__).resolve().parents[2] / "src"
TIMEOUT = 240


def _gate_reason() -> str | None:
    try:
        with httpx.Client(timeout=2.0) as client:
            version = client.get(f"{CONFIG.base_url}/api/version")
            version.raise_for_status()
            tags = client.get(f"{CONFIG.base_url}/api/tags")
            tags.raise_for_status()
    except httpx.HTTPError as exc:
        return f"ollama unreachable at {CONFIG.base_url}: {exc}"
    names = {entry.get("name") for entry in tags.json().get("models", [])}
    if CONFIG.model not in names:
        return f"model {CONFIG.model!r} not present in /api/tags"
    return None


_GATE = _gate_reason()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_GATE is not None, reason=_GATE or ""),
]


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
        timeout=TIMEOUT,
    )


def test_real_model_creates_and_reads_file(tmp_path):
    prompt = (
        "请依次完成两步: "
        "1. 用 write 工具在工作区创建文件 hello_cicada.txt, 内容恰好为: hello cicada; "
        "2. 用 read 工具读取该文件确认内容. "
        "完成后用一句话汇报."
    )
    proc = run_cli("--workspace", str(tmp_path), prompt)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "finished: stop" in proc.stdout
    created = tmp_path / "hello_cicada.txt"
    assert created.exists(), proc.stdout
    assert "hello cicada" in created.read_text(encoding="utf-8")
