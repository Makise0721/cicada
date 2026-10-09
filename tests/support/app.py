"""共享应用夹具: 隔离 Git 工作区 / PromptInfo / fake 剧本 / 启动身份。

原分散在 test_application / test_conversation_view / test_persistence_application /
test_compact_command 的重复定义, 此处为单一来源; 各文件以 `... as _xxx` 转引保持本地命名。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from cicada.persistence.contracts import LaunchIdentity
from cicada.prompting import PromptInfo


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True
    ).stdout


def repo(tmp_path: Path) -> Path:
    """项目内隔离 Git 工作区: init + 一个基线提交。"""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "baseline")
    return root


def prompt_info(root: Path, *, version: str = "test") -> PromptInfo:
    return PromptInfo(
        version=version,
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=hashlib.sha256(version.encode()).hexdigest(),
    )


def script(*entries: dict) -> list:
    """经 parse_script 构建确定性 fake 剧本 (JSON dict -> StreamEvent 列表)。"""
    from cicada.__main__ import parse_script

    return parse_script(json.dumps(list(entries)))


def identity(checks_signature: str | None = None) -> LaunchIdentity:
    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=checks_signature, sources={},
    )
