"""用户级运行状态的位置；工作区路径只用于确定隔离命名空间。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def workspace_state_dir(workspace: Path, *, user_dir: Path | None = None) -> Path:
    """为规范工作区路径生成稳定的用户级状态目录，不创建文件。"""
    canonical = os.path.realpath(Path(workspace).expanduser())
    identity = os.path.normcase(canonical).encode("utf-8")
    name = hashlib.sha256(identity).hexdigest()
    root = user_dir if user_dir is not None else Path.home() / ".cicada"
    return Path(os.path.realpath(root)) / "workspaces" / name


def session_directory(workspace: Path, *, user_dir: Path | None = None) -> Path:
    return workspace_state_dir(workspace, user_dir=user_dir) / "sessions"


def legacy_session_directory(workspace: Path) -> Path:
    return Path(workspace) / ".cicada" / "sessions"
