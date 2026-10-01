"""Coding 工作区: 路径策略 + 同文件变更队列 + 输出工件目录."""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

from cicada.runtime.plugin import PluginDefinition

T = TypeVar("T")


class WorkspaceError(RuntimeError):
    """工作区操作失败基类."""


class PathNotAllowedError(WorkspaceError):
    """写入目标在 workspace root 之外."""


@dataclass
class Workspace:
    """root 与 output_dir 在构造时完成 canonical 化."""

    root: Path
    output_dir: Path
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False, compare=False)
    _output_seq: "itertools.count[int]" = field(default_factory=itertools.count, repr=False, compare=False)

    @classmethod
    def create(cls, root: Path) -> Workspace:
        """canonical root; output_dir 默认 <root>/.cicada/outputs (创建目录)."""
        canonical_root = Path(os.path.realpath(Path(root).expanduser()))
        output_dir = canonical_root / ".cicada" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(root=canonical_root, output_dir=output_dir)

    def resolve(self, raw: str) -> Path:
        """解析合法 Windows 路径: ~ 展开、相对 root、规范化绝对路径; 不做存在性假设."""
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return Path(os.path.normpath(candidate))

    def resolve_within_root(self, raw: str) -> Path:
        """只读边界解析: canonical(realpath 解 junction/symlink, 不存在文件解析最近存在祖先)
        后以大小写不敏感的路径段比较判定必须在 root 内; 越界抛 PathNotAllowedError."""
        target = Path(os.path.realpath(self.resolve(raw)))
        if not self._within_root(target):
            raise PathNotAllowedError(
                f"path {raw!r} resolves to {target} outside workspace root {self.root}"
            )
        return target

    def resolve_for_write(self, raw: str) -> Path:
        """写入沿用同一 canonical 边界；此方法本身不创建文件."""
        return self.resolve_within_root(raw)

    async def mutate(self, path: Path, op: Callable[[], Awaitable[T]]) -> T:
        """同 canonical 路径串行执行 op, 不同路径并行.

        op 一旦开始不被取消抢占 (shield 后等待其完成才释放队列);
        等待队列的调用方被取消时放弃其位置, op 不会执行.
        """
        lock = self._locks.setdefault(self._queue_key(path), asyncio.Lock())
        async with lock:
            task = asyncio.ensure_future(op())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # 取消路径上 op 自身的异常不掩盖取消信号
                try:
                    await task
                except BaseException:
                    pass
                raise

    def new_output_file(self, label: str) -> Path:
        """在 output_dir 分配一个唯一文件路径 (label 净化后作文件名一部分)."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or "output"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.output_dir / f"{safe}-{stamp}-{next(self._output_seq):04d}-{uuid.uuid4().hex[:8]}.txt"

    def _within_root(self, path: Path) -> bool:
        root_parts = [part.casefold() for part in self.root.parts]
        path_parts = [part.casefold() for part in path.parts]
        return path_parts[: len(root_parts)] == root_parts

    @staticmethod
    def _queue_key(path: Path) -> str:
        # canonical 小写化字符串; realpath 对不存在文件回退到规范绝对路径
        return str(Path(os.path.realpath(path))).casefold()


def workspace_plugin(root: Path) -> PluginDefinition:
    """coding-workspace 插件: 提供 coding.workspace 能力."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide("coding.workspace", Workspace.create(root))

    return PluginDefinition(
        name="coding-workspace", setup=setup, provides=frozenset({"coding.workspace"})
    )
