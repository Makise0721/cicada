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
    from cicada.plugins.coding.file_access import CaptureSink, FileAccess
    from cicada.runtime.plugin import PluginContext

from cicada.runtime.plugin import PluginDefinition

T = TypeVar("T")


class WorkspaceError(RuntimeError):
    """工作区操作失败基类."""


class PathNotAllowedError(WorkspaceError):
    """写入目标在 workspace root 之外."""


def _note_handled_cancel(on_caller_cancel: Callable[[], None] | None) -> None:
    """已决意吞掉协程取消换真实结果: 通知回调方并理平取消记账."""
    if on_caller_cancel is not None:
        on_caller_cancel()
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        task.uncancel()


@dataclass
class Workspace:
    """root 与 output_dir 在构造时完成 canonical 化."""

    root: Path
    output_dir: Path
    data_dir: Path = field(init=False)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False, compare=False)
    _output_seq: "itertools.count[int]" = field(default_factory=itertools.count, repr=False, compare=False)
    files: "FileAccess" = field(init=False, repr=False, compare=False)
    """进程内文件访问 Module: 版本依据/队列/暂存/发布 (file_access.py)."""
    _capture_sink: "CaptureSink | None" = field(default=None, repr=False, compare=False)
    """可选还原点捕获汇 (request-snapshot-revert C1); 默认 None = 行为与既有一致."""

    def __post_init__(self) -> None:
        # 延迟导入打破 workspace -> file_access -> workspace 的模块环
        from cicada.plugins.coding.file_access import FileAccess

        self.data_dir = self.output_dir.parent
        self.files = FileAccess(self)

    @classmethod
    def create(cls, root: Path, *, data_dir: Path | None = None) -> Workspace:
        """canonical root; data_dir 由组合层指定，创建输出目录。"""
        canonical_root = Path(os.path.realpath(Path(root).expanduser()))
        effective_data_dir = data_dir if data_dir is not None else canonical_root / ".cicada"
        output_dir = effective_data_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(root=canonical_root, output_dir=output_dir)

    @property
    def capture_sink(self) -> "CaptureSink | None":
        """当前捕获汇; None 表示本次组合不捕获还原点 (C2)."""
        return self._capture_sink

    def install_restore_capture(self, sink: "CaptureSink") -> None:
        """由 coding.restore 插件在启动时安装一次性捕获汇 (不可替换)."""
        if self._capture_sink is not None and self._capture_sink is not sink:
            raise WorkspaceError("a restore capture sink is already installed")
        self._capture_sink = sink

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

    async def mutate(
        self,
        path: Path,
        op: Callable[[], Awaitable[T]],
        *,
        preserve_result_on_cancel: bool = False,
        on_caller_cancel: Callable[[], None] | None = None,
    ) -> T | None:
        """同 canonical 路径串行执行 op, 不同路径并行.

        op 一旦开始不被取消抢占 (shield 后等待其完成才释放队列);
        等待队列的调用方被取消时放弃其位置, op 不会执行.

        preserve_result_on_cancel=True (FileAccess 修改路径): 外层协程取消
        不再向上抛 CancelledError, 而是经 on_caller_cancel 记录后继续
        shield/drain 到 op 的真实结果; 排队中取消则不启动 op, 调用回调后
        返回 None. 默认 False 保持旧的取消/异常语义不变.
        """
        lock = self._locks.setdefault(self._queue_key(path), asyncio.Lock())
        if not preserve_result_on_cancel:
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
        try:
            async with lock:
                task = asyncio.ensure_future(op())
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    _note_handled_cancel(on_caller_cancel)
                    while True:
                        try:
                            return await asyncio.shield(task)
                        except asyncio.CancelledError:
                            if task.done():
                                # op 自身以 cancelled 终结 (当前无此来源): 无真实结果可保留,
                                # 传播取消而不是忙转持有锁
                                raise
                            # 重复外层取消不提前放弃: 持有到真实结果才释放队列
                            _note_handled_cancel(on_caller_cancel)
        except asyncio.CancelledError:
            # 排队中取消: op 未启动, 位置放弃; 取消事实仍传递给回调方
            if on_caller_cancel is not None:
                on_caller_cancel()
            return None

    def new_output_file(self, label: str) -> Path:
        """在 output_dir 分配一个唯一文件路径 (label 净化后作文件名一部分)."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or "output"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.output_dir / f"{safe}-{stamp}-{next(self._output_seq):04d}-{uuid.uuid4().hex[:8]}.txt"

    def resolve_output_for_write(self, candidate: Path) -> Path:
        """只允许内部输出目录的直属文件，拒绝目录链被重定向。"""
        if candidate.parent != self.output_dir or Path(os.path.realpath(self.output_dir)) != self.output_dir:
            raise PathNotAllowedError(f"output path is not in the assigned output directory: {candidate}")
        return candidate

    def _within_root(self, path: Path) -> bool:
        root_parts = [part.casefold() for part in self.root.parts]
        path_parts = [part.casefold() for part in path.parts]
        return path_parts[: len(root_parts)] == root_parts

    @staticmethod
    def _queue_key(path: Path) -> str:
        # canonical 小写化字符串; realpath 对不存在文件回退到规范绝对路径
        return str(Path(os.path.realpath(path))).casefold()


def workspace_plugin(root: Path, *, data_dir: Path | None = None) -> PluginDefinition:
    """coding-workspace 插件: 提供 coding.workspace 能力."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide("coding.workspace", Workspace.create(root, data_dir=data_dir))

    return PluginDefinition(
        name="coding-workspace", setup=setup, provides=frozenset({"coding.workspace"})
    )
