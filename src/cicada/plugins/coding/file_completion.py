"""@ 路径候选: Git 清单与普通目录/子工作区的单层枚举 (C4/C5)。

每次 complete 持有独立 GitInventory (cleanup_timeout_s=None: 共享 read_git 走
实际等待回收的已有分支)。插件拥有全部活跃 complete 调用及其资源: 关闭先封
入口、通知取消、等待实际落定 (runtime scope 原生命周期钩子)，不注册模型工具、
不改变工具名单/schema/fingerprint。仅已分类 not_git / root_mismatch 回退到
当前 workspace 内的局部枚举；其他失败保持 unavailable。
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.file_completion import (
    SCOPE_UNAVAILABLE,
    SCOPE_DIRECTORY,
    FileCompletionResult,
    UnsupportedPathError,
    normalize_query,
    rank_candidates,
)
from cicada.plugins.coding.inventory import (
    DEFAULT_EXCLUSIONS, FileInventory, GitInventory, InventoryError,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginContext, PluginDefinition

FILE_COMPLETION_CAPABILITY = "coding.file_completion"
QUERY_DEADLINE_S = 10.0
MAX_DIRECTORY_ENTRIES = 1024


class CodingFileCompletion:
    """只读候选端口; 每次调用独立有界，取消沿 CancelToken 链继续传播。"""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._closed = False
        self._close_event = asyncio.Event()
        self._active: set[asyncio.Event] = set()

    async def complete(self, query: str, cancel: CancelToken) -> FileCompletionResult:
        """查询候选; 会话关闭后拒绝新入口，进行中的查询不受影响地落定。"""
        if self._closed:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason="session_closed")
        settled = asyncio.Event()
        self._active.add(settled)
        # 内部令牌合并外部取消与关闭通知; 取消事实仍以 CancelledError 向上传播
        token = CancelToken()
        watcher = asyncio.ensure_future(self._forward_cancel(cancel, token))
        try:
            return await self._complete(query, token)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            token.cancel()  # 收尾不再需要内部资源存活; 幂等
            self._active.discard(settled)
            settled.set()

    def close(self) -> None:
        """关闭入口并通知全部活跃查询取消 (scope cleanup, 先于等待落定注册)。"""
        self._closed = True
        self._close_event.set()

    async def wait_settled(self) -> None:
        """等待全部活跃 complete 实际落定; close 之后再无新调用进入。"""
        events = tuple(self._active)
        if events:
            await asyncio.gather(*(event.wait() for event in events))

    async def _forward_cancel(self, outer: CancelToken, inner: CancelToken) -> None:
        outer_wait = asyncio.ensure_future(outer.wait())
        close_wait = asyncio.ensure_future(self._close_event.wait())
        try:
            await asyncio.wait((outer_wait, close_wait), return_when=asyncio.FIRST_COMPLETED)
        finally:
            outer_wait.cancel()
            close_wait.cancel()
            inner.cancel()

    async def _complete(self, query: str, token: CancelToken) -> FileCompletionResult:
        token.throw_if_cancelled()
        try:
            normalized = normalize_query(query)
        except UnsupportedPathError:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason="unsupported_path")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + QUERY_DEADLINE_S
        # 同一 deadline 覆盖探针、清单/局部枚举及排序；OS 调用无硬墙钟保证。
        try:
            async with asyncio.timeout_at(deadline):
                try:
                    inventory = await GitInventory(
                        self._workspace, cleanup_timeout_s=None, classify_not_git=True
                    ).list_files(token)
                except InventoryError as exc:
                    if exc.kind not in ("not_git", "root_mismatch"):
                        raise
                    return await self._directory(normalized, token, deadline)
                pool = _eligible_paths(inventory)
                result = _search(pool, normalized)
                self._checkpoint(token, deadline)
                return result
        except asyncio.CancelledError:
            raise  # 取消继续传播，不转为空匹配
        except InventoryError as exc:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason=exc.kind)
        except TimeoutError:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason="timeout")
        except OSError:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason="path_unreadable")

    @staticmethod
    def _checkpoint(token: CancelToken, deadline: float) -> None:
        token.throw_if_cancelled()
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError

    async def _directory(
        self, normalized: str, token: CancelToken, deadline: float,
    ) -> FileCompletionResult:
        head, _, fragment = normalized.rpartition("/")
        current = self._workspace.root
        # 在原始路径逐级检查，不能先 resolve 丢失 reparse/嵌套仓库事实。
        for part in head.split("/") if head else ():
            self._checkpoint(token, deadline)
            if part in ("", "."):
                continue
            current = current / part
            if not _local_type(current):
                return FileCompletionResult((), SCOPE_DIRECTORY)
        pool: list[tuple[str, bool]] = []
        truncated = False
        self._checkpoint(token, deadline)
        iterator = os.scandir(current)
        try:
            for index, entry in enumerate(iterator):
                if index % 32 == 0:
                    await asyncio.sleep(0)
                    self._checkpoint(token, deadline)
                if index == MAX_DIRECTORY_ENTRIES:
                    truncated = True  # 额外一项只观察存在，不做 stat/匹配。
                    break
                kind = _local_type(Path(entry.path))
                if kind is None:
                    continue
                path = Path(entry.path).relative_to(self._workspace.root).as_posix()
                try:
                    normalize_query(path)
                except UnsupportedPathError:
                    continue
                pool.append((path + "/" if kind else path, kind))
        finally:
            iterator.close()  # 取消/超时/异常都在释放迭代器后交还所有权。
        self._checkpoint(token, deadline)
        candidates, limited = rank_candidates(pool, fragment)
        self._checkpoint(token, deadline)
        return FileCompletionResult(candidates, SCOPE_DIRECTORY, truncated=truncated or limited)


def _local_type(path: Path) -> bool | None:
    """普通目录=True、普通文件=False；缺失/reparse/产物目录/仓库=None。"""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return None
    if stat.S_ISDIR(info.st_mode):
        if path.name.casefold() in DEFAULT_EXCLUSIONS:
            return None
        try:
            (path / ".git").lstat()
        except FileNotFoundError:
            return True
        return None  # 嵌套仓库及已初始化 gitlink 仅查自己的 .git 元信息。
    if stat.S_ISREG(info.st_mode) and path.name.casefold() != ".git":
        return False
    return None


def _eligible_paths(inventory: FileInventory) -> list[tuple[str, bool]]:
    """过滤已删除/reparse 未知/reparse/gitlink 项，再从合法文件派生目录。

    Git 模式不列空目录: 排除项不会出现在任何保留文件的前缀里，派生集合天然
    不含被排除目录。
    """
    files: list[str] = []
    for entry in inventory.files:
        if entry.exists is not True or entry.is_reparse or entry.is_submodule:
            continue
        files.append(entry.relative_path)
    directories: set[str] = set()
    for path in files:
        parts = path.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]) + "/")
    return [(path, False) for path in files] + [(path, True) for path in sorted(directories)]


def _search(
    pool: list[tuple[str, bool]], normalized: str
) -> FileCompletionResult:
    """显式目录部分按原始路径逐级检查，仅保留该目录后代后参加统一评分。"""
    head, _, fragment = normalized.rpartition("/")
    if head:
        prefix = head + "/"
        segments = head.split("/")
        available = {path for path, is_directory in pool if is_directory}
        for index in range(1, len(segments) + 1):
            if "/".join(segments[:index]) + "/" not in available:
                return FileCompletionResult((), "git")
        pool = [
            (path, is_directory)
            for path, is_directory in pool
            if path.startswith(prefix) and path != prefix  # 目录自身不是其后代
        ]
    candidates, truncated = rank_candidates(pool, fragment)
    return FileCompletionResult(candidates, "git", truncated=truncated)


def file_completion_plugin() -> PluginDefinition:
    """coding-file-completion 插件: 仅依赖 coding.workspace 的只读能力。"""

    def setup(ctx: PluginContext) -> None:
        service = CodingFileCompletion(ctx.require("coding.workspace"))
        ctx.provide(FILE_COMPLETION_CAPABILITY, service)
        # LIFO: 先关入口并通知取消，再等待活跃调用实际落定，最后才撤销能力。
        ctx.defer(service.wait_settled)
        ctx.defer(service.close)

    return PluginDefinition(
        name="coding-file-completion", setup=setup,
        provides=frozenset({FILE_COMPLETION_CAPABILITY}),
        requires=frozenset({"coding.workspace"}),
    )
