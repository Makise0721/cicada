"""Git 工作区 @ 路径候选 (project-context-v1 C4): 复用 GitInventory 的实现与策略。

每次 complete 持有独立 GitInventory (cleanup_timeout_s=None: 共享 read_git 走
实际等待回收的已有分支)。插件拥有全部活跃 complete 调用及其资源: 关闭先封
入口、通知取消、等待实际落定 (runtime scope 原生命周期钩子)，不注册模型工具、
不改变工具名单/schema/fingerprint。仅已分类 not_git / root_mismatch 的局部
回退属于票 03; 本票非 Git 失败原样返回 unavailable 短原因，不做目录回退。
"""

from __future__ import annotations

import asyncio

from cicada.core.cancel import CancelToken
from cicada.file_completion import (
    SCOPE_UNAVAILABLE,
    FileCompletionResult,
    UnsupportedPathError,
    normalize_query,
    rank_candidates,
)
from cicada.plugins.coding.inventory import FileInventory, GitInventory, InventoryError
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginContext, PluginDefinition

FILE_COMPLETION_CAPABILITY = "coding.file_completion"
QUERY_DEADLINE_S = 10.0


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
        # 整次 complete 的 10 秒逻辑 deadline 覆盖 Git 探针与清单 (票 03 的局部
        # 回退共享同一 deadline，不重置); 阻塞 OS 调用不保证硬墙钟终止。
        try:
            async with asyncio.timeout(QUERY_DEADLINE_S):
                inventory = await GitInventory(
                    self._workspace, cleanup_timeout_s=None
                ).list_files(token)
        except asyncio.CancelledError:
            raise  # 取消继续传播，不转为空匹配
        except InventoryError as exc:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason=exc.kind)
        except TimeoutError:
            return FileCompletionResult((), SCOPE_UNAVAILABLE, reason="timeout")
        pool = _eligible_paths(inventory)
        return _search(pool, normalized)


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
