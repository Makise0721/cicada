"""资源作用域: 创建与清理写在一起; 严格 LIFO; 失败聚合; 幂等关闭."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

Cleanup = Callable[[], Awaitable[None] | None]


class ScopeClosedError(RuntimeError):
    """向已关闭 scope 登记资源."""


class ResourceScope:
    def __init__(self, name: str) -> None:
        self.name = name
        self._cleanups: list[Cleanup] = []
        self._closed = False
        self._close_errors: BaseExceptionGroup | None = None
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def defer(self, cleanup: Cleanup) -> None:
        """登记清理函数; scope 关闭时按 LIFO 逐个执行."""
        if self._closed:
            raise ScopeClosedError(f"scope {self.name!r} is closed")
        self._cleanups.append(cleanup)

    async def aclose(self) -> None:
        """关闭 scope: 每个清理都执行, 单个失败不中断其余, 聚合为 ExceptionGroup.

        幂等: 重复/并发调用共享同一次清理结果.
        """
        async with self._lock:
            if self._closed:
                if self._close_errors is not None:
                    raise self._close_errors
                return
            self._closed = True
            errors: list[BaseException] = []
            for cleanup in reversed(self._cleanups):
                try:
                    outcome = cleanup()
                    if inspect.isawaitable(outcome):
                        await outcome
                except BaseException as exc:
                    errors.append(exc)
            self._cleanups.clear()
            if errors:
                # BaseExceptionGroup 构造: 全部成员为 Exception 时自动得到 ExceptionGroup 实例,
                # 成员含 BaseException (如 KeyboardInterrupt/CancelledError) 时不再抛 TypeError 掩盖原错误.
                self._close_errors = BaseExceptionGroup(f"scope {self.name!r} cleanup failed", errors)
                raise self._close_errors
