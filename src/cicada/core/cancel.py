"""显式取消令牌: 取消是请求, 不承诺外部副作用已撤销."""

from __future__ import annotations

import asyncio


class CancelToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError
