"""稳定的存储能力引用；管理当前、准备中及待释放的单档资源。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from typing import TypeVar

from cicada.core.messages import Message
from cicada.persistence.contracts import (
    ModelChangeSubmission, NameSubmission, ProcessTerminalFacts, RecoverySnapshot,
    RequestOutcomeFacts, SessionStorePort, StorageUnavailableError, StoredIdentity,
    StoredReceipt, SummaryRecord,
)

T = TypeVar("T")


async def settle(task: Awaitable[T]) -> T:
    """重复外层取消不丢失受拥有资源或已接受事务的结果。"""
    task = asyncio.ensure_future(task)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class SessionStoreRoute:
    def __init__(self, store: SessionStorePort) -> None:
        self._current = store
        self._owned: list[SessionStorePort] = [store]
        self._releases: dict[int, asyncio.Task] = {}
        self._closing = False
        self._close_task: asyncio.Task | None = None

    def capture(self) -> SessionStorePort:
        return self._current

    def retain(self, store: SessionStorePort) -> None:
        if self._closing:
            raise StorageUnavailableError("session route is closing")
        if not any(item is store for item in self._owned):
            self._owned.append(store)

    def publish(self, store: SessionStorePort) -> SessionStorePort:
        if self._closing or not any(item is store for item in self._owned):
            raise StorageUnavailableError("session route cannot publish an unowned/closing store")
        previous = self._current
        self._current = store
        return previous

    async def _release_owned(self, store: SessionStorePort) -> None:
        if not any(item is store for item in self._owned):
            return
        key = id(store)
        task = self._releases.get(key)
        if task is None:
            task = asyncio.create_task(store.aclose())
            self._releases[key] = task
        try:
            await settle(task)
        except BaseException:
            # 失败资源仍归 route；退出可再次取得真实 close 结果。
            raise
        else:
            self._owned = [item for item in self._owned if item is not store]
        finally:
            self._releases.pop(key, None)

    async def release(self, store: SessionStorePort) -> None:
        if store is self._current:
            raise StorageUnavailableError("current session is released only by route close")
        await self._release_owned(store)

    async def aclose(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await settle(self._close_task)

    async def _close(self) -> None:
        errors = []
        for store in tuple(reversed(self._owned)):
            try:
                await self._release_owned(store)
            except BaseException as exc:
                errors.append(f"{store.session_id}: {exc!r}")
        if errors:
            self._close_task = None
            raise StorageUnavailableError("session route cleanup failed: " + "; ".join(errors))

    @property
    def session_id(self) -> str:
        return self.capture().session_id

    def recovery_snapshot(self) -> RecoverySnapshot:
        return self.capture().recovery_snapshot()

    async def start_epoch(self, identity: StoredIdentity) -> str:
        return await self.capture().start_epoch(identity)

    def submit_model_change(self, identity: StoredIdentity, *, expected_epoch_id: str) -> ModelChangeSubmission:
        return self.capture().submit_model_change(identity, expected_epoch_id=expected_epoch_id)

    def submit_name(self, user_name: str | None, *, expected_epoch_id: str) -> NameSubmission:
        return self.capture().submit_name(user_name, expected_epoch_id=expected_epoch_id)

    async def begin_request(self, epoch_id: str, prompt: str) -> str:
        return await self.capture().begin_request(epoch_id, prompt)

    async def commit_request(self, request_id: str, *, messages: tuple[Message, ...],
                             outcome: RequestOutcomeFacts, receipts: tuple[StoredReceipt, ...] = ()) -> int:
        return await self.capture().commit_request(request_id, messages=messages,
                                                   outcome=outcome, receipts=receipts)

    async def begin_process(self, request_id: str, *, tool_name: str, call_id: str,
                            arguments: Mapping[str, object]) -> str:
        return await self.capture().begin_process(request_id, tool_name=tool_name,
                                                  call_id=call_id, arguments=arguments)

    async def end_process(self, activity_id: str, terminal: ProcessTerminalFacts) -> None:
        return await self.capture().end_process(activity_id, terminal)

    async def commit_clear(self, epoch_id: str) -> int:
        return await self.capture().commit_clear(epoch_id)

    async def commit_summary(self, record: SummaryRecord) -> None:
        return await self.capture().commit_summary(record)
