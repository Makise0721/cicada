"""Shell 审批 (P0-1): 应用持有审批事实, 执行档包装器在进程启动前等待决定.

决定只有三种: allow_once / allow_session / deny. 没有 UI 监听 (单次 CLI、fake 剧本)
时一律 deny, 不读 stdin. allow_session 只在当前进程内有效, 不写存档.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from cicada.core.cancel import CancelToken

Decision = Literal["allow_once", "allow_session", "deny"]
DECISIONS: tuple[Decision, ...] = ("allow_once", "allow_session", "deny")
PREVIEW_MAX_CHARS = 500
DENIED_CONTENT = "denied by user"


@dataclass(frozen=True)
class ApprovalRequest:
    """一条待决审批; command 是有界预览, 不是执行依据."""

    request_id: str
    tool_name: str
    command: str


class ApprovalBroker:
    """一次只允许一个待决请求; 取消/关闭都收敛为 deny, 不无限等待."""

    def __init__(self) -> None:
        self._publish: Callable[[ApprovalRequest], None] | None = None
        self._pending: asyncio.Future[Decision] | None = None
        self._pending_request: ApprovalRequest | None = None
        self._session_allowed: set[str] = set()
        self._closed = False

    @property
    def interactive(self) -> bool:
        return self._publish is not None and not self._closed

    def enable(self, publish: Callable[[ApprovalRequest], None]) -> None:
        """绑定 UI 发布回调; 未绑定时 request 直接 deny."""
        self._publish = publish

    def close(self) -> None:
        self._closed = True
        self._settle("deny")

    async def request(
        self, tool_name: str, command: str, cancel: CancelToken
    ) -> Decision:
        if self._closed or cancel.cancelled:
            return "deny"
        if tool_name in self._session_allowed:
            return "allow_session"
        if self._publish is None:
            return "deny"
        request = ApprovalRequest(
            uuid.uuid4().hex, tool_name, _preview(command)
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Decision] = loop.create_future()
        self._pending = future
        self._pending_request = request
        cancelled = asyncio.ensure_future(cancel.wait())
        try:
            self._publish(request)
            await asyncio.wait({future, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            return future.result() if future.done() else "deny"
        finally:
            if not cancelled.done():
                cancelled.cancel()
            if self._pending is future:
                self._pending = None
                self._pending_request = None

    def resolve(self, request_id: str, decision: Decision) -> bool:
        """UI 决定入口; 身份不匹配或非法决定返回 False, 不改变待决状态."""
        if self._pending_request is None or self._pending_request.request_id != request_id:
            return False
        if decision not in DECISIONS:
            return False
        request, self._pending_request = self._pending_request, None
        if decision == "allow_session":
            self._session_allowed.add(request.tool_name)
        self._settle(decision)
        return True

    def _settle(self, decision: Decision) -> None:
        future, self._pending = self._pending, None
        if future is not None and not future.done():
            future.set_result(decision)


def _preview(command: str) -> str:
    flat = " ".join(command.split())
    return flat if len(flat) <= PREVIEW_MAX_CHARS else flat[:PREVIEW_MAX_CHARS] + "…"
