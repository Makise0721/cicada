"""会话状态: 内核拥有, 只允许经 append 演进; 对外快照为不可变 tuple."""

from __future__ import annotations

from collections.abc import Iterable

from cicada.core.messages import Message


class SessionState:
    def __init__(self, messages: Iterable[Message] = ()) -> None:
        # 种子历史须由内核先行校验并隔离调用者可变数据; 应用不得直接持有本列表.
        self._messages: list[Message] = list(messages)

    def append(self, message: Message) -> None:
        self._messages.append(message)

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)
