"""会话状态: 内核拥有, 只允许经 append 演进; 对外快照为不可变 tuple."""

from __future__ import annotations

from cicada.core.messages import Message


class SessionState:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def append(self, message: Message) -> None:
        self._messages.append(message)

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)
