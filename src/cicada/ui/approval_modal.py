"""审批档的最小终端对话框 (P0-1): 显示待决命令并回传 allow/deny.

只负责呈现与按键解释: 决定经回调交回视图, 再由 ApplicationSession.resolve_approval
落到应用层。与登录向导共用同一"全帧替换"渲染入口 (frontend 的 modal 槽)。
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui import theme
from cicada.ui.rendering import sanitize_display

# (决定, 文案); 顺序即默认焦点顺序
OPTIONS: tuple[tuple[str, str], ...] = (
    ("allow_once", "允许一次"),
    ("allow_session", "允许本会话"),
    ("deny", "拒绝"),
)


class ApprovalModal:
    """一次只显示一个待决请求; Esc 等同拒绝."""

    def __init__(self, on_decision: Callable[[Any, str], None], *,
                 on_cancel: Callable[[], None] | None = None) -> None:
        self._on_decision = on_decision
        self._on_cancel = on_cancel
        self.request: Any | None = None
        self.selected = 0

    @property
    def is_open(self) -> bool:
        return self.request is not None

    def open(self, request: Any) -> None:
        self.request = request
        self.selected = 0

    def dismiss(self) -> None:
        self.request = None

    def handle(self, key: KeyPress) -> None:
        if self.request is None:
            return
        if key.key is Keys.ControlC or key.data == "\x03":
            self.dismiss()
            if self._on_cancel is not None:
                self._on_cancel()
        elif key.key is Keys.Up:
            self.selected = (self.selected - 1) % len(OPTIONS)
        elif key.key is Keys.Down:
            self.selected = (self.selected + 1) % len(OPTIONS)
        elif key.key in (Keys.ControlM,) or key.data == "\r":
            self._commit(OPTIONS[self.selected][0])
        elif key.key is Keys.Escape or key.data == "\x1b":
            self._commit("deny")
        elif key.data == "y":
            self._commit("allow_once")
        elif key.data == "a":
            self._commit("allow_session")
        elif key.data == "n":
            self._commit("deny")

    def _commit(self, decision: str) -> None:
        request = self.request
        if request is None:
            return
        self.dismiss()
        self._on_decision(request, decision)

    def render(self, columns: int, rows: int) -> list[str]:
        if self.request is None:
            return []
        width = max(1, columns)
        lines = ["", theme.fg(theme.TOOL_ACCENT_ERROR, "需要确认命令"), ""]
        command = sanitize_display(str(getattr(self.request, "command", "")))
        for chunk in textwrap.wrap(command or "(empty command)", width=max(1, width - 2)) or [""]:
            lines.append("  " + chunk)
        lines.append("")
        for index, (_, label) in enumerate(OPTIONS):
            marker = "›" if index == self.selected else " "
            lines.append(f"{marker} {label}")
        lines.append("")
        lines.append("Enter 确认 · ↑↓ 选择 · Esc 拒绝")
        return lines[: max(1, rows)]


class ModalRouter:
    """frontend 只有一个 modal 槽: 登录向导与审批框互斥, 按 is_open 转交."""

    def __init__(self, login: Any, approval: ApprovalModal) -> None:
        self._login = login
        self._approval = approval

    @property
    def is_open(self) -> bool:
        return self._approval.is_open or bool(getattr(self._login, "is_open", False))

    def _active(self) -> Any:
        return self._approval if self._approval.is_open else self._login

    def dismiss(self) -> None:
        self._active().dismiss()

    def handle(self, key: KeyPress) -> None:
        self._active().handle(key)

    def render(self, columns: int, rows: int) -> list[str]:
        return self._active().render(columns, rows)
