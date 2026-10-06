"""已有斜线命令与编辑器下方的选择菜单；只管理展示，不执行命令。

交互参考 pi-mono@1a584a7a packages/tui/src/components/editor.ts / select-list.ts
（MIT, Copyright (c) 2025 Mario Zechner）：前缀筛选、上下循环选择、
Enter 交回原提交路径、Escape 收起。所有输出仍由主屏 renderer 写入。
"""

from __future__ import annotations

from dataclasses import dataclass

from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width


@dataclass(frozen=True)
class SlashCommand:
    command: str
    description: str


SLASH_COMMANDS = (
    SlashCommand("/help", "显示本帮助"),
    SlashCommand("/status", "显示状态与计量"),
    SlashCommand("/compact", "摘要较早对话以释放上下文（需持久会话）"),
    SlashCommand("/clear", "清空模型历史（已显示记录保留）"),
    SlashCommand("/exit", "退出程序"),
)


class SlashCommandMenu:
    """草稿前缀的候选列表；收起后同一草稿不会因重绘重新打开。"""

    def __init__(self) -> None:
        self._draft: str | None = None
        self._dismissed = False
        self._matches: tuple[SlashCommand, ...] = ()
        self._index = 0

    def sync(self, text: str, *, at_end: bool) -> None:
        if text != self._draft:
            self._draft = text
            self._dismissed = False
            self._index = 0
        eligible = (
            at_end and not self._dismissed and text.startswith("/")
            and not text.startswith("//") and not any(ch.isspace() for ch in text)
        )
        self._matches = tuple(
            item for item in SLASH_COMMANDS if item.command.startswith(text.lower())
        ) if eligible else ()
        self._index = min(self._index, max(0, len(self._matches) - 1))

    @property
    def selected(self) -> SlashCommand | None:
        return self._matches[self._index] if self._matches else None

    def move(self, delta: int) -> bool:
        if not self._matches:
            return False
        self._index = (self._index + delta) % len(self._matches)
        return True

    def dismiss(self) -> bool:
        if not self._matches:
            return False
        self._dismissed = True
        self._matches = ()
        return True

    def render(self, width: int, max_rows: int) -> list[str]:
        if not self._matches or max_rows <= 0:
            return []
        show_count = len(self._matches) > max_rows and max_rows > 1
        count = min(len(self._matches), max_rows - int(show_count))
        start = max(0, min(self._index - count // 2, len(self._matches) - count))
        name_width = max(len(item.command) - 1 for item in SLASH_COMMANDS)
        lines: list[str] = []
        for index in range(start, start + count):
            item = self._matches[index]
            name = item.command[1:].ljust(name_width)
            label = f"→ {name}" if index == self._index else f"  {name}"
            if index == self._index:
                label = theme.fg(theme.MD_LINK, label)
            line = label + "  " + theme.fg(theme.MUTED_GRAY, item.description)
            lines.append(truncate_to_width(line, width))
        if show_count:
            lines.append(theme.fg(theme.FOOTER_DIM, truncate_to_width(
                f"  ({self._index + 1}/{len(self._matches)})", width
            )))
        return lines
