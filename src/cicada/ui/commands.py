"""已有斜线命令与编辑器下方的选择菜单；只管理展示，不执行命令。

交互参考 pi-mono@1a584a7a packages/tui/src/components/editor.ts / select-list.ts
（MIT, Copyright (c) 2025 Mario Zechner）：前缀筛选、上下循环选择、
Enter 交回原提交路径、Escape 收起。所有输出仍由主屏 renderer 写入。

两种候选形态共用同一菜单槽位：输入 `/` 时列出命令；命令带固定参数选项时
（如 `/mode `），补齐后直接列出可选项，方向键选择、Enter 把 `命令 选项` 交回
原提交路径（2026-10-25 用户要求），不需要用户手打参数值。

菜单打开时输入框下方只留 COMMAND_MENU_MAX_ROWS 行候选 (底栏让位), 不显示
分页计数行 (2026-10-18 用户裁决: 从 3 行候选增到 9 行, 其余信息隐去)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.display_style import current_style


@dataclass(frozen=True)
class SlashCommand:
    command: str
    description: str
    # 需要参数的命令 (参数由用户继续输入): 菜单补齐时只把命令写回草稿,
    # 绝不直接提交; 无参数命令仍按选中即执行。
    takes_argument: bool = False
    # 固定参数选项 (值, 说明)。非空时菜单在 `命令 + 空格` 后直接列出可选项,
    # 选中后把 `命令 值` 交回提交路径, 不需要用户手打参数值。
    options: tuple[tuple[str, str], ...] = ()


COMMAND_MENU_MAX_ROWS = 9  # 菜单一次可见的候选行数 (不含已移除的分页计数行)

SLASH_COMMANDS = (
    SlashCommand("/skill", "显示 Skills 目录；/skill <名称> [任务] 发送该 Skill", takes_argument=True),
    SlashCommand("/mode", "执行档：选择 read-only / approval / full", takes_argument=True,
                 options=(("read-only", "只读：拒绝写入与命令"),
                          ("approval", "审批：PowerShell 前确认"),
                          ("full", "完全：默认，无额外确认"))),
    SlashCommand("/diff", "审阅工作区差异"),
    SlashCommand("/rewind", "查看并还原各请求的文件改动"),
    SlashCommand("/model", "选择模型（仅影响后继请求）"),
    SlashCommand("/login", "保存登录配置（可在空闲 DeepSeek 会话保存并应用）"),
    SlashCommand("/compact", "摘要较早对话以释放上下文（需持久会话）"),
    SlashCommand("/clear", "清空模型历史（已显示记录保留）"),
    SlashCommand("/cost", "显示本进程费用与调用限额"),
    SlashCommand("/name", "会话名称：/name <名称> · --clear 清除", takes_argument=True),
    SlashCommand("/resume", "搜索并恢复本工作区会话"),
    SlashCommand("/new", "新建保存会话，验证状态保留"),
    SlashCommand("/export", "导出本次会话为 HTML 并在浏览器打开"),
    SlashCommand("/exit", "退出程序"),
)


_OPTION_TEXT = re.compile(r"^/([a-z]+)\s+(.*)$")


class SlashCommandMenu:
    """草稿前缀的候选列表；收起后同一草稿不会因重绘重新打开。

    候选有两种: 命令前缀匹配 (`/` → 全部命令) 与固定参数选项 (`/mode ` →
    read-only/approval/full)。两种都用同一 index、上下循环与 Enter 交回路径。
    """

    def __init__(self) -> None:
        self._draft: str | None = None
        self._dismissed = False
        self._matches: tuple[SlashCommand, ...] = ()
        self._options: tuple[tuple[str, str], ...] = ()
        self._option_command: SlashCommand | None = None
        self._index = 0

    def sync(self, text: str, *, at_end: bool) -> None:
        if text != self._draft:
            self._draft = text
            self._dismissed = False
            self._index = 0
        self._matches = ()
        self._options = ()
        self._option_command = None
        if not at_end or self._dismissed or not text.startswith("/") or text.startswith("//"):
            return
        if not any(ch.isspace() for ch in text):
            self._matches = tuple(
                item for item in SLASH_COMMANDS if item.command.startswith(text.lower())
            )
        else:
            self._sync_options(text)
        self._index = min(self._index, max(0, self._size - 1))

    def _sync_options(self, text: str) -> None:
        """`命令 + 空格 + 选项前缀` 形态: 列出该命令的固定参数选项。"""
        matched = _OPTION_TEXT.match(text.lower())
        if matched is None:
            return
        command = next(
            (item for item in SLASH_COMMANDS
             if item.command == "/" + matched.group(1) and item.options),
            None,
        )
        if command is None:
            return
        typed = matched.group(2)
        if any(ch.isspace() for ch in typed):
            return
        options = tuple(pair for pair in command.options if pair[0].startswith(typed))
        if not options:
            return
        self._option_command = command
        self._options = options

    @property
    def _size(self) -> int:
        return len(self._options) if self._options else len(self._matches)

    @property
    def selected(self) -> SlashCommand | None:
        return self._matches[self._index] if self._matches else None

    @property
    def selected_option(self) -> tuple[SlashCommand, str] | None:
        """选中的固定参数选项: 调用方拼成 `命令 值` 后走原提交路径。"""
        if not self._options or self._option_command is None:
            return None
        return (self._option_command, self._options[self._index][0])

    @property
    def is_open(self) -> bool:
        return bool(self._matches or self._options)

    def move(self, delta: int) -> bool:
        size = self._size
        if not size:
            return False
        self._index = (self._index + delta) % size
        return True

    def dismiss(self) -> bool:
        if not self.is_open:
            return False
        self._dismissed = True
        self._matches = ()
        self._options = ()
        self._option_command = None
        return True

    def complete(self, text: str) -> None:
        """选中后收起菜单并记住已完成的文案。

        与 `dismiss` 的区别是同时把 `text` 记作当前草稿: 提交被拒后草稿仍是
        这段文案, 重绘不会因为它"文本变了"而重新打开单行候选。
        """
        self._draft = text
        self._dismissed = True
        self._matches = ()
        self._options = ()
        self._option_command = None

    def render(self, width: int, max_rows: int) -> list[str]:
        if max_rows <= 0:
            return []
        if self._options:
            rows = list(self._options)
            name_width = max(len(value) for value, _ in rows)
        elif self._matches:
            rows = [(item.command[1:], item.description) for item in self._matches]
            name_width = max(len(item.command) - 1 for item in SLASH_COMMANDS)
        else:
            return []
        count = min(len(rows), max_rows)
        start = max(0, min(self._index - count // 2, len(rows) - count))
        lines: list[str] = []
        for index in range(start, start + count):
            value, description = rows[index]
            name = value.ljust(name_width)
            # 描述先按剩余宽度截断再套样式 (truncate_to_width 会剥离 SGR)
            desc = truncate_to_width(description, max(0, width - 2 - name_width - 2))
            if index == self._index:
                # 选中行整行背景高亮 (tui-aesthetics 规格 §D):
                # 行内只用只开序列换色, 末尾一次复位 (内层复位会清掉背景)
                used = 2 + name_width + 2 + visible_width(desc)
                pad = " " * max(0, width - used)
                lines.append(
                    theme.bg_open(theme.MENU_SEL_BG)
                    + theme.fg_open(theme.EDITOR_BORDER) + "→ "
                    + theme.fg_open(theme.MD_LINK) + name
                    + theme.fg_open(theme.MUTED_GRAY) + "  " + desc
                    + pad + current_style().sgr("0")
                )
            else:
                lines.append(f"  {name}  " + theme.fg(theme.MUTED_GRAY, desc))
        return lines
