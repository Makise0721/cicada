"""已有斜线命令与编辑器下方的选择菜单；只管理展示，不执行命令。

交互参考 pi-mono@1a584a7a packages/tui/src/components/editor.ts / select-list.ts
（MIT, Copyright (c) 2025 Mario Zechner）：前缀筛选、上下循环选择、
Enter 交回原提交路径、Escape 收起。所有输出仍由主屏 renderer 写入。
"""

from __future__ import annotations

from dataclasses import dataclass

from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.display_style import current_style


@dataclass(frozen=True)
class SlashCommand:
    command: str
    description: str


SLASH_COMMANDS = (
    SlashCommand("/help", "显示本帮助"),
    SlashCommand("/status", "显示状态与计量"),
    SlashCommand("/skill", "显示 Skills 目录；/skill <名称> [任务] 发送该 Skill"),
    SlashCommand("/diff", "审阅工作区差异"),
    SlashCommand("/rewind", "查看并还原各请求的文件改动"),
    SlashCommand("/model", "选择模型（仅影响后继请求）"),
    SlashCommand("/login", "保存登录配置（可在空闲 DeepSeek 会话保存并应用）"),
    SlashCommand("/compact", "摘要较早对话以释放上下文（需持久会话）"),
    SlashCommand("/clear", "清空模型历史（已显示记录保留）"),
    SlashCommand("/cost", "显示本进程费用与调用限额"),
    SlashCommand("/name", "会话名称：/name <名称> · --clear 清除"),
    SlashCommand("/resume", "搜索并恢复本工作区会话"),
    SlashCommand("/new", "新建保存会话，验证状态保留"),
    SlashCommand("/export", "导出诊断报告：/export <路径.md或.json>"),
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
            # 描述先按剩余宽度截断再套样式 (truncate_to_width 会剥离 SGR)
            desc = truncate_to_width(item.description, max(0, width - 2 - name_width - 2))
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
        if show_count:
            lines.append(theme.fg(theme.FOOTER_DIM, truncate_to_width(
                f"  ({self._index + 1}/{len(self._matches)})", width
            )))
        return lines
