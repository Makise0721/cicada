"""会话选择器 (票据 05; 契约 §6 "--tui --resume 启动选择器").

选择发生在打开应用之前: 只读 list_sessions 驱动, 无模型/工具活动、无 TTY
之外的依赖; Escape 取消返回 None (CLI 退出码 0)。输出经现有 sanitize 与
有界排版 (A16); 坏档条目如实标注且不可选 (选中即打开, 校验归共享恢复
路径); 占用条目可选——打开时按既有租约语义明确失败。

键位: ↑/↓ 移动, Enter 恢复, Esc 取消, →/← 或 PgDn/PgUp 翻页。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import strftime, localtime

from prompt_toolkit.keys import Keys

from cicada.persistence.contracts import LIST_SESSIONS_PAGE_SIZE, SessionPage, SessionSummary
from cicada.plugins.session_store import list_sessions
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.rendering import sanitize_display
from cicada.ui.terminal_io import Terminal

SELECTOR_TITLE = "选择要恢复的会话"
SELECTOR_HINT = "↑/↓ 选择  Enter 恢复  Esc 取消  ←/→ 翻页"
_SELECTION_MAX_ROWS = 12
_TITLE_MAX_CHARS = 40
_ERROR_MAX_CHARS = 60

_CLEAR_SCREEN = "\x1b[2J\x1b[H"


class _SelectorState:
    """选择器内部状态: 已取回的页与当前光标; 渲染为纯文本行。"""

    def __init__(self, directory: Path, limit: int) -> None:
        self.directory = directory
        self.limit = limit
        self.pages: list[SessionPage] = []
        self.page_index = 0
        self.row = 0

    # --- 数据 ---------------------------------------------------------------------

    def _fetch_first(self) -> None:
        self.pages = [list_sessions(directory=self.directory, cursor="0", limit=self.limit)]
        self.page_index = 0
        self.row = 0

    def _current(self) -> SessionPage:
        return self.pages[self.page_index]

    def _move_page(self, delta: int) -> None:
        target = self.page_index + delta
        if target < 0:
            return
        while target >= len(self.pages):
            last = self.pages[-1]
            if last.next_cursor is None:
                return
            self.pages.append(
                list_sessions(directory=self.directory, cursor=last.next_cursor, limit=self.limit)
            )
        self.page_index = target
        self.row = 0

    def _move_row(self, delta: int) -> None:
        entries = self._current().entries
        if not entries:
            return
        self.row = min(max(0, self.row + delta), len(entries) - 1)

    def _selected(self) -> SessionSummary | None:
        entries = self._current().entries
        if not entries or self.row >= len(entries):
            return None
        return entries[self.row]

    def _selectable(self, entry: SessionSummary) -> bool:
        return entry.error is None

    # --- 键处理 ----------------------------------------------------------------------

    def handle(self, key, result: dict) -> bool:
        """处理一个按键; 返回 True 表示选择器结束 (result["session_id"] 就绪)。"""
        if key.key in (Keys.Escape, Keys.ControlC):
            result["session_id"] = None
            return True
        if key.key in (Keys.Enter, Keys.ControlM):
            entry = self._selected()
            if entry is not None and self._selectable(entry):
                result["session_id"] = entry.session_id
                return True
            return False
        if key.key in (Keys.Up,):
            self._move_row(-1)
        elif key.key in (Keys.Down,):
            self._move_row(1)
        elif key.key in (Keys.Right, Keys.PageDown):
            self._move_page(1)
        elif key.key in (Keys.Left, Keys.PageUp):
            self._move_page(-1)
        return False

    # --- 渲染 ----------------------------------------------------------------------

    def render_lines(self, columns: int) -> list[str]:
        if not self.pages:
            self._fetch_first()
        page = self._current()
        width = max(20, columns)
        lines = [SELECTOR_TITLE, SELECTOR_HINT, ""]
        if not page.entries:
            lines.append("（本工作区暂无会话存档）")
        for index, entry in enumerate(page.entries):
            marker = ">" if index == self.row else " "
            marker = f"\x1b[7m{marker}\x1b[0m" if index == self.row else marker
            line = _entry_line(entry, width - 4)
            lines.append(f" {marker} {line}")
        lines.append("")
        page_note = (
            f"第 {self.page_index + 1}/{max(len(self.pages), 1)} 页"
            + ("" if page.next_cursor is None else " (→ 下一页)")
        )
        lines.append(page_note)
        return [self._fit(line, width) for line in lines]

    @staticmethod
    def _fit(line: str, width: int) -> str:
        if visible_width(line) <= width:
            return line
        return truncate_to_width(line, width)


def _entry_line(entry: SessionSummary, width: int) -> str:
    updated = (
        strftime("%m-%d %H:%M", localtime(entry.updated_at))
        if entry.updated_at is not None else "无记录"
    )
    title = sanitize_display(entry.short_title)
    if len(title) > _TITLE_MAX_CHARS:
        title = title[: _TITLE_MAX_CHARS - 1] + "…"
    # 标记紧跟标题: 窄窗截断时优先保留占用/阻断/坏档事实
    markers = ""
    if entry.occupied:
        markers += " [占用]"
    if entry.blocked:
        markers += f" [阻断: 活动×{entry.unproven_activities} 回执×{entry.untrusted_receipts}]"
    if entry.error is not None:
        detail = sanitize_display(entry.error)
        if len(detail) > _ERROR_MAX_CHARS:
            detail = detail[: _ERROR_MAX_CHARS - 1] + "…"
        markers += f" [坏档: {detail}]"
    line = f"{entry.session_id}  {title}{markers}  {updated}  请求={entry.committed_requests}"
    if entry.interrupted_requests:
        line += f" 中断={entry.interrupted_requests}"
    return sanitize_display(line)


async def run_session_selector(
    terminal: Terminal, directory: Path, *, limit: int = LIST_SESSIONS_PAGE_SIZE
) -> tuple[str | None, list[str]]:
    """运行选择器; 返回 (session_id | None, 终端恢复错误列表)。"""
    state = _SelectorState(Path(directory), limit)
    done = asyncio.Event()
    result: dict = {"session_id": None}

    def on_input(batch) -> None:
        for key in batch:
            if state.handle(key, result):
                done.set()
                return
        state.render()

    def on_resize() -> None:
        state.render()

    def render() -> None:
        terminal.write(_CLEAR_SCREEN + "\n".join(state.render_lines(terminal.columns)) + "\n")

    state.render = render  # 键处理与 resize 复用同一渲染出口
    terminal.start(on_input, on_resize)
    render()
    try:
        await done.wait()
    finally:
        errors = terminal.stop()
    return result["session_id"], list(errors)


__all__ = ["SELECTOR_TITLE", "run_session_selector"]
