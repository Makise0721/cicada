"""搜索、结果分页与可见窗口的纯状态；所有 render/输入均无 I/O。"""

from __future__ import annotations

import unicodedata
from time import localtime, strftime

from prompt_toolkit.keys import Keys

from cicada.persistence.contracts import (
    CATALOG_QUERY_MAX_CHARS, LIST_SESSIONS_PAGE_SIZE, SessionCatalogEntry, SessionCatalogSnapshot,
)
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.rendering import sanitize_display
from cicada.ui import theme
from cicada.ui.display_style import current_style

SELECTOR_TITLE = "选择要恢复的会话"
SELECTOR_HINT = "↑/↓ 选择  PgUp/PgDn 翻页  ←/→ 光标  Ctrl+R 刷新  Enter 恢复  Esc 取消"


class SessionSelectorState:
    def __init__(self, *, limit: int = LIST_SESSIONS_PAGE_SIZE,
                 current_session_id: str | None = None) -> None:
        self.limit = max(1, int(limit))
        self.current_session_id = current_session_id
        self.snapshot: SessionCatalogSnapshot | None = None
        self.query = ""
        self.cursor = 0
        self.page_index = 0
        self.row = 0
        self.window_offset = 0
        self.entries: tuple[SessionCatalogEntry, ...] = ()
        self.loading = True
        self.error: str | None = None
        self.notice: str | None = None

    def begin_observation(self) -> None:
        self.snapshot = None
        self.entries = ()
        self.page_index = self.row = self.window_offset = 0
        self.loading = True
        self.error = self.notice = None

    def accept_observation(self, snapshot: SessionCatalogSnapshot) -> None:
        self.snapshot = snapshot
        self.loading = False
        self.error = None
        self._filter()

    def fail_observation(self, message: str) -> None:
        self.loading = False
        self.error = message[:512]

    @property
    def page_entries(self) -> tuple[SessionCatalogEntry, ...]:
        start = self.page_index * self.limit
        return self.entries[start:start + self.limit]

    @property
    def selected(self) -> SessionCatalogEntry | None:
        page = self.page_entries
        return page[self.row] if page else None

    @property
    def page_count(self) -> int:
        return max(1, (len(self.entries) + self.limit - 1) // self.limit)

    @property
    def page_key(self) -> tuple[str | None, str, int]:
        return (self.snapshot.observation_id if self.snapshot else None,
                self.query, self.page_index * self.limit)

    def _filter(self) -> None:
        words = self.query.casefold().split()
        self.entries = tuple(entry for entry in (self.snapshot.entries if self.snapshot else ())
                             if all(any(word in field for field in entry.search_fields)
                                    for word in words))
        self.page_index = self.row = self.window_offset = 0

    def _insert(self, text: str) -> bool:
        if (len(self.query) + len(text) > CATALOG_QUERY_MAX_CHARS or
                any(unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") for char in text)):
            self.notice = "搜索限单行 256 字符"
            return False
        self.query = self.query[:self.cursor] + text + self.query[self.cursor:]
        self.cursor += len(text)
        self.notice = None
        return bool(text)

    def handle_batch(self, batch) -> tuple[str | None, str | None]:
        """(action, UUID)；一批最多过滤一次，查询改变使旧分页动作失效。"""
        changed = False
        navigation = []
        action = None
        for key in batch:
            kind = getattr(key, "key", None)
            if kind in (Keys.Escape, Keys.ControlC):
                action = "cancel"
                break
            if kind == Keys.ControlR:
                action = "refresh"
                break
            if kind in (Keys.Enter, Keys.ControlM):
                action = "select"
                break
            if kind in (Keys.Up, Keys.Down, Keys.PageUp, Keys.PageDown):
                navigation.append(kind)
            elif kind == Keys.Left:
                self.cursor = max(0, self.cursor - 1)
            elif kind == Keys.Right:
                self.cursor = min(len(self.query), self.cursor + 1)
            else:
                edited = False
                if kind in (Keys.Backspace, Keys.ControlH) and self.cursor:
                    self.query = self.query[:self.cursor - 1] + self.query[self.cursor:]
                    self.cursor -= 1
                    edited = True
                elif kind == Keys.Delete and self.cursor < len(self.query):
                    self.query = self.query[:self.cursor] + self.query[self.cursor + 1:]
                    edited = True
                elif kind == Keys.BracketedPaste or (isinstance(kind, str) and not isinstance(kind, Keys)):
                    edited = self._insert(key.data)
                if edited:
                    changed = True
                    navigation.clear()
        if changed:
            self._filter()
        for kind in navigation:
            if kind in (Keys.Up, Keys.Down):
                self.row = min(max(0, self.row + (1 if kind == Keys.Down else -1)),
                               max(0, len(self.page_entries) - 1))
            else:
                self.page_index = min(max(0, self.page_index + (1 if kind == Keys.PageDown else -1)),
                                      self.page_count - 1)
                self.row = self.window_offset = 0
        entry = self.selected
        if action == "select":
            if not self.loading and entry is not None and entry.error is None:
                return action, entry.session_id
            action = None
        return action, None

    def render_lines(self, columns: int, rows: int = 24, *, cursor_marker: str = "") -> list[str]:
        width, height = max(1, columns), max(1, rows)
        page = self.page_entries
        show_identity = bool(page) and height >= 7
        spaced = height >= 12
        show_hint = height >= 6
        overhead = 4 + int(show_identity) + int(show_hint) + 2 * int(spaced)
        capacity = max(1, height - overhead)
        self.window_offset = min(self.window_offset, max(0, len(page) - capacity))
        if self.row < self.window_offset:
            self.window_offset = self.row
        elif self.row >= self.window_offset + capacity:
            self.window_offset = self.row - capacity + 1
        if self.loading:
            content = [self._muted_line("正在读取会话…", width)]
        elif self.error:
            content = [theme.fg(theme.DIFF_REMOVED, truncate_to_width(sanitize_display("读取失败: " + self.error), width))]
        elif not self.entries:
            content = [self._muted_line("当前已读取范围无匹配，列表不完整" if self.snapshot and not self.snapshot.complete
                        else "无匹配会话" if self.query else "（本工作区暂无会话存档）", width)]
        else:
            content = [self._entry_line(entry, index, width) for index, entry in
                       enumerate(page[self.window_offset:self.window_offset + capacity], self.window_offset)]
        scope = "列表不完整: " + self.snapshot.reason if self.snapshot and not self.snapshot.complete else "终结事实未读取"
        footer = f"第 {self.page_index + 1}/{self.page_count} 页 · 匹配 {len(self.entries)}"
        if page:
            footer += f" · 行 {self.window_offset + 1}–{min(len(page), self.window_offset + capacity)}/{len(page)}"
        details = []
        if show_identity:
            detail = "ID: " + self.selected.session_id
            if self.selected.error:
                detail += " · 坏档: " + self.selected.error
            details = [self._muted_line(detail, width)]
        title = truncate_to_width(" " + SELECTOR_TITLE, width)
        lines = [theme.fg_open(theme.EDITOR_BORDER) + current_style().sgr("1") + title + current_style().sgr("0")]
        if spaced:
            lines.append("")
        lines.append(self._query_line(width, cursor_marker))
        if spaced:
            lines.append("")
        lines.extend(content)
        lines.extend(details)
        lines.append(self._muted_line(footer, width))
        lines.append(self._muted_line(self.notice or scope, width))
        if show_hint:
            lines.append(self._muted_line(SELECTOR_HINT, width))
        # 极低窗口优先保留搜索及当前候选；正常窗口的行预算包含全部页脚。
        if height < 4:
            lines = [self._query_line(width, cursor_marker), *content]
        return lines[:height]

    @staticmethod
    def _muted_line(text: str, width: int) -> str:
        return theme.fg(theme.MUTED_GRAY, truncate_to_width(sanitize_display(" " + text), width))

    def _query_line(self, width: int, cursor_marker: str = "") -> str:
        prefix = " 搜索 › " if width >= 12 else "› " if width >= 3 else ""
        available = max(0, width - visible_width(prefix) - 1)
        left = self.query[:self.cursor]
        left_budget = available if self.cursor == len(self.query) else available * 2 // 3
        while left and visible_width(left) > left_budget:
            left = left[1:]
        right = truncate_to_width(self.query[self.cursor:], available - visible_width(left))
        hint = ""
        if not self.query:
            hint = truncate_to_width(" 名称 / ID / 首问", available)
        return (theme.fg(theme.EDITOR_BORDER, prefix)
                + theme.fg(theme.ASSISTANT_TEXT, left)
                + cursor_marker + theme.fg(theme.EDITOR_BORDER, "|")
                + theme.fg(theme.ASSISTANT_TEXT, right)
                + theme.fg(theme.EDITOR_PLACEHOLDER, hint))

    def _entry_line(self, entry: SessionCatalogEntry, index: int, width: int) -> str:
        markers = (["当前"] if entry.session_id == self.current_session_id else [])
        if entry.error:
            markers.append("坏档")
        if entry.occupied:
            markers.append("占用")
        try:
            updated = strftime("%m-%d %H:%M", localtime(entry.updated_at)) if entry.updated_at is not None else "无记录"
        except (OSError, OverflowError, ValueError):
            updated = "时间不可显示"
        title = sanitize_display(entry.user_name or entry.short_title or "未命名")
        metadata = " · ".join([*markers, entry.session_id[:8] if width >= 32 else entry.session_id[:4]])
        if width >= 64 and visible_width(metadata) + 14 <= width // 2:
            metadata += " · " + " " * max(0, 11 - visible_width(updated)) + updated
        metadata = truncate_to_width(sanitize_display(metadata), max(0, width // 2))
        prefix = "→ " if index == self.row else "  "
        label = truncate_to_width(title, max(0, width - 4 - visible_width(metadata)))
        gap = " " * max(0, width - visible_width(prefix + label + metadata))
        if width < 6:
            prefix, label, gap, metadata = truncate_to_width(prefix + title, width), "", "", ""
        selected = index == self.row
        color = theme.DIFF_REMOVED if entry.error else theme.MD_LINK if selected else theme.ASSISTANT_TEXT
        return ((theme.bg_open(theme.MENU_SEL_BG) if selected else "")
                + theme.fg_open(theme.EDITOR_BORDER) + prefix
                + theme.fg_open(color) + label
                + gap + theme.fg_open(theme.MUTED_GRAY) + metadata + current_style().sgr("0"))
