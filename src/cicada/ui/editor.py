"""Pi 式编辑器移植 (票据 01): 上下细线、图素光标、内部滚动与撤销。

来源: pi-mono@1a584a7a packages/tui/src/components/editor.ts (MIT,
Copyright (c) 2025 Mario Zechner); 只移植 01 所需子集: 横线几何、按宽度
换行 (wordWrapLine)、硬件光标标记、图素级编辑、撤销栈与历史导航。
斜线命令菜单由 frontend/commands 组合；粘贴折叠标记、kill ring、字符跳转不在本期。

边界:
- 本模块是纯状态+渲染, 不接触终端; 按键语义 (哪个 KeyPress 触发哪个
  方法) 由 frontend 的键分派拥有。
- 行宽按 visible_width (中文双列、组合字符零宽) 计算; 光标索引一律是
  逻辑行内的码点索引, 视觉列由换行块映射。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cicada.ui import theme
from cicada.ui.main_screen import (
    CURSOR_MARKER,
    WrapChunk,
    iter_graphemes,
    truncate_to_width,
    visible_width,
    word_wrap_line,
)

UNDO_LIMIT = 200


@dataclass(frozen=True)
class EditorStyle:
    """v2 视觉样式 (tui-aesthetics 规格 §C): None = 关闭全部装饰 (兼容旧外观)。

    取值由组合层从 theme 注入; 本模块不硬编码颜色。
    """

    line_bg: str | None = None
    prompt_glyph: str | None = None
    prompt_color: str | None = None
    placeholder: str | None = None
    placeholder_color: str | None = None


class _EditorState:
    __slots__ = ("lines", "cursor_line", "cursor_col")

    def __init__(self, lines: list[str], cursor_line: int, cursor_col: int) -> None:
        self.lines = lines
        self.cursor_line = cursor_line
        self.cursor_col = cursor_col

    def copy(self) -> "_EditorState":
        return _EditorState(list(self.lines), self.cursor_line, self.cursor_col)


class UndoStack:
    """简单快照栈 (Pi UndoStack 子集): 每次编辑前压栈, 无合并。"""

    def __init__(self, limit: int = UNDO_LIMIT) -> None:
        self._items: list = []
        self._limit = limit

    def push(self, state) -> None:
        self._items.append(state)
        if len(self._items) > self._limit:
            self._items.pop(0)

    def pop(self):
        return self._items.pop() if self._items else None

    def clear(self) -> None:
        self._items.clear()


class VisualLine:
    """换行后的视觉行: 所属逻辑行、起始码点列、码点长度。"""

    __slots__ = ("logical_line", "start_col", "length")

    def __init__(self, logical_line: int, start_col: int, length: int) -> None:
        self.logical_line = logical_line
        self.start_col = start_col
        self.length = length


def create_scroll_border(direction: str, hidden_line_count: int, width: int) -> str:
    """滚动指示横线: ` ↑ N more ` 居中 (editor.createScrollBorder 移植)。"""
    available = max(0, width)
    label = f" {direction} {hidden_line_count} more "
    label_width = visible_width(label)
    if label_width + 2 <= available:
        left_width = (available - label_width) // 2
        return "─" * left_width + label + "─" * (available - left_width - label_width)
    indicator = f"─── {direction} {hidden_line_count} more "
    remaining = available - visible_width(indicator)
    if remaining >= 0:
        return indicator + "─" * remaining
    ellipsis = "..."[:available]
    return indicator[: max(0, available - len(ellipsis))] + ellipsis


class Editor:
    """Pi 编辑器: 状态、布局与渲染; 提交回调由 frontend 注入。

    max_visible_rows 由前端按终端行数给出 (Pi: max(5, floor(rows*0.3)));
    传 None 表示不限制 (测试/演示用)。
    """

    def __init__(
        self,
        *,
        padding_x: int = 0,
        border_color: Callable[[str], str] | None = None,
        style: EditorStyle | None = None,
    ) -> None:
        self.state = _EditorState([""], 0, 0)
        self.focused = False
        self.padding_x = max(0, padding_x)
        self.border_color = border_color if border_color is not None else (lambda s: s)
        self.style = style
        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.disable_submit = False
        self._scroll_offset = 0
        self._last_width = 80
        self._rendered_visible_count = 1
        self._preferred_visual_col: int | None = None
        self._undo = UndoStack()

    # --- 只读视图 ---------------------------------------------------------------

    def get_text(self) -> str:
        return "\n".join(self.state.lines)

    def is_empty(self) -> bool:
        return len(self.state.lines) == 1 and self.state.lines[0] == ""

    def cursor_at_end(self) -> bool:
        return (
            self.state.cursor_line == len(self.state.lines) - 1
            and self.state.cursor_col == len(self.state.lines[-1])
        )

    @property
    def rendered_visible_count(self) -> int:
        return self._rendered_visible_count

    # --- 状态变更 ----------------------------------------------------------------

    def _snapshot(self) -> None:
        self._undo.push(self.state.copy())

    def _touch(self) -> None:
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _set_cursor_col(self, col: int) -> None:
        self.state.cursor_col = col
        self._preferred_visual_col = None

    def set_text(self, text: str, *, cursor_at_end: bool = True) -> None:
        """整体替换 (清空/恢复草稿用); 行尾归一化 \\r\\n 与 \\r。"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.get_text() != normalized:
            self._snapshot()
        self.state.lines = normalized.split("\n") or [""]
        if cursor_at_end:
            self.state.cursor_line = len(self.state.lines) - 1
            self._set_cursor_col(len(self.state.lines[-1]))
        else:
            self.state.cursor_line = 0
            self._set_cursor_col(0)
        self._scroll_offset = 0
        self._touch()

    def clear(self) -> None:
        self._snapshot()
        self.state = _EditorState([""], 0, 0)
        self._scroll_offset = 0
        self._touch()

    def insert_text(self, text: str) -> None:
        """在光标处插入 (含多行/粘贴原文); 行尾归一化, tab 展开为 4 空格。"""
        if not text:
            return
        self._snapshot()
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        inserted = normalized.split("\n")
        line = self.state.lines[self.state.cursor_line] or ""
        before, after = line[: self.state.cursor_col], line[self.state.cursor_col :]
        if len(inserted) == 1:
            self.state.lines[self.state.cursor_line] = before + normalized + after
            self._set_cursor_col(self.state.cursor_col + len(normalized))
        else:
            self.state.lines = (
                self.state.lines[: self.state.cursor_line]
                + [before + inserted[0]]
                + inserted[1:-1]
                + [inserted[-1] + after]
                + self.state.lines[self.state.cursor_line + 1 :]
            )
            self.state.cursor_line += len(inserted) - 1
            self._set_cursor_col(len(inserted[-1]))
        self._touch()

    def insert_newline(self) -> None:
        self._snapshot()
        line = self.state.lines[self.state.cursor_line] or ""
        before, after = line[: self.state.cursor_col], line[self.state.cursor_col :]
        self.state.lines[self.state.cursor_line] = before
        self.state.lines.insert(self.state.cursor_line + 1, after)
        self.state.cursor_line += 1
        self._set_cursor_col(0)
        self._touch()

    def backspace(self) -> None:
        self._snapshot()
        if self.state.cursor_col > 0:
            line = self.state.lines[self.state.cursor_line] or ""
            before = line[: self.state.cursor_col]
            graphemes = iter_graphemes(before)
            last = graphemes[-1] if graphemes else ""
            cut = len(last) if last else 1
            self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col - cut] + line[self.state.cursor_col :]
            self._set_cursor_col(self.state.cursor_col - cut)
        elif self.state.cursor_line > 0:
            prev = self.state.lines[self.state.cursor_line - 1] or ""
            cur = self.state.lines[self.state.cursor_line] or ""
            self.state.lines[self.state.cursor_line - 1] = prev + cur
            del self.state.lines[self.state.cursor_line]
            self.state.cursor_line -= 1
            self._set_cursor_col(len(prev))
        else:
            self._undo.pop()  # 空编辑器退格无效果, 不留快照
            return
        self._touch()

    def delete_forward(self) -> None:
        line = self.state.lines[self.state.cursor_line] or ""
        if self.state.cursor_col < len(line):
            self._snapshot()
            graphemes = iter_graphemes(line[self.state.cursor_col :])
            cut = len(graphemes[0]) if graphemes else 1
            self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col] + line[self.state.cursor_col + cut :]
            self._set_cursor_col(self.state.cursor_col)
            self._touch()
        elif self.state.cursor_line < len(self.state.lines) - 1:
            self._snapshot()
            nxt = self.state.lines[self.state.cursor_line + 1] or ""
            self.state.lines[self.state.cursor_line] = line + nxt
            del self.state.lines[self.state.cursor_line + 1]
            self._touch()

    def undo(self) -> None:
        snapshot = self._undo.pop()
        if snapshot is None:
            return
        self.state = snapshot
        self._scroll_offset = 0
        self._touch()

    def submit(self) -> str | None:
        """提交当前草稿: 返回并清空; disable_submit 时不动作。"""
        if self.disable_submit:
            return None
        result = self.get_text().strip()
        self.state = _EditorState([""], 0, 0)
        self._scroll_offset = 0
        self._undo.clear()
        self._touch()
        return result

    # --- 光标移动 ----------------------------------------------------------------

    def build_visual_line_map(self, width: int) -> list[VisualLine]:
        visual: list[VisualLine] = []
        for i, line in enumerate(self.state.lines):
            line = line or ""
            if not line:
                visual.append(VisualLine(i, 0, 0))
            elif visible_width(line) <= width:
                visual.append(VisualLine(i, 0, len(line)))
            else:
                for chunk in word_wrap_line(line, width):
                    visual.append(VisualLine(i, chunk.start_index, chunk.end_index - chunk.start_index))
        return visual

    def _find_visual_line_at(self, visual: list[VisualLine], line: int, col: int) -> int:
        for i, vl in enumerate(visual):
            if vl.logical_line != line:
                continue
            offset = col - vl.start_col
            is_last = i == len(visual) - 1 or visual[i + 1].logical_line != vl.logical_line
            if 0 <= offset and (offset < vl.length or (is_last and offset == vl.length)):
                return i
        return max(0, len(visual) - 1)

    def move_cursor(self, delta_line: int, delta_col: int) -> None:
        visual = self.build_visual_line_map(self._last_width)
        current = self._find_visual_line_at(visual, self.state.cursor_line, self.state.cursor_col)

        if delta_line != 0:
            target = current + delta_line
            if 0 <= target < len(visual):
                self._move_to_visual_line(visual, current, target)

        if delta_col != 0:
            line = self.state.lines[self.state.cursor_line] or ""
            if delta_col > 0:
                if self.state.cursor_col < len(line):
                    graphemes = iter_graphemes(line[self.state.cursor_col :])
                    step = len(graphemes[0]) if graphemes else 1
                    self._set_cursor_col(self.state.cursor_col + step)
                elif self.state.cursor_line < len(self.state.lines) - 1:
                    self.state.cursor_line += 1
                    self._set_cursor_col(0)
                else:
                    vl = visual[current]
                    self._preferred_visual_col = self.state.cursor_col - vl.start_col
            else:
                if self.state.cursor_col > 0:
                    graphemes = iter_graphemes(line[: self.state.cursor_col])
                    step = len(graphemes[-1]) if graphemes else 1
                    self._set_cursor_col(self.state.cursor_col - step)
                elif self.state.cursor_line > 0:
                    self.state.cursor_line -= 1
                    self._set_cursor_col(len(self.state.lines[self.state.cursor_line] or ""))

    def _move_to_visual_line(self, visual: list[VisualLine], current: int, target: int) -> None:
        src, dst = visual[current], visual[target]
        src_is_last = current == len(visual) - 1 or visual[current + 1].logical_line != src.logical_line
        dst_is_last = target == len(visual) - 1 or visual[target + 1].logical_line != dst.logical_line
        src_max = src.length if src_is_last else max(0, src.length - 1)
        dst_max = dst.length if dst_is_last else max(0, dst.length - 1)

        current_visual_col = self.state.cursor_col - src.start_col
        has_preferred = self._preferred_visual_col is not None
        cursor_in_middle = current_visual_col < src_max
        target_too_short = dst_max < current_visual_col
        if not has_preferred or cursor_in_middle:
            if target_too_short:
                self._preferred_visual_col = current_visual_col
                move_col = dst_max
            else:
                self._preferred_visual_col = None
                move_col = current_visual_col
        else:
            preferred = self._preferred_visual_col
            if target_too_short or dst_max < preferred:
                move_col = dst_max
            else:
                move_col = preferred
                self._preferred_visual_col = None

        self.state.cursor_line = dst.logical_line
        line = self.state.lines[dst.logical_line] or ""
        self.state.cursor_col = min(dst.start_col + move_col, len(line))

    def move_to_line_start(self) -> None:
        self._set_cursor_col(0)

    def move_to_line_end(self) -> None:
        self._set_cursor_col(len(self.state.lines[self.state.cursor_line] or ""))

    # --- 渲染 ---------------------------------------------------------------------

    def render(self, width: int, max_visible_rows: int | None = None) -> list[str]:
        """渲染编辑器: 上横线 + 可见正文 + 下横线; 聚焦时发出硬件光标标记。

        max_visible_rows: Pi 为 max(5, floor(rows*0.3)), 由前端按当前终端
        行数计算传入。
        """
        max_padding = max(0, (width - 1) // 2)
        padding_x = min(self.padding_x, max_padding)
        content_width = max(1, width - padding_x * 2)
        layout_width = max(1, content_width - (0 if padding_x else 1))
        self._last_width = layout_width

        layout = self._layout_text(layout_width)
        cursor_index = next((i for i, item in enumerate(layout) if item[1]), 0)

        max_visible = max_visible_rows if max_visible_rows is not None else len(layout)
        max_visible = max(1, max_visible)
        if cursor_index < self._scroll_offset:
            self._scroll_offset = cursor_index
        elif cursor_index >= self._scroll_offset + max_visible:
            self._scroll_offset = cursor_index - max_visible + 1
        max_scroll = max(0, len(layout) - max_visible)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll))

        visible = layout[self._scroll_offset : self._scroll_offset + max_visible]
        self._rendered_visible_count = len(visible)

        result: list[str] = []
        left_pad = " " * padding_x
        result.append(self.border_color(create_scroll_border("↑", self._scroll_offset, width) if self._scroll_offset else "─" * width))

        emit_marker = self.focused
        for row_index, (text, has_cursor, cursor_pos) in enumerate(visible):
            display = text
            line_width = visible_width(text)
            cursor_in_padding = False
            if has_cursor and cursor_pos is not None:
                marker = CURSOR_MARKER if emit_marker else ""
                before, after = display[:cursor_pos], display[cursor_pos:]
                if after:
                    graphemes = iter_graphemes(after)
                    first = graphemes[0] if graphemes else ""
                    rest = after[len(first) :]
                    display = before + marker + f"\x1b[7m{first}\x1b[27m" + rest
                else:
                    display = before + marker + "\x1b[7m \x1b[27m"
                    line_width += 1
                    if line_width > content_width and padding_x > 0:
                        cursor_in_padding = True
            padding = " " * max(0, content_width - line_width)
            right_pad = left_pad[1:] if cursor_in_padding else left_pad
            result.append(self._style_line(left_pad, display, padding, right_pad, first=row_index == 0))

        lines_below = len(layout) - (self._scroll_offset + len(visible))
        result.append(self.border_color(create_scroll_border("↓", lines_below, width) if lines_below else "─" * width))
        return result

    def _style_line(self, left_pad: str, display: str, padding: str, right_pad: str, *, first: bool) -> str:
        """v2 装饰 (规格 §C): 首可视行提示符、空态占位、整行底色; 无 style 原样。

        样式纪律: 行内只用只开序列 (fg_open/\\x1b[39m/\\x1b[27m), 末尾一次 RESET;
        内层 \\x1b[0m 会连带清掉底色, 不得出现。
        """
        style = self.style
        if style is None or style.line_bg is None:
            return f"{left_pad}{display}{padding}{right_pad}"
        left = left_pad
        if first and style.prompt_glyph:
            glyph_w = visible_width(style.prompt_glyph)
            if visible_width(left_pad) >= glyph_w:
                left = (
                    theme.fg_open(style.prompt_color or theme.EDITOR_BORDER)
                    + style.prompt_glyph
                    + "\x1b[39m"
                    + left_pad[glyph_w:]
                )
        if self.is_empty() and style.placeholder:
            segment = " " + style.placeholder
            if visible_width(segment) > len(padding):
                segment = truncate_to_width(segment, len(padding))
            segment_w = visible_width(segment)
            padding = (
                theme.fg_open(style.placeholder_color or theme.MUTED_GRAY)
                + segment
                + "\x1b[39m"
                + padding[segment_w:]
            )
        return theme.bg_open(style.line_bg) + left + display + padding + right_pad + "\x1b[0m"

    def _layout_text(self, content_width: int) -> list[tuple[str, bool, int | None]]:
        """逻辑行 -> 换行块列表; 返回 (text, has_cursor, cursor_pos) 元组。"""
        layout: list[tuple[str, bool, int | None]] = []
        if self.is_empty():
            return [("", True, 0)]
        for i, line in enumerate(self.state.lines):
            line = line or ""
            is_current = i == self.state.cursor_line
            if visible_width(line) <= content_width:
                layout.append((line, is_current, self.state.cursor_col if is_current else None))
                continue
            chunks = word_wrap_line(line, content_width)
            for j, chunk in enumerate(chunks):
                is_last_chunk = j == len(chunks) - 1
                has_cursor = False
                adjusted = 0
                if is_current:
                    if is_last_chunk:
                        has_cursor = self.state.cursor_col >= chunk.start_index
                        adjusted = self.state.cursor_col - chunk.start_index
                    else:
                        has_cursor = chunk.start_index <= self.state.cursor_col < chunk.end_index
                        if has_cursor:
                            adjusted = min(self.state.cursor_col - chunk.start_index, len(chunk.text))
                layout.append((chunk.text, has_cursor, adjusted if has_cursor else None))
        return layout


__all__ = ["Editor", "UndoStack", "VisualLine", "create_scroll_border", "word_wrap_line", "WrapChunk"]
