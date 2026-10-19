"""Pi 主屏移植 (票据 01): ANSI 感知文本工具与行差分 renderer.

来源: pi-mono@1a584a7a packages/tui/src/tui-main-screen.ts / tui.ts / utils.ts
(MIT, Copyright (c) 2025 Mario Zechner); 本文件移植其主屏差分/完整重绘与
列宽算法的职责, 只保留 regular 主屏所需子集, 不含 kitty 图片与 overlay。

边界 (01 冻结面, 见 shared-boundary-handoff):
- 本模块是纯同步机制: render_once() 读当前终端尺寸并写一个输出帧;
  调度 (16ms 合并/输入即时帧)、输入解析与生命周期归 frontend。
- 传入组件产出的行必须是"已清理显示文本" (外部 CSI/OSC 已移除), 本模块
  不再净化外部内容; 行内 ANSI 样式由组件按主题生成, renderer 只做
  normalize (tab->3 空格) + 尾部 SGR reset。
- 行宽超过终端宽度时防御性截断并记录 ui_failures, 不抛异常杀死进程
  (Pi 源码此分支为 crash+日志; Cicada 改为有界失败记录, 见交接偏差)。
"""

from __future__ import annotations

import unicodedata
from typing import Callable, Protocol

# APC 光标标记: 零宽, 终端忽略; 组件在焦点光标位置发出, renderer 提取后
# 定位硬件光标 (IME 候选窗跟随), 输出前从行内剥除。
CURSOR_MARKER = "\x1b_pi:c\x07"

# Pi SEGMENT_RESET: 行尾 SGR reset + 关闭 OSC8 超链接
SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"

TAB_WIDTH = 3

# --------------------------------------------------------------------------
# ANSI / 列宽工具 (utils.ts 子集)
# --------------------------------------------------------------------------


def extract_ansi_code(text: str, pos: int) -> tuple[str, int] | None:
    """提取 pos 处的 CSI/OSC/APC 序列, 返回 (code, 长度); 无则 None。

    CSI 按 ECMA-48 解析: 参数字节 (0x30-3F) -> 中间字节 (0x20-2F) ->
    终结字节 (0x40-7E); 比 Pi 源码的简化扫描更严格, 不会把未知终结符
    与后续文本并成一段。
    """
    if pos >= len(text) or text[pos] != "\x1b":
        return None
    nxt = text[pos + 1] if pos + 1 < len(text) else ""
    if nxt == "[":  # CSI
        j = pos + 2
        while j < len(text) and "\x30" <= text[j] <= "\x3f":
            j += 1
        while j < len(text) and "\x20" <= text[j] <= "\x2f":
            j += 1
        if j < len(text) and "\x40" <= text[j] <= "\x7e":
            return text[pos : j + 1], j + 1 - pos
        return None
    if nxt in ("]", "_"):  # OSC/APC: BEL 或 ST (ESC\) 终结
        j = pos + 2
        while j < len(text):
            if text[j] == "\x07":
                return text[pos : j + 1], j + 1 - pos
            if text[j] == "\x1b" and j + 1 < len(text) and text[j + 1] == "\\":
                return text[pos : j + 2], j + 2 - pos
            j += 1
        return None
    return None


def strip_terminal_sequences(text: str) -> str:
    """移除 ANSI/OSC/APC 控制序列, 保留可见文本。"""
    if "\x1b" not in text:
        return text
    result: list[str] = []
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            i += ansi[1]
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _is_printable_ascii(text: str) -> bool:
    return all(0x20 <= ord(ch) <= 0x7E for ch in text)


def iter_graphemes(text: str) -> list[str]:
    """图素分段 (UAX-29 实用子集): 基字符 + 组合附标 + ZWJ/VS/区域指示符连接。

    覆盖 P11 清单: 中文 (单码点宽字符)、组合字符 (基字符+combining)、
    ZWJ emoji (ZWJ 连接)、ANSI 边界 (调用方先剥离序列)。Hangul jamo 聚簇
    与部分新表情不在子集内, 精度边界见交接文档。
    """
    graphemes: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cp = ord(ch)
        # 区域指示符成对 (旗帜 emoji 流式半截也按 2 列保守计)
        if 0x1F1E6 <= cp <= 0x1F1FF and i + 1 < n and 0x1F1E6 <= ord(text[i + 1]) <= 0x1F1FF:
            graphemes.append(text[i : i + 2])
            i += 2
            continue
        j = i + 1
        while j < n:
            nxt = text[j]
            ncp = ord(nxt)
            if nxt == "\u200d":  # ZWJ: 连接前后图素
                j += 2 if j + 1 < n else 1
                continue
            if ncp in (0xFE0F, 0xFE0E) or 0xE0100 <= ncp <= 0xE01EF:  # 变体选择符
                j += 1
                continue
            if unicodedata.combining(nxt) != 0:
                j += 1
                continue
            break
        graphemes.append(text[i:j])
        i = j
    return graphemes


def _grapheme_width(segment: str) -> int:
    """单个图素的终端列宽 (Pi graphemeWidth 实用子集)。"""
    if segment == "\t":
        return TAB_WIDTH
    if not segment:
        return 0
    # 零宽: 控制字符、纯 combining、格式字符
    if segment == "\u200d" or all(
        unicodedata.combining(ch) != 0 or unicodedata.category(ch) in ("Cf", "Cc", "Cn", "Zl", "Zp", "Me", "Mn")
        for ch in segment
    ):
        return 0
    # emoji 保守判定: 主码点在 emoji 区块或带 VS16 按 2 列 (RGI 全表不引入)
    first = segment[0]
    fcp = ord(first)
    if "\ufe0f" in segment or 0x1F000 <= fcp <= 0x1FBFF or 0x2600 <= fcp <= 0x27BF or 0x2B50 <= fcp <= 0x2B55:
        return 2
    eaw = unicodedata.east_asian_width(first)
    width = 2 if eaw in ("W", "F") else 1
    # 后续非零宽可见码点 (半/全角形式) 各占自己的列宽
    for ch in segment[1:]:
        if unicodedata.category(ch) in ("Mn", "Me", "Cf", "Cc"):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


_WIDTH_CACHE: dict[str, int] = {}
_WIDTH_CACHE_MAX = 2048


def visible_width(text: str) -> int:
    """可见列宽: 剥离控制序列, tab 按 3 列, 图素宽度求和 (带缓存)。"""
    if not text:
        return 0
    if _is_printable_ascii(text):
        return len(text)
    cached = _WIDTH_CACHE.get(text)
    if cached is not None:
        return cached
    width = sum(_grapheme_width(g) for g in iter_graphemes(strip_terminal_sequences(text.replace("\t", "   "))))
    if len(_WIDTH_CACHE) >= _WIDTH_CACHE_MAX:
        _WIDTH_CACHE.pop(next(iter(_WIDTH_CACHE)))
    _WIDTH_CACHE[text] = width
    return width


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    """按可见列切片, ANSI 序列随所在区间保留 (utils.sliceByColumn 移植)。"""
    if length <= 0:
        return ""
    end_col = start_col + length
    result: list[str] = []
    result_width = 0
    current_col = 0
    i = 0
    pending_ansi: list[str] = []
    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            code = ansi[0]
            if start_col <= current_col < end_col:
                result.append(code)
            elif current_col < start_col:
                pending_ansi.append(code)
            i += ansi[1]
            continue
        # 无序列文本段: 逐图素推进
        text_end = i
        while text_end < len(line) and not extract_ansi_code(line, text_end):
            text_end += 1
        for g in iter_graphemes(line[i:text_end]):
            w = _grapheme_width(g)
            in_range = start_col <= current_col < end_col
            fits = (not strict) or current_col + w <= end_col
            if in_range and fits:
                if pending_ansi:
                    result.extend(pending_ansi)
                    pending_ansi.clear()
                result.append(g)
                result_width += w
            current_col += w
            if current_col >= end_col:
                break
        i = text_end
        if current_col >= end_col:
            break
    return "".join(result)


def normalize_terminal_output(text: str) -> str:
    """可见 tab 展开为 3 空格 (样式序列内不动), 防止 tab stop 折行破坏版式。"""
    if "\t" not in text:
        return text
    result: list[str] = []
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            result.append(ansi[0])
            i += ansi[1]
            continue
        result.append("   " if text[i] == "\t" else text[i])
        i += 1
    return "".join(result)


def truncate_to_width(text: str, max_width: int, ellipsis: str = "…") -> str:
    """按可见列截断 (超界时加省略号); 组件应主动保证行宽, 这是防御兜底。"""
    if max_width <= 0:
        return ""
    if visible_width(text) <= max_width:
        return text
    target = max(0, max_width - visible_width(ellipsis))
    kept: list[str] = []
    used = 0
    for g in iter_graphemes(strip_terminal_sequences(text)):
        w = _grapheme_width(g)
        if used + w > target:
            break
        kept.append(g)
        used += w
    return "".join(kept) + ellipsis if target > 0 else ellipsis[:max_width]


def wrap_text_ansi(text: str, width: int) -> list[str]:
    """按词换行, CJK 可断; 不感知 ANSI (Pi utils.wrapTextWithAnsi 的纯文本子集)。

    调用方必须先剥离/清理 ANSI (本仓库各组件经 sanitize_display 后进入);
    含转义序列的文本会被按可见内容计宽并可能在块边界截断序列。
    只做换行, 不填充不改背景。供消息/工具组件排版使用; 编辑器用
    word_wrap_line (带原始索引) 而不是本函数。
    """
    lines: list[str] = []
    for logical in text.split("\n"):
        for chunk in word_wrap_line(logical, width):
            lines.append(chunk.text)
    return lines or [""]


class WrapChunk:
    """换行块: 文本与原逻辑行内的码点区间 (编辑器光标映射需要)。"""

    __slots__ = ("text", "start_index", "end_index")

    def __init__(self, text: str, start_index: int, end_index: int) -> None:
        self.text = text
        self.start_index = start_index
        self.end_index = end_index

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"WrapChunk({self.text!r}, {self.start_index}, {self.end_index})"


_CJK_BREAK_RE = __import__("re").compile(
    "["
    "\U00002E80-\U00002EFF\U00003000-\U0000303F\U00003190-\U000031BF"
    "\U00003200-\U000032FF\U00003300-\U000033FF\U00004E00-\U00009FFF"
    "\U00003040-\U0000309F\U000030A0-\U000030FF\U0000AC00-\U0000D7AF"
    "\U00003130-\U0000318F\U0000F900-\U0000FAFF"
    "]"
)


def word_wrap_line(line: str, max_width: int) -> list[WrapChunk]:
    """单词感知换行 (editor.wordWrapLine 移植, 去粘贴标记)。

    断行机会: 空白后非空白前, 或相邻任一侧为 CJK。超长单图素按图素强制断。
    """
    if not line or max_width <= 0:
        return [WrapChunk("", 0, 0)]
    if visible_width(line) <= max_width:
        return [WrapChunk(line, 0, len(line))]

    graphemes = iter_graphemes(line)
    # 预计算每个图素的 (起始码点索引, 宽度, 是否空白, 是否CJK)
    metas: list[tuple[int, str, int, bool, bool]] = []
    offset = 0
    for g in graphemes:
        w = _grapheme_width(g)
        metas.append((offset, g, w, g.isspace(), bool(_CJK_BREAK_RE.search(g))))
        offset += len(g)

    chunks: list[WrapChunk] = []
    chunk_start = 0
    current_width = 0
    wrap_opp_index = -1
    wrap_opp_width = 0

    for idx, (char_index, g, g_width, is_ws, is_cjk) in enumerate(metas):
        if current_width + g_width > max_width:
            if wrap_opp_index >= 0 and current_width - wrap_opp_width + g_width <= max_width:
                chunks.append(WrapChunk(line[chunk_start:wrap_opp_index], chunk_start, wrap_opp_index))
                chunk_start = wrap_opp_index
                current_width -= wrap_opp_width
            elif chunk_start < char_index:
                chunks.append(WrapChunk(line[chunk_start:char_index], chunk_start, char_index))
                chunk_start = char_index
                current_width = 0
            wrap_opp_index = -1

        current_width += g_width
        if idx + 1 < len(metas):
            nxt = metas[idx + 1]
            if is_ws and not nxt[3]:
                wrap_opp_index = nxt[0]
                wrap_opp_width = current_width
            elif not is_ws and not nxt[3] and (is_cjk or nxt[4]):
                wrap_opp_index = nxt[0]
                wrap_opp_width = current_width

    chunks.append(WrapChunk(line[chunk_start:], chunk_start, len(line)))
    return chunks


# --------------------------------------------------------------------------
# 主屏 renderer (TuiMainScreen 移植, 无 kitty/overlay)
# --------------------------------------------------------------------------


class RenderedLineSource(Protocol):
    """主屏内容源: 一个前端持有的组件列表, 按序渲染为行。"""

    def render_lines(self, width: int) -> list[str]: ...


class MainScreen:
    """regular 主屏差分 renderer: 前后行差分、清尾、宽高变化/视口上方
    变化时 fullRender (CSI 2J/H/3J 全文回放), 光标标记定位硬件光标。

    同步机制: 每次调用 render_once() 取当前尺寸渲染一帧并经 terminal 写出;
    write 回调失败向上抛出, 恢复与重试归 frontend 的故障矩阵。
    """

    def __init__(
        self,
        source: Callable[[int], list[str]],
        write: Callable[[str], None],
        *,
        clear_on_shrink: bool = False,
        show_hardware_cursor: bool = True,
    ) -> None:
        self._render_lines = source
        self._write = write
        self.clear_on_shrink = clear_on_shrink
        self.show_hardware_cursor = show_hardware_cursor
        self.ui_failures: list[str] = []
        self.full_redraws = 0
        self.stopped = False
        self._previous_lines: list[str] = []
        self._previous_width = 0
        self._previous_height = 0
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0
        self._cursor_shown: bool | None = None  # None = 未知, 首次必定写

    # --- 状态 -----------------------------------------------------------------

    def reset_render_state(self) -> None:
        """强制下一帧完整重绘 (主题变化/恢复后重放)。"""
        self._previous_lines = []
        self._previous_width = -1
        self._previous_height = -1
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0
        self._cursor_shown = None

    @property
    def viewport_top(self) -> int:
        """当前帧第 0 行对应的屏幕上方偏移 (02): 屏幕行 s 呈现帧行
        viewport_top + s; 自有视口下帧 ≤ rows, 恒为 0。"""
        return self._previous_viewport_top

    # --- 渲染 -----------------------------------------------------------------

    def render_once(self, columns: int, rows: int) -> None:
        if self.stopped:
            return
        width, height = max(1, columns), max(1, rows)
        width_changed = self._previous_width not in (0, width)
        height_changed = self._previous_height not in (0, height)
        prev_viewport_top = self._previous_viewport_top
        if height_changed:
            previous_buffer_length = (
                self._previous_viewport_top + self._previous_height if self._previous_height > 0 else height
            )
            prev_viewport_top = max(0, previous_buffer_length - height)
        viewport_top = prev_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        new_lines = self._render_lines(width)
        cursor_pos = self._extract_cursor_position(new_lines, height)
        new_lines = self._apply_line_resets(new_lines)

        def compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            return (target_row - viewport_top) - current_screen_row

        def full_render(clear: bool) -> None:
            self.full_redraws += 1
            out: list[str] = ["\x1b[?2026h"]
            if clear:
                out.append("\x1b[2J\x1b[H\x1b[3J")
            for i, line in enumerate(new_lines):
                if visible_width(line) > width:
                    # 组件产出超宽行: 有界失败 (截断保版式), 不终止进程
                    self.ui_failures.append(f"line {i} exceeds width ({visible_width(line)} > {width}); truncated")
                    line = truncate_to_width(line, width)
                out.append(("\r\n" if i > 0 else "") + line)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            out.append(self._cursor_sequence(cursor_pos, len(new_lines)))
            out.append("\x1b[?2026l")
            self._write("".join(out))
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            self._previous_lines = new_lines
            self._previous_width = width
            self._previous_height = height

        # 首帧/重置后: 锚定当前可见窗口并清屏重放 (C3 第一次呈现清该帧区域,
        # ADR 0003: 进入时 shell 可见历史可能消失); 帧行与屏幕行自此对齐,
        # 是 02 鼠标坐标映射的前提
        if not self._previous_lines and not width_changed and not height_changed:
            full_render(True)
            return

        # 宽度变化必 fullRender (折行全变); 高度变化同样 (视口对齐)
        if width_changed:
            full_render(True)
            return
        if height_changed:
            full_render(True)
            return

        # 内容缩短: Pi 默认保留空行, clear_on_shrink 开启时清屏重放
        if self.clear_on_shrink and len(new_lines) < self._max_lines_rendered:
            full_render(True)
            return

        # 差分: 找首尾变化行
        first_changed = -1
        last_changed = -1
        for i in range(max(len(new_lines), len(self._previous_lines))):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""
            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i
                last_changed = i
        appended = len(new_lines) > len(self._previous_lines)
        if appended:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1
        append_start = appended and first_changed == len(self._previous_lines) and first_changed > 0

        if first_changed == -1:
            # 无变化: 只需校正硬件光标位置
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_viewport_top = prev_viewport_top
            self._previous_height = height
            return

        # 变化全在已删除行: 清掉多余行
        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                target_row = max(0, len(new_lines) - 1)
                if target_row < prev_viewport_top:
                    full_render(True)
                    return
                out: list[str] = ["\x1b[?2026h"]
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    out.append(f"\x1b[{line_diff}B")
                elif line_diff < 0:
                    out.append(f"\x1b[{-line_diff}A")
                out.append("\r")
                extra_lines = len(self._previous_lines) - len(new_lines)
                if extra_lines > height:
                    full_render(True)
                    return
                clear_start_offset = 1 if new_lines else 0
                if extra_lines > 0 and clear_start_offset > 0:
                    out.append(f"\x1b[{clear_start_offset}B")
                for i in range(extra_lines):
                    out.append("\r\x1b[2K")
                    if i < extra_lines - 1:
                        out.append("\x1b[1B")
                move_back = max(0, extra_lines - 1 + clear_start_offset)
                if move_back > 0:
                    out.append(f"\x1b[{move_back}A")
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
                out.append(self._cursor_sequence(cursor_pos, len(new_lines)))
                out.append("\x1b[?2026l")
                self._write("".join(out))
            else:
                self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = prev_viewport_top
            return

        # 变化行在上一视口之上: 差分写不到, fullRender
        if first_changed < prev_viewport_top:
            full_render(True)
            return

        # 从 first_changed 渲染到 last_changed
        out = ["\x1b[?2026h"]
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                out.append(f"\x1b[{move_to_bottom}B")
            scroll = move_target_row - prev_viewport_bottom
            out.append("\r\n" * scroll)
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            out.append(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            out.append(f"\x1b[{-line_diff}A")
        out.append("\r\n" if append_start else "\r")

        render_end = min(last_changed, len(new_lines) - 1)
        for i in range(first_changed, render_end + 1):
            if i > first_changed:
                out.append("\r\n")
            line = new_lines[i]
            if visible_width(line) > width:
                # 组件产出超宽行: 有界失败 (截断保版式), 不终止进程
                self.ui_failures.append(f"line {i} exceeds width ({visible_width(line)} > {width}); truncated")
                line = truncate_to_width(line, width)
            out.append("\x1b[2K")
            out.append(line)

        final_cursor_row = render_end
        if len(self._previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                out.append(f"\x1b[{len(new_lines) - 1 - render_end}B")
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self._previous_lines) - len(new_lines)
            for _ in range(extra_lines):
                out.append("\r\n\x1b[2K")
            out.append(f"\x1b[{extra_lines}A")
        # 光标定位并入同一同步更新块: 否则终端先提交"光标停在刚写入行"(状态行
        # 动画每帧重绘该行, 表现为 working 后面多一个闪烁光标), 下一笔才挪走
        self._hardware_cursor_row = final_cursor_row
        out.append(self._cursor_sequence(cursor_pos, len(new_lines)))
        out.append("\x1b[?2026l")
        self._write("".join(out))

        self._cursor_row = max(0, len(new_lines) - 1)
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)
        self._previous_lines = new_lines
        self._previous_width = width
        self._previous_height = height

    # --- 光标 -----------------------------------------------------------------

    def _position_hardware_cursor(self, cursor_pos: tuple[int, int] | None, total_lines: int) -> None:
        """定位硬件光标到标记处 (IME 候选窗跟随); 无标记则隐藏光标。

        show/hide 按状态变化去重, 无变化帧不重复写 (P10); 光标移动仍逐帧写。
        """
        data = self._cursor_sequence(cursor_pos, total_lines)
        if data:
            self._write(data)

    def _cursor_sequence(self, cursor_pos: tuple[int, int] | None, total_lines: int) -> str:
        """光标定位序列 (不写出), 供渲染路径并入同一同步更新块。

        单独写出会让终端先提交"光标停在最后一次写入处"(状态行动画每帧都重绘
        那一行, 于是光标停在 working 末尾) 再下一帧才挪回输入框; 表现为状态行
        后面多出一个一直闪烁的光标。
        """
        want_visible = cursor_pos is not None and total_lines > 0 and self.show_hardware_cursor
        if not want_visible:
            if self._cursor_shown is not False:
                self._cursor_shown = False
                return "\x1b[?25l"
            return ""
        target_row = max(0, min(cursor_pos[0], total_lines - 1))
        target_col = max(0, cursor_pos[1])
        row_delta = target_row - self._hardware_cursor_row
        buffer = ""
        if row_delta > 0:
            buffer += f"\x1b[{row_delta}B"
        elif row_delta < 0:
            buffer += f"\x1b[{-row_delta}A"
        buffer += f"\x1b[{target_col + 1}G"
        self._hardware_cursor_row = target_row
        if self._cursor_shown is not True:
            buffer += "\x1b[?25h"
            self._cursor_shown = True
        return buffer

    def _extract_cursor_position(self, lines: list[str], height: int) -> tuple[int, int] | None:
        """从可见视口找 CURSOR_MARKER, 计算可见列并剥除标记。"""
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            line = lines[row]
            marker_index = line.find(CURSOR_MARKER)
            if marker_index != -1:
                col = visible_width(line[:marker_index])
                lines[row] = line[:marker_index] + line[marker_index + len(CURSOR_MARKER) :]
                return (row, col)
        return None

    def _apply_line_resets(self, lines: list[str]) -> list[str]:
        return [normalize_terminal_output(line) + SEGMENT_RESET for line in lines]


__all__ = [
    "CURSOR_MARKER",
    "MainScreen",
    "SEGMENT_RESET",
    "TAB_WIDTH",
    "extract_ansi_code",
    "iter_graphemes",
    "normalize_terminal_output",
    "slice_by_column",
    "strip_terminal_sequences",
    "truncate_to_width",
    "visible_width",
    "word_wrap_line",
    "wrap_text_ansi",
    "WrapChunk",
]
