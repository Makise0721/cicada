"""最小 VT 屏幕解释器 (测试基建, 票据 01)。

只解释本项目 renderer 实际产生的序列子集, 用于断言"真实 VT 输出后的
屏幕/滚动历史/光标", 满足规格 §Testing (不能靠删 ANSI 后文本相等声称
光标/清屏正确)。支持: C0 (\r \n 退格)、CSI A/B/C/D/G/H/J/K、私有模式
?25 (光标可见) / ?2026 (同步输出, 记录不解释)、CSI 3J (清滚动历史)。
宽字符按 display width 占两格; 无序列文本按列写入, 底部滚动推入历史。
"""

from __future__ import annotations

import re
from cicada.ui.main_screen import extract_ansi_code, visible_width

_CSI_FINAL_RE = re.compile(r"^\x1b\[([0-9:;<=>?]*)([ -/]*)([@-~])$")


class VtScreen:
    """固定尺寸屏幕 + 滚动历史; feed() 消费 renderer 的写串。"""

    def __init__(self, columns: int, rows: int) -> None:
        self.columns = columns
        self.rows = rows
        self.history: list[list[str | None]] = []
        self.screen: list[list[str | None]] = [[None] * columns for _ in range(rows)]
        self.cursor_row = 0
        self.cursor_col = 0
        self.cursor_visible = True
        self.sync_output_active = False

    # --- 解释器 -----------------------------------------------------------------

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            if data[i] == "\x1b":
                ansi = extract_ansi_code(data, i)
                if ansi:
                    self._handle_escape(ansi[0])
                    i += ansi[1]
                    continue
                # 孤立 ESC: 不解释, 跳过
                i += 1
                continue
            ch = data[i]
            if ch == "\r":
                self.cursor_col = 0
                i += 1
                continue
            if ch == "\n":
                self._linefeed()
                i += 1
                continue
            if ord(ch) < 32:
                i += 1
                continue
            self._put_char(ch)
            i += 1

    def _handle_escape(self, code: str) -> None:
        m = _CSI_FINAL_RE.match(code)
        if not m:
            return
        params_str, _intermediate, final = m.groups()
        private = params_str.startswith("?")
        if private:
            params_str = params_str[1:]
        params = [int(p) for p in params_str.split(";") if p.isdigit()]
        p1 = params[0] if params else None

        if private:
            if p1 == 25:
                self.cursor_visible = code.endswith("h")
            elif p1 == 2026:
                self.sync_output_active = code.endswith("h")
            return
        if final == "A":  # CUU
            self.cursor_row = max(0, self.cursor_row - (p1 or 1))
        elif final == "B":  # CUD
            self.cursor_row = min(self.rows - 1, self.cursor_row + (p1 or 1))
        elif final == "G":  # CHA (1-based)
            self.cursor_col = max(0, (p1 or 1) - 1)
        elif final == "H":  # CUP (1-based; 无参数 = home)
            self.cursor_row = max(0, (params[0] if params else 1) - 1)
            self.cursor_col = max(0, (params[1] if len(params) > 1 else 1) - 1)
        elif final == "J":
            mode = p1 if p1 is not None else 0
            if mode == 2:
                self.screen = [[None] * self.columns for _ in range(self.rows)]
            elif mode == 3:
                self.history.clear()
        elif final == "K":
            mode = p1 if p1 is not None else 0
            if mode == 2:
                row = self.screen[self.cursor_row]
                for c in range(len(row)):
                    row[c] = None

    def _linefeed(self) -> None:
        if self.cursor_row + 1 >= self.rows:
            self.history.append(self.screen[0])
            self.screen = self.screen[1:] + [[None] * self.columns]
        else:
            self.cursor_row += 1

    def _put_char(self, ch: str) -> None:
        width = visible_width(ch)
        if self.cursor_col + width > self.columns:
            self.cursor_col = 0
            self._linefeed()
        row = self.screen[self.cursor_row]
        row[self.cursor_col] = ch
        for k in range(1, width):
            row[self.cursor_col + k] = ""
        self.cursor_col += width

    # --- 断言视图 -----------------------------------------------------------------

    def line_text(self, row: int) -> str:
        return "".join(cell for cell in self.screen[row] if cell)

    def screen_text(self, *, strip_blank_tail: bool = True) -> list[str]:
        lines = [self.line_text(r) for r in range(self.rows)]
        if strip_blank_tail:
            while lines and lines[-1] == "":
                lines.pop()
        return lines

    def history_text(self) -> list[str]:
        return ["".join(cell for cell in row if cell) for row in self.history]

    def find_row(self, needle: str) -> int | None:
        for r in range(self.rows):
            if needle in self.line_text(r):
                return r
        return None
