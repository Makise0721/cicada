"""票据 02 坐标误触最小回归: 帧命中/点击配对/滚轮跟随 (契约 C3/C4/D5).

只覆盖票据 Validation 点名的关键长期缺口: CJK/ANSI 命中、同批旧 pointer、
拖动/双击/修饰符不误触、滚轮跟随与「最新」、几何不符忽略、写失败失效、
文档高度 0 安全。原生 Win32 解码 (滚轮符号/origin/lock 位/双击/水平消费/
失焦 cancel) 用合成 MOUSE_EVENT_RECORD 离线断言; 真实终端人工核对归 04,
自动化不能代签。
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui.commands import SlashCommandMenu
from cicada.ui.document_layout import (
    ACTION_TOGGLE,
    TARGET_GROUP,
    TARGET_TOOL,
    DisplayAction,
    DocumentLayout,
    EntryAnchor,
    HitSpan,
)
from cicada.ui.editor import Editor
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.main_screen import visible_width
from cicada.ui.terminal_io import FakeTerminal, PointerEvent, TerminalGeometry
from tests.ui.vt_screen import VtScreen

ACT_GROUP = DisplayAction(ACTION_TOGGLE, (TARGET_GROUP, "run1", 0))
ACT_TOOL = DisplayAction(ACTION_TOGGLE, (TARGET_TOOL, "run1", 0, "c9"))

TITLE_A = "\x1b[36m运行命令 · 2 次\x1b[0m"  # ANSI + CJK 混排标题
TITLE_B = "读取文件 · 1 次"
BODY = "正文行没有任何动作"


class PointerHarness:
    """合成同源布局 + FakeTerminal: 命中图/点击/滚轮的离线驱动。"""

    def __init__(self, *, columns: int = 40, rows: int = 14, long_doc: bool = False,
                 commands: bool = False) -> None:
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.editor = Editor()
        self.editor.focused = True
        self.executed: list[DisplayAction] = []
        self.submitted: list[str] = []
        self.long_doc = long_doc
        self.lines: tuple[str, ...] = (
            tuple(f"历史第{i:02d}行 稳定内容" for i in range(1, 41)) if long_doc else
            (TITLE_A, BODY, TITLE_B)
        )
        self.runner: asyncio.Task | None = None
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=lambda width: list(self.lines),
            layout_document=self._layout_document,
            on_display_action=self.executed.append,
            editor=self.editor,
            on_submit=self.submitted.append,
            command_menu=SlashCommandMenu() if commands else None,
        )

    def _layout_document(self, width: int, selected) -> DocumentLayout:
        hits: list[HitSpan] = []
        anchors: list[EntryAnchor | None] = []
        for row, line in enumerate(self.lines):
            if line is TITLE_A:
                hits.append(HitSpan(row, 0, visible_width(line), ACT_GROUP))
            elif line is TITLE_B:
                hits.append(HitSpan(row, 0, visible_width(line), ACT_TOOL))
            anchors.append(EntryAnchor(("entry", row), 0) if not self.long_doc else None)
        return DocumentLayout(tuple(self.lines), tuple(hits), tuple(anchors))

    async def start(self) -> None:
        self.runner = asyncio.ensure_future(self.frontend.run_async())
        await self.settle()

    async def settle(self, ticks: int = 3) -> None:
        await asyncio.sleep(0.05)
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def stop(self) -> list[str]:
        self.frontend.request_exit()
        return await asyncio.wait_for(self.runner, timeout=5)

    def geo(self) -> TerminalGeometry:
        return self.terminal.geometry

    def set_size(self, columns: int, rows: int) -> None:
        """尺寸变化: 终端与 VT 镜像同步重建 (同 StageTerminal 纪律)。"""
        self.terminal.set_size(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed

    def click(self, x: int, y: int, **kwargs) -> None:
        self.terminal.emit([
            PointerEvent("down", x, y, self.geo(), button="left", **kwargs),
            PointerEvent("up", x, y, self.geo()),
        ])

    def wheel(self, delta: int) -> None:
        self.terminal.emit([PointerEvent("wheel", 0, 0, self.geo(), wheel_delta=delta)])


@pytest.mark.asyncio
async def test_click_hits_cjk_title_and_body_misses():
    """点击只改对应标题; 正文/越界列无动作; 草稿与提交不受影响。"""
    h = PointerHarness()
    try:
        await h.start()
        h.editor.insert_text("草稿")
        h.click(3, 0)  # 「运行命令」双列字内
        await h.settle()
        assert h.executed == [ACT_GROUP]
        h.click(1, 1)  # 正文行: 无动作
        h.click(visible_width(TITLE_A) + 2, 0)  # 标题右侧越界列
        await h.settle()
        assert h.executed == [ACT_GROUP]
        h.click(0, 2)  # 第二个标题: 独立动作
        await h.settle()
        assert h.executed == [ACT_GROUP, ACT_TOOL]
        assert h.editor.get_text() == "草稿"
        assert h.submitted == []
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_same_batch_stale_pointer_dropped_until_new_frame():
    """动作执行后同批旧 pointer 丢弃; 未渲染前下一批点击也不受理 (C4)。"""
    h = PointerHarness()
    try:
        await h.start()
        h.terminal.emit([
            PointerEvent("down", 1, 0, h.geo(), button="left"),
            PointerEvent("up", 1, 0, h.geo()),
            PointerEvent("down", 1, 2, h.geo(), button="left"),  # 同批第二击: 丢弃
            PointerEvent("up", 1, 2, h.geo()),
        ])
        h.click(1, 2)  # 新帧未呈现: 命中图仍失效
        await h.settle()
        assert h.executed == [ACT_GROUP]
        h.click(1, 2)  # 新帧成功后恢复受理
        await h.settle()
        assert h.executed == [ACT_GROUP, ACT_TOOL]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_move_drag_and_cancel_clear_pending():
    """拖出又拖回、坐标变化 move、cancel 均清除按下, 释放无动作 (C4)。"""
    h = PointerHarness()
    try:
        await h.start()
        geo = h.geo()
        h.terminal.emit([  # 拖出又拖回同一标题释放
            PointerEvent("down", 1, 0, geo, button="left"),
            PointerEvent("move", 20, 0, geo),
            PointerEvent("move", 1, 0, geo),
            PointerEvent("up", 1, 0, geo),
        ])
        h.terminal.emit([  # 按下期间失焦 cancel
            PointerEvent("down", 1, 2, geo, button="left"),
            PointerEvent("cancel", -1, -1, geo),
            PointerEvent("up", 1, 2, geo),
        ])
        await h.settle()
        assert h.executed == []
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_double_click_second_press_and_modifiers_consumed():
    """原生双击第二按不误触展开后的另一行; 修饰符点击归宿主选择 (C4)。"""
    h = PointerHarness()
    try:
        await h.start()
        h.click(1, 0)
        await h.settle()
        assert h.executed == [ACT_GROUP]
        geo = h.geo()
        h.terminal.emit([
            PointerEvent("down", 1, 2, geo, button="left", double_click=True),  # 双击第二按
            PointerEvent("up", 1, 2, geo),
        ])
        h.terminal.emit([
            PointerEvent("down", 1, 2, geo, button="left", modifiers=("shift",)),
            PointerEvent("up", 1, 2, geo),
        ])
        h.terminal.emit([
            PointerEvent("down", 1, 2, geo, button="left"),
            PointerEvent("up", 1, 2, geo, modifiers=("shift",)),  # 释放带修饰符
        ])
        h.terminal.emit([
            PointerEvent("down", 1, 2, geo, button="right"),  # 非左键不配对
            PointerEvent("up", 1, 2, geo),
        ])
        await h.settle()
        assert h.executed == [ACT_GROUP]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_geometry_mismatch_and_unknown_input_consumed():
    """事件几何与呈现帧不符忽略等待新帧; 未知鼠标 KeyPress 不写编辑器。"""
    h = PointerHarness()
    try:
        await h.start()
        stale = TerminalGeometry(40, 13, 0, 0)  # 与呈现几何 (rows=14) 不符
        h.terminal.emit([
            PointerEvent("down", 1, 0, stale, button="left"),
            PointerEvent("up", 1, 0, stale),
        ])
        h.terminal.emit([KeyPress(Keys.WindowsMouseEvent, "left;MOUSE_DOWN;1;0")])  # 上游产物: 消费
        await h.settle()
        assert h.executed == []
        assert h.editor.get_text() == ""
        h.click(1, 0)
        await h.settle()
        assert h.executed == [ACT_GROUP]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_write_failure_invalidates_hits_until_recovery():
    """写失败使命中图立即失效; 恢复成功后重新受理 (C4)。"""
    h = PointerHarness()
    try:
        await h.start()
        h.terminal.fail_write = True
        h.terminal.emit([KeyPress("x", "x")])  # 触发一次必然失败的渲染
        await h.settle(ticks=1)
        assert h.frontend.ui_failures
        assert h.frontend._frame_hits is None
        h.click(1, 0)
        assert h.executed == []
        h.terminal.fail_write = False
        h.terminal.emit([KeyPress("y", "y")])  # 恢复渲染成功 → 命中图重新提升
        await h.settle()
        h.click(1, 0)
        await h.settle()
        assert h.executed == [ACT_GROUP]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_command_menu_captures_pointer():
    """菜单活跃时 pointer 被消费不穿透; 收起后点击恢复 (C4)。"""
    from prompt_toolkit.keys import Keys

    h = PointerHarness(commands=True)
    try:
        await h.start()
        h.terminal.emit([KeyPress("/", "/")])  # 经键派发打开菜单
        await h.settle()  # 菜单打开
        h.click(1, 0)
        await h.settle()
        assert h.executed == []
        h.terminal.emit([KeyPress(Keys.Escape, "\x1b")])  # 收起菜单 (草稿保留)
        await h.settle()
        h.click(1, 0)
        await h.settle()
        assert h.executed == [ACT_GROUP]
        assert h.editor.get_text() == "/"
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_resize_keeps_reading_position_and_rearms_hits():
    """resize 保留回看阅读位置, 新帧后命中图重新受理 (C3/C4)。"""
    h = PointerHarness(long_doc=True)
    try:
        await h.start()
        h.wheel(-120)
        await h.settle()
        assert "历史第28行" in h.vt.line_text(0)
        h.set_size(60, 14)
        await h.settle()
        assert "历史第28行" in h.vt.line_text(0)  # 阅读位置保持
        assert h.frontend._frame_hits is not None  # 新帧命中图已重新提升
        h.wheel(-120)
        await h.settle()
        assert "历史第25行" in h.vt.line_text(0)  # 滚动继续可用
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_wheel_review_anchor_and_latest_indicator():
    """滚轮回看脱离跟随、新输出不拉回、点「最新」恢复跟随 (C3)。"""
    h = PointerHarness(long_doc=True)
    try:
        await h.start()
        assert h.vt.find_row("历史第40行") is not None  # 默认跟随尾部
        h.wheel(-120)  # 上滚一个 notch = 3 行 (窗口 10 行: 顶部第31行 -> 第28行)
        await h.settle()
        anchor_text = h.vt.line_text(0)
        assert "历史第28行" in anchor_text
        assert h.vt.find_row("有更新 · 最新") is not None  # 指示行可见
        h.lines = (*h.lines, "历史第41行 新输出", "历史第42行 新输出")
        h.frontend.request_render()
        await h.settle()
        assert "历史第28行" in h.vt.line_text(0)  # 新输出不拉回 (锚点保持)
        assert h.vt.find_row("有更新 · 最新") is not None
        indicator_row = h.vt.find_row("有更新 · 最新")
        h.click(3, indicator_row)  # 点击「最新」恢复跟随
        await h.settle()
        screen = "\n".join(h.vt.screen_text())
        assert "历史第42行" in screen
        assert h.vt.find_row("有更新 · 最新") is None  # 跟随后指示行消失
        h.wheel(120)  # 到底后下滚: clamp 且恢复跟随
        await h.settle()
        assert "历史第42行" in "\n".join(h.vt.screen_text())
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_wheel_accumulates_small_deltas():
    """触控板小 delta 累积到 notch (每 120 一格), 每 notch 3 行 (C3)。"""
    h = PointerHarness(long_doc=True)
    try:
        await h.start()
        h.wheel(-30)
        h.wheel(-30)
        await h.settle()
        assert "历史第40行" in "\n".join(h.vt.screen_text())  # 60 尚不足一格
        h.wheel(-30)
        h.wheel(-30)
        await h.settle()
        anchor_text = h.vt.line_text(0)
        assert "历史第28行" in anchor_text  # 累积满 120 → 恰好一格
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_doc_height_zero_safe():
    """文档高度 0: 无命中/滚动, 点击与滚轮安全消费 (C3)。"""
    h = PointerHarness(columns=40, rows=3)
    try:
        await h.start()
        h.wheel(-120)
        h.click(1, 0)
        await h.settle()
        assert h.executed == []
    finally:
        await h.stop()


# --- 原生 Win32 记录级解码 (合成 MOUSE_EVENT_RECORD, 离线) ----------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Win32 记录路径仅 Windows")
class TestWin32PointerDecode:
    from prompt_toolkit.win32_types import COORD, MOUSE_EVENT_RECORD

    def make_reader(self):
        from cicada.ui.terminal_io import _ShiftEnterConsoleInputReader

        reader = _ShiftEnterConsoleInputReader()
        reader.geometry_provider = lambda: TerminalGeometry(80, 24, 5, 3)
        return reader

    def mouse(self, x, y, button, state, flags):
        ev = self.MOUSE_EVENT_RECORD()
        ev.MousePosition = self.COORD(x, y)
        ev.ButtonState = button
        ev.ControlKeyState = state
        ev.EventFlags = flags
        return ev

    def test_wheel_signed_high_word_and_origin(self):
        r = self.make_reader()
        up = r._decode_pointer(self.mouse(10, 7, 0x00780000, 0, 0x0004))
        assert (up.kind, up.wheel_delta, up.x, up.y) == ("wheel", 120, 5, 4)
        # 上游 button_state>0 文本解析会把下滚误判为上滚; 记录级按符号解码
        down = r._decode_pointer(self.mouse(10, 7, 0xFF880000, 0, 0x0004))
        assert (down.kind, down.wheel_delta) == ("wheel", -120)

    def test_lock_bits_not_modifiers_and_button_decode(self):
        r = self.make_reader()
        # NUMLOCK|CAPSLOCK|SCROLLLOCK|ENHANCED 不计入修饰符 (C4)
        pe = r._decode_pointer(self.mouse(3, 2, 0x0001, 0x01E0, 0))
        assert (pe.kind, pe.button, pe.modifiers) == ("down", "left", ())
        assert r._decode_pointer(self.mouse(1, 1, 0x0002, 0, 0)).button == "right"
        assert r._decode_pointer(self.mouse(1, 1, 0x0004, 0, 0)).button == "middle"
        assert r._decode_pointer(self.mouse(1, 1, 0, 0, 0)).kind == "up"

    def test_double_click_flag_and_horizontal_consumed(self):
        r = self.make_reader()
        pe = r._decode_pointer(self.mouse(4, 2, 0x0001, 0, 0x0002))
        assert (pe.kind, pe.double_click) == ("down", True)
        assert r._decode_pointer(self.mouse(0, 0, 0x00780000, 0, 0x0008)) is None

    def test_move_carries_state_and_modifiers(self):
        r = self.make_reader()
        pe = r._decode_pointer(self.mouse(4, 2, 0x0001, 0x0010, 0x0001))
        assert (pe.kind, pe.button, pe.modifiers) == ("move", "left", ("shift",))
