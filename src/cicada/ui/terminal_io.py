"""终端输入/输出适配 (票据 01/02): prompt_toolkit 公开低层 API 的 Python 适配.

来源职责: pi-mono@1a584a7a packages/tui/src/terminal.ts (MIT, Copyright
(c) 2025 Mario Zechner)。平台实现按设计用 prompt_toolkit 公开低层能力替换
Node stdin/stdout:
- Input: create_input() 的 raw_mode/attach/read_keys/flush_keys。附着的
  回调由宿主事件循环触发 (Windows 经 executor + call_soon_threadsafe),
  读到的输入批次 (KeyPress | PointerEvent) 交 on_input; 批后立即 flush_keys
  强制解析器落出孤立 Escape ( Application 同款), 键前缀超时归 frontend 所有。
- Output: create_output() 的 write_raw/flush/get_size。Windows Terminal
  下是 Windows10_Output (VT 直写); 行级光标/清屏序列由 renderer 产生,
  本模块不解释内容。
- 适配器拥有: bracketed paste 开关序列、raw 模式生命周期、尺寸轮询、
  停止时的模式恢复与恢复错误收集。write 失败向上抛, 恢复矩阵归 frontend。
- 票据 02 (契约 C4/D6): TerminalGeometry 暴露尺寸与可见窗口 buffer origin;
  PointerEvent 是记录级解码的原生指针事件 —— buffer 坐标减 origin、滚轮按
  有符号 high word 解码、lock 状态位不当作修饰符、水平滚轮安全消费、失焦
  产出 cancel。启用 ENABLE_MOUSE_INPUT 前保存原始 input mode, 正常退出/
  启动部分失败/输入失败统一经 stop() 恢复, 不调用上游 disable_mouse_support
  (其不恢复完整原模式), 不改第三方安装源码。鼠标启用失败保留键盘并带
  mouse_setup_error 短提示事实; 非 Windows 后端不注册鼠标能力。

FakeTerminal 供 headless 检查与演示注入: 可控尺寸、脚本化按键、写捕获与
write 故障注入 (fail_write)、停止恢复错误注入 (restore_errors)、可注入
窗口 origin 与鼠标可用性; get_size/输入读取故障由测试对真实适配器以替身
注入 (见 test_frontend_headless)。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

RESIZE_POLL_INTERVAL_S = 0.2
WHEEL_DELTA = 120  # MSDN: 一个标准滚轮 notch (win32 高字有符号 delta 的单位)

POINTER_KINDS = ("down", "up", "move", "wheel", "cancel")


@dataclass(frozen=True)
class TerminalGeometry:
    """采集时终端几何 (C4): columns/rows 与可见窗口左上角的 buffer 坐标。"""

    columns: int
    rows: int
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class PointerEvent:
    """原生指针事件 (C4)。

    kind ∈ POINTER_KINDS; x/y 是可见窗口 0 基单元格坐标 (buffer 减 origin);
    geometry 是采集时几何, 与呈现几何不一致的事件被前端忽略等待新帧。
    wheel_delta 有符号, 正 = 向上, 120 为一个标准 notch; modifiers 只含
    shift/ctrl/alt (lock 状态位不计入); double_click 是原生 DOUBLE_CLICK
    的第二次按下, 前端清除配对并消费。
    """

    kind: str
    x: int
    y: int
    geometry: TerminalGeometry
    button: str = "none"  # left | right | middle | other | none
    wheel_delta: int = 0
    modifiers: tuple[str, ...] = ()
    double_click: bool = False

    def __post_init__(self) -> None:
        if self.kind not in POINTER_KINDS:
            raise ValueError(f"unknown pointer kind: {self.kind!r}")


InputBatchItem = KeyPress | PointerEvent
InputBatchCallback = Callable[[list[InputBatchItem]], None]
ResizeCallback = Callable[[], None]


class Terminal(Protocol):
    """前端唯一终端依赖面 (01 冻结调用形状; 02 扩展 geometry/指针批次);
    实现不得偷用 prompt_toolkit Application。"""

    def start(self, on_input: InputBatchCallback, on_resize: ResizeCallback) -> None: ...

    def stop(self) -> list[str]:
        """恢复终端状态, 返回恢复过程中的错误描述 (空列表 = 干净恢复)。"""
        ...

    def write(self, data: str) -> None: ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    @property
    def geometry(self) -> TerminalGeometry: ...

    def flush_keys(self) -> list[InputBatchItem]: ...


_ShiftEnterConsoleInputReader = None  # type: ignore[assignment]; POSIX 首版不在范围
if sys.platform == "win32":
    import ctypes
    from ctypes import byref, pointer, windll
    from ctypes.wintypes import DWORD, HANDLE

    from prompt_toolkit.eventloop.win32 import wait_for_handles
    from prompt_toolkit.input.win32 import ConsoleInputReader
    from prompt_toolkit.win32_types import (
        INPUT_RECORD,
        KEY_EVENT_RECORD,
        MOUSE_EVENT_RECORD,
        STD_INPUT_HANDLE,
        STD_OUTPUT_HANDLE,
        CONSOLE_SCREEN_BUFFER_INFO,
        EventTypes,
    )

    # console input mode 位 (MSDN GetConsoleMode)
    _ENABLE_MOUSE_INPUT = 0x0010
    _ENABLE_WINDOW_INPUT = 0x0008
    _ENABLE_QUICK_EDIT_MODE = 0x0040
    _ENABLE_EXTENDED_FLAGS = 0x0080
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

    # MOUSE_EVENT_RECORD.EventFlags (MSDN)
    _MOUSE_MOVED = 0x0001
    _DOUBLE_CLICK_FLAG = 0x0002
    _MOUSE_WHEELED = 0x0004
    _MOUSE_HWHEELED = 0x0008

    # ControlKeyState 修饰位: 只把 shift/ctrl/alt 计入修饰符 (C4);
    # CAPSLOCK/NUMLOCK/SCROLLLOCK/ENHANCED 等状态位不掺入。
    _SHIFT_PRESSED = 0x0010
    _LEFT_CTRL_PRESSED = 0x0008
    _RIGHT_CTRL_PRESSED = 0x0004
    _LEFT_ALT_PRESSED = 0x0002
    _RIGHT_ALT_PRESSED = 0x0001

    # ButtonState 按钮 位 (非滚轮记录)
    _BUTTON_LEFT = 0x0001
    _BUTTON_RIGHT = 0x0002
    _BUTTON_MIDDLE = 0x0004

    def _modifiers_of(state: int) -> tuple[str, ...]:
        mods: list[str] = []
        if state & _SHIFT_PRESSED:
            mods.append("shift")
        if state & (_LEFT_CTRL_PRESSED | _RIGHT_CTRL_PRESSED):
            mods.append("ctrl")
        if state & (_LEFT_ALT_PRESSED | _RIGHT_ALT_PRESSED):
            mods.append("alt")
        return tuple(mods)

    def _button_of(button_state: int) -> str:
        if button_state & _BUTTON_LEFT:
            return "left"
        if button_state & _BUTTON_RIGHT:
            return "right"
        if button_state & _BUTTON_MIDDLE:
            return "middle"
        return "other" if button_state else "none"

    class _ShiftEnterConsoleInputReader(ConsoleInputReader):  # type: ignore[misc,valid-type]
        """记录级读取器 (01 Shift+Enter + 02 原生鼠标/焦点解码, 契约 C4/D6)。

        Shift+Enter 确定性编码 (2026-10-03 用户裁决: 主换行键): 非 VT 记录
        读取器的每个 KEY_EVENT_RECORD 自带 ControlKeyState; Enter 按下时
        SHIFT_PRESSED 且无 Ctrl → 编码为 Keys.ControlJ (换行), 与
        Textual/crossterm 等 Windows TUI 的记录级修饰键方案同源。VT 字节流
        路径丢失记录级修饰键, 事后 GetAsyncKeyState 采样有竞态 (实测快速
        敲击/中文 IME 下不可靠), 已弃用。

        鼠标/焦点不走上游文本解析 (丢修饰符、滚轮符号错、坐标是 buffer
        原值, 不能作为证据): _get_keys 直接产出 PointerEvent —— buffer 坐标
        减可见窗口 origin, 垂直滚轮按有符号 high word 解码 (正 = 向上),
        水平滚轮安全消费, 焦点移出产出 cancel; 批次内保持记录顺序。按键的
        粘贴识别/代理对合并等后处理只作用于 KeyPress 子序列。
        """

        def __init__(self, recognize_paste: bool = True) -> None:
            super().__init__(recognize_paste)
            # 几何提供者由 PromptToolkitTerminal 注入; 读取失败退原点 (0,0),
            # 事件几何与呈现几何不一致时前端忽略, 不会误触发。
            self.geometry_provider: Callable[[], TerminalGeometry] = TerminalGeometry
            # 上一次非滚轮记录的 ButtonState 快照 (D1); 失焦时重置 (见 _get_keys)
            self._last_buttons = 0

        def _event_to_key_presses(self, ev) -> list[KeyPress]:
            keys = list(super()._event_to_key_presses(ev))
            if (
                len(keys) == 1
                and keys[0].key is Keys.ControlM
                and ev.ControlKeyState & self.SHIFT_PRESSED
                and not ev.ControlKeyState & (self.LEFT_CTRL_PRESSED | self.RIGHT_CTRL_PRESSED)
            ):
                return [KeyPress(Keys.ControlJ, "\n")]
            return keys

        # --- 记录循环: 按键走上游, 鼠标/焦点产 PointerEvent ---------------------

        def _get_keys(self, read: DWORD, input_records) -> Iterator[InputBatchItem]:  # type: ignore[no-untyped-def]
            for i in range(read.value):
                ir = input_records[i]
                if ir.EventType not in EventTypes:  # cmder 等会发 EventType 0
                    continue
                ev = getattr(ir.Event, EventTypes[ir.EventType])
                if isinstance(ev, KEY_EVENT_RECORD) and ev.KeyDown:
                    yield from self._event_to_key_presses(ev)
                elif isinstance(ev, MOUSE_EVENT_RECORD):
                    event = self._decode_pointer(ev)
                    if event is not None:
                        yield event
                elif EventTypes[ir.EventType] == "FocusEvent" and not ev.SetFocus:
                    # 失焦: 焦点外的按下/松开不会投递到本窗口, 按钮状态已不可信。
                    # 重置快照, 使回窗后的第一次按下无论按钮位是否变化都能产 down
                    # (D1 双向自愈); 前端 cancel 已清除未完成配对 (C4)。
                    self._last_buttons = 0
                    yield PointerEvent("cancel", -1, -1, self._capture_geometry())

        def _capture_geometry(self) -> TerminalGeometry:
            try:
                return self.geometry_provider()
            except Exception:
                return TerminalGeometry(0, 0)

        def _decode_pointer(self, ev: MOUSE_EVENT_RECORD) -> PointerEvent | None:
            """按按钮位差分解码 (输入处理 v1 D1)。

            ButtonState 是绝对快照, 与上一非滚轮记录差分: 新增位 = 按下, 归零 =
            松开。因此弦按(按下时仍有其他按钮)不再把第二个按钮解成可配对的左键
            down, 也不会被随后的 up 配成一次点击; 同时带 MOUSE_MOVED 的记录不再
            吞掉其中的按下。失焦时重置快照(焦点外记录不投递), 回窗后第一次按下
            必产 down, 使自愈双向成立; 滚轮记录不参与也不更新按钮状态。
            """
            flags = ev.EventFlags
            if flags & _MOUSE_HWHEELED:
                return None  # 水平滚轮: 安全消费, 不产事件 (C4)
            geometry = self._capture_geometry()
            x = ev.MousePosition.X - geometry.origin_x
            y = ev.MousePosition.Y - geometry.origin_y
            modifiers = _modifiers_of(ev.ControlKeyState)
            if flags & _MOUSE_WHEELED:
                # ButtonState 高字是有符号 delta (正 = 向上); 按 c_short 解码
                delta = ctypes.c_short((ev.ButtonState >> 16) & 0xFFFF).value
                return PointerEvent("wheel", x, y, geometry, "none",
                                    wheel_delta=int(delta), modifiers=modifiers)
            button_state = ev.ButtonState
            previous = self._last_buttons
            self._last_buttons = button_state
            if flags & _DOUBLE_CLICK_FLAG:
                # 原生双击的第二次按下: 交前端清除配对并消费 (C4)
                return PointerEvent("down", x, y, geometry, _button_of(button_state),
                                    modifiers=modifiers, double_click=True)
            if button_state & ~previous:
                # 按下: 仅"此前无按钮持有且本记录只按下一个按钮"才可与后续 up
                # 配对; 弦按(已有按钮持有, 或同一记录多键齐按) 报 "other"。
                pairable = not previous and button_state & (button_state - 1) == 0
                return PointerEvent("down", x, y, geometry,
                                    _button_of(button_state) if pairable else "other",
                                    modifiers=modifiers)
            if previous & ~button_state:
                if button_state:
                    return None  # 部分松开: 仍有按钮持有, 不产 up 配对
                return PointerEvent("up", x, y, geometry, "none", modifiers=modifiers)
            if flags & _MOUSE_MOVED:
                return PointerEvent("move", x, y, geometry, _button_of(button_state),
                                    modifiers=modifiers)
            return None  # 无按钮位变化且非移动: 不产事件

        # --- 按键后处理: 容忍批次内 PointerEvent (保持记录顺序) ---------------

        def read(self) -> list[InputBatchItem]:
            """上游 read() 的批次结构复刻; 后处理各步骤对 PointerEvent 透传。"""
            max_count = 2048  # Max events to read at the same time.
            read = DWORD(0)
            arrtype = INPUT_RECORD * max_count
            input_records = arrtype()
            if not wait_for_handles([self.handle], timeout=0):
                return []
            windll.kernel32.ReadConsoleInputW(
                self.handle, pointer(input_records), max_count, pointer(read)
            )
            items = list(self._get_keys(read, input_records))
            items = [
                item if isinstance(item, PointerEvent) else self._insert_key_data(item)
                for item in items
            ]
            items = list(self._merge_paired_surrogates(items))
            if self.recognize_paste and self._is_paste(items):
                return list(self._group_paste(items))
            return items

        def _insert_key_data(self, key_press) -> KeyPress:  # type: ignore[no-untyped-def]
            if isinstance(key_press, PointerEvent):
                return key_press
            return super()._insert_key_data(key_press)

        def _merge_paired_surrogates(self, key_presses) -> Iterator[InputBatchItem]:  # type: ignore[no-untyped-def]
            buffered_high_surrogate: KeyPress | None = None
            for key in key_presses:
                if isinstance(key, PointerEvent):
                    if buffered_high_surrogate is not None:
                        yield buffered_high_surrogate
                        buffered_high_surrogate = None
                    yield key
                    continue
                is_text = not isinstance(key.key, Keys)
                is_high_surrogate = is_text and "\ud800" <= key.key <= "\udbff"
                is_low_surrogate = is_text and "\udc00" <= key.key <= "\udfff"
                if buffered_high_surrogate is not None:
                    if is_low_surrogate:
                        fullchar = (
                            (buffered_high_surrogate.key + key.key)
                            .encode("utf-16-le", "surrogatepass")
                            .decode("utf-16-le")
                        )
                        key = KeyPress(fullchar, fullchar)
                    else:
                        yield buffered_high_surrogate
                    buffered_high_surrogate = None
                if is_high_surrogate:
                    buffered_high_surrogate = key
                else:
                    yield key
            if buffered_high_surrogate is not None:
                yield buffered_high_surrogate

        def _is_paste(self, keys) -> bool:  # type: ignore[no-untyped-def]
            text_count = 0
            newline_count = 0
            for k in keys:
                if isinstance(k, PointerEvent):
                    continue
                if not isinstance(k.key, Keys):
                    text_count += 1
                if k.key == Keys.ControlM:
                    newline_count += 1
            return newline_count >= 1 and text_count >= 1

        def _group_paste(self, items) -> Iterator[InputBatchItem]:  # type: ignore[no-untyped-def]
            gen = iter(items)
            k: InputBatchItem | None = next(gen, None)
            while k is not None:
                data: list[str] = []
                while k is not None and isinstance(k, KeyPress) and (
                    not isinstance(k.key, Keys)
                    or k.key in {Keys.ControlJ, Keys.ControlM}
                ):
                    data.append(k.data)
                    k = next(gen, None)
                if data:
                    yield KeyPress(Keys.BracketedPaste, "".join(data))
                if k is not None:
                    yield k
                    k = next(gen, None)


class PromptToolkitTerminal:
    """真实终端: prompt_toolkit 公开低层组合。"""

    def __init__(self, input=None, output=None) -> None:
        from prompt_toolkit.output.defaults import create_output as _create_output

        self._input = input if input is not None else self._create_input()
        self._output = output if output is not None else _create_output()
        self._raw_mode = None
        self._attach_cm = None
        self._poll_task: asyncio.Task | None = None
        self._on_resize: ResizeCallback | None = None
        self._last_size: tuple[int, int] | None = None
        self.failures: list[str] = []
        self.input_failed = False  # 可选故障事实；不改变 Terminal 必需接口。
        self.mouse_supported = False  # start() 按平台与能力判定 (C4)
        self.mouse_setup_error: str | None = None
        self._original_input_mode: int | None = None

    def _create_input(self):
        if sys.platform == "win32":
            # Windows Terminal 下默认走 VT 字节流路径 (修饰键信息丢失);
            # 覆盖为非 VT 记录读取器, ControlKeyState 确定性区分 Shift+Enter,
            # 并在记录级解码原生鼠标 (契约 C4/D6)。
            from prompt_toolkit.input.win32 import Win32Input

            inp = Win32Input()
            reader = _ShiftEnterConsoleInputReader()
            reader.geometry_provider = self._query_geometry
            inp.console_input_reader = reader
            return inp
        from prompt_toolkit.input.defaults import create_input

        return create_input()

    def _enter_raw_mode(self):
        """raw 模式; Windows 下不启用 ENABLE_VIRTUAL_TERMINAL_INPUT,
        否则 conhost 改发 VT 字节流, 记录级修饰键 (Shift 位) 失效。"""
        if sys.platform == "win32":
            from prompt_toolkit.input.win32 import raw_mode as win32_raw_mode

            return win32_raw_mode(use_win10_virtual_terminal_input=False)
        return self._input.raw_mode()

    # --- 尺寸/几何 ----------------------------------------------------------------

    def _read_size(self) -> tuple[int, int]:
        size = self._output.get_size()
        return max(1, size.columns), max(1, size.rows)

    @property
    def columns(self) -> int:
        try:
            self._last_size = self._read_size()
        except Exception as exc:  # 尺寸读取失败: 沿用上次良好值并记录
            self.failures.append(f"get_size failed: {exc!r}")
        if self._last_size is None:
            return 80
        return self._last_size[0]

    @property
    def rows(self) -> int:
        try:
            self._last_size = self._read_size()
        except Exception as exc:
            self.failures.append(f"get_size failed: {exc!r}")
        if self._last_size is None:
            return 24
        return self._last_size[1]

    @staticmethod
    def _window_origin() -> tuple[int, int]:
        """可见窗口左上角的 buffer 坐标 (C4); 非控制台输出退 (0, 0)。"""
        if sys.platform != "win32":
            return (0, 0)
        try:
            handle = HANDLE(windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE))
            info = CONSOLE_SCREEN_BUFFER_INFO()
            if windll.kernel32.GetConsoleScreenBufferInfo(handle, byref(info)):
                return int(info.srWindow.Left), int(info.srWindow.Top)
        except Exception:
            pass
        return (0, 0)

    def _query_geometry(self) -> TerminalGeometry:
        columns, rows = self._read_size()
        origin_x, origin_y = self._window_origin()
        return TerminalGeometry(columns, rows, origin_x, origin_y)

    @property
    def geometry(self) -> TerminalGeometry:
        try:
            return self._query_geometry()
        except Exception as exc:
            self.failures.append(f"get_size failed: {exc!r}")
            if self._last_size is None:
                return TerminalGeometry(80, 24, *self._window_origin())
            return TerminalGeometry(*self._last_size, *self._window_origin())

    # --- 鼠标模式 (C4/D6) ----------------------------------------------------------

    def _enable_mouse(self) -> None:
        """进入 raw mode 前记录原始 input mode 并启用原生鼠标。

        启用 ENABLE_MOUSE_INPUT + WINDOW_INPUT (可观察失焦), 配合
        EXTENDED_FLAGS 关闭 QuickEdit, 保持 VT input 关闭。任一步失败都保留
        键盘回退并带 mouse_setup_error; 不尝试就不承诺恢复。
        """
        self.mouse_supported = False
        self.mouse_setup_error = None
        if sys.platform != "win32":
            return  # 非 Windows 后端不注册鼠标能力, 键盘路径保持
        handle = HANDLE(windll.kernel32.GetStdHandle(STD_INPUT_HANDLE))
        mode = DWORD()
        if not windll.kernel32.GetConsoleMode(handle, byref(mode)):
            self.mouse_setup_error = "console-mode-unavailable"
            self.failures.append("mouse setup failed: GetConsoleMode")
            return
        self._original_input_mode = mode.value
        new_mode = (
            mode.value
            | _ENABLE_MOUSE_INPUT
            | _ENABLE_WINDOW_INPUT
            | _ENABLE_EXTENDED_FLAGS
        ) & ~(_ENABLE_QUICK_EDIT_MODE | _ENABLE_VIRTUAL_TERMINAL_INPUT)
        if not windll.kernel32.SetConsoleMode(handle, DWORD(new_mode)):
            self._original_input_mode = None
            self.mouse_setup_error = "mouse-enable-failed"
            self.failures.append("mouse setup failed: SetConsoleMode")
            return
        self.mouse_supported = True

    def _restore_input_mode(self, errors: list[str]) -> None:
        """恢复进入前记录的原始 input mode; 不能证明恢复时如实记错。"""
        if self._original_input_mode is None:
            return
        mode, self._original_input_mode = self._original_input_mode, None
        try:
            handle = HANDLE(windll.kernel32.GetStdHandle(STD_INPUT_HANDLE))
            if not windll.kernel32.SetConsoleMode(handle, DWORD(mode)):
                errors.append("input mode restore failed: SetConsoleMode")
        except Exception as exc:
            errors.append(f"input mode restore failed: {exc!r}")

    # --- 生命周期 ----------------------------------------------------------------

    def start(self, on_input: InputBatchCallback, on_resize: ResizeCallback) -> None:
        self.input_failed = False
        self._on_resize = on_resize
        try:
            self._enable_mouse()  # 先记录原始 mode, 再进入 raw (C4)
            self._raw_mode = self._enter_raw_mode()
            self._raw_mode.__enter__()
            # bracketed paste: 粘贴内容以单事件到达, 换行不会触发提交
            self._output.write_raw("\x1b[?2004h")
            self._output.flush()

            def input_ready() -> None:
                try:
                    batch = list(self._input.read_keys())
                    batch.extend(self._input.flush_keys())  # 强制解析器落出孤立 Escape
                except Exception as exc:
                    self.input_failed = True
                    self.failures.append(f"input read failed: {exc!r}")
                    return
                if batch:
                    on_input(batch)

            self._attach_cm = self._input.attach(input_ready)
            self._attach_cm.__enter__()
            self._last_size = self._read_size()
            loop = asyncio.get_running_loop()
            self._poll_task = loop.create_task(self._poll_size())
        except Exception as exc:
            # 部分开启失败: 回滚已开启状态 (raw/paste/attach/mode), 不留下半开终端
            errors = self.stop()
            self.failures = [f"terminal start failed: {exc!r}", *errors]
            raise RuntimeError("terminal start failed") from exc

    async def _poll_size(self) -> None:
        """尺寸轮询: Windows 无 SIGWINCH, 适配器负责周期发现变化。"""
        try:
            while True:
                await asyncio.sleep(RESIZE_POLL_INTERVAL_S)
                try:
                    size = self._read_size()
                except Exception as exc:
                    self.failures.append(f"get_size failed: {exc!r}")
                    continue
                if size != self._last_size:
                    self._last_size = size
                    if self._on_resize is not None:
                        self._on_resize()
        except asyncio.CancelledError:
            pass

    def stop(self) -> list[str]:
        errors: list[str] = list(self.failures)
        self.failures = []
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
        if self._attach_cm is not None:
            try:
                self._attach_cm.__exit__(None, None, None)
            except Exception as exc:
                errors.append(f"input detach failed: {exc!r}")
            self._attach_cm = None
        try:
            self._output.write_raw("\x1b[?2004l")  # 关闭 bracketed paste
            self._output.flush()
        except Exception as exc:
            errors.append(f"paste mode restore failed: {exc!r}")
        if self._raw_mode is not None:
            try:
                self._raw_mode.__exit__(None, None, None)
            except Exception as exc:
                errors.append(f"raw mode restore failed: {exc!r}")
            self._raw_mode = None
        self._restore_input_mode(errors)  # 原始 input mode (含鼠标位) 最后恢复
        return errors

    # --- 输出/输入 ---------------------------------------------------------------

    def write(self, data: str) -> None:
        self._output.write_raw(data)
        self._output.flush()

    def flush_keys(self) -> list[InputBatchItem]:
        try:
            return list(self._input.flush_keys())
        except Exception as exc:
            self.input_failed = True
            self.failures.append(f"flush_keys failed: {exc!r}")
            return []


class FakeTerminal:
    """headless 终端: 脚本化按键、可控尺寸、写捕获与故障注入。"""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self.columns = columns
        self.rows = rows
        self.origin_x = 0  # 可见窗口 origin 可注入 (C4)
        self.origin_y = 0
        self.mouse_supported = True
        self.mouse_setup_error: str | None = None
        self.written: list[str] = []
        self.feed_screen: Callable[[str], None] | None = None  # 写镜像 (VT 屏断言用)
        self.fail_write = False
        self.restore_errors: list[str] = []
        self.started = False
        self.stopped = False
        self._on_input: InputBatchCallback | None = None
        self._on_resize: ResizeCallback | None = None

    # --- 测试驱动接口 ---------------------------------------------------------------

    def emit(self, keys: list[InputBatchItem]) -> None:
        """推送一批输入 (同步调用 on_input, 模拟一次 read_keys 批次)。"""
        if self._on_input is not None:
            self._on_input(list(keys))

    def set_size(self, columns: int, rows: int) -> None:
        self.columns, self.rows = columns, rows
        if self._on_resize is not None:
            self._on_resize()

    @property
    def geometry(self) -> TerminalGeometry:
        return TerminalGeometry(self.columns, self.rows, self.origin_x, self.origin_y)

    @property
    def output_text(self) -> str:
        return "".join(self.written)

    # --- Terminal 协议 ---------------------------------------------------------------

    def start(self, on_input: InputBatchCallback, on_resize: ResizeCallback) -> None:
        self.started = True
        self._on_input = on_input
        self._on_resize = on_resize

    def stop(self) -> list[str]:
        self.stopped = True
        return list(self.restore_errors)

    def write(self, data: str) -> None:
        if self.fail_write:
            raise OSError("injected write failure")
        self.written.append(data)
        if self.feed_screen is not None:
            self.feed_screen(data)

    def flush_keys(self) -> list[InputBatchItem]:
        return []


__all__ = [
    "FakeTerminal",
    "PointerEvent",
    "POINTER_KINDS",
    "PromptToolkitTerminal",
    "RESIZE_POLL_INTERVAL_S",
    "Terminal",
    "TerminalGeometry",
]
