"""终端输入/输出适配 (票据 01): prompt_toolkit 公开低层 API 的 Python 适配.

来源职责: pi-mono@1a584a7a packages/tui/src/terminal.ts (MIT, Copyright
(c) 2025 Mario Zechner)。平台实现按设计用 prompt_toolkit 公开低层能力替换
Node stdin/stdout:
- Input: create_input() 的 raw_mode/attach/read_keys/flush_keys。附着的
  回调由宿主事件循环触发 (Windows 经 executor + call_soon_threadsafe),
  读到的 KeyPress 批次交 on_input; 批后立即 flush_keys 强制解析器落出
  孤立 Escape ( Application 同款), 键前缀超时归 frontend 所有。
- Output: create_output() 的 write_raw/flush/get_size。Windows Terminal
  下是 Windows10_Output (VT 直写); 行级光标/清屏序列由 renderer 产生,
  本模块不解释内容。
- 适配器拥有: bracketed paste 开关序列、raw 模式生命周期、尺寸轮询、
  停止时的模式恢复与恢复错误收集。write 失败向上抛, 恢复矩阵归 frontend。

FakeTerminal 供 headless 检查与演示注入: 可控尺寸、脚本化按键、写捕获与
write 故障注入 (fail_write)、停止恢复错误注入 (restore_errors); get_size/
输入读取故障由测试对真实适配器以替身注入 (见 test_frontend_headless)。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Callable, Protocol

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

RESIZE_POLL_INTERVAL_S = 0.2

InputBatchCallback = Callable[[list[KeyPress]], None]
ResizeCallback = Callable[[], None]


class Terminal(Protocol):
    """前端唯一终端依赖面 (01 冻结); 实现不得偷用 prompt_toolkit Application。"""

    def start(self, on_input: InputBatchCallback, on_resize: ResizeCallback) -> None: ...

    def stop(self) -> list[str]:
        """恢复终端状态, 返回恢复过程中的错误描述 (空列表 = 干净恢复)。"""
        ...

    def write(self, data: str) -> None: ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    def flush_keys(self) -> list[KeyPress]: ...


_ShiftEnterConsoleInputReader = None  # type: ignore[assignment]; POSIX 首版不在范围
if sys.platform == "win32":
    from prompt_toolkit.input.win32 import ConsoleInputReader

    class _ShiftEnterConsoleInputReader(ConsoleInputReader):  # type: ignore[misc,valid-type]
        """Shift+Enter 确定性编码 (2026-10-03 用户裁决: 主换行键)。

        非 VT 记录读取器的每个 KEY_EVENT_RECORD 自带 ControlKeyState;
        Enter 按下时 SHIFT_PRESSED 且无 Ctrl → 编码为 Keys.ControlJ (换行),
        与 Textual/crossterm 等 Windows TUI 的记录级修饰键方案同源。
        VT 字节流路径丢失记录级修饰键, 事后 GetAsyncKeyState 采样有竞态
        (实测快速敲击/中文 IME 下不可靠), 已弃用。
        """

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

    @staticmethod
    def _create_input():
        if sys.platform == "win32":
            # Windows Terminal 下默认走 VT 字节流路径 (修饰键信息丢失);
            # 覆盖为非 VT 记录读取器, ControlKeyState 确定性区分 Shift+Enter。
            from prompt_toolkit.input.win32 import Win32Input

            inp = Win32Input()
            inp.console_input_reader = _ShiftEnterConsoleInputReader()
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

    # --- 尺寸 -----------------------------------------------------------------

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

    # --- 生命周期 ----------------------------------------------------------------

    def start(self, on_input: InputBatchCallback, on_resize: ResizeCallback) -> None:
        self.input_failed = False
        self._on_resize = on_resize
        self._raw_mode = self._enter_raw_mode()
        self._raw_mode.__enter__()
        try:
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
            # 部分开启失败: 回滚已开启状态 (raw/paste/attach), 不留下半开终端
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
        return errors

    # --- 输出/输入 ---------------------------------------------------------------

    def write(self, data: str) -> None:
        self._output.write_raw(data)
        self._output.flush()

    def flush_keys(self) -> list[KeyPress]:
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
        self.written: list[str] = []
        self.feed_screen: Callable[[str], None] | None = None  # 写镜像 (VT 屏断言用)
        self.fail_write = False
        self.restore_errors: list[str] = []
        self.started = False
        self.stopped = False
        self._on_input: InputBatchCallback | None = None
        self._on_resize: ResizeCallback | None = None

    # --- 测试驱动接口 ---------------------------------------------------------------

    def emit(self, keys: list[KeyPress]) -> None:
        """推送一批按键 (同步调用 on_input, 模拟一次 read_keys 批次)。"""
        if self._on_input is not None:
            self._on_input(list(keys))

    def set_size(self, columns: int, rows: int) -> None:
        self.columns, self.rows = columns, rows
        if self._on_resize is not None:
            self._on_resize()

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

    def flush_keys(self) -> list[KeyPress]:
        return []


__all__ = [
    "FakeTerminal",
    "PromptToolkitTerminal",
    "RESIZE_POLL_INTERVAL_S",
    "Terminal",
]
