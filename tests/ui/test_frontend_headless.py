"""票据 01 无头组合检查: MainScreenFrontend + FakeTerminal (P06/P08/P10 最小场景).

覆盖计划 01 的故障门槛: 孤立 Escape、Alt+Enter 跨 read 批次/键前缀超时、
paste 换行无误提交; write 故障恢复; 退出恢复序列。真实 IME/缩放
手感由 Windows Terminal 人工探针核对, 不在本文件范围。
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.key_processor import KeyPress

from cicada.ui.editor import Editor
from cicada.ui.frontend import ESCAPE_PREFIX_TIMEOUT_S, MainScreenFrontend
from cicada.ui.terminal_io import FakeTerminal
from tests.ui.vt_screen import VtScreen

ENTER = KeyPress(Keys.ControlM, "\r")
ESCAPE = KeyPress(Keys.Escape, "\x1b")
CTRL_C = KeyPress(Keys.ControlC, "\x03")
CTRL_D = KeyPress(Keys.ControlD, "\x04")
BACKSPACE = KeyPress(Keys.Backspace, "\x7f")


def text_keys(text: str) -> list[KeyPress]:
    return [KeyPress(ch, ch) for ch in text]


class Harness:
    """headless 前端驱动: FakeTerminal + VtScreen 镜像 + 可控文档。"""

    def __init__(self, *, escape_timeout: float = ESCAPE_PREFIX_TIMEOUT_S, columns=40, rows=12):
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        self.doc_lines: list[str] = []
        self.editor = Editor()
        self.editor.focused = True
        self.submitted: list[str] = []
        self.escape_count = 0
        self.ctrl_c_count = 0
        self.exit_requested = False
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=lambda width: list(self.doc_lines),
            editor=self.editor,
            on_submit=self.submitted.append,
            on_escape=self._on_escape,
            on_ctrl_c=self._on_ctrl_c,
            on_exit=self._on_exit,
            escape_prefix_timeout=escape_timeout,
        )
        self.runner: asyncio.Task | None = None

    def _on_escape(self) -> None:
        self.escape_count += 1

    def _on_ctrl_c(self) -> None:
        self.ctrl_c_count += 1

    def _on_exit(self) -> None:
        self.exit_requested = True
        self.frontend.request_exit()

    async def start(self) -> None:
        self.runner = asyncio.ensure_future(self.frontend.run_async())
        await self.settle()

    async def settle(self, ticks: int = 3) -> None:
        # 真实短睡眠: 节流渲染 (16ms call_later) 与 Escape 定时器需要真实时间
        await asyncio.sleep(0.05)
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def stop(self) -> list[str]:
        self.frontend.request_exit()
        return await asyncio.wait_for(self.runner, timeout=5)

    def screen(self) -> list[str]:
        return self.vt.screen_text()

    def find(self, needle: str) -> bool:
        return self.vt.find_row(needle) is not None


@pytest.mark.asyncio
async def test_enter_submits_and_clears_draft():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("你好"))
        await h.settle()
        assert h.find("你好")  # 草稿可见
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.submitted == ["你好"]
        assert h.editor.is_empty()
    finally:
        errors = await h.stop()
    assert errors == []


@pytest.mark.asyncio
async def test_background_stream_updates_entry_while_typing_chinese():
    """P06 最小场景: 后台原位更新同一条目期间输入中文草稿不被破坏。"""

    async def stream(h: Harness) -> None:
        for chunk in ["回答第一段。", "回答第二段,包含中文与标点。"]:
            await asyncio.sleep(0.01)
            h.doc_lines[0] += chunk
            h.frontend.request_render()

    h = Harness()
    try:
        await h.start()
        h.doc_lines = ["条目: "]
        h.frontend.request_render()
        await h.settle()
        streamer = asyncio.ensure_future(stream(h))
        h.terminal.emit(text_keys("中文草稿"))
        await streamer
        await h.settle()
        joined = "\n".join(h.screen())
        assert joined.count("回答第一段。") == 1  # 原位更新, 不二次打印
        assert "回答第二段" in joined
        assert h.find("中文草稿")  # 后台输出期间草稿保持
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_alt_enter_same_batch_newline_no_submit():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("第一行"))
        h.terminal.emit([ESCAPE, ENTER])  # 同一批次: Alt+Enter
        h.terminal.emit(text_keys("第二行"))
        await h.settle()
        assert h.submitted == []
        assert h.editor.get_text() == "第一行\n第二行"
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_alt_enter_cross_batch_within_window_newline():
    # 加大前缀窗口保证跨批次确定性 (生产默认 10ms, 真实手感由人工探针核对)
    h = Harness(escape_timeout=1.0)
    try:
        await h.start()
        h.terminal.emit(text_keys("跨批"))
        h.terminal.emit([ESCAPE])
        await asyncio.sleep(0.02)  # 跨 read 批次 (窗口内)
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.submitted == []
        assert h.editor.get_text() == "跨批\n"
        assert h.escape_count == 0  # 未被拆成取消
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_lone_escape_dispatches_without_submit():
    h = Harness(escape_timeout=0.001)
    try:
        await h.start()
        h.terminal.emit(text_keys("草稿"))
        h.terminal.emit([ESCAPE])
        await asyncio.sleep(0.05)
        await h.settle()
        assert h.escape_count == 1
        assert h.submitted == []
        assert h.editor.get_text() == "草稿"
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_enter_after_escape_timeout_is_fresh_submit():
    """已公布超时规则: 窗口外的 Enter 是新提交, 不是 Alt+Enter。"""
    h = Harness(escape_timeout=0.001)
    try:
        await h.start()
        h.terminal.emit(text_keys("内容"))
        h.terminal.emit([ESCAPE])
        await asyncio.sleep(0.05)  # 超过窗口
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.escape_count == 1
        assert h.submitted == ["内容"]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_bracketed_paste_multiline_does_not_submit():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit([KeyPress(Keys.BracketedPaste, "粘贴一\n粘贴二\r\n粘贴三")])
        await h.settle()
        assert h.submitted == []
        assert h.editor.get_text() == "粘贴一\n粘贴二\n粘贴三"  # \r\n 归一化
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_idle_frames_write_nothing():
    h = Harness()
    try:
        await h.start()
        await h.settle()
        baseline = len(h.terminal.written)
        await asyncio.sleep(0.05)
        await h.settle()
        assert len(h.terminal.written) == baseline  # P10: 稳定后无新写
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_resize_rewraps_document():
    h = Harness(columns=40, rows=12)
    try:
        await h.start()
        h.doc_lines = ["这一行内容在窄屏下必须重新换行排版显示"]
        h.frontend.request_render()
        await h.settle()
        h.terminal.set_size(16, 12)  # P08: 尺寸变化触发重排
        await h.settle()
        body = [line for line in h.screen() if line]
        assert all(len(line) <= 16 for line in body)
        assert "".join(body).count("这一行内容") >= 1  # 内容完整回放
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_write_failure_recovers_on_retry():
    h = Harness()
    try:
        await h.start()
        await h.settle()
        h.terminal.fail_write = True
        h.doc_lines = ["故障行"]
        h.frontend.request_render()
        await asyncio.sleep(0.06)  # 首次渲染失败已记录, 恢复重试 (200ms) 未到
        await h.settle(0)
        assert any("render failed" in f for f in h.frontend.ui_failures)
        assert not h.frontend.terminal_failed  # 已排程延迟重试
        h.terminal.fail_write = False  # 瞬时故障消除
        await asyncio.sleep(0.35)  # 等待恢复重试
        await h.settle()
        assert not h.frontend.terminal_failed
        assert h.find("故障行")  # P10: 恢复后完整重放成功
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_permanent_write_failure_sets_terminal_failed_and_exits():
    h = Harness()
    try:
        await h.start()
        await h.settle()
        h.terminal.fail_write = True
        h.doc_lines = ["必败"]
        h.frontend.request_render()
        await asyncio.wait_for(h.runner, timeout=5)
    except asyncio.TimeoutError:
        raise AssertionError("永久写失败应触发退出")
    assert h.frontend.terminal_failed
    assert any("permanently" in f for f in h.frontend.ui_failures)
    assert h.terminal.stopped  # 终端恢复流程已执行


def test_prompt_toolkit_size_failure_falls_back_to_last_good():
    """适配层 get_size 失败: 记录失败并沿用上次良好值 (P10 故障矩阵)。"""
    from prompt_toolkit.data_structures import Size

    class FlakyOutput:
        def __init__(self) -> None:
            self.calls = 0
            self.size = Size(rows=30, columns=100)

        def get_size(self) -> Size:
            self.calls += 1
            if self.calls == 1:
                raise OSError("injected size failure")
            return self.size

        def write_raw(self, data: str) -> None:
            pass

        def flush(self) -> None:
            pass

    from cicada.ui.terminal_io import PromptToolkitTerminal

    output = FlakyOutput()
    terminal = PromptToolkitTerminal(input=object(), output=output)
    assert terminal.columns == 80  # 无良好值时回退默认并记录失败
    assert any("get_size failed" in f for f in terminal.failures)
    assert terminal.rows == 30  # 第三次成功后获得真实尺寸
    assert terminal.columns == 100


@pytest.mark.asyncio
async def test_ctrl_c_and_ctrl_d_semantics():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("草稿"))
        h.terminal.emit([CTRL_C])
        await h.settle()
        assert h.ctrl_c_count == 1
        h.terminal.emit([CTRL_D])  # 非空草稿: 不触发退出
        await h.settle()
        assert h.exit_requested is False
        h.editor.clear()
        await h.settle()
        h.terminal.emit([CTRL_D])  # 空草稿: 退出
        await h.settle()
        assert h.exit_requested
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_backspace_and_undo_through_frontend():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("中文abc"))
        h.terminal.emit([BACKSPACE])  # 删 c
        await h.settle()
        assert h.editor.get_text() == "中文ab"
        h.terminal.emit([KeyPress("\x1f", "\x1f")])  # Ctrl+_ 撤销
        await h.settle()
        assert h.editor.get_text() == "中文abc"
    finally:
        await h.stop()


def test_prompt_toolkit_start_failure_rolls_back_partial_modes():
    """故障门槛: raw/attach/paste 部分开启失败 -> 回滚已开启状态并如实报错。"""

    class RawRecorder:
        def __init__(self) -> None:
            self.entered = 0
            self.exited = 0

        def __call__(self):
            return self

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *args):
            self.exited += 1

    class AttachFailingInput:
        def __init__(self) -> None:
            self.raw = RawRecorder()

        def raw_mode(self):
            return self.raw()

        def attach(self, callback):
            raise OSError("injected attach failure")

        def read_keys(self):
            return []

        def flush_keys(self):
            return []

    class RecordingOutput:
        def __init__(self) -> None:
            self.written: list[str] = []

        def write_raw(self, data: str) -> None:
            self.written.append(data)

        def flush(self) -> None:
            pass

        def get_size(self):
            from prompt_toolkit.data_structures import Size

            return Size(rows=24, columns=80)

    import asyncio

    from cicada.ui.terminal_io import PromptToolkitTerminal

    inp = AttachFailingInput()
    out = RecordingOutput()
    terminal = PromptToolkitTerminal(input=inp, output=out)
    terminal._enter_raw_mode = lambda: inp.raw()  # 隔离真实控制台 raw 模式
    with pytest.raises(RuntimeError, match="terminal start failed"):
        asyncio.run(terminal.start(lambda keys: None, lambda: None))
    assert inp.raw.exited == 1  # raw 模式已回滚
    assert any("\x1b[?2004l" in w for w in out.written)  # 粘贴模式已关闭
    assert any("terminal start failed" in f for f in terminal.failures)


def test_record_reader_encodes_shift_enter_deterministically():
    """非 VT 记录读取器: KEY_EVENT_RECORD 的 SHIFT_PRESSED 位确定性区分
    Shift+Enter (Textual/crossterm 同款记录级方案; VT 字节流+GetAsyncKeyState
    采样因竞态弃用)。"""
    from prompt_toolkit.win32_types import KEY_EVENT_RECORD

    from cicada.ui.terminal_io import _ShiftEnterConsoleInputReader

    def make_event(u_char: str, control_state: int) -> KEY_EVENT_RECORD:
        ev = KEY_EVENT_RECORD()
        ev.KeyDown = 1
        ev.RepeatCount = 1
        ev.VirtualKeyCode = 13
        ev.VirtualScanCode = 28
        ev.uChar.UnicodeChar = u_char
        ev.ControlKeyState = control_state
        return ev

    reader = _ShiftEnterConsoleInputReader()
    shift, ctrl, alt = reader.SHIFT_PRESSED, reader.LEFT_CTRL_PRESSED, reader.LEFT_ALT_PRESSED
    # Shift+Enter -> ControlJ (前端换行), 确定性无竞态
    assert reader._event_to_key_presses(make_event("\r", shift)) == [KeyPress(Keys.ControlJ, "\n")]
    # 无修饰 Enter -> ControlM (提交)
    assert reader._event_to_key_presses(make_event("\r", 0)) == [KeyPress(Keys.ControlM, "\r")]
    # Ctrl+Enter 保持 pt 既有: Escape + ControlJ (换行)
    assert reader._event_to_key_presses(make_event("\n", ctrl)) == [
        KeyPress(Keys.Escape, ""),
        KeyPress(Keys.ControlJ, "\n"),
    ]
    # Alt+Enter -> Escape + ControlM (前端 Escape 前缀配对为换行, 兼容保留)
    assert reader._event_to_key_presses(make_event("\r", alt)) == [
        KeyPress(Keys.Escape, ""),
        KeyPress(Keys.ControlM, "\r"),
    ]


@pytest.mark.asyncio
async def test_control_j_newline_no_submit():
    """记录读取器把 Shift+Enter 编码为 ControlJ; 前端收到后换行不提交。"""
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("第一行"))
        h.terminal.emit([KeyPress(Keys.ControlJ, "\n")])
        h.terminal.emit(text_keys("第二行"))
        await h.settle()
        assert h.submitted == []
        assert h.editor.get_text() == "第一行\n第二行"
    finally:
        await h.stop()


def test_prompt_toolkit_size_failure_falls_back_to_last_good():
    """适配层 get_size 失败: 记录失败并沿用上次良好值 (P10 故障矩阵)。"""
    from prompt_toolkit.data_structures import Size

    class FlakyOutput:
        def __init__(self) -> None:
            self.calls = 0
            self.size = Size(rows=30, columns=100)

        def get_size(self) -> Size:
            self.calls += 1
            if self.calls == 1:
                raise OSError("injected size failure")
            return self.size

        def write_raw(self, data: str) -> None:
            pass

        def flush(self) -> None:
            pass

    from cicada.ui.terminal_io import PromptToolkitTerminal

    output = FlakyOutput()
    terminal = PromptToolkitTerminal(input=object(), output=output)
    assert terminal.columns == 80  # 无良好值时回退默认并记录失败
    assert any("get_size failed" in f for f in terminal.failures)
    assert terminal.rows == 30  # 第三次成功后获得真实尺寸
    assert terminal.columns == 100


@pytest.mark.asyncio
async def test_ctrl_c_and_ctrl_d_semantics():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("草稿"))
        h.terminal.emit([CTRL_C])
        await h.settle()
        assert h.ctrl_c_count == 1
        h.terminal.emit([CTRL_D])  # 非空草稿: 不触发退出
        await h.settle()
        assert h.exit_requested is False
        h.editor.clear()
        await h.settle()
        h.terminal.emit([CTRL_D])  # 空草稿: 退出
        await h.settle()
        assert h.exit_requested
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_backspace_and_undo_through_frontend():
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("中文abc"))
        h.terminal.emit([BACKSPACE])  # 删 c
        await h.settle()
        assert h.editor.get_text() == "中文ab"
        h.terminal.emit([KeyPress("\x1f", "\x1f")])  # Ctrl+_ 撤销
        await h.settle()
        assert h.editor.get_text() == "中文abc"
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_plain_enter_after_paste_text_still_submits():
    """换行键机制回归哨兵: 无 Shift 的 Enter 保持提交语义。"""
    h = Harness()
    try:
        await h.start()
        h.terminal.emit(text_keys("普通提交"))
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.submitted == ["普通提交"]
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_submit_acceptance_contract():
    """受理制提交: on_submit 返回 False = 拒绝, 草稿保留 (02 准入语义前提);
    None/True = 接受, 草稿清空; 空白草稿直接清空不派发。"""
    h = Harness()
    results = [False, None]

    def gate(text: str):
        return results.pop(0)

    try:
        await h.start()
        h.frontend._on_submit = gate
        h.terminal.emit(text_keys("第一问"))
        h.terminal.emit([ENTER])  # False: 拒绝
        await h.settle()
        assert h.submitted == []
        assert h.editor.get_text() == "第一问"
        h.terminal.emit([ENTER])  # None: 接受
        await h.settle()
        assert h.editor.is_empty()
        h.terminal.emit(text_keys("   "))
        h.terminal.emit([ENTER])  # 空白草稿: 清空不派发
        await h.settle()
        assert h.editor.is_empty()
        assert h.submitted == []
    finally:
        await h.stop()


@pytest.mark.asyncio
async def test_ctrl_o_hook():
    """Ctrl+O 钩子 (03 工具展开切换的接缝); 无宿主时无动作不崩溃。"""
    h = Harness()
    count = []
    try:
        await h.start()
        h.frontend._on_ctrl_o = lambda: count.append(1)
        h.terminal.emit([KeyPress(Keys.ControlO, "\x0f")])
        h.terminal.emit(text_keys("草稿"))
        await h.settle()
        assert count == [1]
        assert h.editor.get_text() == "草稿"  # Ctrl+O 不进草稿
    finally:
        await h.stop()


async def test_run_async_cancel_still_restores_terminal():
    """任务被取消: run_async 收尾仍执行 terminal.stop (US21 取消退出的终端恢复)。"""
    h = Harness()
    await h.start()
    h.runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(h.runner, timeout=5)
    assert h.terminal.stopped, "取消路径不跳过终端模式恢复"
    assert h.frontend._stopped
