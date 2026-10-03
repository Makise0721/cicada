"""生产组合的窗口缩放回归: 实际尺寸 + Rich + prompt_toolkit VT 输出.

仅检查活动布局和下一条记录; 已打印 scrollback 不由应用清屏重放.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.utils import get_cwidth

from cicada.application import ApplicationConfig, open_application
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui import build_production_executor
from cicada.ui.contracts import ActivePreview, PresentationRecord, PresentationUpdate, StatusSnapshot


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _text(fragments) -> str:
    return "".join(text for _, text in fragments)


def _line_widths(text: str) -> list[int]:
    return [get_cwidth(line) for line in text.splitlines()]


async def _session(root):
    return await open_application(ApplicationConfig(
        workspace=root,
        system_prompt="test",
        prompt_info=PromptInfo(
            version="test", workspace=root, instructions_path=None,
            instructions_sha256=None, prompt_sha256=hashlib.sha256(b"test").hexdigest(),
        ),
        model_plugin=fake_model_plugin(FakeModel([])),
        model_label="fake",
    ))


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
async def test_next_record_uses_resized_output_in_real_vt_pipeline(tmp_path, role):
    session = await _session(tmp_path)
    stream = io.StringIO()
    size = Size(rows=40, columns=43)
    output = Vt100_Output(stream, lambda: size, enable_cpr=False)
    body = "中文列宽安全" * 20
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        executor = build_production_executor(session, input=pipe, output=output)
        try:
            for index, columns in enumerate((43, 139, 32)):
                size = Size(rows=40, columns=columns)
                record = PresentationRecord(
                    key=f"record-{index}", role=role, title="记录", status="done", body=(body,),
                )
                before = stream.tell()
                executor._apply(PresentationUpdate(appended=(record,)))
                visible = _ANSI.sub("", stream.getvalue()[before:]).replace("\r", "")
                widths = _line_widths(visible)
                assert max(widths) <= columns
                assert max(widths) >= columns - 2, "放宽后的记录不应仍硬换行在100列"
                assert body == record.body[0], "显示排版不改记录原文"
                assert "\x1b[" in stream.getvalue()[before:], "必须经过真实styled VT输出"
        finally:
            await session.aclose()


async def test_active_preview_reflows_without_changing_draft_or_original(tmp_path):
    session = await _session(tmp_path)
    size = Size(rows=40, columns=43)
    output = Vt100_Output(io.StringIO(), lambda: size, enable_cpr=False)
    preview = ActivePreview(run_id="run-1", turn_index=0, text="中文列宽安全" * 20)
    draft = "下一问\n草稿内容"
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=output)
        executor.buffer.text = draft
        executor._apply(PresentationUpdate(preview=preview))
        try:
            for columns in (43, 139, 32):
                size = Size(rows=40, columns=columns)
                widths = _line_widths(_text(executor._get_preview_lines()))
                assert max(widths) <= columns
                assert max(widths) >= columns - 2
                assert executor.buffer.text == draft
                assert executor._display_preview == preview
        finally:
            await session.aclose()


async def test_narrow_status_keeps_phase_and_check_ahead_of_long_workspace(tmp_path):
    session = await _session(tmp_path)
    size = Size(rows=30, columns=60)
    output = Vt100_Output(io.StringIO(), lambda: size, enable_cpr=False)
    snapshot = StatusSnapshot(
        model="qwen3.5:9b", workspace="C:/" + "很长的中文工作区/" * 40,
        phase="finalizing", check_status="check-1=passed/current",
        verification_run_id="v-1", input_tokens=1234, output_tokens=123,
    )
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=output)
        executor._apply(PresentationUpdate(status=snapshot))
        try:
            line = _text(executor._get_status_line())
            assert "finalizing" in line and "check-1=passed/current" in line
            assert "\n" not in line and get_cwidth(line) <= 60
            for columns in (1, 8, 20, 80, 160):
                size = Size(rows=30, columns=columns)
                line = _text(executor._get_status_line())
                assert "\n" not in line and get_cwidth(line) <= columns
            assert executor._display_status == snapshot
        finally:
            await session.aclose()


@pytest.mark.parametrize("check", [
    "check-1=passed/stale",
    "check-1=passed/current | check-2=failed/current",
])
async def test_narrow_status_marks_omitted_check_summary_instead_of_partial_pass(tmp_path, check):
    session = await _session(tmp_path)
    output = Vt100_Output(io.StringIO(), lambda: Size(rows=8, columns=40), enable_cpr=False)
    snapshot = StatusSnapshot(phase="finalizing", check_status=check)
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=output)
        executor._apply(PresentationUpdate(status=snapshot))
        try:
            line = _text(executor._get_status_line())
            assert "finalizing" in line
            assert "省略" in line and "/status" in line
            assert "passed" not in line
            assert get_cwidth(line) <= 40
            assert executor._display_status == snapshot
            assert check in executor._status_summary(), "完整事实仍可由/status查看"
        finally:
            await session.aclose()


async def test_windows_size_poll_redraws_active_layout_without_input(tmp_path):
    session = await _session(tmp_path)
    size = Size(rows=40, columns=120)
    output = Vt100_Output(io.StringIO(), lambda: size, enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        executor = build_production_executor(session, input=pipe, output=output)
        executor._apply(PresentationUpdate(preview=ActivePreview(
            run_id="run-1", turn_index=0, text="中文列宽安全" * 20,
        ), status=StatusSnapshot(phase="executing", check_status="pending")))
        draft = "保留草稿\n下一行"
        executor.buffer.text = draft
        frames = []
        executor.app.terminal_size_polling_interval = 0.02
        def capture_frame(app):
            screen = app.renderer.last_rendered_screen
            if screen is None or app.renderer._last_size is None:
                return  # 退出时 renderer 已 reset, 不属于活动布局.
            rows = ["".join(char.char for _, char in sorted(row.items()))
                    for _, row in sorted(screen.data_buffer.items())]
            frames.append((
                app.renderer._last_size,
                max(_line_widths(_text(executor._get_preview_lines()))),
                executor.buffer.text,
                rows,
            ))
        executor.app.after_render += capture_frame
        runner = asyncio.create_task(executor.run_async())
        try:
            await _wait(lambda: frames)
            # 首次 poll 先建立旧尺寸, 然后只改Output.get_size, 不发按键/应用事件.
            await asyncio.sleep(0.05)
            size = Size(rows=8, columns=48)
            await _wait(lambda: any(frame[0] == size for frame in frames))
            frame = next(frame for frame in frames if frame[0] == size)
            assert frame[1] <= 48
            assert frame[2] == draft
            assert len(frame[3]) <= 8
            visible = "\n".join(frame[3])
            assert "保留草稿" in visible and "下一行" in visible
            assert "executing" in visible and "pending" in visible
        finally:
            executor.app.exit()
            await asyncio.wait_for(runner, timeout=5)


async def _wait(predicate) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)
