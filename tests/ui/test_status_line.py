"""tui-aesthetics 06: 输入框上方动态状态行 (规格 §F)。

组合层接缝: ConversationView.status_lines(width) — idle 无行, 忙时 1 行
(braille spinner + 真实状态文案), spinner 帧随注入时钟前进; 动画任务在
忙时驱动 on_update, 转闲即停, 关闭不泄漏。前端接缝 (status_lines 注入
MainScreenFrontend) 的位置与行预算测试在 test_frontend_headless.py。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cicada.ui.conversation_view import build
from cicada.ui.main_screen import strip_terminal_sequences
from tests.ui.test_conversation_view import (
    ChunkedFakeModel,
    _repo,
    _prompt_info,
    _wait_until,
)
from cicada.application import ApplicationConfig, open_application
from cicada.plugins.fake_model import fake_model_plugin

SPINNER_FG = "\x1b[38;2;178;148;187m"  # spinner 紫 #b294bb
TEXT_FG = "\x1b[38;2;128;128;128m"     # 文案 #808080


def _slow_config(root, *, delay: float = 0.05) -> ApplicationConfig:
    script = json.dumps([{"text": "分段回答内容, 足够长以覆盖多个延迟块。" * 3, "stop": True}])
    return ApplicationConfig(
        workspace=root,
        system_prompt="status line test",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(ChunkedFakeModel(script, chunks=6, delay=delay)),
        plan=None,
        max_turns=6,
        model_label="fake",
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0


async def _open_view(tmp_path, clock: _Clock):
    root = _repo(tmp_path)
    session = await open_application(_slow_config(root))
    renders: list[int] = []
    bundle = build(session, on_update=lambda: renders.append(1), clock=lambda: clock.now)
    return session, bundle, renders


@pytest.mark.asyncio
async def test_idle_has_no_status_line(tmp_path):
    session, bundle, _ = await _open_view(tmp_path, _Clock())
    try:
        assert bundle["status_lines"](80) == []
    finally:
        await session.aclose()


async def test_late_model_deltas_do_not_revive_or_replace_current_stage(tmp_path):
    """完成轮次与旧轮次的迟到事件不能把状态行重新切到思考/输出。"""
    from cicada.core.events import (
        AssistantCompleted, AssistantReasoningDelta, AssistantTextDelta,
        RunFinished, RunStarted, TurnStarted,
    )
    from cicada.core.messages import AssistantMessage

    session, bundle, _ = await _open_view(tmp_path, _Clock())
    view = bundle["view"]
    try:
        assert bundle["on_submit"]("你好")
        project = view._project_agent_event
        project(RunStarted("late-events"))
        project(TurnStarted("late-events", 0))
        project(AssistantCompleted("late-events", AssistantMessage(text="完成")))
        for event_type in (AssistantReasoningDelta, AssistantTextDelta):
            project(event_type("late-events", 0, "迟到"))
            assert "正在执行" in strip_terminal_sequences(bundle["status_lines"](80)[0])
        project(TurnStarted("late-events", 1))
        for event_type in (AssistantReasoningDelta, AssistantTextDelta):
            project(event_type("late-events", 0, "旧轮次"))
            assert "等待模型" in strip_terminal_sequences(bundle["status_lines"](80)[0])
        project(AssistantReasoningDelta("late-events", 1, ""))
        assert "正在思考" in strip_terminal_sequences(bundle["status_lines"](80)[0])
        project(AssistantTextDelta("late-events", 1, "当前答案"))
        assert "正在输出" in strip_terminal_sequences(bundle["status_lines"](80)[0])
        project(RunFinished("late-events", "stop"))
        project(AssistantReasoningDelta("late-events", 1, "运行已结束"))
        assert "正在执行" in strip_terminal_sequences(bundle["status_lines"](80)[0])
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_status_line_extreme_narrow_width_takes_no_row(tmp_path):
    """审查 06-2: 宽度 < 4 时状态行不占行 (不产生防御截断噪音)。"""
    session, bundle, _ = await _open_view(tmp_path, _Clock())
    try:
        assert bundle["on_submit"]("你好") is True
        for width in (1, 2, 3):
            assert bundle["status_lines"](width) == []
        assert bundle["status_lines"](4) != []
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.usefixtures("color_env")
async def test_executing_shows_spinner_and_frame_follows_clock(tmp_path):
    session, bundle, _ = await _open_view(tmp_path, clock := _Clock())
    try:
        assert bundle["on_submit"]("你好") is True
        line = bundle["status_lines"](80)[0]
        plain = strip_terminal_sequences(line)
        # 票据 02 (D4): 提交后未收到 TurnStarted, 状态是"正在执行"而非"正在输出"
        assert "正在执行…" in plain
        assert "⠋" in plain  # clock=0.0 -> 第 0 帧
        assert SPINNER_FG in line and TEXT_FG in line
        clock.now = 0.15  # 第 1 帧
        assert "⠙" in strip_terminal_sequences(bundle["status_lines"](80)[0])
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_status_line_disappears_after_completion(tmp_path):
    session, bundle, _ = await _open_view(tmp_path, _Clock())
    try:
        assert bundle["on_submit"]("你好") is True
        assert await _wait_until(lambda: session.state == "idle" and bundle["status_lines"](80) == [])
        # 完成后状态行不占行
        assert bundle["status_lines"](80) == []
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancelling_shows_cancelling_text(tmp_path):
    session, bundle, _ = await _open_view(tmp_path, _Clock())
    view = bundle["view"]
    try:
        assert bundle["on_submit"]("你好") is True
        assert await _wait_until(lambda: session.state == "executing")
        view.handle_ctrl_c()
        assert await _wait_until(lambda: session.state in ("cancelling", "finalizing", "idle"))
        if session.state == "cancelling":
            assert "正在取消…" in strip_terminal_sequences(bundle["status_lines"](80)[0])
        assert await _wait_until(lambda: bundle["status_lines"](80) == [])
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_animation_task_drives_render_and_stops_cleanly(tmp_path):
    clock = _Clock()
    session, bundle, renders = await _open_view(tmp_path, clock)
    view = bundle["view"]
    try:
        assert bundle["on_submit"]("你好") is True
        assert await _wait_until(lambda: view._spinner_task is not None)
        baseline = len(renders)
        await asyncio.sleep(0.35)  # 无新应用事件, 动画任务自行驱动重绘
        assert len(renders) >= baseline + 2
        assert await _wait_until(lambda: session.state == "idle")
        assert await _wait_until(lambda: view._spinner_task is None)
    finally:
        await session.aclose()
    await asyncio.sleep(0.05)
    assert view._spinner_task is None  # 关闭后无残留任务引用
