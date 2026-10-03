"""05 组合路径 e2e: --tui 生产组合 (共享应用 + 正式 presenter + 渲染 + 终端执行器).

经注入 input/output (create_pipe_input + DummyOutput) 驱动真实
ProductionTerminalExecutor, 覆盖 T04/T05/T07/T11 组合路径与 T14 的进程内
退出码; CLI 子进程层面的 T14 (非 TTY/缺依赖/零 UI 导入) 在 tests/test_cli.py.
真实终端人工核对 (IME/粘贴/resize/恢复) 留待 06, 自动化不能代签.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cicada.__main__ import parse_script
from cicada.application import AgentEventReceived
from cicada.core.cancel import CancelToken
from cicada.core.events import AssistantCompleted, AssistantTextDelta, TurnStarted
from cicada.core.messages import AssistantMessage
from cicada.core.ports import StreamDone, StreamEvent, TextDelta
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.plugins.fake_model import fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui import build_production_executor, run_tui_session
from cicada.ui.rendering import record_fragments

# --- 夹具 (与 test_terminal / test_tui_verification 同构; 测试文件保持自包含) ---------------------


class ChunkedFakeModel:
    """parse_script JSON 剧本 + 可控分块/延迟的 fake 模型."""

    def __init__(self, script_json: str, *, chunks: int = 4, delay: float = 0.0) -> None:
        self._script = parse_script(script_json)
        self._chunks = chunks
        self._delay = delay
        self.requests: list = []

    async def stream(self, request, cancel: CancelToken):
        self.requests.append(request)
        entry: list[StreamEvent] = self._script.pop(0) if self._script else [StreamDone("stop")]
        for event in entry:
            if cancel.cancelled:
                yield StreamDone("aborted")
                return
            if isinstance(event, TextDelta) and event.text and self._chunks > 1:
                size = -(-len(event.text) // self._chunks)
                for start in range(0, len(event.text), size):
                    if cancel.cancelled:
                        yield StreamDone("aborted")
                        return
                    await asyncio.sleep(self._delay)
                    yield TextDelta(event.text[start : start + size])
            else:
                await asyncio.sleep(self._delay)
                yield event


def _script(*entries: dict) -> str:
    return json.dumps(list(entries))


def _prompt_info(root: Path) -> PromptInfo:
    return PromptInfo(
        version="test",
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=hashlib.sha256(b"test").hexdigest(),
    )


def _config(root: Path, script_json: str, *, chunks: int = 4, delay: float = 0.0, plan=None):
    from cicada.application import ApplicationConfig

    model = ChunkedFakeModel(script_json, chunks=chunks, delay=delay)
    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(model),
        plan=plan,
        max_turns=6,
        model_label="fake",
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "baseline"], check=True)
    return root


class RecordingRenderSink:
    """捕获生产渲染: print_record 存渲染纯文本, print_line 存行文本."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.rendered: list[str] = []

    def print_line(self, text: str = "") -> None:
        self.lines.append(text)

    def print_record(self, record) -> None:
        fragments = record_fragments(record)
        self.rendered.append("".join(text for _, text in fragments))

    def rendered_count(self, needle: str) -> int:
        return sum(1 for text in self.rendered if needle in text)


async def wait_until(predicate, timeout: float = 10.0, step: float = 0.01) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


def _done_count(sink: RecordingRenderSink) -> int:
    return sink.rendered_count("运行结束")


def _model_of(session) -> ChunkedFakeModel:
    model = session._app.agent._model
    if type(model).__name__ == "RunPolicy":
        model = model.inner
    assert isinstance(model, ChunkedFakeModel)
    return model


# --- T04: 增量预览 -> 最终回答一次 -> 预览收起 (生产渲染管线) ------------------------------------


async def test_t04_production_streaming_final_once_and_preview_collapse(tmp_path):
    answer = "第一段流式内容。\n第二段内容。"
    session = await open_app(_config(tmp_path, _script({"text": answer, "stop": True}), chunks=6, delay=0.02))
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=DummyOutput(), sink=sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            assert await wait_until(lambda: executor._consumer_task is not None)
            pipe.send_text("问题\r")
            assert await wait_until(lambda: any("[user]" in line for line in sink.lines))
            assert await wait_until(lambda: executor._display_preview is not None)
            assert await wait_until(
                lambda: executor._display_preview is not None
                and "第一段流式内容" in executor._display_preview.text
            )
            assert await wait_until(lambda: _done_count(sink) >= 1)
            assert executor._display_preview is None, "最终完成收起活动预览"
            assert sink.rendered_count("第一段流式内容") == 1, "最终回答只渲染一次"
            assert any("第二段内容" in text for text in sink.rendered)
            assert all("\x1b" not in text for text in sink.rendered), "可见产物无控制码"
        finally:
            pipe.send_text("/exit\r")
            await asyncio.wait_for(runner, timeout=15)


async def open_app(config):
    from cicada.application import open_application

    return await open_application(config)


# --- T05: 忙碌提交拒绝不排队 (生产组合) ---------------------------------------------------------


async def test_t05_production_busy_submit_rejected_draft_kept(tmp_path):
    # chunks=4/delay=0.15: 忙碌窗口 ~0.6s, 保证草稿输入与忙碌提交落在窗口内
    session = await open_app(
        _config(tmp_path, _script({"text": "慢慢回答", "stop": True}), chunks=4, delay=0.15)
    )
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=DummyOutput(), sink=sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            assert await wait_until(lambda: executor._consumer_task is not None)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: session.state == "executing")
            pipe.send_text("忙碌草稿")
            assert await wait_until(lambda: executor.buffer.text == "忙碌草稿")
            pipe.send_text("\r")
            await asyncio.sleep(0.1)
            assert any("提交被拒绝" in line for line in sink.lines)
            assert len(_model_of(session).requests) == 1
            assert executor.buffer.text == "忙碌草稿"
            assert await wait_until(lambda: _done_count(sink) >= 1)
        finally:
            executor.buffer.reset()  # 草稿有意保留: 清掉再 /exit, 避免拼进草稿
            pipe.send_text("/exit\r")
            await asyncio.wait_for(runner, timeout=15)


# --- T07: finalizing 阶段拒绝新请求 (生产组合) --------------------------------------------------


class SlowService:
    """把真实验证服务的 refresh 延迟化, 制造可观察的 finalizing 窗口."""

    def __init__(self, inner, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    @property
    def plan(self):
        return self._inner.plan

    def mark_process_uncertain(self, reason: str) -> None:
        self._inner.mark_process_uncertain(reason)

    async def initialize(self, cancel: CancelToken):
        return await self._inner.initialize(cancel)

    async def run_check(self, check_id: str, cancel: CancelToken):
        return await self._inner.run_check(check_id, cancel)

    async def refresh(self, cancel: CancelToken):
        await asyncio.sleep(self._delay)
        return await self._inner.refresh(cancel)

    async def finalize(self, cancel: CancelToken):
        return await self._inner.finalize(cancel)


async def test_t07_production_finalizing_submit_rejected(tmp_path):
    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    session = await open_app(_config(root, _script({"text": "直接回答", "stop": True}), plan=plan))
    session._service = SlowService(session._service, delay=0.8)  # type: ignore[arg-type]
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=DummyOutput(), sink=sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            assert await wait_until(lambda: executor._consumer_task is not None)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: session.state == "finalizing", timeout=10)
            pipe.send_text("插队\r")
            await asyncio.sleep(0.1)
            assert any("提交被拒绝" in line for line in sink.lines)
            assert await wait_until(lambda: _done_count(sink) >= 1)
            assert session.state == "idle"
            assert len(_model_of(session).requests) == 1
        finally:
            executor.buffer.reset()  # 插队草稿有意保留: 清掉再 /exit
            pipe.send_text("/exit\r")
            await asyncio.wait_for(runner, timeout=15)


# --- T11 + T14: run_tui_session 退出码 / 首问 / 取消后关闭 --------------------------------------


async def test_t14_run_tui_exit_code_and_initial_prompt(tmp_path):
    session = await open_app(_config(tmp_path, _script({"text": "回答", "stop": True})))
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        task = asyncio.ensure_future(
            run_tui_session(session, initial_prompt="首问", input=pipe, output=DummyOutput(), sink=sink)
        )
        assert await wait_until(lambda: any("首问" in line for line in sink.lines))
        assert await wait_until(lambda: _done_count(sink) >= 1)
        assert len(_model_of(session).requests) == 1, "首问只提交一次"
        pipe.send_text("/exit\r")
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0, "正常请求退出码 0"
        assert session.state == "closed"


async def test_t11_run_tui_cancel_yields_aborted_exit_code_and_closes(tmp_path):
    session = await open_app(
        _config(tmp_path, _script({"text": "慢慢回答", "stop": True}), chunks=4, delay=0.15)
    )
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        task = asyncio.ensure_future(
            run_tui_session(session, input=pipe, output=DummyOutput(), sink=sink)
        )
        pipe.send_text("问题\r")
        assert await wait_until(lambda: session.state == "executing")
        pipe.send_text("\x03")  # Ctrl+C 协作取消
        assert await wait_until(lambda: any("aborted" in text for text in sink.rendered))
        pipe.send_text("/exit\r")
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 1, "取消请求的退出码 1 (aborted)"
        assert session.state == "closed"


# --- 检查模式组合渲染 (T08/T10 经生产管线) ------------------------------------------------------


async def test_structural_events_before_text_flush_multi_turn(tmp_path):
    """同批渲染: 结构事件先于文本 flush——后一轮的待渲染增量不得抢先改写轮号,
    否则前一轮最终回答错标并与后一轮键碰撞而被去重静默丢失."""
    from cicada.ui.terminal import _TEXT_DIRTY

    session = await open_app(_config(tmp_path, _script({"text": "x", "stop": True})))
    sink = RecordingRenderSink()
    executor = build_production_executor(session, output=DummyOutput(), sink=sink)
    try:
        executor._on_update(AgentEventReceived(TurnStarted("run-1", 0)))
        turn0 = executor._mailbox.get_nowait()
        executor._render_batch([turn0])
        # turn-1 的增量被 subscriber 合并 (脏信号入队), 随后 turn-0 的完成事件入队
        executor._on_update(AgentEventReceived(AssistantTextDelta("run-1", 1, "第二轮开头")))
        dirty = executor._mailbox.get_nowait()
        assert dirty is _TEXT_DIRTY
        executor._on_update(AgentEventReceived(AssistantCompleted(
            "run-1", AssistantMessage(text="第一轮回答", stop_reason="stop"),
        )))
        completed0 = executor._mailbox.get_nowait()
        executor._render_batch([completed0])  # 同一批: turn-1 增量尚在 pending
        assert sink.rendered_count("第一轮回答") == 1
        executor._render_batch([dirty])  # flush turn-1 增量: 只影响 turn-1 预览
        executor._on_update(AgentEventReceived(AssistantCompleted(
            "run-1", AssistantMessage(text="第二轮最终回答", stop_reason="stop"),
        )))
        completed1 = executor._mailbox.get_nowait()
        executor._render_batch([completed1])
        assert sink.rendered_count("第二轮最终回答") == 1, "轮 1 最终回答不被键碰撞吞掉"
        assert sink.rendered_count("第一轮回答") == 1
    finally:
        await session.aclose()


async def test_production_check_mode_delivery_rendered(tmp_path):
    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    script = _script(
        {"tool_calls": [
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "第一轮完成", "stop": True},
    )
    session = await open_app(_config(root, script, plan=plan))
    sink = RecordingRenderSink()
    with create_pipe_input() as pipe:
        executor = build_production_executor(session, input=pipe, output=DummyOutput(), sink=sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            assert await wait_until(lambda: executor._consumer_task is not None)
            pipe.send_text("改好并检查\r")
            assert await wait_until(lambda: _done_count(sink) >= 1, timeout=15)
            assert sink.rendered_count("程序判定") >= 1, "交付判定经生产渲染输出"
            assert any("来源=本次检查" in text for text in sink.rendered)
            status = executor._display_status
            assert status is not None
            assert status.check_status is not None and "check-1=passed" in status.check_status
            assert status.change_scope is not None and "累计" in status.change_scope
        finally:
            pipe.send_text("/exit\r")
            await asyncio.wait_for(runner, timeout=15)


async def test_run_tui_session_fallback_closes_session_when_runner_fails(tmp_path, monkeypatch):
    """run_tui_session 兜底分支: runner 在自身收尾前异常 → 异常上抛且 session 被关闭."""
    from cicada.ui import ProductionTerminalExecutor

    async def boom(self):
        raise RuntimeError("runner exploded before cleanup")

    monkeypatch.setattr(ProductionTerminalExecutor, "run_async", boom)
    session = await open_app(_config(tmp_path, _script({"text": "答", "stop": True})))
    with create_pipe_input() as pipe:
        with pytest.raises(RuntimeError, match="runner exploded"):
            await run_tui_session(session, input=pipe, output=DummyOutput())
    assert session.state == "closed"
