"""票据 03 聚焦检查: 终端对话入口 (headless: create_pipe_input + DummyOutput).

覆盖 T04 (预览按 run/turn、最终回答一次、零增量显示)、用户记录/忙碌拒绝、
T05 (忙碌可编辑、提交拒绝不排队、finalizing 拒绝)、输入裁决 (Enter/Alt+Enter/
粘贴/中文编辑)、命令 (/help /status /clear /exit //转义/未知)、T11 终端部分
(Ctrl+C 协作取消、退出顺序、关闭不可逆)、T13 (显示净化、规范历史原始)、
T15 (文本合并不逐字排队、结构溢出失败并取消、subscriber 不写终端).

真实 IME / 终端模式恢复按 01 探针裁决由人工在 Windows Terminal 复核,
自动化不能代签 (见票据回执).
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
from cicada.application import (
    AgentEventReceived,
    HistoryCleared,
    open_application,
)
from cicada.core.cancel import CancelToken
from cicada.core.events import AssistantCompleted, AssistantTextDelta
from cicada.core.messages import AssistantMessage
from cicada.core.ports import StreamDone, StreamEvent, TextDelta
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.plugins.fake_model import fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui.contracts import PresentationRecord, PresentationUpdate
from cicada.ui.terminal import (
    MAILBOX_CAPACITY,
    BasicTextPresenter,
    TerminalExecutor,
    sanitize_external_text,
)


# --- 夹具 -------------------------------------------------------------------------------------


class ChunkedFakeModel:
    """parse_script JSON 剧本 + 可控分块/延迟的 fake 模型 (与 demo 夹具同构)."""

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
                    if cancel.cancelled:  # 分块之间也响应协作取消
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


def _config(
    root: Path,
    script_json: str,
    *,
    chunks: int = 4,
    delay: float = 0.0,
    plan: VerificationPlan | None = None,
):
    model = ChunkedFakeModel(script_json, chunks=chunks, delay=delay)
    from cicada.application import ApplicationConfig

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


class RecordingSink:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print_line(self, text: str = "") -> None:
        self.lines.append(text)

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for line in self.lines if line.startswith(prefix))

    def joined(self) -> str:
        return "\n".join(self.lines)


def _make_executor(session, pipe, sink: RecordingSink) -> TerminalExecutor:
    return TerminalExecutor(session, sink=sink, input=pipe, output=DummyOutput())


def _make_offline_executor(session, sink: RecordingSink) -> TerminalExecutor:
    """不启动主屏的执行器: 直接驱动 subscriber/渲染路径 (T15 无头单元检查)."""
    return TerminalExecutor(session, sink=sink, output=DummyOutput())


async def wait_until(predicate, timeout: float = 5.0, step: float = 0.01) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


async def _start(executor: TerminalExecutor, runner) -> None:
    """屏障: 等 run_async 已订阅并启动消费任务后, 再开始注入按键."""
    assert await wait_until(lambda: executor._consumer_task is not None)


async def _wait_request_done(sink: RecordingSink, index: int = 1) -> None:
    """同步点: 第 index 条 '请求完成' 记录已被消费任务渲染 (mailbox 已排空到该点)."""
    assert await wait_until(lambda: _done_count(sink) >= index, timeout=15)


def _done_count(sink: RecordingSink) -> int:
    return sum(
        1 for line in sink.lines if line.startswith("[system]") and "请求完成" in line
    )


async def _close_by_exit(executor: TerminalExecutor, pipe, runner) -> None:
    executor.buffer.reset()  # 清掉草稿, 避免 /exit 被拼进草稿当新提问提交
    pipe.send_text("/exit\r")
    await asyncio.wait_for(runner, timeout=15)


def _model_of(session) -> ChunkedFakeModel:
    model = session._app.agent._model
    if type(model).__name__ == "RunPolicy":  # 检查模式: RunPolicy 包装原适配器
        model = model.inner
    assert isinstance(model, ChunkedFakeModel)
    return model


# --- T04: 预览按 run/turn 更新、最终回答一次、忙碌可编辑 ---------------------------------------


async def test_streaming_preview_final_once_and_busy_draft_edit(tmp_path):
    answer = "这是分块流式回答。\n第二行内容。\n第三行内容。"
    session = await open_application(
        _config(tmp_path, _script({"text": answer, "stop": True}), chunks=8, delay=0.02)
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一个问题\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            assert await wait_until(lambda: session.state == "executing")
            # 分块流式期间活动预览按 run/turn 更新
            assert await wait_until(lambda: executor._display_preview is not None, timeout=10)
            preview = executor._display_preview
            assert preview is not None
            assert preview.run_id == "run-1" and preview.turn_index == 0
            assert preview.text  # 增量预览有内容
            # 忙碌时可继续编辑草稿且不被输出破坏
            pipe.send_text("草稿")
            assert await wait_until(lambda: executor.buffer.text == "草稿")
            await asyncio.sleep(0.05)
            assert executor.buffer.text == "草稿"
            await _wait_request_done(sink, 1)
            assert session.state == "idle"
            # 最终回答只进入终端记录一次 (即使有多条增量)
            assert sink.count_prefix("[assistant]") == 1
            assistant_record = next(line for line in sink.lines if line.startswith("[assistant]"))
            assert "第三行内容" in assistant_record
            # 预览已收起; 用户记录一次且原文保留
            assert executor._display_preview is None
            assert sink.count_prefix("[user]") == 1
            assert any("第一个问题" in line for line in sink.lines)
            assert executor.buffer.text == "草稿"  # 提交后的草稿与输出互不影响
        finally:
            await _close_by_exit(executor, pipe, runner)
    assert session.state == "closed"


async def test_zero_delta_final_answer_still_displayed_once():
    """零增量轮: AssistantCompleted 的最终回答仍显示一次 (T04)."""
    presenter = BasicTextPresenter()
    update = presenter.accept(
        AgentEventReceived(
            AssistantCompleted("run-1", AssistantMessage(text="零增量的最终回答"))
        )
    )
    assert len(update.appended) == 1
    assert update.appended[0].role == "assistant"
    assert update.appended[0].body == ("零增量的最终回答",)
    again = presenter.accept(
        AgentEventReceived(
            AssistantCompleted("run-1", AssistantMessage(text="零增量的最终回答"))
        )
    )
    assert again.appended == ()  # 已完成记录不重复追加


# --- 用户记录: 多行原文、清空输入不清正文、忙碌拒绝无假消息 ------------------------------------


async def test_multiline_prompt_kept_once_and_canonical_history(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "回答", "stop": True}))
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一行")
            pipe.send_text("\x1b\r")  # Alt+Enter 换行
            pipe.send_text("第二行")
            assert await wait_until(lambda: executor.buffer.text == "第一行\n第二行")
            pipe.send_text("\r")  # Enter 提交
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            await _wait_request_done(sink, 1)
            assert sink.count_prefix("[user]") == 1
            user_record = next(line for line in sink.lines if line.startswith("[user]"))
            assert "第一行\n第二行" in user_record  # 原始多行内容保留为一次记录
            assert executor.buffer.text == ""  # 输入框清空只为下一条草稿, 正文仍在记录里
            assert session.history[0].text == "第一行\n第二行"  # 规范历史为原始内容
        finally:
            await _close_by_exit(executor, pipe, runner)


async def test_display_sanitized_but_canonical_raw(tmp_path):
    """T13: 显示路径去除 CSI/控制字符; 规范消息保持原始内容."""
    dirty_prompt = "看见\x1b[31m红色\x1b[0m\x07了吗"
    dirty_answer = "模型\x1b[32m输出\x1b[0m\x07完毕"
    session = await open_application(
        _config(tmp_path, _script({"text": dirty_answer, "stop": True}))
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            executor.buffer.text = dirty_prompt  # 外部含控制字符的草稿
            pipe.send_text("\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            await _wait_request_done(sink, 1)
            user_record = next(line for line in sink.lines if line.startswith("[user]"))
            assert "红色" in user_record and "了吗" in user_record
            assert "\x1b" not in user_record and "\x07" not in user_record
            assistant_record = next(line for line in sink.lines if line.startswith("[assistant]"))
            assert "模型输出完毕" in assistant_record
            assert "\x1b" not in assistant_record and "\x07" not in assistant_record
            # 规范历史保持原始内容 (不修改规范消息)
            assert session.history[0].text == dirty_prompt
            assert session.history[1].text == dirty_answer
            assert sanitize_external_text(dirty_prompt) == "看见红色了吗"
        finally:
            await _close_by_exit(executor, pipe, runner)


# --- T05: 忙碌提交拒绝不排队; finalizing 拒绝 --------------------------------------------------


async def test_busy_submit_rejected_keeps_draft_not_queued(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "慢慢回答", "stop": True}), chunks=10, delay=0.05)
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            assert await wait_until(lambda: session.state == "executing")
            pipe.send_text("忙碌草稿")
            assert await wait_until(lambda: executor.buffer.text == "忙碌草稿")
            pipe.send_text("\r")  # 忙碌提交
            await asyncio.sleep(0.1)
            assert any("提交被拒绝" in line for line in sink.lines)
            assert sink.count_prefix("[user]") == 1  # 忙碌拒绝不产生假用户消息
            await _wait_request_done(sink, 1)
            assert len(_model_of(session).requests) == 1  # 不排队: 只执行了一次请求
            assert executor.buffer.text == "忙碌草稿"  # 草稿保留
        finally:
            await _close_by_exit(executor, pipe, runner)


class SlowService:
    """把真实验证服务的 refresh 延迟化; 其余委托 (制造可观察的 finalizing 窗口)."""

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


async def test_finalizing_submit_rejected_and_draft_kept(tmp_path):
    """T05: finalizing 阶段提交同样被拒绝、不执行、不排队."""
    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    session = await open_application(
        _config(root, _script({"text": "直接回答", "stop": True}), plan=plan)
    )
    session._service = SlowService(session._service, delay=0.5)  # type: ignore[arg-type]
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            assert await wait_until(lambda: session.state == "finalizing", timeout=10)
            pipe.send_text("插队\r")
            await asyncio.sleep(0.1)
            assert any(
                "finalizing" in line and "提交被拒绝" in line for line in sink.lines
            )
            await _wait_request_done(sink, 1)
            assert session.state == "idle"
            assert len(_model_of(session).requests) == 1  # 插队请求未执行
            assert executor.buffer.text == "插队"  # 草稿保留
        finally:
            await _close_by_exit(executor, pipe, runner)


# --- 输入裁决: Alt+Enter / 粘贴不触发提交 ------------------------------------------------------


async def test_alt_enter_and_bracketed_paste_do_not_submit(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "回答", "stop": True}))
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("A")
            pipe.send_text("\x1b\r")  # Alt+Enter
            pipe.send_text("B")
            assert await wait_until(lambda: executor.buffer.text == "A\nB")
            assert session.state == "idle" and not _model_of(session).requests
            pipe.send_text("\x1b[200~粘贴行一\n粘贴行二\x1b[201~")  # bracketed paste 单事件
            assert await wait_until(
                lambda: "粘贴行一\n粘贴行二" in executor.buffer.text
            )
            assert session.state == "idle"  # 粘贴的换行不触发提交
            pipe.send_text("\r")  # Enter 提交整段多行
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            await _wait_request_done(sink, 1)
            assert session.history[0].text == "A\nB粘贴行一\n粘贴行二"
        finally:
            await _close_by_exit(executor, pipe, runner)


# --- 命令 -------------------------------------------------------------------------------------


async def test_commands_help_status_unknown_and_slash_escape(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "回答", "stop": True}))
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("/help\r")
            assert await wait_until(lambda: "可用命令" in sink.joined())
            assert "/status" in sink.joined() and "/clear" in sink.joined()
            pipe.send_text("/status\r")
            assert await wait_until(lambda: "phase=" in sink.joined())
            assert "model=fake" in sink.joined() and "workspace=" in sink.joined()
            pipe.send_text("/nope\r")
            assert await wait_until(lambda: "未知命令 /nope" in sink.joined())
            assert executor.buffer.text == "/nope"  # 未知命令只提示, 草稿保留可修正
            executor.buffer.reset()
            pipe.send_text("//是普通斜杠文本\r")  # // 转义
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            await _wait_request_done(sink, 1)
            assert session.history[0].text == "/是普通斜杠文本"
            assert any("/是普通斜杠文本" in line for line in sink.lines)
        finally:
            await _close_by_exit(executor, pipe, runner)


async def test_clear_command_idle_ok_and_busy_rejected(tmp_path):
    session = await open_application(
        _config(
            tmp_path,
            _script({"text": "第一答", "stop": True}, {"text": "第二答", "stop": True}),
            chunks=10,
            delay=0.05,
        )
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            await _wait_request_done(sink, 1)
            assert len(session.history) == 2
            pipe.send_text("第二问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 2)
            assert await wait_until(lambda: session.state == "executing")
            pipe.send_text("/clear\r")  # 忙碌清空
            await asyncio.sleep(0.1)
            assert any("清空被拒绝" in line for line in sink.lines)
            assert executor.buffer.text == "/clear"  # 忙碌清空拒绝且保留草稿
            await _wait_request_done(sink, 2)
            assert len(session.history) == 4  # 未清空
            executor.buffer.reset()
            pipe.send_text("/clear\r")  # 空闲清空
            assert await wait_until(lambda: "已清空模型历史" in sink.joined(), timeout=10)
            assert session.history == ()
        finally:
            await _close_by_exit(executor, pipe, runner)


# --- T11 终端部分: Ctrl+C / Escape 协作取消, 退出顺序, 关闭不可逆 -------------------------------


async def test_ctrl_c_cancels_busy_then_session_reusable(tmp_path):
    session = await open_application(
        _config(
            tmp_path,
            _script({"text": "会被取消的回答", "stop": True}, {"text": "重试后的回答", "stop": True}),
            chunks=8,
            delay=0.15,
        )
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("第一问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            assert await wait_until(lambda: session.state == "executing")
            await asyncio.sleep(0.05)
            pipe.send_text("\x03")  # Ctrl+C: 协作取消
            assert await wait_until(lambda: "请求完成: aborted" in sink.joined(), timeout=10)
            # 取消路径不打印完整最终回答 (如有记录只能是已取消的部分前缀)
            assistant_lines = [line for line in sink.lines if line.startswith("[assistant]")]
            assert all("会被取消的回答" not in line for line in assistant_lines)
            # 取消后可继续提交并完成
            pipe.send_text("第二问\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 2)
            assert await wait_until(lambda: "重试后的回答" in sink.joined(), timeout=10)
        finally:
            await _close_by_exit(executor, pipe, runner)


async def test_escape_cancels_busy_after_disambiguation(tmp_path):
    # 流须长于 1.0s 消歧窗口: 12 字分 6 块 x 0.3s = 1.8s
    session = await open_application(
        _config(
            tmp_path,
            _script({"text": "流式回答流式回答流式回答", "stop": True}),
            chunks=6,
            delay=0.3,
        )
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("问题\r")
            assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
            assert await wait_until(lambda: session.state == "executing")
            await asyncio.sleep(0.05)
            # 孤立 Escape 经 KeyProcessor 消歧 (timeoutlen=1.0s), 显式模拟定时 flush
            from prompt_toolkit.key_binding.key_processor import KeyPress, _Flush
            from prompt_toolkit.keys import Keys

            executor.app.key_processor.feed_multiple([KeyPress(Keys.Escape, "\x1b")])
            executor.app.key_processor.process_keys()
            await asyncio.sleep(1.1)
            executor.app.key_processor.feed_multiple([_Flush])
            executor.app.key_processor.process_keys()
            assert await wait_until(lambda: "请求完成: aborted" in sink.joined(), timeout=10)
        finally:
            await _close_by_exit(executor, pipe, runner)


async def test_exit_closes_in_order_and_is_irreversible(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "回答", "stop": True}))
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        await _start(executor, runner)
        pipe.send_text("问题\r")
        assert await wait_until(lambda: sink.count_prefix("[user]") == 1)
        await _wait_request_done(sink, 1)
        pipe.send_text("/exit\r")
        await asyncio.wait_for(runner, timeout=15)
        # T11 顺序: 主屏退出(终端模式恢复) -> 退订 -> 停 mailbox 消费 -> session.aclose
        assert session.state == "closed"
        assert executor._unsubscribe is None  # 已退订
        assert executor._consumer_task is None  # 消费任务已停止
        assert executor._mailbox.qsize() == 0
    # 关闭不可逆: execute 被拒绝
    with pytest.raises(RuntimeError, match="closed"):
        await session.execute("after close")


# --- 预览有界 ---------------------------------------------------------------------------------


async def test_preview_bounded_lines_and_truncation_marked(tmp_path):
    answer = "\n".join(f"第{i}行" for i in range(1, 21))  # 20 行 > 12 行上限
    session = await open_application(
        _config(tmp_path, _script({"text": answer, "stop": True}), chunks=6, delay=0.02)
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("问题\r")
            assert await wait_until(lambda: executor._display_preview is not None, timeout=10)
            assert await wait_until(
                lambda: executor._display_preview is not None
                and executor._display_preview.truncated
            )
            preview = executor._display_preview
            assert preview is not None
            assert len(preview.text.split("\n")) <= 12
            await _wait_request_done(sink, 1)
            assert executor._display_preview is None  # 完成后预览收起
            assert "第20行" in sink.joined()  # 完整文本以最终回答为准
        finally:
            await _close_by_exit(executor, pipe, runner)


async def test_preview_bounded_bytes_single_long_line(tmp_path):
    answer = "x" * 6000  # 单行 6000 bytes > 4096
    session = await open_application(
        _config(tmp_path, _script({"text": answer, "stop": True}), chunks=6, delay=0.02)
    )
    sink = RecordingSink()
    with create_pipe_input() as pipe:
        executor = _make_executor(session, pipe, sink)
        runner = asyncio.ensure_future(executor.run_async())
        try:
            await _start(executor, runner)
            pipe.send_text("问题\r")
            assert await wait_until(
                lambda: executor._display_preview is not None
                and executor._display_preview.truncated
            )
            preview = executor._display_preview
            assert preview is not None
            assert len(preview.text.encode("utf-8")) <= 4096
            await _wait_request_done(sink, 1)
            # 最终回答完整保留
            assistant_record = next(line for line in sink.lines if line.startswith("[assistant]"))
            assert "x" * 6000 in assistant_record
        finally:
            await _close_by_exit(executor, pipe, runner)


# --- T15: 文本合并 / 溢出失败并取消 / subscriber 不写终端 ---------------------------------------


async def test_text_deltas_merge_not_queued_per_char(tmp_path):
    from cicada.ui.terminal import _TEXT_DIRTY

    session = await open_application(
        _config(tmp_path, _script({"text": "回答", "stop": True}))
    )
    sink = RecordingSink()
    executor = _make_offline_executor(session, sink)
    try:
        # 同步 subscriber: 5 条增量只入队 1 个合并信号, 不逐字排队, 不写终端
        for piece in ("a", "b", "c", "d", "e"):
            executor._on_update(AgentEventReceived(AssistantTextDelta("run-1", 0, piece)))
        assert executor._mailbox.qsize() == 1
        assert sink.lines == []  # subscriber 不直接写终端
        item = executor._mailbox.get_nowait()
        assert item is _TEXT_DIRTY
        executor._render_batch([item])
        preview = executor._display_preview
        assert preview is not None
        assert preview.run_id == "run-1" and preview.turn_index == 0
        assert preview.text == "abcde"  # 合并进当前活动预览
        assert sink.lines == []
        # 结构通知单独入队
        executor._on_update(HistoryCleared())
        assert executor._mailbox.qsize() == 1
        # 已完成记录按键去重: 重复 apply 只打印一次
        record = PresentationRecord(
            key="assistant:run-1:0", role="assistant", title="", status="done", body=("答",)
        )
        executor._apply(PresentationUpdate(appended=(record,)))
        executor._apply(PresentationUpdate(appended=(record,)))
        assert sink.count_prefix("[assistant]") == 1
    finally:
        await session.aclose()


async def test_structural_overflow_fails_explicitly_and_cancels(tmp_path):
    session = await open_application(
        _config(tmp_path, _script({"text": "慢慢回答", "stop": True}), chunks=10, delay=0.05)
    )
    sink = RecordingSink()
    executor = _make_offline_executor(session, sink)
    unsubscribe = session.subscribe(executor._on_update)
    try:
        task = asyncio.create_task(session.execute("慢请求"))
        assert await wait_until(lambda: session.state == "executing")
        # 灌满 mailbox 后再投递结构通知: 显式记录 UI failure 并调用 session.cancel()
        while executor._mailbox.qsize() < MAILBOX_CAPACITY:
            executor._mailbox.put_nowait(HistoryCleared())
        executor._on_update(HistoryCleared())
        assert executor.ui_failures, "结构通知溢出必须记录 UI failure"
        assert "overflow" in executor.ui_failures[0]
        assert session.state == "cancelling"  # cancel() 已被调用, 不伪造 idle
        assert sink.lines == []  # subscriber 不写终端 (记录者只会在消费任务里被调用)
        outcome = await asyncio.wait_for(task, timeout=15)
        assert outcome.result.stop_reason == "aborted"  # RequestOutcome 仍是收尾权威
    finally:
        unsubscribe()
        await session.aclose()
