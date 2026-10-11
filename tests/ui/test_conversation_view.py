"""票据 02 组合检查: 真实 ApplicationSession 驱动 conversation_view.build.

覆盖票据验收 P02/P03/P07 与准入门槛: 连续两问结构化历史、同一条目原位
更新、完成不二次打印、屏障内 Enter 零请求零提示且草稿保留、同步
reservation 先于 create_task、取消意图绑定本次任务、/clear 不擦显示文档、
错误/取消条目如实、process_uncertain 不锁死 idle 命令、closing/closed 不
重开输入。确定性: fake 模型剧本 + 真实 Git 工作区, 无真实模型调用。
渲染行为 (VT 屏) 归 01 已冻结的 frontend 检查; 本文件只驱动组合层回调。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from cicada.application import ApplicationConfig, RequestPhaseChanged, open_application
from cicada.core.cancel import CancelToken
from cicada.core.ports import StreamDone, StreamEvent, TextDelta
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.ui.admission import AdmissionGate
from cicada.ui.conversation_view import ConversationView, build
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.transcript import (
    AssistantEntry,
    CancelledEntry,
    ErrorEntry,
    SystemEntry,
    UserEntry,
)
from cicada.ui.welcome import WelcomeEntry


# --- 夹具 (工作区/剧本/PromptInfo 单一来源在 tests.support.app; ChunkedFakeModel 与 e2e 夹具同构) ---


async def test_diff_page_consumes_input_and_returns_without_model_history(tmp_path):
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys
    from cicada.ui.commands import SlashCommandMenu
    from cicada.ui.frontend import MainScreenFrontend
    from cicada.ui.terminal_io import FakeTerminal, PointerEvent

    root = _repo(tmp_path)
    (root / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    session = await open_application(_config(root, []))
    terminal = FakeTerminal(80, 24)
    bundle = build(session)
    view = bundle["view"]
    frontend = MainScreenFrontend(
        terminal, document_lines=view.document_lines, layout_document=view.layout_document,
        editor=view.editor, on_submit=view.on_submit, on_display_action=view.execute_display_action,
        review=view.review_view, command_menu=SlashCommandMenu(),
    )
    enter = KeyPress(Keys.ControlM, "\r")
    try:
        view.editor.set_text("/diff")
        frontend._on_input_batch([enter, KeyPress("x", "x"), enter])
        await view.review_view._task
        assert view.review_view.active and view.editor.is_empty()
        frontend._do_render()
        span = next(h for h in frontend._frame_hits if h.action.target_id[1] == "file")
        frontend._on_input_batch([
            PointerEvent("down", span.start_column, span.row, terminal.geometry, button="left"),
            PointerEvent("up", span.start_column, span.row, terminal.geometry, button="left"),
            KeyPress(Keys.BracketedPaste, "must not enter the draft"), KeyPress(Keys.BackTab, "\x1b[Z"),
        ])
        await view.review_view._task
        assert "+VALUE = 2" in view.review_view.preview.text
        entered = asyncio.Event()
        original_read = session._review_reader.read_file

        async def slow_read(snapshot, target, token):
            entered.set()
            await token.wait()
            return await original_read(snapshot, target, token)

        session._review_reader.read_file = slow_read
        view.review_view.execute(span.action)
        await entered.wait()
        frontend._on_input_batch([KeyPress(Keys.Escape, "\x1b"), KeyPress("y", "y"), enter])
        await asyncio.gather(view.review_view._task, return_exceptions=True)
        await asyncio.sleep(0)
        assert not view.review_view.active and not session.review_active
        assert view.editor.is_empty() and session.history == ()
        assert not any(isinstance(entry, UserEntry) for entry in view.transcript.entries())
        assert not any(isinstance(entry, ErrorEntry) for entry in view.transcript.entries())
    finally:
        frontend._stopped = True
        await session.aclose()
        view.close()

from tests.support.app import prompt_info as _shared_prompt_info, repo as _repo, script as _script


def _prompt_info(root: Path):
    return _shared_prompt_info(root, version="ticket02")


def _config(root: Path, script: list, *, plan: VerificationPlan | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="ticket02 system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(script)),
        plan=plan,
        max_turns=6,
        model_label="fake",
    )


def _chunked_config(tmp_path: Path, script_json: str, *, chunks: int, delay: float) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=_repo(tmp_path),
        system_prompt="ticket02 system prompt",
        prompt_info=_prompt_info(tmp_path / "repo"),
        model_plugin=fake_model_plugin(ChunkedFakeModel(script_json, chunks=chunks, delay=delay)),
        plan=None,
        max_turns=6,
        model_label="fake",
    )


def _uncertain_plan(root: Path) -> VerificationPlan:
    """慢检查计划: 0.5s 超时制造 process_uncertain 锁存 (与 T09 同构)。"""
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "Start-Sleep -Seconds 5", timeout_s=0.5),),
    )


class ChunkedFakeModel:
    """长文本分块 + 可控延迟的 fake 模型: 流式与取消窗口的可控驱动。"""

    def __init__(self, script_json: str, *, chunks: int = 6, delay: float = 0.02) -> None:
        from cicada.__main__ import parse_script

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


async def _wait_until(predicate, timeout: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


class Harness:
    """build() 组合 + 捕获退出请求; session 由测试用例 open 后注入。"""

    def __init__(self, session) -> None:
        self.session = session
        self.exit_requested = 0
        self.render_requests = 0
        self.bundle = build(session, on_exit=self._request_exit, on_update=self._note_update)
        self.view: ConversationView = self.bundle["view"]

    def _request_exit(self) -> None:
        self.exit_requested += 1

    def _note_update(self) -> None:
        self.render_requests += 1

    def submit(self, text: str) -> bool:
        return self.bundle["on_submit"](text)

    def entries(self) -> list:
        return self.view.transcript.entries()

    def lines(self, width: int = 80) -> list[str]:
        return self.bundle["document_lines"](width)

    def assistant_entries(self) -> list[AssistantEntry]:
        return [e for e in self.entries() if isinstance(e, AssistantEntry)]

    def assistant_texts(self) -> list[str]:
        return [e.text for e in self.assistant_entries()]

    def model(self):
        model = self.session._app.agent._model
        if type(model).__name__ == "RunPolicy":
            model = model.inner
        return model

    async def wait_request_done(self) -> None:
        assert await _wait_until(lambda: session_idle(self.session)), self.session.state
        assert await _wait_until(lambda: self.view._gate.busy() is False), "reservation 未释放"

    def close(self) -> None:
        self.view.close()


def session_idle(session) -> bool:
    return session.state == "idle"


# --- 连续两问: 结构化历史 + 原位更新 ---------------------------------------------------


async def test_two_questions_carry_full_history_and_update_entries_in_place(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"text": "第一轮回答", "stop": True},
        {"text": "第二轮回答", "stop": True},
    )))
    h = Harness(session)
    try:
        assert h.submit("第一个问题") is True
        await h.wait_request_done()
        assert h.submit("第二个问题") is True
        await h.wait_request_done()
        # 第二次请求一次且仅一次携带完整结构化历史
        model = h.model()
        assert isinstance(model, FakeModel)
        assert len(model.requests) == 2
        second = [(type(m).__name__, getattr(m, "text", None)) for m in model.requests[1].messages]
        assert second == [
            ("UserMessage", "第一个问题"),
            ("AssistantMessage", "第一轮回答"),
            ("UserMessage", "第二个问题"),
        ]
        assert model.requests[1].system_prompt == "ticket02 system prompt"
        # 同一 assistant 条目原位更新: 每问各一条, 完成校准不二次追加
        assert h.assistant_texts() == ["第一轮回答", "第二轮回答"]
        kinds = [type(e) for e in h.entries()]
        # welcome 01: 显示文档首条为启动欢迎, 其后才是会话条目
        assert kinds == [WelcomeEntry, UserEntry, AssistantEntry, UserEntry, AssistantEntry]
        assert [(e.run_id, e.turn_index, e.finalized) for e in h.assistant_entries()] == [
            ("run-1", 0, True),
            ("run-2", 0, True),
        ]
        # 显示文档每行可见列宽不超过宽度 (约束 ⑤)
        for line in h.lines(80):
            assert visible_width(line) <= 80
        assert h.render_requests > 0  # 事件经 on_update 投递渲染请求
    finally:
        h.close()
        await session.aclose()


async def test_streaming_deltas_update_single_entry_then_finalize(tmp_path):
    text = "这是一段流式长回答。" * 30
    session = await open_application(_chunked_config(
        tmp_path, json.dumps([{"text": text, "stop": True}]), chunks=6, delay=0.03
    ))
    h = Harness(session)
    try:
        assert h.submit("长问题") is True
        # 流式中途: 只有一条 assistant 条目, 文本是已收到的前缀 (完成需 ~180ms+)
        assert await _wait_until(lambda: len(h.assistant_texts()) == 1 and h.assistant_texts()[0])
        partial_len = len(h.assistant_texts()[0])
        assert 0 < partial_len < len(text)
        await h.wait_request_done()
        assert h.assistant_texts() == [text]  # 校准为完整正文
        assert len(h.assistant_entries()) == 1  # 同一条目, 无第二份
        assert h.assistant_entries()[0].finalized
    finally:
        h.close()
        await session.aclose()


# --- 忙碌屏障: 零重复请求、零命令、零提示, 草稿保留 --------------------------------------


async def test_busy_enter_and_commands_rejected_silently_draft_kept(tmp_path):
    session = await open_application(_chunked_config(
        tmp_path, json.dumps([{"text": "慢回答" * 20, "stop": True}]), chunks=8, delay=0.03
    ))
    h = Harness(session)
    try:
        assert h.submit("第一问") is True
        # reservation 已同步占用: 任务尚未起跑时同批二次 Enter/命令零动作
        assert h.view._gate.busy() is True
        assert h.submit("第二问") is False
        assert h.submit("/status") is False
        assert h.submit("/exit") is False
        assert h.exit_requested == 0  # 忙碌 /exit 不退出
        # 任务起跑后屏障继续生效
        assert await _wait_until(lambda: session.state == "executing")
        assert h.submit("第三问") is False
        assert h.submit("/help") is False
        # 零重复请求、零伪用户条目、零提示
        assert len(h.model().requests) == 1
        assert len([e for e in h.entries() if isinstance(e, UserEntry)]) == 1
        assert not [e for e in h.entries() if isinstance(e, SystemEntry)]
        # 拒绝路径不触碰草稿
        h.view.editor.set_text("预备的下一条问题")
        assert h.submit("第四问") is False
        assert h.view.editor.get_text() == "预备的下一条问题"
        await h.wait_request_done()
        assert session_idle(session) and h.view._gate.busy() is False  # 空闲即释放
    finally:
        h.close()
        await session.aclose()


async def test_reservation_released_only_after_task_done_and_idle(tmp_path):
    session = await open_application(_chunked_config(
        tmp_path, json.dumps([{"text": "收尾慢一点的回答" * 10, "stop": True}]), chunks=4, delay=0.02
    ))
    h = Harness(session)
    try:
        assert h.submit("占用") is True
        assert h.view._gate.busy() is True
        assert await _wait_until(lambda: session.state == "executing")
        # 拥有任务运行全程 reservation 保持 (含内核结束后的 finalizing 收尾段;
        # 普通模式该段瞬时不可观察, 屏障拒绝由 state 检查兜底, 见 admission)
        assert h.view._gate.busy() is True
        assert h.submit("插队") is False
        await h.wait_request_done()
        assert h.view._gate.busy() is False  # 任务终结且真正 idle 后立即释放
        assert h.submit("下一个") is True
        await h.wait_request_done()
    finally:
        h.close()
        await session.aclose()


# --- 斜线命令 -----------------------------------------------------------------------


async def test_commands_idle_only_with_escape_and_unknown_hint(tmp_path):
    session = await open_application(_config(_repo(tmp_path), _script(
        {"text": "转义后的回答", "stop": True},
    )))
    h = Harness(session)
    try:
        assert h.submit("/help") is True
        assert any("/status" in line for line in h.lines())
        assert h.submit("/status") is True
        joined = "\n".join(strip_terminal_sequences(line) for line in h.lines())
        assert "状态: idle" in joined and "模型: fake" in joined
        assert "计量: 不可得" in joined  # 无计量数据不当作 0
        assert h.submit("/未知命令") is False  # 未知命令提示且草稿保留
        assert any("未知命令" in line for line in h.lines())
        h.view.editor.set_text("/foo 保留草稿")
        assert h.submit("/foo") is False
        assert h.view.editor.get_text() == "/foo 保留草稿"
        # // 转义: 去一个斜杠按普通编码请求发送
        assert h.submit("//这是/斜杠开头的普通文本") is True
        await h.wait_request_done()
        users = [e.text for e in h.entries() if isinstance(e, UserEntry)]
        assert users == ["/这是/斜杠开头的普通文本"]
        assert h.model().requests[0].messages[0].text == "/这是/斜杠开头的普通文本"
    finally:
        h.close()
        await session.aclose()


async def test_status_multiline_field_keeps_its_section_when_total_is_clipped(tmp_path):
    from dataclasses import replace

    config = replace(_config(_repo(tmp_path), _script()), model_label="fake\n模型续行")
    session = await open_application(config)
    view = ConversationView(session, diagnostics=lambda: [f"诊断{i}: 原始值" for i in range(20)])
    try:
        assert view.on_submit("/status") is True
        lines = [strip_terminal_sequences(line).strip() for line in view.document_lines(100)]
        session_heading, metrics_heading, config_heading = (
            lines.index(name) for name in ("会话", "计量", "配置")
        )
        assert "模型续行" in lines[session_heading:metrics_heading]
        assert "计量: 不可得" in lines[metrics_heading:config_heading]
        assert "模型续行" not in lines[config_heading:]
        assert any("…[已截断: 显示" in line for line in lines)
        assert session.history == ()
    finally:
        view.close()
        await session.aclose()


async def test_clear_keeps_display_and_appends_system_entry(tmp_path):
    session = await open_application(_config(_repo(tmp_path), _script(
        {"text": "第一轮", "stop": True},
    )))
    h = Harness(session)
    try:
        assert h.submit("第一问") is True
        await h.wait_request_done()
        before = [type(e) for e in h.entries()]
        assert h.submit("/clear") is True
        assert session.history == ()  # 模型历史清空
        # 显示文档保留 (原条目仍在) 并追加一条系统条目
        after = h.entries()
        assert [type(e) for e in after[: len(before)]] == before
        assert "已清空模型历史" in "\n".join(h.lines())
        # 清空后空闲命令继续可用
        assert h.submit("/status") is True
    finally:
        h.close()
        await session.aclose()


async def test_exit_only_when_idle(tmp_path):
    session = await open_application(_chunked_config(
        tmp_path, json.dumps([{"text": "慢慢回答" * 10, "stop": True}]), chunks=6, delay=0.03
    ))
    h = Harness(session)
    try:
        assert h.submit("占住") is True
        h.bundle["on_exit"]()  # 忙碌: 屏障内无动作
        assert h.exit_requested == 0
        await h.wait_request_done()
        h.bundle["on_exit"]()  # idle: 触发退出回调
        assert h.exit_requested == 1
    finally:
        h.close()
        await session.aclose()


# --- 错误 / 取消条目 -----------------------------------------------------------------


async def test_error_entry_truthful_and_followup_possible(tmp_path):
    session = await open_application(_config(_repo(tmp_path), _script(
        {"error": "provider exploded"},
        {"text": "恢复后回答", "stop": True},
    )))
    h = Harness(session)
    try:
        assert h.submit("第一问") is True
        await h.wait_request_done()
        joined = "\n".join(h.lines())
        assert "错误: provider exploded" in joined  # 如实表达, 不吞没
        assert h.assistant_texts() == [""]  # 错误轮 assistant 条目存在但无正文
        assert h.submit("第二问") is True
        await h.wait_request_done()
        assert h.assistant_texts() == ["", "恢复后回答"]
        assert "错误: provider exploded" in "\n".join(h.lines())  # 错误事实保留
    finally:
        h.close()
        await session.aclose()


async def test_cancel_intent_in_reservation_window_binds_to_own_task(tmp_path):
    session = await open_application(_chunked_config(
        tmp_path,
        json.dumps([
            {"text": "这一轮回答得很慢" * 30, "stop": True},
            {"text": "取消后的新请求", "stop": True},
        ]),
        chunks=10,
        delay=0.03,
    ))
    h = Harness(session)
    try:
        phases: list[str] = []

        def track(update) -> None:
            if isinstance(update, RequestPhaseChanged):
                phases.append(update.phase)

        unsubscribe = session.subscribe(track)
        assert h.submit("取消我") is True
        # reservation 窗口内 (任务尚未起跑) Ctrl+C: 记录取消意图
        h.view.handle_ctrl_c()
        assert h.view._gate.pending_cancel is True
        # 任务起跑后意图交给公共 cancel: 意图移交可早于首轮产出 (请求直接 aborted)
        await h.wait_request_done()
        assert "cancelling" in phases, phases
        assert h.view._gate.pending_cancel is False  # 终结即清除, 不带到下次请求
        assert "已取消" in "\n".join(h.lines())  # 取消条目如实
        # 取消意图不带到下次请求: 第二次请求正常完成 (stop, 非 aborted), 且只有一条取消条目
        assert h.submit("下一个请求") is True
        await h.wait_request_done()
        last = h.assistant_entries()[-1]
        assert last.finalized and last.stop_reason == "stop"
        assert sum(1 for e in h.entries() if isinstance(e, CancelledEntry)) == 1
        unsubscribe()
    finally:
        h.close()
        await session.aclose()


async def test_ctrl_c_executing_cancels_and_idle_clears_draft(tmp_path):
    session = await open_application(_chunked_config(
        tmp_path, json.dumps([{"text": "慢慢地回答" * 40, "stop": True}]), chunks=10, delay=0.03
    ))
    h = Harness(session)
    try:
        assert h.submit("占住") is True
        assert await _wait_until(lambda: session.state == "executing")
        h.view.handle_ctrl_c()  # executing: 公共 cancel (幂等)
        assert await _wait_until(lambda: session.state in ("cancelling", "finalizing"))
        await h.wait_request_done()
        assert "已取消" in "\n".join(h.lines())
        # 空闲 Ctrl+C: 清草稿, 不退出
        h.view.editor.set_text("草稿内容")
        h.view.handle_ctrl_c()
        assert h.view.editor.is_empty()
        assert h.exit_requested == 0
        # 空闲 Escape: 无动作
        h.view.editor.set_text("不会被清掉")
        h.bundle["on_escape"]()
        assert h.view.editor.get_text() == "不会被清掉"
    finally:
        h.close()
        await session.aclose()


# --- process_uncertain / closing / closed ----------------------------------------------


async def test_process_uncertain_error_entry_then_idle_commands_usable(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "stop": True},
    ), plan=_uncertain_plan(root)))
    h = Harness(session)
    try:
        assert h.submit("跑一个会超时的检查") is True
        await h.wait_request_done()
        assert session.has_process_uncertain() is True
        # 编码请求被应用拒绝: 受理真实发生 (用户条目+任务), execute 拒绝如实成条
        assert h.submit("继续改代码") is True
        assert await _wait_until(
            lambda: any(
                isinstance(e, ErrorEntry) and "process_uncertain" in e.text for e in h.entries()
            )
        )
        # 任务终结且 idle: reservation 释放, 诊断/清空/退出不被锁死
        assert h.view._gate.busy() is False
        assert h.submit("/status") is True
        assert h.submit("/clear") is True
        assert session.history == ()
        assert session.has_process_uncertain() is True  # 清空历史不解锁: 锁存是 session 层事实
        h.bundle["on_exit"]()
        assert h.exit_requested == 1
    finally:
        h.close()
        await session.aclose()


async def test_closing_and_closed_reopen_refused(tmp_path):
    session = await open_application(
        _config(_repo(tmp_path), _script({"text": "回答", "stop": True}))
    )
    h = Harness(session)
    try:
        await session.aclose()
        assert session.state == "closed"
        assert h.submit("关闭后再提交") is False
        h.bundle["on_exit"]()
        assert h.exit_requested == 0  # closed 不重开输入
    finally:
        h.close()


async def test_admission_gate_unit_rejects_without_loop_side_effects(tmp_path):
    session = await open_application(
        _config(_repo(tmp_path), _script({"text": "回答", "stop": True}))
    )
    gate = AdmissionGate(session)
    try:
        assert gate.try_admit("") is None  # 空提示拒绝
        assert gate.busy() is False
        task = gate.try_admit("正常问题")
        assert task is not None
        assert gate.busy() is True
        assert gate.try_admit("同批二次") is None  # reservation 挡住
        await asyncio.wait_for(task, timeout=10)
        assert session_idle(session) and gate.busy() is False
    finally:
        await session.aclose()
