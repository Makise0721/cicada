"""审批框与底栏档位显示的最小回归 (P0-1).

只验证可观察行为: 渲染包含命令与动作、按键映射、路由器互斥、底栏档位段。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui.approval_modal import ApprovalModal, ModalRouter
from cicada.ui.conversation_view import ConversationView
from cicada.ui.theme import FooterComponent
from cicada.application import RequestCompleted, open_application
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.ui import run_tui_session
from cicada.ui.terminal_io import FakeTerminal
from test_application import _config
from tests.support.app import script
from tests.ui.test_frontend_headless import CTRL_C, CTRL_D


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r1", tool_name="powershell", command="Remove-Item -Recurse x"
    )


class LoginStub:
    is_open = False
    dismissed = False

    def dismiss(self) -> None:
        self.dismissed = True

    def handle(self, key) -> None:
        raise AssertionError("login stub should not receive approval keys")

    def render(self, columns: int, rows: int) -> list[str]:
        return ["login"]


def test_modal_renders_command_and_actions():
    shown: list = []
    modal = ApprovalModal(lambda request, decision: shown.append((request, decision)))
    request = _request()
    modal.open(request)
    text = "\n".join(modal.render(80, 20))
    assert "需要确认命令" in text
    assert "Remove-Item -Recurse x" in text
    assert "允许一次" in text and "允许本会话" in text and "拒绝" in text


def test_enter_after_down_selects_allow_session():
    shown: list = []
    modal = ApprovalModal(lambda request, decision: shown.append((request, decision)))
    request = _request()
    modal.open(request)
    modal.handle(KeyPress(Keys.Down, ""))
    modal.handle(KeyPress(Keys.ControlM, "\r"))
    assert shown == [(request, "allow_session")]
    assert not modal.is_open


def test_escape_denies_and_closes():
    shown: list = []
    modal = ApprovalModal(lambda request, decision: shown.append((request, decision)))
    request = _request()
    modal.open(request)
    modal.handle(KeyPress(Keys.Escape, "\x1b"))
    assert shown == [(request, "deny")]
    assert not modal.is_open


async def test_ctrl_c_during_approval_cancels_request_before_shell_runs(tmp_path, monkeypatch):
    model = FakeModel(script(
        {"tool_calls": [{"id": "shell", "name": "powershell", "arguments": {
            "command": "Set-Content should-not-exist.txt executed"}}]},
        {"text": "should not continue", "stop": True},
    ))
    session = await open_application(replace(
        _config(tmp_path, script=[]), model_plugin=fake_model_plugin(model), approval=True,
        runtime_dir=tmp_path / "runtime",
    ))
    terminal = FakeTerminal()
    views = []
    original_view = ConversationView

    def capture_view(*args, **kwargs):
        view = original_view(*args, **kwargs)
        views.append(view)
        return view

    monkeypatch.setattr("cicada.ui.ConversationView", capture_view)
    completed = asyncio.Event()
    outcomes = []

    def observe(update):
        if isinstance(update, RequestCompleted):
            outcomes.append(update.outcome)
            completed.set()

    unsubscribe = session.subscribe(observe)
    runner = asyncio.create_task(run_tui_session(session, terminal=terminal, initial_prompt="request shell"))
    try:
        async with asyncio.timeout(3):
            while not views or not views[0].approval_modal.is_open:
                await asyncio.sleep(0)
        terminal.emit([CTRL_C])
        await asyncio.wait_for(completed.wait(), 3)
        assert outcomes[0].result.stop_reason == "aborted"
        assert len(model.requests) == 1
        assert not (tmp_path / "should-not-exist.txt").exists()
        assert not views[0].approval_modal.is_open
        async with asyncio.timeout(3):
            while views[0]._gate.busy():
                await asyncio.sleep(0)
        terminal.emit([CTRL_D])
        await asyncio.wait_for(runner, 3)
    finally:
        unsubscribe()
        if not runner.done():
            session.cancel()
            runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        await session.aclose()


def test_router_prefers_approval_then_falls_back_to_login():
    login = LoginStub()
    approval = ApprovalModal(lambda request, decision: None)
    router = ModalRouter(login, approval)
    assert router.is_open is False
    router.render(80, 20)  # 登录框关闭时也不应抛错
    approval.open(_request())
    assert router.is_open is True
    assert "需要确认命令" in "\n".join(router.render(80, 20))
    router.dismiss()
    assert approval.is_open is False


def test_footer_shows_mode_segment():
    metrics = lambda: (None, False, None, False)
    plain = FooterComponent("C:/repo", "flash", metrics).render(120, 1)[0]
    assert "只读" not in plain and "审批" not in plain and "完全" not in plain
    for label in ("只读", "审批", "完全"):
        line = FooterComponent("C:/repo", "flash", metrics, mode=label).render(120, 1)[0]
        assert label in line


class _SessionStub:
    execution_mode = "read-only"

    def set_execution_mode(self, mode: str) -> bool:
        changed = mode != self.execution_mode
        self.execution_mode = mode
        return changed


class _TranscriptStub:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_system(self, text: str) -> None:
        self.lines.append(text)


def test_mode_command_cycles_and_sets():
    view = ConversationView.__new__(ConversationView)
    view._session = _SessionStub()
    view.transcript = _TranscriptStub()
    assert view._handle_mode_command("") is True
    assert view._session.execution_mode == "approval"
    assert "审批" in view.transcript.lines[-1]
    view._handle_mode_command("full")
    assert view._session.execution_mode == "full"
    assert "完全" in view.transcript.lines[-1]
    view._handle_mode_command("bogus")
    assert "用法" in view.transcript.lines[-1]
    assert view._session.execution_mode == "full"
