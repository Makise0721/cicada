"""审批框与底栏档位显示的最小回归 (P0-1).

只验证可观察行为: 渲染包含命令与动作、按键映射、路由器互斥、底栏档位段。
"""

from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui.approval_modal import ApprovalModal, ModalRouter
from cicada.ui.conversation_view import ConversationView
from cicada.ui.theme import FooterComponent


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


def test_footer_shows_mode_only_when_selected():
    metrics = lambda: (None, False, None, False)
    plain = FooterComponent("C:/repo", "flash", metrics).render(120, 1)[0]
    assert "只读" not in plain and "审批" not in plain
    approval = FooterComponent("C:/repo", "flash", metrics, mode="审批").render(120, 1)[0]
    assert "审批" in approval


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
