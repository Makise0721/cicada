"""TUI 04 e2e: 真实 ApplicationSession/工具/Git/PowerShell/MockTransport 的连续投影.

覆盖票据 04 的 T06/T08/T09/T10/T12/T13 各点与"普通模式未启用检查":
- 检查模式用项目内隔离 Git 工作区 (pytest 私有 basetemp 下) + 真实 PowerShell;
- T12 用 httpx.MockTransport 记录 POST 次数, 核对 65536 bytes 请求守卫零 POST;
- T13/输出回归经 prompt_toolkit Vt100 输出管线对最终可见产物断言 (做法同 01 探针),
  不只检查 RecordingSink/正文; presenter 记录保留原始文本 (规范事实不受显示清理影响).
全部确定性: fake 剧本或 MockTransport, 不启动 Ollama, 无真实模型调用.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

import httpx
import pytest

from cicada.application import ApplicationConfig, open_application
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.runtime.plugin import PluginDefinition
from cicada.ui.presenter import (
    HISTORICAL_PASS_LABEL,
    PROCESS_UNCERTAIN_LABEL,
    ProjectionPresenter,
)
from cicada.ui.rendering import record_fragments, status_fragments

CHECK_COMMAND = (
    "if ((Get-Content -LiteralPath sample.py -Raw) -like '*VALUE = 2*') "
    "{ Write-Output 'check-ok'; exit 0 } else { Write-Output 'check-fail'; exit 1 }"
)

# 01 探针同一形态的"可见问号控制码"签名: ESC 丢失/未被解析后残留的 SGR 文本
VISIBLE_SGR_RE = re.compile(r"\?\[[0-9;]*m")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True
    ).stdout


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8", newline="")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "baseline")
    return root


def _prompt_info(root: Path) -> PromptInfo:
    return PromptInfo(
        version="test",
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=hashlib.sha256(b"test").hexdigest(),
    )


def _script(*entries: dict) -> list:
    from cicada.__main__ import parse_script

    return parse_script(json.dumps(list(entries)))


def _plan(root: Path, command: str, timeout_s: float = 30.0) -> VerificationPlan:
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", command, timeout_s=timeout_s),),
    )


def _config(root: Path, *, script: list, plan: VerificationPlan | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel(script)),
        plan=plan,
        max_turns=6,
        model_label="fake",
    )


class PresenterTap:
    """把 presenter 接进 session.subscribe; 记录每轮 PresentationUpdate."""

    def __init__(self, session) -> None:
        self.presenter = ProjectionPresenter(
            model=session.config.model_label or None,
            workspace=str(session.config.workspace),
            verification_run_id=session.verification_run_id,
        )
        self.updates = []
        self._unsubscribe = session.subscribe(self)

    def __call__(self, update) -> None:
        self.updates.append(self.presenter.accept(update))

    def record(self, key: str):
        return next(r for r in self.presenter.records if r.key == key)

    def try_record(self, key: str):
        return next((r for r in self.presenter.records if r.key == key), None)

    def close(self) -> None:
        self._unsubscribe()


def _delivery(tap: PresenterTap, run_id: str):
    return tap.record(f"{run_id}:delivery")


# --- T08: 累计变更 / 历史 PASS 来源 / 失败与 stale ---------------------------------------------


async def test_t08_cumulative_changes_and_historical_pass_source(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "第一轮检查完成", "stop": True},
        {"text": "第二轮只汇报，不再检查", "stop": True},
    ), plan=plan))
    tap = PresenterTap(session)
    first = await session.execute("改成 VALUE = 2 并检查")
    assert first.exit_code == 0, first.decision.blocking_reasons if first.decision else None
    second = await session.execute("再汇报一次")
    assert second.exit_code == 0

    # 请求 1: 本次检查来源 + 累计口径 + 工件身份
    delivery_1 = _delivery(tap, first.result.run_id)
    body_1 = "\n".join(delivery_1.body)
    assert "本次检查" in body_1 and "可交付" in body_1
    assert "程序判定" in body_1
    assert "相对启动基线累计改动: 1 项" in body_1
    assert str(first.evidence.artifact_path) in body_1
    assert first.evidence.artifact_sha256 in body_1
    assert delivery_1.status == "done" and delivery_1.role == "system"

    # 请求 2: 历史 PASS 标明"此前检查，仍有效"; 绑定自己的独立工件
    delivery_2 = _delivery(tap, second.result.run_id)
    body_2 = "\n".join(delivery_2.body)
    assert HISTORICAL_PASS_LABEL in body_2
    assert "可交付" in body_2, "代码未再变化时历史 current PASS 仍可交付"
    assert str(second.evidence.artifact_path) in body_2
    assert second.evidence.artifact_path != first.evidence.artifact_path, "每请求独立工件"
    # 第二请求没有 check 工具事件 -> 该 PASS 不标为本次检查
    assert "本次检查" not in body_2

    # 状态栏: 检查模式事实
    status = tap.presenter.status
    assert status.verification_run_id == plan.verification_run_id
    assert "check-1=passed/current" in status.check_status
    assert "累计改动口径" in status.change_scope
    tap.close()
    await session.aclose()


async def test_t08_latest_failure_not_shown_as_deliverable(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 3"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "检查失败了", "stop": True},
    ), plan=plan))
    tap = PresenterTap(session)
    outcome = await session.execute("改坏再检查")
    assert outcome.exit_code == 3 and not outcome.decision.can_deliver
    delivery = _delivery(tap, outcome.result.run_id)
    assert delivery.role == "error" and delivery.status == "failed"
    body = "\n".join(delivery.body)
    assert "不可交付" in body and "check check-1 is failed" in body
    assert "程序判定（非模型自然语言）: 不可交付" in body
    tap.close()
    await session.aclose()


async def test_t08_stale_check_not_shown_as_deliverable(tmp_path):
    root = _repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "第一轮通过", "stop": True},
        {"tool_calls": [
            {"id": "e2", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 2", "new_text": "VALUE = 4"}],
            }},
        ], "tool_use": True},
        {"text": "改了但没重检", "stop": True},
    ), plan=plan))
    tap = PresenterTap(session)
    first = await session.execute("改好并检查")
    assert first.exit_code == 0
    second = await session.execute("再改一次不检查")
    assert second.exit_code == 3 and not second.decision.can_deliver
    delivery = _delivery(tap, second.result.run_id)
    body = "\n".join(delivery.body)
    assert delivery.status == "failed"
    assert "stale" in body and "不可交付" in body
    tap.close()
    await session.aclose()


# --- T09: process_uncertain 锁存 / 清空保留 / 应用拒绝呈现 --------------------------------------


async def test_t09_uncertain_latch_survives_clear_and_rejection_shown(tmp_path):
    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "Start-Sleep -Seconds 5", timeout_s=0.5),),
    )
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "c1", "name": "check",
                         "arguments": {"action": "run", "check_id": "check-1"}}],
         "stop": True},
        {"text": "汇报超时", "stop": True},
    ), plan=plan))
    tap = PresenterTap(session)
    outcome = await session.execute("跑一个会超时的检查")
    assert outcome.view is not None and outcome.view.process_uncertain
    assert tap.presenter.has_process_uncertain()
    body = "\n".join(_delivery(tap, outcome.result.run_id).body)
    assert PROCESS_UNCERTAIN_LABEL in body and "不可交付" in body
    assert PROCESS_UNCERTAIN_LABEL in tap.presenter.status.check_status

    # 清空历史: 展示层记录锁存保留事实
    session.clear_history()
    cleared = tap.try_record("app:history-cleared")
    assert cleared is not None
    assert PROCESS_UNCERTAIN_LABEL in "\n".join(cleared.body)
    assert tap.presenter.has_process_uncertain()

    # 未知进程时新编码请求被应用拒绝; RuntimeError 文本如实呈现
    with pytest.raises(RuntimeError, match="process_uncertain"):
        await session.execute("清空后再改代码")
    rejection = tap.presenter.accept_error(
        "process_uncertain: new coding requests are refused; only status, clear or exit remain"
    )
    record = rejection.appended[0]
    assert record.role == "error" and "process_uncertain" in "\n".join(record.body)
    tap.close()
    await session.aclose()


# --- T10: 每请求独立工件与 hash; 缺失/篡改显示失败 ----------------------------------------------


async def _run_two_requests_with_artifact_tampered(tmp_path: Path, *, delete: bool):
    root = _repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "e1", "name": "edit", "arguments": {
                "path": "sample.py",
                "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
            }},
            {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
        ], "tool_use": True},
        {"text": "第一轮通过", "stop": True},
        {"text": "第二轮汇报", "stop": True},
    ), plan=plan))
    tap = PresenterTap(session)
    first = await session.execute("改好并检查")
    assert first.exit_code == 0
    artifact = first.view.checks[0].receipt.output_artifact_path
    assert artifact is not None and artifact.exists()
    if delete:
        artifact.unlink()
    else:
        with open(artifact, "ab") as handle:  # 篡改字节, hash 不再匹配
            handle.write(b"tampered\n")
    second = await session.execute("再汇报一次")
    assert not second.decision.can_deliver
    return session, tap, first, second, artifact


@pytest.mark.parametrize("mode", ["tamper", "missing"])
async def test_t10_artifact_tamper_or_missing_shows_failure(tmp_path, mode):
    session, tap, first, second, artifact = await _run_two_requests_with_artifact_tampered(
        tmp_path, delete=(mode == "missing")
    )
    assert second.exit_code == 3
    reasons = "\n".join(second.decision.blocking_reasons)
    if mode == "tamper":
        assert "no longer matches the hash" in reasons
    else:
        assert "missing or unreadable" in reasons
    delivery = _delivery(tap, second.result.run_id)
    body = "\n".join(delivery.body)
    assert delivery.status == "failed" and "不可交付" in body
    if mode == "tamper":
        assert "no longer matches the hash" in body
    else:
        assert "missing or unreadable" in body
    # 每请求结果绑定自己的工件路径与 hash (两条记录都不缺)
    body_1 = "\n".join(_delivery(tap, first.result.run_id).body)
    assert str(first.evidence.artifact_path) in body_1
    assert first.evidence.artifact_sha256 in body_1
    assert str(second.evidence.artifact_path) in body
    assert second.evidence.artifact_sha256 in body
    assert second.evidence.artifact_path != first.evidence.artifact_path
    tap.close()
    await session.aclose()


# --- T12: 实际 HTTP 请求 65536 bytes 守卫 (MockTransport, 零 POST) ------------------------------


def _ndjson(text: str, prompt_eval: int = 3) -> bytes:
    chunk = json.dumps(
        {"message": {"role": "assistant", "content": text}, "done": False},
        ensure_ascii=False,
    )
    done = json.dumps({
        "message": {"role": "assistant", "content": ""}, "done": True,
        "done_reason": "stop", "prompt_eval_count": prompt_eval, "eval_count": 2,
    })
    return f"{chunk}\n{done}\n".encode("utf-8")


class _CountingTransport(httpx.MockTransport):
    """记录实际 POST 的 MockTransport; 依次回放剧本响应."""

    def __init__(self, responses: list[bytes]) -> None:
        super().__init__(self._handle)
        self.posts: list[httpx.Request] = []
        self._responses = list(responses)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.posts.append(request)
        body = self._responses.pop(0) if self._responses else _ndjson("ok")
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=body)


def _mock_ollama_plugin(transport: _CountingTransport) -> PluginDefinition:
    from cicada.plugins.ollama.model import OllamaModel
    from cicada.plugins.ollama.protocol import OllamaConfig

    def setup(ctx) -> None:
        client = httpx.AsyncClient(transport=transport)
        ctx.defer(client.aclose)
        ctx.provide("model", OllamaModel(
            client,
            OllamaConfig(
                base_url="http://mock.local", model="fake-model",
                max_request_bytes=65536,
            ),
        ))

    return PluginDefinition(name="mock-ollama", setup=setup, provides=frozenset({"model"}))


async def test_t12_oversized_history_request_sends_zero_post_and_recovers(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    big_text = "汉" * 20000  # 60000 UTF-8 bytes -> 进入历史后下一请求超限
    transport = _CountingTransport([_ndjson(big_text, prompt_eval=100), _ndjson("小回复")])
    config = ApplicationConfig(
        workspace=workspace,
        system_prompt="S" * 2000,
        prompt_info=_prompt_info(workspace),
        model_plugin=_mock_ollama_plugin(transport),
        plan=None,
        max_turns=6,
        model_label="fake-model",
    )
    session = await open_application(config)
    tap = PresenterTap(session)

    first = await session.execute("第一问")
    assert first.exit_code == 0
    assert len(transport.posts) == 1
    sent = json.loads(transport.posts[0].content)
    # 实际请求体带系统提示 + 工具 (首个请求尚无历史)
    assert sent["messages"][0]["role"] == "system" and sent["messages"][0]["content"]
    assert sent["tools"], "工具 schema 必须在请求体中"
    assert len(transport.posts[0].content) <= 65536

    # 带历史 (60KB 助手文本) + 系统提示 + 工具 -> 超 65536 bytes -> 零 POST
    second = await session.execute("第二问")
    assert len(transport.posts) == 1, "超限请求不得发出 POST"
    assert second.result.stop_reason == "error" and second.exit_code == 1
    assert "request too large" in (second.result.error or "")
    error_record = tap.record(f"{second.result.run_id}:summary")
    assert "request too large" in "\n".join(error_record.body)
    assert tap.presenter.status.check_status == "未启用检查"

    # 清空上下文后按状态恢复提交
    session.clear_history()
    third = await session.execute("第三问")
    assert len(transport.posts) == 2, "清空历史后恢复 POST"
    assert third.exit_code == 0
    assert len(json.loads(transport.posts[1].content)["messages"]) >= 2
    tap.close()
    await session.aclose()


# --- T13 + 输出回归: 最终可见产物经 prompt_toolkit 输出管线断言 ----------------------------------


def _render_updates_to_vt(tap: PresenterTap) -> str:
    """presenter 全部记录与状态栏经 prompt_toolkit Vt100 输出管线产出最终字节流."""
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.shortcuts import print_formatted_text

    stream = io.StringIO()
    output = Vt100_Output(stream, lambda: Size(rows=40, columns=120), enable_cpr=False)
    for record in tap.presenter.records:
        print_formatted_text(record_fragments(record), output=output)
    print_formatted_text(status_fragments(tap.presenter.status), output=output)
    return stream.getvalue()


async def test_t13_final_visible_output_has_no_control_codes(tmp_path):
    root = _repo(tmp_path)
    inject_command = (
        "Write-Output \"$([char]27)[35;46mINJ\"; "
        "Write-Output '[bold]not markup[/bold]'"
    )
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [{"id": "p1", "name": "powershell",
                         "arguments": {"command": inject_command, "timeout": 15}}],
         "tool_use": True},
        {"text": "### 结论\n\n**加粗** 正文与 `code` 片段", "stop": True},
    )))
    tap = PresenterTap(session)
    outcome = await session.execute("输出一些不可信内容")
    assert outcome.exit_code == 0

    stream = _render_updates_to_vt(tap)
    # styled fragments 路径成立: 程序样式以真实 SGR 输出
    assert "\x1b[" in stream
    # 完成后的可见产物无控制码残留 (不退化为 ?[..m 问号形态)
    assert not VISIBLE_SGR_RE.search(stream), "可见问号控制码出现"
    assert "\x1b[35;46m" not in stream, "注入 SGR 不得进入最终可见输出"
    assert "?[35;46m" not in stream
    visible = ANSI_RE.sub("", stream).replace("\r", "")
    assert "not markup" in visible, "工具文本应作为正文可见"
    assert "INJ" in visible
    assert "结论" in visible and "加粗" in visible, "Markdown 排版后正文可见"
    assert "[bold]not markup[/bold]" in visible, "markup 方括号只作字面文本"

    # 规范事实不受显示清理影响: presenter 记录与会话历史保留原始文本
    tool_card = next(r for r in tap.presenter.records if r.role == "tool")
    raw_body = "\n".join(tool_card.body)
    assert "\x1b[35;46m" in raw_body and "[bold]not markup[/bold]" in raw_body
    canonical = "".join(
        m.result.content for m in session.history
        if getattr(m, "result", None) is not None
    )
    assert "\x1b[35;46m" in canonical
    assistant = next(r for r in tap.presenter.records if r.role == "assistant")
    assert "**加粗**" in "\n".join(assistant.body), "最终记录保留 markdown 原文"
    tap.close()
    await session.aclose()


async def test_t06_projection_through_real_application_events(tmp_path):
    """真实应用事件流中的工具卡: key=(run,turn,call), 参数来自 AssistantCompleted."""
    root = _repo(tmp_path)
    session = await open_application(_config(root, script=_script(
        {"tool_calls": [
            {"id": "g1", "name": "glob", "arguments": {"pattern": "**/*.py"}},
        ], "error": "provider exploded"},
        {"text": "恢复", "stop": True},
    )))
    tap = PresenterTap(session)
    bad = await session.execute("触发模型错误")
    assert bad.exit_code == 1
    # 模型错误路径: 无 ToolStarted 的 ToolCompleted -> failed/未执行卡, 内核固定文本
    card = tap.record("run-1:0:g1")
    assert card.status == "failed"
    body = "\n".join(card.body)
    assert "model stream failed; tool call not executed" in body
    assert "未执行（无 ToolStarted）" in body
    assert '{"pattern": "**/*.py"}' in body, "参数取自规范完成事件"
    tap.close()
    await session.aclose()
