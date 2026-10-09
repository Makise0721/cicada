"""05 组合 e2e: 真实 ApplicationSession/工具/Git/PowerShell/MockTransport 的新前端投影.

覆盖票据 05/06 的 T06/T08/T09/T10/T12/T13 语义迁移 (旧 ProjectionPresenter 断言
改为 Pi 主屏 VT 屏可见事实):
- 检查模式用项目内隔离 Git 工作区 (pytest 私有 basetemp 下) + 真实 PowerShell;
- T12 用 httpx.MockTransport 记录 POST 次数, 核对 65536 bytes 请求守卫零 POST;
- 显示断言合并滚动历史与屏幕 (stage.flat()), 长路径/hash 允许跨行拼接;
- 规范事实与会话历史保持原始内容, 显示副本经清理 (T13)。
全部确定性: fake 剧本或 MockTransport, 不启动 Ollama, 无真实模型调用。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
import pytest

from cicada.application import open_application
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.runtime.plugin import PluginDefinition
from cicada.ui import run_tui_session
from tui_fixture import (
    ENTER,
    StageTerminal,
    git_repo,
    make_config,
    model_of,
    script_json,
    text_keys,
    wait_until,
)

CHECK_COMMAND = (
    "if ((Get-Content -LiteralPath sample.py -Raw) -like '*VALUE = 2*') "
    "{ Write-Output 'check-ok'; exit 0 } else { Write-Output 'check-fail'; exit 1 }"
)

# 01 探针同一形态的"可见问号控制码"签名: ESC 丢失/未被解析后残留的 SGR 文本
VISIBLE_SGR_RE = re.compile(r"\?\[[0-9;]*m")


def _plan(root: Path, command: str, timeout_s: float = 30.0) -> VerificationPlan:
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", command, timeout_s=timeout_s),),
    )


def _edit_call(call_id: str, old: str, new: str) -> dict:
    return {"id": call_id, "name": "edit", "arguments": {
        "path": "sample.py",
        "edits": [{"old_text": old, "new_text": new}],
    }}


def _check_call(call_id: str) -> dict:
    return {"id": call_id, "name": "check",
            "arguments": {"action": "run", "check_id": "check-1"}}


async def _run_until_idle(session, stage: StageTerminal, question: str, *, requests: int) -> None:
    # 前端首帧 (terminal.start + 初始渲染) 就绪后才接受输入
    assert await wait_until(lambda: stage.terminal.started and stage.terminal.written), "前端未就绪"
    stage.emit(text_keys(question))
    stage.emit([ENTER])
    assert await wait_until(
        lambda: len(model_of(session).requests) >= requests
    ), f"请求未到 {requests}"
    assert await wait_until(lambda: session.state == "idle"), "请求未回 idle"
    # 渲染节流 (16ms) 落后于状态翻转: 等最终帧落屏再断言可见事实
    await asyncio.sleep(0.05)


# --- T08: 累计变更 / 历史 PASS 来源 / 每请求独立工件 --------------------------------------------


async def test_t08_cumulative_changes_and_historical_pass_source(tmp_path):
    root = git_repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    script = script_json(
        {"tool_calls": [_edit_call("e1", "VALUE = 1", "VALUE = 2"), _check_call("c1")],
         "tool_use": True},
        {"text": "第一轮检查完成", "stop": True},
        {"text": "第二轮只汇报，不再检查", "stop": True},
    )
    session = await open_application(make_config(root, script, chunks=1, plan=plan))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "改成 VALUE = 2 并检查", requests=2)
        flat = stage.flat()
        assert "程序判定（非模型自然语言）: 可交付" in flat
        assert "来源=本次检查" in flat
        assert "相对启动基线累计改动: 1 项" in flat
        assert "累计变更工件" in flat and "sha256=" in flat

        await _run_until_idle(session, stage, "再汇报一次", requests=3)
        flat = stage.flat()
        assert "此前检查，仍有效" in flat, "历史 current PASS 标注仍有效"
        assert flat.count("此前检查，仍有效") == 1
        assert "来源=本次检查" in flat, "第二轮无 check 事件不改标本次"
        assert stage.contains("第二轮只汇报")
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


async def test_t08_latest_failure_and_stale_not_deliverable(tmp_path):
    root = git_repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    # 请求 1 检查失败 (VALUE=3 不满足); 请求 2 通过后第三轮再改不重检 -> stale
    script = script_json(
        {"tool_calls": [_edit_call("e1", "VALUE = 1", "VALUE = 3"), _check_call("c1")],
         "tool_use": True},
        {"text": "检查失败了", "stop": True},
        {"tool_calls": [_edit_call("e2", "VALUE = 3", "VALUE = 2"), _check_call("c2")],
         "tool_use": True},
        {"text": "第二轮通过", "stop": True},
        {"tool_calls": [_edit_call("e3", "VALUE = 2", "VALUE = 9")], "tool_use": True},
        {"text": "又改了但没重检", "stop": True},
    )
    session = await open_application(make_config(root, script, chunks=1, plan=plan))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "改坏再检查", requests=2)
        flat = stage.flat()
        assert "程序判定（非模型自然语言）: 不可交付" in flat
        assert "check check-1 is failed" in flat

        await _run_until_idle(session, stage, "修好再检查", requests=4)
        await _run_until_idle(session, stage, "再改一次不检查", requests=5)
        flat = stage.flat()
        assert "freshness=stale" in flat and "不可交付" in flat
        # stale 的历史 PASS 只标中性来源, 不得宣称为仍有效
        assert "此前检查" in flat
        assert "仍有效" not in flat
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 3
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T09: process_uncertain 锁存 / 清空保留 / 编码被拒但 idle 命令可用 ---------------------------


async def test_t09_uncertain_latch_survives_clear_and_commands_remain(tmp_path):
    root = git_repo(tmp_path)
    plan = _plan(root, "Start-Sleep -Seconds 5", timeout_s=0.5)
    script = script_json(
        {"tool_calls": [_check_call("c1")], "stop": True},
        {"text": "汇报超时", "stop": True},
    )
    session = await open_application(make_config(root, script, chunks=1, plan=plan))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "跑一个会超时的检查", requests=1)
        assert stage.contains("进程终结未知（锁存）"), "锁存事实可见"
        assert stage.contains("不可交付")

        # 清空历史: 锁存保留事实成条; 编码请求被应用拒绝且如实呈现
        stage.emit(text_keys("/clear"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("已清空模型历史"))
        assert stage.count("进程终结未知（锁存）") >= 2, "清空条目保留锁存说明"
        stage.emit(text_keys("清空后再改代码"))
        stage.emit([ENTER])
        # 等待被拒请求的错误条目 (唯一文本; 错误条目逐行折行, needle 须在首折行内)
        assert await wait_until(
            lambda: "process_uncertain: new coding requests" in stage.flat()
        ), "拒绝文本如实成条"

        # 未知锁存后 idle 的 /status、/clear、/exit 仍可用 (P14)
        stage.emit(text_keys("/status"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("状态: idle"))
        assert "进程终结未知（锁存）" in stage.flat()
        stage.emit(text_keys("/clear"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.text().count("已清空模型历史") >= 2)
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 3, "最后完成请求: 模型 stop 但检查条件不满足"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T10: 每请求独立工件与 hash; 缺失/篡改显示失败 ----------------------------------------------


def _check_artifact_from_history(session) -> tuple[Path, str]:
    """check 回执的输出工件路径与 hash (会话历史 ToolResult.details 公开事实)。

    篡改/删除是测试动作的定位手段; 判定后果的断言仍全部基于显示文档与退出码。
    """
    details = next(
        m.result.details
        for m in session.history
        if getattr(m, "result", None) is not None and m.result.name == "check"
    )
    return Path(details["output_artifact_path"]), details["output_artifact_sha256"]


@pytest.mark.parametrize("mode", ["tamper", "missing"])
async def test_t10_artifact_tamper_or_missing_shows_failure(tmp_path, mode):
    root = git_repo(tmp_path)
    plan = _plan(root, CHECK_COMMAND)
    script = script_json(
        {"tool_calls": [_edit_call("e1", "VALUE = 1", "VALUE = 2"), _check_call("c1")],
         "tool_use": True},
        {"text": "第一轮通过", "stop": True},
        {"text": "第二轮汇报", "stop": True},
    )
    session = await open_application(make_config(root, script, chunks=1, plan=plan))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "改好并检查", requests=2)
        artifact, sha = _check_artifact_from_history(session)
        assert artifact.exists() and sha
        if mode == "missing":
            artifact.unlink()
        else:
            with open(artifact, "ab") as handle:  # 篡改字节, hash 不再匹配
                handle.write(b"tampered\n")
        await _run_until_idle(session, stage, "再汇报一次", requests=3)
        flat = stage.flat()
        assert "不可交付" in flat
        if mode == "tamper":
            assert "no longer matches the hash" in flat
        else:
            assert "missing or unreadable" in flat
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 3
        assert session.state == "closed"
    finally:
        if session.state != "closed":
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
    from cicada.application import ApplicationConfig
    from cicada.prompting import PromptInfo

    workspace = tmp_path / "ws"
    workspace.mkdir()
    big_text = "汉" * 20000  # 60000 UTF-8 bytes -> 进入历史后下一请求超限
    transport = _CountingTransport([_ndjson(big_text, prompt_eval=100), _ndjson("小回复")])
    config = ApplicationConfig(
        workspace=workspace,
        system_prompt="S" * 2000,
        prompt_info=PromptInfo(
            version="test", workspace=workspace, instructions_path=None,
            instructions_sha256=None, prompt_sha256="0" * 64,
        ),
        model_plugin=_mock_ollama_plugin(transport),
        plan=None,
        max_turns=6,
        model_label="fake-model",
    )
    session = await open_application(config)
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        assert await wait_until(lambda: stage.terminal.started and stage.terminal.written)
        # mock ollama 无 ChunkedFakeModel: 以 transport POST 计数与 idle 等待
        stage.emit(text_keys("第一问"))
        stage.emit([ENTER])
        assert await wait_until(
            lambda: len(transport.posts) == 1 and session.state == "idle"
        )
        await asyncio.sleep(0.05)
        assert len(transport.posts) == 1
        sent = json.loads(transport.posts[0].content)
        assert sent["messages"][0]["role"] == "system" and sent["messages"][0]["content"]
        assert sent["tools"], "工具 schema 必须在请求体中"
        assert len(transport.posts[0].content) <= 65536

        # 带历史 (60KB 助手文本) + 系统提示 + 工具 -> 超 65536 bytes -> 零 POST
        stage.emit(text_keys("第二问"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("request too large")), "守卫错误如实成条"
        assert len(transport.posts) == 1, "超限请求不得发出 POST"
        assert await wait_until(lambda: session.state == "idle")

        # 清空上下文后按状态恢复提交
        stage.emit(text_keys("/clear"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("已清空模型历史"))
        stage.emit(text_keys("第三问"))
        stage.emit([ENTER])
        assert await wait_until(
            lambda: len(transport.posts) == 2 and session.state == "idle"
        )
        assert len(transport.posts) == 2, "清空历史后恢复 POST"
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0, "最后完成请求正常"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T13 + T06: 可见产物控制码清理 / 内核错误闭合工具卡 ------------------------------------------


# NO_COLOR 存在(含空值) 与 TERM=dumb 都走 plain; "" 即 NO_COLOR 存在的代表值
@pytest.mark.parametrize("no_color,term", [(None, "xterm-256color"), ("", "xterm-256color"),
                                          (None, "dumb")])
async def test_t13_visible_output_has_no_control_codes(tmp_path, monkeypatch, no_color, term):
    monkeypatch.setenv("TERM", term)
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    root = git_repo(tmp_path)
    # Console 直接写原始字节，绕开 PowerShell 按 TERM/NO_COLOR 格式化对象的降级。
    # 工具真实捕获带 SGR 的输出后，UI 净化和规范历史保留才能分别被验证。
    inject_command = (
        "[Console]::Out.WriteLine(\"$([char]27)[35;46mINJ\"); "
        "Write-Output '[bold]not markup[/bold]'"
    )
    script = script_json(
        {"tool_calls": [{"id": "p1", "name": "powershell",
                         "arguments": {"command": inject_command, "timeout": 15}}],
         "tool_use": True},
        {"text": "### 结论\n\n**加粗** 正文与 `code` 片段", "stop": True},
    )
    session = await open_application(make_config(root, script, chunks=1))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "输出一些不可信内容", requests=1)
        raw = "".join(stage.terminal.written)
        if no_color is not None or term == "dumb":
            assert all(code in ("\x1b[0m",) for code in re.findall(r"\x1b\[[0-9;]*m", raw))
        else:
            assert "38;2;" in raw, "有色组合确有样式"
        assert not VISIBLE_SGR_RE.search(raw), "可见问号控制码出现 (ESC 丢失形态)"
        assert "\x1b[35;46m" not in raw, "注入 SGR 不得进入最终可见输出"
        flat = stage.flat()
        assert "INJ" in flat and "not markup" in flat, "工具文本作为正文可见"
        assert "结论" in flat and "加粗" in flat, "Markdown 排版后正文可见"
        assert "[bold]not markup[/bold]" in flat, "markup 方括号只作字面文本"
        # 规范事实不受显示清理影响: 会话历史保留原始文本
        canonical = "".join(
            m.result.content for m in session.history
            if getattr(m, "result", None) is not None
        )
        assert "\x1b[35;46m" in canonical
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


async def test_t06_unstarted_tool_completion_after_model_error(tmp_path):
    """模型错误路径: 无 ToolStarted 的 ToolCompleted 以终态卡显示内核固定文本."""
    root = git_repo(tmp_path)
    script = script_json(
        {"tool_calls": [{"id": "g1", "name": "glob", "arguments": {"pattern": "**/*.py"}}],
         "error": "provider exploded"},
    )
    session = await open_application(make_config(root, script, chunks=1))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "触发模型错误", requests=1)
        flat = stage.flat()
        assert "glob" in flat, "工具卡可见"
        assert "model stream failed; tool call not executed" in flat, "内核闭合文本如实显示"
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 1, "模型错误请求退出码 1"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- 普通模式 /status: 未启用检查 ----------------------------------------------------------------


async def test_status_normal_mode_reports_check_not_enabled(tmp_path):
    session = await open_application(
        make_config(git_repo(tmp_path), script_json({"text": "普通回答", "stop": True}), chunks=1)
    )
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await _run_until_idle(session, stage, "普通问题", requests=1)
        stage.emit(text_keys("/status"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("状态: idle"))
        assert stage.contains("检查: 未启用检查")
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()
