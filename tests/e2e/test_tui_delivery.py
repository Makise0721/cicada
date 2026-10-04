"""05 组合路径 e2e: --tui 生产组合 (Pi 主屏前端 + 共享应用) 语义断言.

经 run_tui_session 注入 FakeTerminal (VtScreen 真实 VT 输出) 驱动公开组合,
覆盖 T04/T05/T07/T11/T14 组合路径的迁移语义: 流式原位更新一次完成、屏障内
Enter 静默且草稿保留、退出码与关闭观察者保留订阅、缩放重排与不可恢复失败;
CLI 子进程层面的 T14 (非 TTY/缺依赖/零 UI 导入) 在 tests/test_cli.py。
真实终端人工核对 (P16) 留待 06, 自动化不能代签。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress

from cicada.application import open_application
from cicada.ui import MainScreenFrontend, run_tui_session
from tui_fixture import (
    CTRL_C,
    ENTER,
    StageTerminal,
    clear_draft,
    make_config,
    model_of,
    script_json,
    text_keys,
    wait_until,
)

DRAFT_KEYS = [KeyPress(ch, ch) for ch in "忙碌草稿"]


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "baseline"], check=True)
    return root


# --- T04: 流式增量同一条目原位更新, 完成校准一次, 无重复正文 -------------------------


async def test_t04_streaming_in_place_and_final_once(tmp_path):
    answer = "第一段流式内容,先出现在同一 assistant 条目中。\n第二段内容随后追加。"
    session = await open_application(make_config(_repo(tmp_path), script_json({"text": answer, "stop": True}), chunks=6, delay=0.05))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("问题"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("第一段流式内容")), "流式增量可见"
        assert stage.count("第一段流式内容") == 1, "同一 assistant 条目原位更新"
        assert await wait_until(
            lambda: stage.count("第一段流式内容") == 1 and stage.count("第二段内容随后追加。") == 1
        ), "完成校准不重复打印"
        assert await wait_until(lambda: session.state == "idle")
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0, "正常请求退出码 0"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T05: 忙碌提交静默拒绝, 不排队, 草稿保留 -----------------------------------------


async def test_t05_busy_submit_silent_draft_kept(tmp_path):
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "慢慢回答", "stop": True}), chunks=4, delay=0.2)
    )
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("第一问"))
        stage.emit([ENTER])
        assert await wait_until(lambda: session.state == "executing")
        stage.emit(DRAFT_KEYS)  # 忙碌窗口内草稿可编辑
        assert await wait_until(lambda: stage.contains("忙碌草稿"))
        stage.emit([ENTER])  # 屏障内 Enter: 静默无动作
        await asyncio.sleep(0.15)
        assert len(model_of(session).requests) == 1, "不排队, 零重复请求"
        assert stage.contains("忙碌草稿"), "草稿保留"
        assert "拒绝" not in stage.text() and "提示" not in stage.text(), "不输出说明 (安静界面)"
        assert await wait_until(lambda: session.state == "idle")
        # 空闲 Ctrl+C 清草稿后 /exit
        assert await clear_draft(stage, "忙碌草稿")
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T07: finalizing 阶段提交静默拒绝 -------------------------------------------------


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

    async def initialize(self, cancel):
        return await self._inner.initialize(cancel)

    async def run_check(self, check_id: str, cancel):
        return await self._inner.run_check(check_id, cancel)

    async def refresh(self, cancel):
        await asyncio.sleep(self._delay)
        return await self._inner.refresh(cancel)

    async def finalize(self, cancel):
        return await self._inner.finalize(cancel)


async def test_t07_finalizing_submit_silent(tmp_path):
    from cicada.plugins.coding.verification import generate_verification_run_id
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan

    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    session = await open_application(
        make_config(root, script_json({"text": "直接回答", "stop": True}), plan=plan)
    )
    session._service = SlowService(session._service, delay=0.8)  # type: ignore[arg-type]
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("第一问"))
        stage.emit([ENTER])
        assert await wait_until(lambda: session.state == "finalizing", timeout=10)
        stage.emit(text_keys("插队草稿"))
        stage.emit([ENTER])
        await asyncio.sleep(0.1)
        assert len(model_of(session).requests) == 1, "finalizing 内 Enter 零请求"
        assert stage.contains("插队草稿"), "草稿保留"
        assert "拒绝" not in stage.text(), "不输出旧等待说明"
        assert await wait_until(lambda: session.state == "idle")
        assert await clear_draft(stage, "插队草稿")
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 3, "检查模式无检查执行: 模型 stop 但条件不满足"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- T14 + T11: run_tui_session 首问 / 退出码 / 取消后关闭 ----------------------------


async def test_t14_initial_prompt_submitted_once_exit_zero(tmp_path):
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "回答", "stop": True}))
    )
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(
            run_tui_session(session, initial_prompt="首问", terminal=stage.terminal)
        )
        assert await wait_until(lambda: len(model_of(session).requests) == 1), "首问只提交一次"
        assert await wait_until(lambda: stage.contains("首问") and stage.contains("回答"))
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0, "正常关闭取最后请求退出码"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


async def test_t11_cancel_yields_aborted_exit_code_and_closes(tmp_path):
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "慢慢回答", "stop": True}), chunks=4, delay=0.2)
    )
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("问题"))
        stage.emit([ENTER])
        assert await wait_until(lambda: session.state == "executing")
        stage.emit([CTRL_C])
        assert await wait_until(lambda: stage.contains("已取消"))
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 1, "取消请求的退出码 1 (aborted)"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- 关闭回执: 观察者保留订阅到 ApplicationClosed; 晚到关闭错误优先 1 -------------------


async def test_close_observer_retained_until_application_closed(tmp_path):
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "回答", "stop": True}))
    )
    stage = StageTerminal()
    observed: list[str] = []

    def extra_observer(update) -> None:
        observed.append(type(update).__name__)

    unsubscribe_extra = session.subscribe(extra_observer)
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
        assert "ApplicationClosed" in observed, "外部观察者保留订阅直到关闭通知"
        assert observed[-1] == "ApplicationClosed"
        assert session._observers == [], "关闭完成后 session 清空全部观察者"
    finally:
        unsubscribe_extra()
        if session.state != "closed":
            await session.aclose()


async def test_runner_failure_closes_session_and_exits_one(tmp_path, monkeypatch):
    """frontend.run_async 异常: UI 不可恢复优先退出 1, 且 session 经关闭监督收尾."""

    async def boom(self):
        raise RuntimeError("frontend exploded")

    monkeypatch.setattr(MainScreenFrontend, "run_async", boom)
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "答", "stop": True}))
    )
    try:
        code = await run_tui_session(session, terminal=StageTerminal().terminal)
        assert code == 1
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


async def test_unrecoverable_write_failure_exits_one(tmp_path):
    """写失败恢复重试仍失败: terminal_failed -> 退出 1, 会话仍有序关闭 (P10/P15)."""
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "答", "stop": True}))
    )
    stage = StageTerminal()
    stage.terminal.fail_write = True
    try:
        code = await asyncio.wait_for(
            run_tui_session(session, terminal=stage.terminal), timeout=15
        )
        assert code == 1, "不可恢复 UI 错误优先退出 1"
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


# --- P08 组合级: 缩放后当前显示文档按新宽度重排 ----------------------------------------


async def test_resize_reflows_visible_document(tmp_path):
    long_line = "这是一段很长的中文回答用于缩放重排验收" * 6
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": long_line, "stop": True}))
    )
    stage = StageTerminal(columns=120, rows=30)
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))
        await wait_until(lambda: stage.contains("fake"))
        stage.emit(text_keys("缩放问题"))
        stage.emit([ENTER])
        assert await wait_until(
            lambda: len(model_of(session).requests) >= 1 and session.state == "idle"
        )
        assert await wait_until(lambda: stage.contains("这是一段很长的中文回答")), "正文落屏"
        requests_before = len(model_of(session).requests)
        stage.set_size(60, 24)  # 触发 on_resize -> full redraw 回放
        await asyncio.sleep(0.15)
        from cicada.ui.main_screen import visible_width

        for line in stage.visible_lines():
            assert visible_width(line) <= 60, f"新宽度下无超宽行: {line!r}"
        assert stage.contains("这是一段很长的中文回答"), "缩放后正文仍完整可见"
        assert len(model_of(session).requests) == requests_before, "缩放不触发请求"
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
    finally:
        if session.state != "closed":
            await session.aclose()


# --- P12: /status 诊断注入与有界限额 (CLI 提供不可变诊断对象) ---------------------------


async def test_status_diagnostics_injected_and_bounded(tmp_path):
    session = await open_application(
        make_config(_repo(tmp_path), script_json({"text": "回答", "stop": True}), chunks=1)
    )
    stage = StageTerminal()

    def diagnostics() -> list[str]:
        return [
            "prompt 版本: builtin:p4-v1",
            "prompt sha256: " + "a" * 64,
            "超长字段: " + "汉" * 400,  # > 512 bytes: 必须截断并给标记
        ]

    try:
        task = asyncio.ensure_future(
            run_tui_session(session, terminal=stage.terminal, diagnostics=diagnostics)
        )
        assert await wait_until(lambda: stage.terminal.started and stage.terminal.written)
        stage.emit(text_keys("/status"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("状态: idle"))
        flat = stage.flat()
        assert "prompt 版本: builtin:p4-v1" in flat
        assert "prompt sha256: " + "a" * 64 in flat
        assert "…[已截断: 512/" in flat, "超长字段按 512 bytes 限额截断并标记"
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=15)
        assert code == 0
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()


async def test_run_tui_cli_assembles_diagnostics_from_config(tmp_path):
    """__main__._run_tui: 诊断快照来自 ApplicationConfig 已发布字段 (P12 组装路径)."""
    from cicada.__main__ import _run_tui
    from cicada.plugins.coding.verification import generate_verification_run_id
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
    from cicada.ui import run_tui_session as run_fn

    root = _repo(tmp_path)
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )
    app_config = make_config(root, script_json({"text": "答", "stop": True}), chunks=1, plan=plan)
    stage = StageTerminal()
    captured: list[str] = []

    async def runner(session, **kwargs):
        assert kwargs.get("diagnostics") is not None, "CLI 必须注入诊断"
        captured.extend(kwargs["diagnostics"]())

        async def driver():
            assert await wait_until(lambda: stage.terminal.started and stage.terminal.written)
            stage.emit(text_keys("/status"))
            stage.emit([ENTER])
            assert await wait_until(lambda: stage.contains("状态: idle"))
            stage.emit(text_keys("/exit"))
            stage.emit([ENTER])

        driver_task = asyncio.ensure_future(driver())
        code = await run_fn(session, terminal=stage.terminal, **kwargs)
        await driver_task
        return code

    code = await asyncio.wait_for(
        _run_tui(runner, app_config, initial_prompt=None), timeout=30
    )
    assert code == 0, "无请求完成: 正常关闭退出码 0"
    # 诊断快照字段与 ApplicationConfig/prompt_info/plan 一致
    assert f"工作区: {root}" in " ".join(captured) or any("工作区:" in line for line in captured)
    assert any("prompt 版本: stage-acceptance" in line for line in captured)
    assert any("prompt sha256: " for line in captured if line.startswith("prompt sha256"))
    assert any("检查配置: ids=check-1" in line for line in captured)
    assert any("检查超时: 30s" in line for line in captured)
    assert any("轮次上限: 6" in line for line in captured)
    # /status 实际显示包含注入诊断
    assert stage.contains("prompt 版本: stage-acceptance")
