"""票据 05 组合检查: 持久会话的恢复显示、异步清空屏障、保存失败显示与状态页.

真实共享应用 + fake 模型 (与 test_conversation_view 同构的 Harness 驱动);
存档经真实 open_application 保存/恢复。失败路径用 port seam 注入。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace
from pathlib import Path

from test_conversation_view import (
    Harness,
    _config,
    _prompt_info,
    _repo,
    _script,
    _wait_until,
    session_idle,
)

from cicada.application import PersistenceConfig, open_application
from cicada.ui.main_screen import strip_terminal_sequences
from cicada.persistence.contracts import LaunchIdentity


def _persistence_config(root: Path, script: list, *, session_id: str | None):
    from cicada.application import PersistenceConfig as _PC

    info = _prompt_info(root)
    identity = LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version=info.version, prompt_sha256=info.prompt_sha256,
        instructions_sha256=None, checks_signature=None, sources={},
    )
    return dc_replace(
        _config(root, script),
        persistence=_PC(session_id=session_id, identity=identity),
    )


async def test_restored_display_projected_before_live_requests(tmp_path):
    """A13: 订阅前注入只读恢复显示; 新请求追加, 旧回答不重放。"""
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "第一轮回答", "stop": True}), session_id=None)
    )
    first = Harness(session)
    assert first.submit("第一问")
    await first.wait_request_done()
    session_id = session.persistence.session_id
    await session.aclose()
    first.close()

    resumed_session = await open_application(
        _persistence_config(root, _script({"text": "恢复后的回答", "stop": True}), session_id=session_id)
    )
    resumed = Harness(resumed_session)
    try:
        rendered = chr(10).join(resumed.lines(80))
        assert "已恢复历史：1 条请求" in rendered
        assert "第一问" in rendered and "第一轮回答" in rendered
        assert "恢复为查看状态" not in rendered
        assert resumed_session.persistence.restored_request_count == 1
        assert resumed.submit("第二问")
        await resumed.wait_request_done()
        assert "恢复后的回答" in chr(10).join(resumed.lines(80))
        assert resumed_session.persistence.saved_request_count == 1
    finally:
        await resumed_session.aclose()
        resumed.close()


async def test_view_only_restore_shows_reason_and_refuses_requests(tmp_path):
    root = _repo(tmp_path)
    script = _script(
        {"tool_calls": [{"id": "c1", "name": "powershell",
                         "arguments": {"command": "Start-Sleep -Seconds 30", "timeout": 30}}],
         "tool_use": True},
        {"text": "done", "stop": True},
    )
    session = await open_application(_persistence_config(root, script, session_id=None))
    task = asyncio.create_task(session.execute("会留下未证终结的请求"))
    await asyncio.sleep(0.3)
    session.cancel()
    await asyncio.wait_for(task, timeout=15)
    session_id = session.persistence.session_id
    await session.aclose()

    resumed_session = await open_application(
        _persistence_config(root, _script(), session_id=session_id)
    )
    resumed = Harness(resumed_session)
    try:
        assert resumed_session.view_only_reason is not None
        rendered = chr(10).join(resumed.lines(80))
        assert "恢复为查看状态" in rendered and "powershell" in rendered
        assert resumed.submit("新请求")
        assert await _wait_until(
            lambda: "view-only" in chr(10).join(resumed.lines(80))
        ), "查看状态的拒绝事实如实成条"
    finally:
        await resumed_session.aclose()
        resumed.close()


async def test_persistent_clear_barriers_input_until_settled(tmp_path):
    """A14/A10: /clear 经共享 async 入口; 提交期间 Enter 拒绝草稿保留; 落定后释放。"""
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()

        assert harness.submit("/clear")
        assert harness.view._gate.clearing, "命令处置时屏障已同步置位"
        harness.bundle["editor"].set_text("清空中的新请求")
        assert not harness.submit("清空中的新请求"), "屏障内 Enter 不提交"
        assert "清空中的新请求" in harness.bundle["editor"].get_text(), "草稿保留"
        assert await _wait_until(lambda: not harness.view._gate.clearing), "清空落定后释放屏障"
        rendered = chr(10).join(harness.lines(80))
        assert "已清空模型历史" in rendered
        assert harness.submit("清空后的新请求")
        await harness.wait_request_done()
        assert session.persistence.context_generation == 1
    finally:
        await session.aclose()
        harness.close()


async def test_save_failure_shown_and_followups_refused(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    from cicada.persistence.contracts import StorageWriteError

    async def failing(*args, **kwargs):
        raise StorageWriteError("disk gone")

    session._store.commit_request = failing  # type: ignore[method-assign]
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        assert await _wait_until(lambda: session_idle(session))
        assert await _wait_until(
            lambda: "[session save failed:" in chr(10).join(harness.lines(80))
        ), "保存失败如实显示"
        assert harness.submit("第二问")
        assert await _wait_until(
            lambda: "storage_failed" in chr(10).join(harness.lines(80))
        ), "锁存拒绝后继编码"
    finally:
        await session.aclose()
        harness.close()


async def test_status_persistence_section_after_save(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("/status")
        rendered = strip_terminal_sequences(chr(10).join(harness.lines(80)))
        assert "会话 ID: " in rendered
        assert "模式: save · epoch " in rendered
        assert "上下文代次: 0" in rendered
        assert "已保存 1 条" in rendered
    finally:
        await session.aclose()
        harness.close()


async def test_clear_failure_shows_error_and_releases_barrier(tmp_path, monkeypatch):
    """审查 05-F9: 清空提交失败 → 屏障释放 + 错误如实成条 (内存历史由应用保证不变)."""
    from cicada.persistence.contracts import StorageWriteError

    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()

        async def failing(epoch_id):
            raise StorageWriteError("disk gone")

        session._store.commit_clear = failing  # type: ignore[method-assign]
        assert harness.submit("/clear")
        assert await _wait_until(lambda: not harness.view._gate.clearing), "失败后屏障释放"
        rendered = strip_terminal_sequences(chr(10).join(harness.lines(80)))
        assert "清空提交失败" in rendered
        assert "disk gone" in rendered
    finally:
        await session.aclose()
        harness.close()
