"""02 独立审查处置测试 (2026-10-06): F6/F8/F10/F12 应用层补测.

复用 test_compaction 的手工会话夹具; 全部为审查发现的直接钉子。
PA03 事件序钉子已并入 test_compaction 的接受后取消测试 (同一 gated-commit
场景); A11 扫描钉子由 test_auto_trigger 的自动重试复核代表。
"""

from __future__ import annotations

import asyncio

import pytest

from test_compaction import (
    _GatedSummary,
    _ScriptedSummary,
    _budget,
    _config,
    _identity,
    _manual_persistent_session,
    _two_committed_requests,
)

from cicada.application import ApplicationSession
from cicada.boot import bootstrap
from cicada.budget import BudgetConfig
from cicada.context_policy import BudgetPolicy
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.core.ports import StreamDone
from cicada.persistence.contracts import (
    PersistenceError,
    StorageWriteError,
    StoredIdentity,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store import open_session_store

from tests.support.budget import admission_total


async def test_admission_uses_projection_after_compaction_not_canonical(tmp_path):
    """01-F10 集成证据 (审查 02-F10): 压缩后预估按有效消息重估.

    num_ctx 卡在窗口内: 第二次 execute 通过、压缩后旧规范历史+新 prompt 超限
    (若预估计量面未切换则必拒)、有效投影+新 prompt 可容纳 → 放行且模型所见为投影。
    """
    u1, a1 = UserMessage(text="x" * 9000), AssistantMessage(text="")
    u2, a2 = UserMessage(text="y" * 2000), AssistantMessage(text="")
    probe = BudgetConfig(num_ctx=1, num_predict=2048)
    t_second = admission_total("test system prompt", (u1, a1, u2), probe)
    t_summary = admission_total("test system prompt", (u1, a1), probe)  # 摘要输入下界 (提示+覆盖区间)
    t_final_canonical = admission_total(
        "test system prompt", (u1, a1, u2, a2, UserMessage(text="第三问")), probe
    )
    assert t_second < t_final_canonical and t_summary < t_final_canonical
    num_ctx = (max(t_second, t_summary) + t_final_canonical) // 2
    config = BudgetConfig(num_ctx=num_ctx, num_predict=2048)
    session, main_model, _channel, store = await _manual_persistent_session(
        tmp_path,
        script=[[StreamDone("stop")]] * 3,
        channel=_ScriptedSummary(text="较早请求的摘要正文"),
        num_ctx=num_ctx,
    )
    try:
        await session.execute(u1.text)
        await session.execute(u2.text)
        compacted = await session.compact_context()
        assert compacted is not None and compacted.end_message_index == 2
        # 前置断言: 若预估仍按规范历史 (含被摘要的 req1 原文), 此请求必被拒绝
        assert admission_total(
            "test system prompt", (*session._history, UserMessage(text="第三问")), config
        ) > num_ctx, "前置条件: 规范历史口径必须超限"
        outcome = await session.execute("第三问")
        assert outcome.result.stop_reason == "stop", "有效投影口径放行 (01-F10)"
        sent = main_model.requests[-1]
        assert sent.messages[0].text.startswith("[程序生成的对话摘要]")
        assert sent.messages[1:] == (u2, a2, UserMessage(text="第三问"))
    finally:
        await session.aclose()
        await store.aclose()


class _FailingScript:
    """按剧本产出的摘要通道替身 (失败分类集成用)."""

    def __init__(self, events) -> None:
        self._events = events
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        for event in self._events:
            yield event


async def test_summary_failure_never_persists_or_switches(tmp_path):
    """A06 集成 (审查 02-F6): 摘要失败 → 零落库、投影不变、屏障落定、健康存储不锁存.

    失败分类矩阵 (length/error/tool_use/empty/too_long/aborted) 由底层
    test_context_projection 逐类覆盖; 应用层所有分类进入 _run_compaction
    同一拒绝分支, 此处保留 error 类作集成代表。
    """
    session, _main, _channel, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_FailingScript([StreamDone("error", "boom")]),
    )
    try:
        await _two_committed_requests(session)
        with pytest.raises(RuntimeError, match="summary generation failed") as exc_info:
            await session.compact_context()
        assert "error" in str(exc_info.value)
        assert store.recovery_snapshot().summaries == (), "无摘要记录落库"
        assert session._context_projection.blocks == (), "投影不变"
        assert session._compact_done.done(), "屏障落定"
        assert session.persistence.storage_failed is None, "失败分类不是存储故障"
    finally:
        await session.aclose()
        await store.aclose()


async def test_execute_and_clear_refused_while_compaction_in_progress(tmp_path):
    """A05 应用层互斥 (审查 02-F8): 压缩在途时 execute/clear 双向拒绝 (不依赖 UI 屏障)."""
    channel = _GatedSummary()
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=channel,
    )
    try:
        await _two_committed_requests(session)
        compact_task = asyncio.ensure_future(session.compact_context())
        while not channel.requests:
            await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="compaction is in progress"):
            await session.execute("压缩中的新请求")
        with pytest.raises(RuntimeError, match="compaction is in progress"):
            await session.clear_context()
        assert session._context_generation == 0, "clear 未推进代次"
        channel.gate.set()
        compacted = await asyncio.wait_for(compact_task, timeout=5)
        assert compacted is not None
    finally:
        channel.gate.set()
        await session.aclose()
        await store.aclose()


async def test_compact_refused_while_clear_in_progress(tmp_path):
    """A05 反向互斥 (审查 02-F8): 清空在途时压缩拒绝."""
    store = open_session_store(
        directory=tmp_path / ".cicada" / "sessions", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    original_clear = store.commit_clear
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_clear(epoch):
        result = await original_clear(epoch)
        entered.set()
        await release.wait()
        return result

    main_model = FakeModel([[StreamDone("stop")], [StreamDone("stop")]])
    config = _config(tmp_path, main_model, budget=_budget())
    channel = _ScriptedSummary()
    policy = BudgetPolicy(main_model, config.budget, summarize=channel)
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save",
        budget_policy=policy,
    )
    object.__setattr__(store, "commit_clear", gated_clear)
    try:
        await _two_committed_requests(session)
        clear_task = asyncio.ensure_future(session.clear_context())
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="clear is in progress"):
            await session.compact_context()
        release.set()
        await asyncio.wait_for(clear_task, timeout=5)
    finally:
        release.set()
        await session.aclose()
        await store.aclose()


@pytest.mark.parametrize(
    "exception,expect_latch",
    [
        (PersistenceError("summary generation does not match the current generation"), False),
        (StorageWriteError("session store SQL failed: disk I/O error"), True),
    ],
)
async def test_commit_failure_latching_follows_storage_fault_whitelist(
    tmp_path, exception, expect_latch
):
    """02-F12: commit_summary 的关系冲突不锁存 storage_failed (存储健康);
    真实落盘故障仍锁存。"""
    session, _main, _channel, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
    )

    async def failing_commit(record, _exc=exception):
        raise _exc

    object.__setattr__(store, "commit_summary", failing_commit)
    try:
        await _two_committed_requests(session)
        with pytest.raises(RuntimeError, match="failed to persist"):
            await session.compact_context()
        assert session._context_projection.blocks == ()
        latched = session.persistence.storage_failed is not None
        assert latched is expect_latch, str(exception)
    finally:
        await session.aclose()
        await store.aclose()
