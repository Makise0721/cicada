"""票据 04 UI 状态检查: /status 预算段与摘要段 (A12) 及只读出口的显示消费.

真实共享应用 (手工组装独立摘要通道, 与 test_compact_command 同构) + Harness:
占用/上限/余量/来源等级/锚状态/校准偏差、摘要计量单列与压缩事实、有界裁剪与
strip_terminal_sequences 口径。压缩后保存→恢复的组合检查 (摘要记录一次重建)
在 test_conversation_persistence; 机器读取口径直接断言 session.budget 与
session.summary_records。
"""

from __future__ import annotations

import pytest

from test_compact_command import _ScriptedSummary, _manual_session
from test_conversation_view import Harness, _wait_until

from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.persistence.contracts import StorageWriteError
from cicada.ui.checks import STATUS_MAX_LINES
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.transcript import InformationEntry


def _status_render(harness: Harness, width: int = 90) -> str:
    assert harness.submit("/status")
    return strip_terminal_sequences("\n".join(harness.lines(width)))


def _status_sections(harness: Harness) -> list[tuple[str, list[str]]]:
    entries = [e for e in harness.entries() if isinstance(e, InformationEntry)]
    return entries[-1].sections


def _status_rows(harness: Harness) -> list[str]:
    """最新一次 /status 的事实行 (显示文档保留历史条目, 整页渲染会含旧状态页)."""
    return [row for _, rows in _status_sections(harness) for row in rows]


async def _wait_compact_settled(harness: Harness) -> bool:
    return await _wait_until(lambda: not harness.view._gate.compacting)


class _TimedSummary:
    """带时长的摘要通道: 验证时长计量单列。"""

    def __init__(self) -> None:
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        yield TextDelta("较早请求的摘要正文")
        yield StreamDone("stop", metrics=ModelMetrics(
            input_tokens=9, output_tokens=2, provider_duration_s=1.5,
        ))


async def test_status_budget_disabled_single_line_without_summary_section(tmp_path):
    session, _m, _c, _s = await _manual_session(
        tmp_path, script=[[StreamDone("stop")]], budget=None, store=False
    )
    harness = Harness(session)
    try:
        rendered = _status_render(harness)
        assert "未启用" in rendered
        assert "占用估计" not in rendered
        assert "摘要计量" not in rendered and "压缩:" not in rendered
        headings = [heading for heading, _ in _status_sections(harness)]
        assert "预算" in headings and "摘要" not in headings
        # 票据 03: 底栏经 session.budget 显示未启用事实, 不补 0 (A08)
        footer = strip_terminal_sequences(harness.view.footer.render(80, 24)[1])
        assert footer == "预算未启用"
    finally:
        await session.aclose()


async def test_status_budget_before_first_request_and_empty_compaction_facts(tmp_path):
    session, _m, _c, store = await _manual_session(tmp_path, script=[[StreamDone("stop")]])
    harness = Harness(session)
    try:
        rendered = _status_render(harness)
        assert "预算: 上限 1000000 · 占用估计不可得（当前上下文尚无预算报告）" in rendered
        assert "压缩: 0 次 · 摘要边界: 无（未压缩）" in rendered
        assert "摘要计量: 不可得（本代次尚无已提交摘要）" in rendered
        # 占用 (当前窗口) 与 计量 (跨调用累计) 分行分区, 文字不混用
        assert "占用估计" in rendered and "计量: " in rendered
        assert session.budget is not None, "预算快照机器出口可用"
        assert session.summary_records == ()
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()


async def test_status_budget_anchor_and_deviation_after_requests(tmp_path):
    script = [
        [TextDelta("答一"), StreamDone("stop", metrics=ModelMetrics(input_tokens=500, output_tokens=10))],
        [TextDelta("答二"), StreamDone("stop", metrics=ModelMetrics(input_tokens=640, output_tokens=10))],
    ]
    session, _m, _c, store = await _manual_session(tmp_path, script=script)
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        rendered = _status_render(harness)
        assert "占用估计 2321 · 余量 " in rendered, "上限-占用-余量成行"
        snapshot = session.budget
        assert snapshot is not None and snapshot.report is not None
        report = snapshot.report
        assert report.source == "estimated"
        assert f"来源: {report.source} · 估计器 {report.estimator} v{report.estimator_version}" in rendered
        assert "anchor_measured_prefix" not in rendered
        assert "消息(user)x1~11" in rendered, "消息组件按类别合并"
        assert "output_reserve~2048" in rendered and "safety_margin~256" in rendered
        assert "锚: 已立锚 · 实测输入 500 · 锚定请求含 1 条消息" in rendered
        # 首请求报告先于锚确立: 如实显示启发式, 不虚构"前缀失配"
        assert "最近估计未使用锚" in rendered
        assert "校准偏差: 估计 17 − 实测 500 = -483（正=启发式偏高）" in rendered
        assert "累计输入 500" in rendered, "计量段保持跨调用累计, 与占用分开"

        assert harness.submit("第二问")
        await harness.wait_request_done()
        rendered = _status_render(harness)
        assert "来源: anchored" in rendered, "前缀延长按锚估计"
        assert "anchor_measured_prefix~500" in rendered
        assert "最近估计沿用锚" in rendered
        snapshot = session.budget
        assert snapshot is not None and snapshot.report is not None
        assert snapshot.report.source == "anchored"
        report = snapshot.report
        assert (
            f"来源: {report.source} · 估计器 {report.estimator} v{report.estimator_version}"
            in " ".join(_status_rows(harness))
        )
        assert snapshot.anchor is not None
        assert snapshot.anchor.measured_input_tokens == 640
        assert snapshot.anchor.estimated_input_tokens is not None
        assert snapshot.anchor.estimated_input_tokens - snapshot.anchor.measured_input_tokens < 0
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()


async def test_status_summary_metrics_single_listed_after_manual_compact(tmp_path):
    session, _m, _c, store = await _manual_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_TimedSummary(),
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("第二问")
        await harness.wait_request_done()
        assert harness.submit("/compact")
        assert await _wait_compact_settled(harness)
        rendered = _status_render(harness)
        assert "压缩: 1 次 · 最近原因 manual" in rendered
        assert "已摘要至消息 " in rendered and "保留起点，其后原文可见" in rendered
        assert "摘要计量: 1 次 · 累计输入 9 · 累计输出 2 · 累计时长 1.5s" in rendered
        first = _status_render(harness)
        assert "摘要计量: 1 次 · 累计输入 9" in first, "不随显示重复累计"
        # 机器出口: 摘要记录与预算快照可直接读取 (05 校准取数)
        records = session.summary_records
        assert len(records) == 1 and records[0].reason == "manual"
        assert records[0].metrics is not None
        assert records[0].metrics.provider_duration_s == 1.5
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()


@pytest.mark.parametrize("persistent", [False, True], ids=["sync", "persistent"])
async def test_status_budget_invalidated_after_clear_until_next_request(tmp_path, persistent):
    """成功清空丢弃报告和锚; 显示文档保留旧页, 最新状态与下一问使用当前事实。"""
    script = [
        [TextDelta("答一"), StreamDone("stop", metrics=ModelMetrics(input_tokens=500, output_tokens=10))],
        [TextDelta("答二"), StreamDone("stop", metrics=ModelMetrics(input_tokens=640, output_tokens=10))],
    ]
    session, model, _c, store = await _manual_session(tmp_path, script=script, store=persistent)
    harness = Harness(session)
    try:
        # 票据 03 (A09): 底栏读 session.budget——首问前待评估, 请求后最近预算,
        # 清空后退回待评估, 下一问恢复; 全程不新增刷新任务
        def footer_segment() -> str:
            return strip_terminal_sequences(harness.view.footer.render(100, 24)[1])

        assert "待评估 · 上限 1.0M" in footer_segment(), "首问前待评估并显示已知上限"
        assert harness.submit("第一问")
        await harness.wait_request_done()
        first = footer_segment()
        assert first.startswith("最近预算 ~") and "输入估计" in first, "请求后显示最近预算"
        assert "累计" in first, "累计与预算同段共存, 口径分开"
        assert harness.submit("/status")
        snapshot = session.budget
        assert snapshot is not None and snapshot.report is not None and snapshot.anchor is not None
        old_sections = _status_sections(harness)
        old_budget_rows = next(rows for heading, rows in old_sections if heading == "预算")
        assert any("占用估计 2321" in row for row in old_budget_rows)
        assert harness.submit("/clear")
        assert await _wait_until(lambda: not harness.view._gate.clearing), "清空落定"
        cleared = session.budget
        assert cleared is not None and cleared.report is None and cleared.anchor is None
        assert "待评估" in footer_segment() and "~" not in footer_segment(), "清空后退回待评估"
        assert session.history == ()
        assert harness.submit("/status")
        sections = _status_sections(harness)
        budget_rows = next(rows for heading, rows in sections if heading == "预算")
        assert budget_rows == ["预算: 上限 1000000 · 占用估计不可得（当前上下文尚无预算报告）"]
        assert old_sections in [
            e.sections for e in harness.entries() if isinstance(e, InformationEntry)
        ], "旧状态页保留为显示记录"
        if persistent:
            assert "压缩: 0 次" in " ".join(_status_rows(harness)), "当前代次摘要段一致"
        assert harness.submit("第二问")
        await harness.wait_request_done()
        assert "输入估计" in footer_segment(), "下一问后恢复最近预算"
        assert harness.submit("/status")
        current = session.budget
        assert current is not None and current.report is not None and current.anchor is not None
        assert current.report is not snapshot.report, "重新生成报告"
        assert current.report.source == "estimated", "清空前的锚不参与下一问"
        assert current.anchor.measured_input_tokens == 640 and current.anchor.message_count == 1
        assert len(model.requests) == 2 and len(model.requests[-1].messages) == 1
        assert model.requests[-1].messages[0].text == "第二问", "下一问不发送旧历史"
        rows = " ".join(_status_rows(harness))
        assert f"占用估计 {current.report.total_estimate} · 余量 {current.report.margin_remaining}" in rows
        assert "来源: estimated" in rows and "最近估计未使用锚" in rows
        assert "不可得（当前上下文尚无预算报告）" not in rows
    finally:
        await session.aclose()
        harness.close()
        if store is not None:
            await store.aclose()


async def test_status_budget_preserved_after_failed_persistent_clear(tmp_path, monkeypatch):
    """提交失败不清空历史、报告或锚, 最新状态继续显示既有事实与存储故障。"""
    script = [
        [TextDelta("答一"), StreamDone("stop", metrics=ModelMetrics(input_tokens=500, output_tokens=10))],
        [TextDelta("答二"), StreamDone("stop", metrics=ModelMetrics(input_tokens=640, output_tokens=10))],
    ]
    session, _m, _c, store = await _manual_session(tmp_path, script=script)
    harness = Harness(session)
    try:
        for prompt in ("第一问", "第二问"):
            assert harness.submit(prompt)
            await harness.wait_request_done()
        snapshot = session.budget
        assert snapshot is not None and snapshot.report is not None and snapshot.anchor is not None
        assert snapshot.report.source == "anchored"
        history = session.history
        assert harness.submit("/status")
        old_budget_rows = next(rows for heading, rows in _status_sections(harness) if heading == "预算")

        async def failing_clear(epoch_id):
            raise StorageWriteError("disk gone")

        assert store is not None
        monkeypatch.setattr(store, "commit_clear", failing_clear)
        assert harness.submit("/clear")
        assert await _wait_until(lambda: not harness.view._gate.clearing), "失败后屏障释放"
        assert session.history == history and session.budget == snapshot
        assert session.persistence is not None and session.persistence.storage_failed
        assert "清空提交失败" in strip_terminal_sequences("\n".join(harness.lines(90)))
        assert harness.submit("/status")
        budget_rows = next(rows for heading, rows in _status_sections(harness) if heading == "预算")
        assert budget_rows == old_budget_rows, "失败不伪造失效或清空成功"
        assert "保存失败已锁存: disk gone" in " ".join(_status_rows(harness))
    finally:
        await session.aclose()
        harness.close()
        if store is not None:
            await store.aclose()


async def test_status_summary_duration_unknown_without_unit(tmp_path):
    """审查 04-F2: 摘要计量字段全缺时显示'不可得', 单位不拼在'不可得'之后."""
    session, _m, _c, store = await _manual_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_ScriptedSummary(),  # metrics 无 provider_duration_s
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("第二问")
        await harness.wait_request_done()
        assert harness.submit("/compact")
        assert await _wait_compact_settled(harness)
        rendered = _status_render(harness)
        assert "累计输入 9 · 累计输出 2 · 累计时长 不可得" in rendered
        assert "不可得s" not in rendered
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()


async def test_status_budget_sections_stay_bounded_and_width_safe(tmp_path):
    script = [
        [TextDelta("答"), StreamDone("stop", metrics=ModelMetrics(input_tokens=500, output_tokens=10))],
    ]
    session, _m, _c, store = await _manual_session(tmp_path, script=script)
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("/status")
        sections = _status_sections(harness)
        rows = sum(len(rows) for _, rows in sections)
        assert rows <= STATUS_MAX_LINES + 1, "状态页整体有界 (含截断注记)"
        headings = [heading for heading, _ in sections]
        assert headings[0] == "会话" and "预算" in headings and "摘要" in headings
        width = 90
        for line in harness.lines(width):
            assert visible_width(strip_terminal_sequences(line)) <= width
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()
