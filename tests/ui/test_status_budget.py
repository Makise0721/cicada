"""票据 04 UI 状态检查: /status 预算段与摘要段 (A12) 及只读出口的显示消费.

真实共享应用 (手工组装独立摘要通道, 与 test_compact_command 同构) + Harness:
占用/上限/余量/来源等级/锚状态/校准偏差、摘要计量单列与压缩事实、有界裁剪与
strip_terminal_sequences 口径。压缩后保存→恢复的组合检查 (摘要记录一次重建)
在 test_conversation_persistence; 机器读取口径直接断言 session.budget 与
session.summary_records。
"""

from __future__ import annotations

from test_compact_command import _manual_session
from test_conversation_view import Harness, _wait_until

from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.ui.checks import STATUS_MAX_LINES
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.transcript import InformationEntry


def _status_render(harness: Harness, width: int = 90) -> str:
    assert harness.submit("/status")
    return strip_terminal_sequences("\n".join(harness.lines(width)))


def _status_sections(harness: Harness) -> list[tuple[str, list[str]]]:
    entries = [e for e in harness.entries() if isinstance(e, InformationEntry)]
    return entries[-1].sections


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
    finally:
        await session.aclose()


async def test_status_budget_before_first_request_and_empty_compaction_facts(tmp_path):
    session, _m, _c, store = await _manual_session(tmp_path, script=[[StreamDone("stop")]])
    harness = Harness(session)
    try:
        rendered = _status_render(harness)
        assert "预算: 上限 1000000 · 占用估计不可得（本进程尚未发送请求）" in rendered
        assert "压缩: 0 次 · 摘要边界: 无（未压缩）" in rendered
        assert "摘要计量: 不可得（本代次尚无摘要调用）" in rendered
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
        assert "来源: estimated · 估计器 utf8-bytes-over-3-plus-template-overhead v1" in rendered
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
