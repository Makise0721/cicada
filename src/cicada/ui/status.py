"""只读状态快照的文字格式化与状态页裁剪，不读取资源或改变应用状态。"""

from __future__ import annotations

from cicada.budget import BudgetReport, BudgetSnapshot
from cicada.models import CacheRequestObservation
from cicada.persistence.contracts import SummaryRecord
from cicada.ui.checks import (
    STATUS_MAX_BYTES,
    STATUS_MAX_LINES,
    clip_lines,
    clip_status_field,
    truncation_note,
)
from cicada.ui.transcript import CacheMetricsTotal

# 可选分词观察行在缓存与发送诊断之后省略，保留必需状态的限额优先级。
TOKEN_OBSERVATION_PREFIXES = ("本地计数:", "同请求偏差:")


def clip_status_sections(sections: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    # 保留原有字段顺序及整体限额; 装饰标题/空行不挤占事实行预算。
    clipped_sections = [
        (heading, [clip_status_field(row) for row in rows]) for heading, rows in sections
    ]
    # 新缓存/发送诊断/分词观察行不挤掉原有检查/阻断事实。超限先省略缓存与发送诊断行,
    # 仍超限再省略分词观察行, 最后沿用原裁剪。
    cache_omitted = False
    observation_omitted = False
    token_omitted = False

    def over_limit(sections: list[tuple[str, list[str]]]) -> bool:
        fields = [row for _, rows in sections for row in rows]
        return (
            len(fields) > STATUS_MAX_LINES
            or len("\n".join(fields).encode("utf-8")) > STATUS_MAX_BYTES
        )

    def without(sections: list[tuple[str, list[str]]], predicate) -> list[tuple[str, list[str]]]:
        return [
            (heading, [row for row in rows if not predicate(heading, row)])
            for heading, rows in sections
        ]

    def present(sections: list[tuple[str, list[str]]], predicate) -> bool:
        return any(predicate(heading, row) for heading, rows in sections for row in rows)

    if over_limit(clipped_sections):
        cache_omitted = present(clipped_sections, _is_cache_row)
        observation_omitted = present(clipped_sections, _is_observation_row)
        clipped_sections = without(
            clipped_sections,
            lambda heading, row: _is_cache_row(heading, row)
            or _is_observation_row(heading, row),
        )
        if over_limit(clipped_sections):
            token_omitted = present(clipped_sections, _is_token_row)
            clipped_sections = without(clipped_sections, _is_token_row)
    fields = [row for _, rows in clipped_sections for row in rows]
    total = "\n".join(fields)
    omitted = "/".join(
        name for flag, name in (
            (cache_omitted, "缓存计量"), (observation_omitted, "发送诊断"),
            (token_omitted, "分词观察"),
        ) if flag
    )
    if len(fields) <= STATUS_MAX_LINES and len(total.encode("utf-8")) <= STATUS_MAX_BYTES:
        if omitted:
            clipped_sections.append(("", [f"…[{omitted}因状态页限额省略]"]))
        return clipped_sections
    kept, original_lines, original_bytes, truncated = clip_lines(
        total, STATUS_MAX_LINES, STATUS_MAX_BYTES,
    )
    result: list[tuple[str, list[str]]] = []
    offset = 0
    for heading, rows in clipped_sections:
        # clip_lines 展开字段内换行; 加入字段间换行以保留尾部空行归属。
        count = sum(len((row + "\n").splitlines()) for row in rows)
        group = list(kept[offset:offset + count])
        if group:
            result.append((heading, group))
        offset += count
    notes = truncation_note(kept, original_lines, original_bytes, truncated)
    if notes:
        suffix = f"（{omitted}因状态页限额省略）" if omitted else ""
        result.append(("", [note + suffix for note in notes]))
    return result



def observation_line(observation: CacheRequestObservation | None) -> str:
    """缓存 03 (契约 C7): 最近实际发送相对前次的本地形状变化行。

    无能力/未观察显示不可得; 只显示发送次数、变化与保留前导消息数,
    不显示正文、hash、凭据或回执, 不作为服务端 token 前缀或命中证明。
    """
    if observation is None:
        return "发送诊断: 不可得"
    labels = {
        "first_request": "首次可观察",
        "system_changed": "system变化",
        "tools_changed": "工具变化",
        "messages_extended": "历史延长",
        "messages_replaced": "历史替换",
        "request_context_changed": "尾部变化",
        "unchanged": "与上次相同",
    }
    parts = [f"第{observation.sequence}次发送"]
    parts.extend(labels.get(code, code) for code in observation.changes)
    if observation.preserved_messages is not None:
        parts.append(f"保留前导消息 {observation.preserved_messages} 条")
    return f"发送诊断: {' · '.join(parts)}（本地对比，非服务端命中）"


def token_observation_lines(snapshot: BudgetSnapshot | None) -> list[str]:
    """03 (契约 C7): 最近正式普通调用的本地计数与同请求偏差观察 (至多两条)。

    只读 session.budget 已发布快照, 不加载资源、不重复计数: 无预算/无报告/未
    配置本地计数时不出行; 本地不可用时只报安全原因; profile 为固定公开参考,
    线上一致性未验证, 短显示身份可回溯完整 manifest。
    """
    if snapshot is None or snapshot.report is None:
        return []
    local = snapshot.report.local_count
    if local is None:
        return []
    if local.tokens is None:
        return [f"本地计数: 不可用（{local.reason}）"]
    lines = [
        f"本地计数: {local.tokens} tokens · {_short_profile(local.profile)}"
        "（固定公开参考，线上一致性未验证）"
    ]
    comparison = snapshot.comparison
    if comparison is None:
        return lines
    actual = comparison.input_tokens
    if actual is None:
        lines.append("同请求偏差: 实际输入不可得（本地计数偏高与否未知）")
    else:
        deviation = comparison.local_count.tokens - actual
        lines.append(
            f"同请求偏差: 实际输入 {actual} · 本地 − 实际 = {deviation:+d}（正=本地偏高）"
        )
    return lines


def _short_profile(profile: str) -> str:
    """短显示身份 (03): profile 名 + manifest hash 前 8 位, 可回溯完整 manifest。"""
    name, separator, digest = profile.partition(":manifest-sha256:")
    return f"{name} · manifest {digest[:8]}…" if separator else profile


def _is_cache_row(heading: str, row: str) -> bool:
    """可先省略的缓存观察行: 计量段缓存行与摘要段已提交摘要缓存行。"""
    return (heading == "计量" and row.startswith("缓存（")) or (
        heading == "摘要" and row.startswith("已提交摘要缓存:")
    )


def _is_observation_row(heading: str, row: str) -> bool:
    return heading == "计量" and row.startswith("发送诊断:")


def _is_token_row(heading: str, row: str) -> bool:
    return heading == "计量" and row.startswith(TOKEN_OBSERVATION_PREFIXES)

# --- 预算与摘要段 (04, A12) ------------------------------------------------------


def budget_sections(
    snapshot: BudgetSnapshot | None,
    records: tuple[SummaryRecord, ...],
    *,
    persistent: bool,
) -> list[tuple[str, list[str]]]:
    """/status 预算段 + 摘要段: 只读出口 session.budget / session.summary_records.

    占用 (当前窗口估计) 与 计量 (跨调用累计) 分行分区, 文字不混用; 来源
    等级与估计偏差如实显示, 缺实测/缺估计时如实不可得。预算未启用时单行
    说明, 摘要段省略 (无压缩能力)。校准事实的机器出口是 session.budget
    (BudgetSnapshot.anchor) 与 session.summary_records, 05 直接读取。
    """
    if snapshot is None:
        return [("预算", ["未启用（真实模型按 --num-ctx 自动启用；fake 模式无预算）"])]
    report = snapshot.report
    lines: list[str] = []
    if report is None:
        # 首次发送、成功清空或配置切换后, 均以应用公开快照的失效事实为准。
        lines.append(f"预算: 上限 {snapshot.limit} · 占用估计不可得（当前上下文尚无预算报告）")
    else:
        lines.append(
            f"预算: 上限 {report.limit} · 占用估计 {report.total_estimate}"
            f" · 余量 {report.margin_remaining}"
        )
        lines.append(f"来源: {report.source} · 估计器 {report.estimator} v{report.estimator_version}")
        lines.append(f"占用分解: {_budget_breakdown(report)}")
        lines.extend(_budget_anchor_lines(snapshot, report))
    return [("预算", lines), ("摘要", _summary_section_lines(records, persistent=persistent))]


def _budget_breakdown(report: BudgetReport) -> str:
    """占用逐项分解: 非消息组件逐项, 消息组件按类别合并 (条数+合计)。

    各项之和恒等于总估计; 显示聚合只省行数不省事实 (状态页有界裁剪口径)。
    """
    parts: list[str] = []
    message_sums: dict[str, list[int]] = {}  # kind -> [条数, tokens 合计]
    for component in report.components:
        if component.label.startswith("message:"):
            kind = component.label.rsplit(":", 1)[-1]
            count, total = message_sums.get(kind, [0, 0])
            message_sums[kind] = [count + 1, total + (component.tokens or 0)]
        else:
            tokens = "不可得" if component.tokens is None else str(component.tokens)
            parts.append(f"{component.label}~{tokens}")
    for kind, (count, total) in message_sums.items():
        parts.append(f"消息({kind})x{count}~{total}")
    return " · ".join(parts)


def _budget_anchor_lines(snapshot: BudgetSnapshot, report: BudgetReport) -> list[str]:
    """锚状态与校准偏差 (04/A12): 如实文字, 不虚构失效原因。"""
    anchor = snapshot.anchor
    if anchor is None:
        return ["锚: 未立锚（尚无带实测计量的成功请求；/clear 或恢复后重置）"]
    lines = [
        f"锚: 已立锚 · 实测输入 {anchor.measured_input_tokens}"
        f" · 锚定请求含 {anchor.message_count} 条消息",
        (
            "最近估计沿用锚"
            if report.source in ("measured", "anchored")
            else "最近估计未使用锚（估计先于本锚确立，或请求前缀已变化）"
        ),
    ]
    if anchor.estimated_input_tokens is None:
        lines.append("校准偏差: 不可得（锚缺少立锚时的估计值）")
    else:
        deviation = anchor.estimated_input_tokens - anchor.measured_input_tokens
        lines.append(
            f"校准偏差: 估计 {anchor.estimated_input_tokens} − "
            f"实测 {anchor.measured_input_tokens} = {deviation:+d}（正=估计偏高）"
        )
    return lines


def _summary_section_lines(records: tuple[SummaryRecord, ...], *, persistent: bool) -> list[str]:
    """摘要段: 压缩次数/最近原因/当前代次边界 + 摘要调用计量单列。

    计量从摘要记录派生 (恢复后随记录一次可得), 不进累计计量、不随显示
    重复累计; None 字段保持不可得。
    """
    if not persistent:
        return ["压缩: 不可用（需持久会话）"]
    lines: list[str] = []
    if records:
        last = records[-1]
        lines.append(
            f"压缩: {len(records)} 次 · 最近原因 {last.reason}"
            f" · 已摘要至消息 {last.end_message_index}（保留起点，其后原文可见）"
        )
    else:
        lines.append("压缩: 0 次 · 摘要边界: 无（未压缩）")
    if not records:
        lines.append("摘要计量: 不可得（本代次尚无已提交摘要）")
        lines.append("已提交摘要缓存: 尚无调用")
        return lines
    inputs = [r.metrics.input_tokens for r in records if r.metrics is not None and r.metrics.input_tokens is not None]
    outputs = [r.metrics.output_tokens for r in records if r.metrics is not None and r.metrics.output_tokens is not None]
    durations = [r.metrics.provider_duration_s for r in records if r.metrics is not None and r.metrics.provider_duration_s is not None]

    def fmt(values: list, partial: bool, unit: str = "") -> str:
        if not values:
            return "不可得"
        text = f"{sum(values):.1f}{unit}" if unit else str(sum(values))
        return f"{text}（部分调用不可得）" if partial else text

    partial_in = any(r.metrics is None or r.metrics.input_tokens is None for r in records)
    partial_out = any(r.metrics is None or r.metrics.output_tokens is None for r in records)
    partial_dur = any(r.metrics is None or r.metrics.provider_duration_s is None for r in records)
    lines.append(
        f"摘要计量: {len(records)} 次 · 累计输入 {fmt(inputs, partial_in)}"
        f" · 累计输出 {fmt(outputs, partial_out)} · 累计时长 {fmt(durations, partial_dur, 's')}"
    )
    cache = CacheMetricsTotal()
    for record in records:
        cache.add(record.metrics)
    lines.append(cache.summary("已提交摘要缓存"))
    return lines
