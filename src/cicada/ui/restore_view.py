"""恢复历史的只读显示投影 (票据 05; 契约 §7; 04 补压缩边界条目).

从 ApplicationSession 的只读恢复事实 (restored_requests/restored_interrupted)
直接重建显示文档条目: 不调用 execute、不重新发布 ToolStarted/ToolCompleted、
不改应用规范历史。每条请求一个来源头 (epoch/request/seq), 助手条目与工具卡
的键命名空间化为 restored:<request_id>, 与本进程新 run_id (run-1)、跨组同
call_id 不碰撞 (A13); 历史计量经 transcript.add_metrics 一次重建, None 保持
不可得; 历史失败只成显示条目, 不冒充本进程最后请求 (退出码仍由本进程
RequestCompleted 决定)。

04 (A12): 压缩边界插入程序条目——当前代次摘要记录覆盖的请求序列末尾处,
声明覆盖范围与"模型所见为该摘要而非本段原文"语义, 附代次/时间/原因事实;
原始消息全部仍成条, 不清空不隐藏。摘要调用计量不进累计计量 (与显示重建
分开, 由 /status 摘要段从摘要记录派生)。
"""

from __future__ import annotations

import time

from cicada.core.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from cicada.persistence.contracts import InterruptedRequest, RestoredRequest, SummaryRecord
from cicada.ui.tool_view import ToolCardEntry, ToolDocument
from cicada.ui.transcript import Transcript

RESTORED_RUN_PREFIX = "restored"
CLEAR_BOUNDARY_TEXT = "（此处历史上已清空上下文）"
COMPACT_BOUNDARY_HEAD = "── 压缩边界"
INTERRUPTED_PROMPT_MAX_CHARS = 80
_HISTORICAL_CHECK_NOTE = "历史检查回执 {receipt}（运行 {run}）· 不适用于当前验证运行"


def _compaction_boundary_text(record: SummaryRecord) -> str:
    """压缩边界程序条目 (04/A12): 覆盖范围 + 模型所见语义 + 时间/代次/原因事实."""
    committed = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created_at))
    return (
        f"{COMPACT_BOUNDARY_HEAD} · 覆盖 seq {record.first_request_seq}.."
        f"{record.last_request_seq}（{record.message_count} 条消息）──\n"
        "此处之前的较早内容已摘要：恢复后模型所见为该摘要而非本段原文；"
        "原始记录保留可回看。\n"
        f"代次 {record.context_generation} · 摘要提交于 {committed} · 原因 {record.reason}"
    )


def project_restored_history(
    transcript: Transcript,
    tools: ToolDocument,
    *,
    requests: tuple[RestoredRequest, ...],
    interrupted: tuple[InterruptedRequest, ...],
    summaries: tuple[SummaryRecord, ...] = (),
) -> None:
    """把恢复快照投影进显示文档; 无恢复事实时不产生任何条目.

    summaries 是当前代次的已提交摘要记录 (调用方保证与投影口径一致, 见
    ApplicationSession.summary_records); 边界条目插在其覆盖的末请求之后。
    """
    if not requests and not interrupted:
        return
    header = f"已恢复历史：{len(requests)} 条请求"
    if interrupted:
        header += f"，{len(interrupted)} 条未完成"
    transcript.append_system(header)
    boundaries: dict[tuple[int, int], list[SummaryRecord]] = {}
    for record in summaries:
        boundaries.setdefault(
            (record.context_generation, record.last_request_seq), []
        ).append(record)
    last_generation: int | None = None
    for request in requests:
        if last_generation is not None and request.context_generation != last_generation:
            transcript.append_system(CLEAR_BOUNDARY_TEXT)
        last_generation = request.context_generation
        _project_request(transcript, tools, request)
        for record in boundaries.get((request.context_generation, request.seq), ()):
            transcript.append_system(_compaction_boundary_text(record))
    for item in interrupted:
        prompt = item.prompt if len(item.prompt) <= INTERRUPTED_PROMPT_MAX_CHARS else (
            item.prompt[: INTERRUPTED_PROMPT_MAX_CHARS - 1] + "…"
        )
        transcript.append_system(f"中断请求（未完成）：{prompt}")


def _project_request(
    transcript: Transcript, tools: ToolDocument, request: RestoredRequest
) -> None:
    key = f"{RESTORED_RUN_PREFIX}:{request.request_id}"
    transcript.append_system(f"── 历史请求 #{request.seq} · epoch {request.epoch_id[:8]} ──")
    turn = 0
    assistant_count = 0
    # 结果优先: 结果消息到了才建结果卡; 助手声明而无结果的调用 (中断轮) 在
    # 组尾补无输出占位卡, 避免同键两卡互相覆盖
    pending: dict[str, tuple[int, str, str]] = {}  # call_id -> (turn, name, args_json)
    for message in request.messages:
        if isinstance(message, UserMessage):
            transcript.append_user(message.text)
        elif isinstance(message, AssistantMessage):
            transcript.restore_assistant(key, turn, message)
            for call in message.tool_calls:
                pending[call.id] = (turn, call.name, call.arguments_json)
            assistant_count += 1
            turn = assistant_count
        elif isinstance(message, ToolResultMessage):
            declared = pending.pop(message.result.call_id, None)
            _add_tool_card(
                transcript, tools, key=key,
                turn=max(0, assistant_count - 1), call_id=message.result.call_id,
                name=message.result.name,
                arguments_json=declared[2] if declared else None,
                result=message.result,
            )
    for call_id, (call_turn, name, arguments_json) in pending.items():
        _add_tool_card(
            transcript, tools, key=key, turn=call_turn, call_id=call_id, name=name,
            arguments_json=arguments_json, result=None,
        )
    outcome = request.outcome
    if outcome.stop_reason == "aborted":
        transcript.append_cancelled()
    elif outcome.stop_reason == "error":
        transcript.append_error(outcome.error or "历史请求以错误结束，但未记录错误信息。")
    for model_call in outcome.model_calls:
        transcript.add_metrics(model_call.metrics)


def _add_tool_card(
    transcript: Transcript,
    tools: ToolDocument,
    *,
    key: str,
    turn: int,
    call_id: str,
    name: str,
    arguments_json: str | None,
    result: ToolResult | None,
) -> None:
    """挂一张工具卡; check 工具带旧检查归属行 (receipt/运行来源, 只显示)。"""
    if result is not None:
        content = result.content
        is_error = result.is_error
        details = result.details
        display_name = result.name
    else:
        # 模型发起了调用但存档没有对应结果 (中断轮): 以无输出终态卡如实显示
        content = ""
        is_error = True
        details = None
        display_name = name
    artifact = None
    if display_name == "check" and isinstance(details, dict):
        receipt_id = details.get("receipt_id")
        verification_run_id = details.get("verification_run_id")
        if isinstance(receipt_id, str) and receipt_id:
            artifact = _HISTORICAL_CHECK_NOTE.format(
                receipt=receipt_id,
                run=(verification_run_id if isinstance(verification_run_id, str) and verification_run_id else "不可得"),
            )
    index = tools.add_completed_card(
        run_id=key,
        turn_index=turn,
        call_id=call_id,
        name=display_name,
        arguments_json=arguments_json,
        content=content,
        is_error=is_error,
        artifact=artifact,
    )
    transcript.add_entry(ToolCardEntry(tools, index))


__all__ = [
    "CLEAR_BOUNDARY_TEXT",
    "COMPACT_BOUNDARY_HEAD",
    "RESTORED_RUN_PREFIX",
    "project_restored_history",
]
