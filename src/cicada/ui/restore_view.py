"""恢复历史的只读显示投影 (票据 05; 契约 §7).

从 ApplicationSession 的只读恢复事实 (restored_requests/restored_interrupted)
直接重建显示文档条目: 不调用 execute、不重新发布 ToolStarted/ToolCompleted、
不改应用规范历史。每条请求一个来源头 (epoch/request/seq), 助手条目与工具卡
的键命名空间化为 restored:<request_id>, 与本进程新 run_id (run-1)、跨组同
call_id 不碰撞 (A13); 历史计量经 transcript.add_metrics 一次重建, None 保持
不可得; 历史失败只成显示条目, 不冒充本进程最后请求 (退出码仍由本进程
RequestCompleted 决定)。
"""

from __future__ import annotations

from cicada.core.messages import AssistantMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from cicada.persistence.contracts import InterruptedRequest, RestoredRequest
from cicada.ui.tool_view import ToolCardEntry, ToolDocument
from cicada.ui.transcript import Transcript

RESTORED_RUN_PREFIX = "restored"
CLEAR_BOUNDARY_TEXT = "（此处历史上已清空上下文）"
INTERRUPTED_PROMPT_MAX_CHARS = 80
_HISTORICAL_CHECK_NOTE = "历史检查回执 {receipt}（运行 {run}）· 不适用于当前验证运行"


def project_restored_history(
    transcript: Transcript,
    tools: ToolDocument,
    *,
    requests: tuple[RestoredRequest, ...],
    interrupted: tuple[InterruptedRequest, ...],
) -> None:
    """把恢复快照投影进显示文档; 无恢复事实时不产生任何条目。"""
    if not requests and not interrupted:
        return
    header = f"已恢复历史：{len(requests)} 条请求"
    if interrupted:
        header += f"，{len(interrupted)} 条未完成"
    transcript.append_system(header)
    last_generation: int | None = None
    for request in requests:
        if last_generation is not None and request.context_generation != last_generation:
            transcript.append_system(CLEAR_BOUNDARY_TEXT)
        last_generation = request.context_generation
        _project_request(transcript, tools, request)
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


__all__ = ["CLEAR_BOUNDARY_TEXT", "RESTORED_RUN_PREFIX", "project_restored_history"]
