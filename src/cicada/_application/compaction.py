"""上下文压缩协议: 规划切点 → 摘要 → 校验 → 封口核对 → 接受提交 → 切换投影.

压缩是唯一允许嵌套在请求内部的事务 (自动压缩), 也是唯一有"封口先于接受"语义的
事务 (close 期间零写入)。这两条规则原来散在 `ApplicationSession` 的五个方法里,
与会话的准入/所有权代码交错; 这里收成一个模块: 会话只提供一次状态快照, 协议自己
决定何时派发、何时放弃、何时切换。

状态快照在调用方持有独占槽时建立, 压缩期间不会被会话改写; 唯一可能异步变化的是
close 封口, 因此它是回调而不是快照值。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from cicada.budget import budget_exceeded_text
from cicada.context_policy import (
    SUMMARY_PROMPT_SHA256,
    SUMMARY_PROMPT_VERSION,
    BudgetPolicy,
    ContextProjection,
    SummaryBlock,
    SummaryOutcome,
    build_summary_request,
    canonical_prefix_sha256,
    generate_summary,
    summary_block_message,
    summary_input_messages,
)
from cicada.core.cancel import CancelToken
from cicada.core.messages import Message
from cicada.core.ports import ModelRequest
from cicada.persistence.contracts import (
    ArchiveLimitError,
    PersistenceError,
    SessionStorePort,
    StorageUnavailableError,
    StorageWriteError,
    SummaryRecord,
)

from .contracts import ApplicationConfig, ApplicationUpdate, CompactionSealedError, ContextCompacted


@dataclass(frozen=True)
class CompactionPlan:
    """一次压缩的切点计划 (D4): 覆盖区间与保留边界."""

    request: ModelRequest             # 完整摘要输入: 原始历史区间及固定最后 user
    first_request_seq: int
    last_request_seq: int
    message_count: int                # 本块覆盖消息数 (end - start)
    start_index: int                  # 覆盖起点 (= 投影保留起点)
    end_index: int                    # 覆盖前缀终点 = 保留边界


@dataclass(frozen=True)
class CompactionState:
    """压缩看到的状态快照; 由会话在持有独占槽时建立.

    会话保证压缩期间 store/代次/配置/历史/区间不变 (持有独占槽, 无并发写入);
    `sealed` 是唯一的活回调——close 随时可能封口, 接受点必须现场核对。
    """

    budget: BudgetPolicy
    store: SessionStorePort
    config: ApplicationConfig
    epoch_id: str | None
    context_generation: int
    history: tuple[Message, ...]
    projection: ContextProjection
    spans: tuple[tuple[int, int, int], ...]
    summary_records: list[SummaryRecord]  # 提交成功后追加; 会话只读出口据此取数
    summary_calls: list[SummaryOutcome]   # 每次摘要调用登记 (诊断)
    sealed: Callable[[], bool]            # close 封口: 先封口则零写入
    closed: Callable[[], bool]            # 应用已关闭/正在关闭
    notify: Callable[[ApplicationUpdate], None]
    latch_storage_failed: Callable[[str], None]


def check_summary_admission(state: CompactionState, cancel_token: CancelToken | None) -> None:
    """摘要派发前核对: 封口/关闭/取消都不再发往模型."""
    if (state.sealed() or state.closed()
            or (cancel_token is not None and cancel_token.cancelled)):
        raise RuntimeError(
            "summary generation failed (aborted); the model projection is unchanged")


def plan_compaction(state: CompactionState, cancel_token: CancelToken | None = None) -> CompactionPlan:
    """D4: 当前代次、已提交完整请求、保留最近 1 个; 摘要输入自身须满足预算.

    覆盖起点 = 当前投影保留起点 (再次压缩只追加新区间, 不重写旧摘要);
    区间按请求边界收缩 (最长最旧前缀), 最旧单个请求也装不下时明确拒绝。
    摘要输入是原始规范历史 (D5: 不经 RunPolicy/BudgetPolicy 投影)。
    """
    config = state.budget.summary_budget
    assert config is not None
    spans = state.spans
    if len(spans) < 2:
        raise RuntimeError(
            "nothing to compact: at least two committed full requests are required "
            "(the most recent one is always retained)"
        )
    start = state.projection.retained_start
    eligible = [span for span in spans[:-1] if span[2] > start]
    if not eligible:
        raise RuntimeError(
            "nothing to compact: the committed requests before the retained "
            "boundary are already summarized"
        )
    for span in eligible:
        if span[1] < start:
            raise RuntimeError(
                "compaction bookkeeping is inconsistent with the projection"
            )
    chosen_span: tuple[int, int, int] | None = None
    for span in reversed(eligible):
        check_summary_admission(state, cancel_token)
        request = build_summary_request(summary_input_messages(
            tuple(state.history[start : span[2]])
        ))
        report = state.budget.assess_summary(request)
        check_summary_admission(state, cancel_token)
        if report.total_estimate <= config.num_ctx:
            chosen_span = span
            break
    if chosen_span is None:
        raise RuntimeError(
            "compaction unavailable: even the oldest single request does not fit "
            "the summary request budget"
        )
    return CompactionPlan(
        request=request,
        first_request_seq=eligible[0][0],
        last_request_seq=chosen_span[0],
        message_count=chosen_span[2] - start,
        start_index=start,
        end_index=chosen_span[2],
    )


async def run_compaction(
    state: CompactionState, plan: CompactionPlan, *, reason: str, cancel_token: CancelToken,
) -> ContextCompacted | None:
    """压缩核心协议 (D4-D6, PA03/PA05): 手动与自动共用同一实现 (03).

    摘要生成 → 响应校验 → 封口核对且未取消 → store 接受 → 明确成功 →
    切换投影 + ContextCompacted 事件。close 封口先于接受时返回 None
    (归关闭取消, 零写入, 旧投影不变); 摘要失败/提交失败抛 RuntimeError
    (投影不变; 锁存按 begin 同一白名单); store 接受后取消不取消事务,
    成功先切换再传播外层 CancelledError。
    """
    channel = state.budget.summarize_channel
    if state.sealed():
        return None
    check_summary_admission(state, cancel_token)
    report = state.budget.assess_summary(plan.request)
    check_summary_admission(state, cancel_token)
    if report.total_estimate > report.limit:
        raise RuntimeError("compaction unavailable: " + budget_exceeded_text(report))
    try:
        outcome = await generate_summary(channel, plan.request, cancel_token)
    except asyncio.CancelledError:
        state.summary_calls.append(SummaryOutcome(ok=False, failure="aborted"))
        raise
    state.summary_calls.append(outcome)
    if not outcome.ok:
        if outcome.failure == "aborted" and state.sealed():
            # PA05: close 封口协作取消 → 归关闭取消, 静默放弃 (旧投影不变)
            return None
        # D5/PA03: 摘要失败不派发提交, 投影不变, 无摘要记录落库
        raise RuntimeError(
            f"summary generation failed ({outcome.failure}); the model projection "
            "is unchanged" + (f": {outcome.detail}" if outcome.detail else "")
        )
    # D6: 摘要模型身份取启动层有效身份 (PersistenceConfig.identity 本就是
    # LaunchIdentity; 审查修复: 此前误取 .launch 使真实持久组合的压缩必失败)
    identity_launch = state.config.persistence.identity if state.config.persistence is not None else None
    summary_identity = state.budget.summary_identity
    record = SummaryRecord(
        summary_id=str(uuid.uuid4()),
        epoch_id=state.epoch_id or "",
        context_generation=state.context_generation,
        first_request_seq=plan.first_request_seq,
        last_request_seq=plan.last_request_seq,
        message_count=plan.message_count,
        end_message_index=plan.end_index,
        prefix_sha256=canonical_prefix_sha256(state.history[: plan.end_index]),
        text=outcome.text,
        provider=(
            summary_identity.provider if summary_identity is not None else (
            identity_launch.provider
            if identity_launch is not None
            else (state.config.model_label or "unknown")
            )
        ),
        model=(
            summary_identity.model if summary_identity is not None else (
            identity_launch.model
            if identity_launch is not None
            else (state.config.model_label or "unknown")
            )
        ),
        prompt_version=SUMMARY_PROMPT_VERSION,
        prompt_sha256=SUMMARY_PROMPT_SHA256,
        metrics=outcome.metrics,
        reason=reason,
        created_at=time.time(),
        call_identity=summary_identity,
        previous_projection_identity=canonical_prefix_sha256(
            state.projection.project(state.history)
        ),
    )
    commit_task = asyncio.ensure_future(_dispatch_commit(state, record))
    commit_cancelled = False
    while True:
        try:
            await asyncio.shield(commit_task)
            break
        except asyncio.CancelledError:
            # PA03: 派发后取消不取消事务; 重复取消也等明确结果
            commit_cancelled = True
        except CompactionSealedError:
            # PA05: 封口先于接受 → 零写入; 归关闭取消, 旧投影不变,
            # 不锁存 storage_failed, 迟到响应丢弃
            return None
        except PersistenceError as exc:
            # 提交失败: 旧投影不变。锁存面与 begin 路径同一白名单 (审查
            # 02-F12): 真实落盘故障才锁存 storage_failed; 关系冲突
            # (代次/seq/重复) 是状态问题, 存储健康, 误锁存会不可逆拒绝
            # 本进程后续请求
            if isinstance(exc, (StorageUnavailableError, StorageWriteError, ArchiveLimitError)):
                state.latch_storage_failed(str(exc))
            raise RuntimeError(
                "compaction failed to persist; the model projection is unchanged: "
                f"{exc}"
            ) from exc
    # 明确成功 → 切换投影 + 事件; 已被取消时先落定再传播 (PA03)
    # 04: store 已接受即已提交摘要事实, 进入只读出口 (摘要计量/压缩次数取数)
    state.summary_records.append(record)
    block = SummaryBlock(
        message=summary_block_message(
            first_request_seq=plan.first_request_seq,
            last_request_seq=plan.last_request_seq,
            message_count=plan.message_count,
            text=record.text,
        ),
        end_message_index=plan.end_index,
        first_request_seq=plan.first_request_seq,
        last_request_seq=plan.last_request_seq,
        message_count=plan.message_count,
        prefix_sha256=record.prefix_sha256,
    )
    state.projection.apply_summary(block, canonical_length=len(state.history))
    state.budget.invalidate()
    compacted = ContextCompacted(
        summary_id=record.summary_id,
        first_request_seq=record.first_request_seq,
        last_request_seq=record.last_request_seq,
        message_count=record.message_count,
        end_message_index=record.end_message_index,
        reason=reason,
    )
    state.notify(compacted)
    if commit_cancelled:
        raise asyncio.CancelledError()
    return compacted


async def _dispatch_commit(state: CompactionState, record: SummaryRecord) -> None:
    """PA05 接受落定点: 封口核对与 store 派发之间无 await, 不可交错.

    store._dispatch 的 closing 检查与入队同为同步前缀, 因此"封口先 → 零
    写入 / 接受先 → 等明确结果"在本协程首个执行块内一次判定;
    create_task 不越过核对 (任务未起跑时封口先生效即零提交)。
    """
    if state.sealed():
        raise CompactionSealedError(
            "compaction is sealed by close; the summary commit was not sent"
        )
    await state.store.commit_summary(record)
