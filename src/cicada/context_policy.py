"""应用层 BudgetPolicy: 每轮发送前的权威预算准入与同身份实测锚 (01) +
模型可见投影容器与摘要协议 (02).

与 RunPolicy 同构的 ModelPort 装饰器 (spec D1): 安装次序
Agent → RunPolicy → BudgetPolicy → 适配器; 内核零改动, 规范历史不因本策略
改变——压缩投影是本策略对已通过核对消息的有记录替换 (D4), 验证扫描仍在上游
全量进行。

- 每轮 stream 先按投影容器把请求消息替换为 [摘要块] + 保留后缀 (无块时原样
  透传), 再按 BudgetConfig 估计完整输入 (system prompt 已含 RunPolicy 控制
  段、消息已含 freshness 标记: 计量覆盖验证段最终形态)。总需求 > num_ctx 时
  产出 StreamDone("error") 且适配器零发送, 错误文本带估计值、上限与主要占用。
- 实测锚 (spec D3): 指纹基准是本策略实际所见 (投影后) 的完整请求。前次实测
  input_tokens 非 None/非 0 且流级终结非 error/aborted 时立锚; 后继请求为该锚
  的严格前缀延长时按 anchored 估计 = 实测 + 后缀增量; 与锚完全相同则整体
  measured。clear (committed_count == 0)、恢复 (新策略实例)、压缩 (投影版本
  变化) 与配置/工具/提示/验证视图变化经前缀失配自然失效; 后者是预期的保守
  回退, 不是缺陷。
- 请求边界接缝 (spec D1/PA02): note_request_start(committed_count) 由应用在
  每次 execute 准入通过后、首次预算预估之前调用一次, 参数为该时刻规范历史
  消息数; 它建立本次 execute 尝试的预算状态范围, begin 不再调用或清零。
  committed_count == 0 视为代次重置 (clear) 并丢弃锚。03 起同一接缝重置本次
  execute 的两项独立额度: 准入自动压缩至多一次 (begin 前预估失败与 begin 后
  每轮准入失败共享), overflow 重试至多一次; 失败不返还。
- 自动压缩与 overflow 重试 (spec D7/PA06, 03): 每轮准入失败且尚未尝试时, 经
  auto_compact 回调先消费额度再压缩, 对有效投影重估; 仍超限按原两道口拒绝。
  adapter 的 StreamDone("error") 经 classify_overflow 确认上下文超限、该发送
  零输出 (无任何 TextDelta 含空片段/ToolCallEvent) 且重试额度未用时, 由策略
  内部暂存该终结错误, 压缩后重发一次; 只向上游发布一次终结。任何可见输出后
  的上下文错误不重试 (透明重发禁令); 取消不重试且收尾为 aborted 不按 error。
- 摘要通道 (02) 与 overflow 分类器 (03) 经构造参数注入, 组装接线在
  application 层; 本模块不导入 plugins (导入边界测试钉住)。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal

from cicada.budget import (
    AnchorInfo,
    BudgetConfig,
    BudgetReport,
    BudgetSnapshot,
    ComponentEstimate,
    assemble_report,
    budget_exceeded_text,
    estimate_input_components,
    estimate_message_tokens,
    message_kind,
)
from cicada.core.cancel import CancelToken
from cicada.core.messages import Message, UserMessage
from cicada.core.ports import (
    ModelMetrics,
    ModelPort,
    ModelRequest,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
)
from cicada.persistence.codec import CodecError, encode_messages

SUMMARY_PROMPT_VERSION = "2"  # v2: 摘要请求末尾追加固定 user 指令 (05 收口修正)

# 摘要请求的结尾 user 指令 (summary_input_messages 追加): 对话切片以 assistant
# 消息结尾时, 模板紧跟的生成提示使模型立即终结 (实测 qwen3.5/ollama 0.20.7:
# stop + content 空 + eval_count=1, 人工核对 M2 复现)——固定 user 结尾保证摘要
# 有生成起点。该消息是请求整形, 不属规范历史、不计入压缩覆盖范围。
SUMMARY_TURN_TEXT = "以上是需要摘要的较早对话记录。请现在输出摘要正文。"


def summary_input_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    """摘要请求的完整输入 (D5): 原始对话切片 + 结尾固定 user 指令.

    压缩计划的消息构造与摘要输入预算估计共用本函数, 保证估计面与实际发送
    逐消息一致; 调用方 (generate_summary) 原样发送, 不再自行整形。
    """
    return tuple(messages) + (UserMessage(text=SUMMARY_TURN_TEXT),)

# auto_compact 回调的额度种类: admission=准入自动压缩, overflow=overflow 重试压缩;
# 两种额度独立, 同属本次 execute 状态 (D7)
AutoCompactQuota = Literal["admission", "overflow"]


def summary_prompt_text() -> str:
    """摘要通道的专用程序系统提示 (版本化, hash 随摘要记录落库, spec D5)."""
    return (
        "你是 Cicada coding agent 的会话摘要器。给你一段较早的对话记录"
        "(用户请求、助手回答、工具调用与结果), 请生成一份供后续模型续写的摘要。\n"
        "要求:\n"
        "1. 覆盖: 任务目标、已完成的修改 (文件与要点)、关键来源/证据、失败与未决项、下一步。\n"
        "2. 只陈述记录中可追溯的事实, 不推断、不编造; 保留关键路径与标识符原样。\n"
        "3. 不要声称任何验证/检查已通过: 验证状态由程序控制段给出, 摘要无此权威。\n"
        "4. 只输出摘要正文本身, 不要寒暄或解释任务。"
    )


SUMMARY_PROMPT_SHA256 = hashlib.sha256(summary_prompt_text().encode("utf-8")).hexdigest()


def summary_block_message(
    *, first_request_seq: int, last_request_seq: int, message_count: int, text: str
) -> UserMessage:
    """程序生成的摘要块 UserMessage (D4): 固定头声明非原文/覆盖范围/权威边界.

    压缩与恢复重建共用本函数, 保证块字节逐位一致 (A08)。
    """
    header = (
        "[程序生成的对话摘要] 下面是较早请求的摘要，不是原始对话记录。\n"
        f"覆盖范围: 请求 seq {first_request_seq}..{last_request_seq}"
        f"（{message_count} 条消息）。\n"
        "验证与进程状态以程序控制段为准；本摘要不携带验证权威，"
        "不得作为任务已完成或检查已通过的证据。\n"
        "---- 摘要开始 ----\n"
    )
    return UserMessage(text=header + text)


def canonical_prefix_sha256(messages: Sequence[Message]) -> str:
    """规范历史前缀 [0:end) 的规范编码 sha256 (D6 覆盖前缀指纹)."""
    try:
        encoded = encode_messages(tuple(messages))
    except CodecError as exc:
        raise ValueError(f"canonical prefix is not encodable: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SummaryBlock:
    """一个已提交摘要的投影事实: 程序生成的块消息 + 覆盖范围 (D4/D6).

    块按覆盖区间有序拼接; 区间连续 (下一块起点 = 上一块终点), 覆盖前缀指纹
    针对规范历史 [0:end) 全前缀——恢复重建据此逐块核对。
    """

    message: UserMessage
    end_message_index: int          # 覆盖前缀结束处 = 保留后缀起点 (规范历史下标)
    first_request_seq: int
    last_request_seq: int
    message_count: int              # 本块自身覆盖的消息数
    prefix_sha256: str              # 规范历史 [0:end) 的规范编码 sha256


class ProjectionRelationError(ValueError):
    """投影关系错误: 块区间不连续/越界或覆盖前缀与规范历史不符 (fail-loud)."""


class ContextProjection:
    """应用持有的模型可见投影容器 (D4); 规范历史不因压缩改变.

    有效消息 = [各摘要块消息] + 规范历史[最后覆盖终点:]。压缩把新区间追加为
    新摘要块 (不合并重写旧摘要, spec 范围外裁决); clear 后重置为空 (新代次
    投影为空); 恢复按摘要记录确定性重建。
    """

    def __init__(self) -> None:
        self._blocks: list[SummaryBlock] = []

    @property
    def blocks(self) -> tuple[SummaryBlock, ...]:
        return tuple(self._blocks)

    @property
    def retained_start(self) -> int:
        """保留后缀在规范历史中的起点 (= 最后覆盖终点; 无块为 0)."""
        return self._blocks[-1].end_message_index if self._blocks else 0

    def adopt(self, blocks: Sequence[SummaryBlock]) -> None:
        """整体替换块序列 (恢复重建用); 区间连续性即刻核对."""
        previous_end = 0
        for block in blocks:
            if block.end_message_index <= previous_end:
                raise ProjectionRelationError(
                    "summary blocks must cover strictly increasing canonical ranges"
                )
            previous_end = block.end_message_index
        self._blocks = list(blocks)

    def apply_summary(self, block: SummaryBlock, *, canonical_length: int) -> None:
        """压缩成功后追加新块; 起点必须衔接上一块终点且不越过规范历史 (fail-loud)."""
        if block.end_message_index <= self.retained_start:
            raise ProjectionRelationError(
                "a new summary block must cover messages beyond the current retained start"
            )
        if block.end_message_index > canonical_length:
            raise ProjectionRelationError(
                f"summary coverage end {block.end_message_index} exceeds the "
                f"canonical history length {canonical_length}"
            )
        self._blocks.append(block)

    def reset(self) -> None:
        """clear 成功后重置: 新代次投影为空 (与内存历史同一同步块调用)."""
        self._blocks.clear()

    def project(self, messages: Sequence[Message]) -> tuple[Message, ...]:
        """[摘要块] + messages[retained_start:]; 越界属关系错误 (调用方 fail-loud)."""
        start = self.retained_start
        if start > len(messages):
            raise ProjectionRelationError(
                f"projection retained start {start} exceeds the message count {len(messages)}"
            )
        if not self._blocks:
            return tuple(messages)
        return tuple(block.message for block in self._blocks) + tuple(messages[start:])


@dataclass(frozen=True)
class _UsageAnchor:
    """一次实测锚: 投影后完整请求的组成 + 实测输入 token (不可变)."""

    system_prompt: str
    tools: tuple
    messages: tuple[Message, ...]
    input_tokens: int
    estimated_input_tokens: int | None


def _is_prefix_extension(anchor: _UsageAnchor, request: ModelRequest) -> bool:
    """本次请求是否为锚请求的 (弱) 前缀延长: 投影后逐位相同的前缀 (spec D3)."""
    return (
        request.system_prompt == anchor.system_prompt
        and request.tools == anchor.tools
        and len(request.messages) >= len(anchor.messages)
        and request.messages[: len(anchor.messages)] == anchor.messages
    )


class BudgetPolicy:
    """ModelPort 装饰器: 每轮权威预算准入 + 实测锚; 默认不启用 (由启动层显式组入)."""

    def __init__(
        self,
        inner: ModelPort,
        config: BudgetConfig,
        *,
        summarize: ModelPort | Callable[..., Any] | None = None,
        classify_overflow: Callable[[str], bool] | None = None,
        auto_compact: Callable[[AutoCompactQuota], Any] | None = None,
        projection: ContextProjection | None = None,
    ) -> None:
        if not callable(getattr(inner, "stream", None)):
            raise TypeError("inner must be a ModelPort")
        if not isinstance(config, BudgetConfig):
            raise TypeError("config must be a BudgetConfig")
        if summarize is not None and not callable(getattr(summarize, "stream", None)) and not callable(summarize):
            raise TypeError("summarize must be a ModelPort or callable or None")
        if classify_overflow is not None and not callable(classify_overflow):
            raise TypeError("classify_overflow must be a callable or None")
        if auto_compact is not None and not callable(auto_compact):
            raise TypeError("auto_compact must be a callable or None")
        self.inner = inner
        self.config = config
        # 02/03 注入接缝: summarize 是摘要通道 (02 起为 ModelPort, 生产=同一无状态
        # 适配器实例); classify_overflow 与 auto_compact 03 接线。
        self._summarize = summarize
        self._classify_overflow = classify_overflow
        self._auto_compact = auto_compact
        # 模型可见投影容器 (D4): 应用持有同一实例, 压缩/恢复/clear 由应用驱动
        self._projection = projection if projection is not None else ContextProjection()
        self._anchor: _UsageAnchor | None = None
        self._last_report: BudgetReport | None = None
        self._last_committed_count: int | None = None  # PA02 范围事实
        # 本次 execute 的两项独立额度 (D7, 03): 接缝重置, begin 不重置, 失败不返还
        self._admission_auto_used = False
        self._overflow_retry_used = False

    def set_auto_compact(self, auto_compact: Callable[[AutoCompactQuota], Any] | None) -> None:
        """装配期注入自动压缩回调 (open_application: 会话构造后绑定, D7 接缝)."""
        if auto_compact is not None and not callable(auto_compact):
            raise TypeError("auto_compact must be a callable or None")
        self._auto_compact = auto_compact

    def try_consume_admission_auto_attempt(self) -> bool:
        """消费本次 execute 的准入自动压缩额度 (至多一次, 失败不返还)."""
        if self._admission_auto_used:
            return False
        self._admission_auto_used = True
        return True

    def try_consume_overflow_retry(self) -> bool:
        """消费本次 execute 的 overflow 重试额度 (独立于准入额度, 失败不返还)."""
        if self._overflow_retry_used:
            return False
        self._overflow_retry_used = True
        return True

    def note_request_start(self, committed_count: int) -> None:
        """请求边界接缝 (D1/PA02): 应用在准入通过后、首次预估前调用一次.

        参数为该时刻规范历史消息数; 它建立本次 execute 尝试的预算状态范围。
        committed_count == 0 视为代次重置 (clear) 丢弃锚; 同时重置两项额度,
        跨请求/跨预估不残留 (A09)。
        """
        if isinstance(committed_count, bool) or not isinstance(committed_count, int):
            raise ValueError("committed_count must be an int")
        if committed_count < 0:
            raise ValueError("committed_count must be nonnegative")
        self._last_committed_count = committed_count
        self._admission_auto_used = False
        self._overflow_retry_used = False
        if committed_count == 0:
            # 代次重置 (clear): 锚随历史一起失效; 恢复/压缩等其余失效面经前缀失配
            self._anchor = None

    @property
    def projection(self) -> ContextProjection:
        """本策略使用的投影容器 (应用持有的同一实例)."""
        return self._projection

    @property
    def summarize_channel(self) -> Any:
        """注入的摘要通道 (D5: ModelPort; None=未接线, 压缩不可用)."""
        return self._summarize

    @property
    def last_report(self) -> BudgetReport | None:
        return self._last_report

    @property
    def anchor(self) -> AnchorInfo | None:
        """当前可用锚的只读事实; 失效 (clear/前缀失配前仍保留旧值, 由快照消费方判定) 为 None.

        注意: 前缀失配不主动清除锚 (后继请求可能重新匹配), 此处反映的是最近一次
        立锚事实; 是否可用于本次估计由估计路径即时判定。
        """
        current = self._anchor
        if current is None:
            return None
        return AnchorInfo(
            measured_input_tokens=current.input_tokens,
            estimated_input_tokens=current.estimated_input_tokens,
            message_count=len(current.messages),
        )

    def snapshot(self) -> BudgetSnapshot:
        """应用只读出口 (04): 最新每轮报告 + 当前锚状态."""
        return BudgetSnapshot(
            limit=self.config.num_ctx,
            report=self._last_report,
            anchor=self.anchor,
        )

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        if cancel.cancelled:
            yield StreamDone("aborted")
            return
        # D4: 投影替换发生在 RunPolicy 核对之后 (上游已全量扫描/证明), 本策略把
        # 消息替换为 [摘要块] + 保留后缀; 无块时原样透传同一请求对象
        projected = self._project(request)
        report = self._assess(projected)
        self._last_report = report
        if report.total_estimate > self.config.num_ctx:
            # 权威准入失败: 先尝试本次 execute 至多一次的自动压缩 (D7), 对有效
            # 投影重估; 仍超限零发送, 由内核既有错误路径收尾 (D7 下道口)
            if await self._attempt_admission_auto_compact(cancel):
                projected = self._project(request)
                report = self._assess(projected)
                self._last_report = report
            if report.total_estimate > self.config.num_ctx:
                if cancel.cancelled:
                    # begin 后取消按 aborted 收尾 (D7), 不把压缩期取消伪装成 error
                    yield StreamDone("aborted")
                    return
                yield StreamDone("error", budget_exceeded_text(report))
                return
        # 发送 (含至多一次 overflow 重试, PA06): 见 _send_with_overflow_retry
        async for event in self._send_with_overflow_retry(request, projected, report, cancel):
            yield event

    def _project(self, request: ModelRequest) -> ModelRequest:
        """按投影容器替换请求消息; 无块时返回原请求对象 (02 语义不变)."""
        if self._projection.blocks:
            return ModelRequest(
                messages=self._projection.project(request.messages),
                tools=request.tools,
                system_prompt=request.system_prompt,
            )
        return request

    async def _attempt_admission_auto_compact(self, cancel: CancelToken) -> bool:
        """准入失败的自动压缩尝试: 至多一次, 条件判定与额度消费在回调内 (D7)."""
        if self._admission_auto_used or self._auto_compact is None:
            return False
        return bool(await self._auto_compact("admission"))

    def _overflow_retry_eligible(self, done: StreamDone, seen_output: bool) -> bool:
        """PA06 重试资格: 分类超限 + 该发送零输出 + 重试额度未用 + 回调已接线."""
        return (
            done.stop_reason == "error"
            and not seen_output
            and not self._overflow_retry_used
            and self._classify_overflow is not None
            and self._auto_compact is not None
            and self._classify_overflow(done.error or "")
        )

    async def _send_with_overflow_retry(
        self,
        original: ModelRequest,
        projected: ModelRequest,
        report: BudgetReport,
        cancel: CancelToken,
    ) -> AsyncIterator[StreamEvent]:
        """一次发送 + 至多一次有界 overflow 重试 (PA06/A10).

        分类终结错误先在策略内暂存 (不发布): 资格全满足且回调压缩成功才重发;
        重试发送的终结是唯一向上游发布的终结, 首次错误要么被重试取代要么原样
        发布, 绝不先发布 error 再继续流。零输出判定固定在该发送点: 任何
        TextDelta (含空片段) 或 ToolCallEvent 都禁止透明重发; 发送点后无工具
        副作用由本层结构性保证 (工具只在内核终结交付后派发)。
        """
        seen_output = False
        async for event in self.inner.stream(projected, cancel):
            if isinstance(event, StreamDone):
                if self._overflow_retry_eligible(event, seen_output):
                    if await self._auto_compact("overflow"):
                        retried = self._project(original)
                        retried_report = self._assess(retried)
                        self._last_report = retried_report
                        if retried_report.total_estimate > self.config.num_ctx:
                            if cancel.cancelled:
                                yield StreamDone("aborted")
                                return
                            # 压缩后二次准入仍超限: 按每轮权威口径拒绝 (D7/D8)
                            yield StreamDone("error", budget_exceeded_text(retried_report))
                            return
                        async for retry_event in self.inner.stream(retried, cancel):
                            if isinstance(retry_event, StreamDone):
                                self._consider_anchor(retried, retried_report, retry_event)
                            yield retry_event
                        return
                    if cancel.cancelled:
                        yield StreamDone("aborted")
                        return
                self._consider_anchor(projected, report, event)
                yield event
                return
            if isinstance(event, (TextDelta, ToolCallEvent)):
                seen_output = True
            yield event

    def _assess(self, request: ModelRequest) -> BudgetReport:
        anchor = self._anchor
        if anchor is not None and _is_prefix_extension(anchor, request):
            prefix = len(anchor.messages)
            input_total = anchor.input_tokens
            components: list[ComponentEstimate] = [
                ComponentEstimate(
                    "anchor_measured_prefix",
                    anchor.input_tokens,
                    "measured",
                    note=(
                        "measured input of an earlier identical request prefix "
                        f"(system prompt, {len(anchor.tools)} tool schemas, "
                        f"first {prefix} messages)"
                    ),
                )
            ]
            for index in range(prefix, len(request.messages)):
                tokens = estimate_message_tokens(request.messages[index], self.config)
                components.append(
                    ComponentEstimate(
                        f"message:{index}:{message_kind(request.messages[index])}",
                        tokens,
                        "estimated",
                    )
                )
                input_total += tokens
            source = "measured" if prefix == len(request.messages) else "anchored"
            return assemble_report(
                self.config, input_estimate=input_total, input_components=tuple(components), source=source
            )
        input_estimate, components = estimate_input_components(
            request.system_prompt, request.tools, request.messages, self.config
        )
        return assemble_report(
            self.config,
            input_estimate=input_estimate,
            input_components=components,
            source="estimated",
        )

    def _consider_anchor(self, request: ModelRequest, report: BudgetReport, done: StreamDone) -> None:
        """终结非 error/aborted 且实测 input_tokens 可得非 0 时立锚 (spec D3)."""
        if done.stop_reason in ("error", "aborted"):
            return
        metrics = done.metrics
        if metrics is None:
            return
        tokens = metrics.input_tokens
        if tokens is None or tokens <= 0:
            return
        if not request.messages:
            return
        self._anchor = _UsageAnchor(
            system_prompt=request.system_prompt,
            tools=request.tools,
            messages=request.messages,
            input_tokens=tokens,
            estimated_input_tokens=report.input_estimate,
        )


# --- 摘要生成协议 (D5, 02) ----------------------------------------------------------------

SUMMARY_FAILURES = ("length", "error", "tool_use", "aborted", "empty", "too_long")


@dataclass(frozen=True)
class SummaryOutcome:
    """一次摘要调用的结果: ok=True 时 text/metrics 可用; 否则 failure 逐类标注."""

    ok: bool
    text: str = ""
    metrics: Any = None              # ModelMetrics | None (终结事件自带; None=不可得)
    failure: str | None = None       # SUMMARY_FAILURES 之一
    detail: str | None = None        # 失败补充说明 (如异常文本); 成功为 None


async def generate_summary(
    channel: ModelPort, messages: tuple[Message, ...], cancel: CancelToken
) -> SummaryOutcome:
    """独立无工具摘要调用 (D5): tools=() + 版本化摘要系统提示, 收集文本与终结.

    channel 是注入的 ModelPort (生产复用同一无状态适配器实例; 测试注入独立
    FakeModel, 不消费共享剧本队列)。响应校验: stop_reason=="stop" 且文本非空
    且 ≤ SUMMARY_TEXT_MAX_BYTES; length/error/tool_use/aborted/空/超长逐类失败。
    取消经 cancel 协作传播: 通道返回 aborted 时按失败分类, 外层 Task.cancel
    照常在 await 点传播。
    """
    from cicada.persistence.contracts import SUMMARY_TEXT_MAX_BYTES

    request = ModelRequest(messages=messages, tools=(), system_prompt=summary_prompt_text())
    text_parts: list[str] = []
    done: StreamDone | None = None
    try:
        async for event in channel.stream(request, cancel):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
            elif isinstance(event, ToolCallEvent):
                return SummaryOutcome(ok=False, failure="tool_use")
            elif isinstance(event, StreamDone):
                done = event
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return SummaryOutcome(ok=False, failure="error", detail=str(exc))
    if done is None:
        return SummaryOutcome(ok=False, failure="error", detail="summary stream ended without a terminal event")
    if done.stop_reason == "aborted":
        return SummaryOutcome(ok=False, failure="aborted", metrics=done.metrics)
    if done.stop_reason in ("length", "tool_use", "error"):
        return SummaryOutcome(ok=False, failure=done.stop_reason, metrics=done.metrics)
    text = "".join(text_parts)
    if not text.strip():
        return SummaryOutcome(ok=False, failure="empty", metrics=done.metrics)
    if len(text.encode("utf-8")) > SUMMARY_TEXT_MAX_BYTES:
        return SummaryOutcome(ok=False, failure="too_long", metrics=done.metrics)
    return SummaryOutcome(ok=True, text=text, metrics=done.metrics)
