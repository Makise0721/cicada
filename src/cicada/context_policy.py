"""应用层 BudgetPolicy: 每轮发送前的权威预算准入与同身份实测锚 (01).

与 RunPolicy 同构的 ModelPort 装饰器 (spec D1): 安装次序
Agent → RunPolicy → BudgetPolicy → 适配器; 内核零改动, 规范历史不因本策略
改变 (压缩投影属 02+)。

- 每轮 stream 先估计完整输入。本策略位于 RunPolicy 下游: 所见 system prompt
  已含控制段、消息已含 freshness 标记, 计量因此覆盖动态验证段最终形态。
  总需求 (输入+输出预留+安全余量) > num_ctx 时产出 StreamDone("error") 且
  适配器零发送, 错误文本带估计值、上限与主要占用组成。
- 实测锚 (spec D3): 指纹基准是本策略实际所见 (投影后) 的完整请求。前次实测
  input_tokens 非 None/非 0 且终结非 error/aborted 时立锚; 后继请求为该锚的
  严格前缀延长 (system prompt 与 tools 逐位相同、消息前缀相等) 时按 anchored
  估计 = 实测 + 后缀增量; 与锚完全相同则整体 measured。clear (committed_count
  == 0)、恢复 (新策略实例) 与配置/工具/提示/验证视图变化经前缀失配自然失效;
  后者是预期的保守回退, 不是缺陷。
- 请求边界接缝 (spec D1): note_request_start(committed_count) 由应用在每个
  请求开始 (持久模式 begin 成功后、agent.run 之前) 调用一次, 参数为调用时
  规范历史长度; committed_count == 0 视为代次重置并丢弃锚。计数语义由 03 消费。
- 摘要通道 (02) 与 overflow 分类器 (03) 经构造参数注入, 组装接线在
  application 层; 本模块不导入 plugins (导入边界测试钉住)。01 阶段两个注入
  参数为 None 且不产生任何自动行为。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable

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
from cicada.core.messages import Message
from cicada.core.ports import ModelPort, ModelRequest, StreamDone, StreamEvent


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
        summarize: Callable[..., Any] | None = None,
        classify_overflow: Callable[[str], bool] | None = None,
    ) -> None:
        if not callable(getattr(inner, "stream", None)):
            raise TypeError("inner must be a ModelPort")
        if not isinstance(config, BudgetConfig):
            raise TypeError("config must be a BudgetConfig")
        if summarize is not None and not callable(summarize):
            raise TypeError("summarize must be a callable or None")
        if classify_overflow is not None and not callable(classify_overflow):
            raise TypeError("classify_overflow must be a callable or None")
        self.inner = inner
        self.config = config
        # 02/03 注入接缝: 01 不消费, 不产生任何自动行为
        self._summarize = summarize
        self._classify_overflow = classify_overflow
        self._anchor: _UsageAnchor | None = None
        self._last_report: BudgetReport | None = None

    def note_request_start(self, committed_count: int) -> None:
        """请求边界接缝 (D1): 应用在每个请求开始调用, 参数为当前规范历史长度."""
        if isinstance(committed_count, bool) or not isinstance(committed_count, int):
            raise ValueError("committed_count must be an int")
        if committed_count < 0:
            raise ValueError("committed_count must be nonnegative")
        if committed_count == 0:
            # 代次重置 (clear): 锚随历史一起失效; 恢复/压缩等其余失效面经前缀失配
            self._anchor = None

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
        report = self._assess(request)
        self._last_report = report
        if report.total_estimate > self.config.num_ctx:
            # 权威准入失败: 该轮零发送, 由内核既有错误路径收尾 (D7)
            yield StreamDone("error", budget_exceeded_text(report))
            return
        async for event in self.inner.stream(request, cancel):
            if isinstance(event, StreamDone):
                self._consider_anchor(request, report, event)
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
