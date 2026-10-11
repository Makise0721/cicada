"""中立预算契约: 预算配置、开销分解报告与启发式估计器 (上下文预算与压缩 01).

只依赖 core (消息/端口类型); 不导入 persistence/runtime/plugins/boot (导入边界
测试钉住)。预算口径 (spec D2): 上限 = 有效 num_ctx; 完整输入 = system prompt
(含 RunPolicy 固定验证规则) + 实际工具 schema + 有效消息 + 非空 request_context
(程序验证上下文尾部); 总需求另加输出预留 (配置 num_predict, 缺省固定值) 与安全
余量 max(下限, ⌈比例 × 估计输入⌉)。

来源等级如实标注 (spec D3): measured (同身份实测) / anchored (实测锚+增量) /
estimated (启发式) / unknown。启发式: 文本 token ≈ ⌈UTF-8 字节 / 3⌉ (英文保守
偏高, 中文约 1 token/字符), 每条消息加固定模板开销; 工具 schema 按其规范 JSON
字节同法估计。近似估计不得称为精确 token, 也不保证服务端不截断;
max_request_bytes (P4 §8) 的 wire 字节守卫是另一道独立检查, 语义互不替代。

估计器名与版本随报告输出; 调整算法须推进版本号 (04 校准的基线)。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from cicada.core.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ToolSpec

BUDGET_ESTIMATOR_NAME = "utf8-bytes-over-3-plus-template-overhead"
BUDGET_ESTIMATOR_VERSION = "3"

OUTPUT_RESERVE_DEFAULT_TOKENS = 4096
SAFETY_MARGIN_MIN_TOKENS = 256
SAFETY_MARGIN_RATIO = 0.10
MESSAGE_TEMPLATE_OVERHEAD_TOKENS = 8

EstimateSource = Literal["measured", "anchored", "estimated", "unknown"]


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BudgetConfig:
    """预算配置; num_ctx 是唯一必填项 (有效窗口请求值, 与 wire 字节守卫无关)."""

    num_ctx: int
    num_predict: int | None = None
    message_template_overhead: int = MESSAGE_TEMPLATE_OVERHEAD_TOKENS
    safety_margin_ratio: float = SAFETY_MARGIN_RATIO
    safety_margin_min: int = SAFETY_MARGIN_MIN_TOKENS
    output_reserve_default: int = OUTPUT_RESERVE_DEFAULT_TOKENS

    def __post_init__(self) -> None:
        _require_positive_int(self.num_ctx, "num_ctx")
        if self.num_predict is not None:
            _require_positive_int(self.num_predict, "num_predict")
        if (
            isinstance(self.message_template_overhead, bool)
            or not isinstance(self.message_template_overhead, int)
            or self.message_template_overhead < 0
        ):
            raise ValueError("message_template_overhead must be a nonnegative integer")
        if (
            isinstance(self.safety_margin_ratio, bool)
            or not isinstance(self.safety_margin_ratio, (int, float))
            or not 0 <= self.safety_margin_ratio <= 1
        ):
            raise ValueError("safety_margin_ratio must be a number in [0, 1]")
        if (
            isinstance(self.safety_margin_min, bool)
            or not isinstance(self.safety_margin_min, int)
            or self.safety_margin_min < 0
        ):
            raise ValueError("safety_margin_min must be a nonnegative integer")
        _require_positive_int(self.output_reserve_default, "output_reserve_default")

    @property
    def effective_output_reserve(self) -> int:
        """输出预留: 配置的 num_predict (正整数) 生效, 否则用固定缺省值."""
        return self.num_predict if self.num_predict is not None else self.output_reserve_default


@dataclass(frozen=True)
class ComponentEstimate:
    """一个开销组成部分的估计; tokens=None 表示不可得 (来源 unknown)."""

    label: str
    tokens: int | None
    source: EstimateSource
    note: str = ""


@dataclass(frozen=True)
class BudgetReport:
    """一次发送前的完整预算报告; 各组成部分之和 = 总估计, 无静默省略.

    anchored 来源下 system prompt/工具 schema/被锚覆盖的前缀消息合并在
    anchor_measured_prefix 一个组件里 (实测无法逐段归因), 组件 note 说明
    覆盖范围; 这是有意呈现, 不是省略。
    """

    limit: int
    components: tuple[ComponentEstimate, ...]
    input_estimate: int
    output_reserve: int
    safety_margin: int
    total_estimate: int
    source: EstimateSource
    estimator: str = BUDGET_ESTIMATOR_NAME
    estimator_version: str = BUDGET_ESTIMATOR_VERSION

    @property
    def margin_remaining(self) -> int:
        return self.limit - self.total_estimate

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的显示消费视图 (04)."""
        return asdict(self)


@dataclass(frozen=True)
class AnchorInfo:
    """最近一次实测锚的只读事实; 估计偏差 = measured 与 estimated_input_tokens 之差."""

    measured_input_tokens: int
    estimated_input_tokens: int | None
    message_count: int


@dataclass(frozen=True)
class BudgetSnapshot:
    """应用只读预算出口: 最新一次每轮报告与当前锚状态 (04 显示消费)."""

    limit: int
    report: BudgetReport | None
    anchor: AnchorInfo | None


def estimate_text_tokens(text: str) -> int:
    """启发式文本估计: ⌈UTF-8 字节 / 3⌉; 不声称精确 token."""
    return math.ceil(len(text.encode("utf-8")) / 3)


def _canonical_json_bytes(value: Any) -> int:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        return len(repr(value).encode("utf-8"))
    return len(encoded.encode("utf-8"))


def estimate_tool_spec_tokens(spec: ToolSpec) -> int:
    """工具 schema 按其规范 JSON 字节同法估计 (spec D3)."""
    payload = {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
    }
    return math.ceil(_canonical_json_bytes(payload) / 3)


def _message_bytes(message: Message) -> int:
    if isinstance(message, UserMessage):
        return len(message.text.encode("utf-8"))
    if isinstance(message, AssistantMessage):
        total = len(message.text.encode("utf-8"))
        if message.reasoning is not None:
            total += len(message.reasoning.text.encode("utf-8"))
        if message.error:
            total += len(message.error.encode("utf-8"))
        for call in message.tool_calls:
            total += (
                len(call.id.encode("utf-8"))
                + len(call.name.encode("utf-8"))
                + len(call.arguments_json.encode("utf-8"))
            )
        return total
    if isinstance(message, ToolResultMessage):
        result = message.result
        total = (
            len(result.call_id.encode("utf-8"))
            + len(result.name.encode("utf-8"))
            + len(result.content.encode("utf-8"))
        )
        if result.details is not None:
            total += _canonical_json_bytes(result.details)
        return total
    raise ValueError(f"unsupported message type: {type(message).__name__}")


def estimate_message_tokens(message: Message, config: BudgetConfig) -> int:
    """单条消息估计 = 内容字节/3 + 固定模板开销."""
    return math.ceil(_message_bytes(message) / 3) + config.message_template_overhead


def message_kind(message: Message) -> str:
    """报告标签用的消息类别词: user / assistant / tool; 未知类型 fail-loud."""
    if isinstance(message, UserMessage):
        return "user"
    if isinstance(message, AssistantMessage):
        return "assistant"
    if isinstance(message, ToolResultMessage):
        return "tool"
    raise ValueError(f"unsupported message type: {type(message).__name__}")


def estimate_input_components(
    system_prompt: str,
    tools: Sequence[ToolSpec],
    messages: Sequence[Message],
    config: BudgetConfig,
    *,
    request_context: str = "",
) -> tuple[int, tuple[ComponentEstimate, ...]]:
    """纯启发式输入分解: system prompt / 工具 schema / 逐条消息 / 请求上下文, 来源全为 estimated.

    非空 `request_context` 是本轮程序验证观察 (不是规范 Message), 单独组成计:
    ⌈UTF-8 字节 / 3⌉ + 一条消息模板开销; 空串不产生组件。
    """
    components: list[ComponentEstimate] = [
        ComponentEstimate("system_prompt", estimate_text_tokens(system_prompt), "estimated")
    ]
    total = components[0].tokens
    assert total is not None
    for spec in tools:
        tokens = estimate_tool_spec_tokens(spec)
        components.append(ComponentEstimate(f"tool_schema:{spec.name}", tokens, "estimated"))
        total += tokens
    for index, message in enumerate(messages):
        tokens = estimate_message_tokens(message, config)
        components.append(
            ComponentEstimate(
                f"message:{index}:{message_kind(message)}", tokens, "estimated"
            )
        )
        total += tokens
    if request_context:
        tokens = math.ceil(len(request_context.encode("utf-8")) / 3) + config.message_template_overhead
        components.append(ComponentEstimate("request_context", tokens, "estimated"))
        total += tokens
    return total, tuple(components)


def assemble_report(
    config: BudgetConfig,
    *,
    input_estimate: int,
    input_components: tuple[ComponentEstimate, ...],
    source: EstimateSource,
) -> BudgetReport:
    """输入分解 + 输出预留 + 安全余量 -> 完整报告; 各部分之和 = 总估计."""
    reserve = config.effective_output_reserve
    margin = max(
        config.safety_margin_min,
        math.ceil(config.safety_margin_ratio * input_estimate),
    )
    total = input_estimate + reserve + margin
    components = tuple(input_components) + (
        ComponentEstimate(
            "output_reserve",
            reserve,
            "estimated",
            note="configured num_predict" if config.num_predict is not None else "default reserve",
        ),
        ComponentEstimate(
            "safety_margin",
            margin,
            "estimated",
            note=f"max({config.safety_margin_min}, ceil({config.safety_margin_ratio} x input))",
        ),
    )
    return BudgetReport(
        limit=config.num_ctx,
        components=components,
        input_estimate=input_estimate,
        output_reserve=reserve,
        safety_margin=margin,
        total_estimate=total,
        source=source,
    )


def budget_exceeded_text(report: BudgetReport) -> str:
    """超限解释文本: 估计值、上限与主要占用组成 (A02); 与 wire 字节守卫文本不同源."""
    ranked = sorted(
        (component for component in report.components if component.tokens is not None),
        key=lambda component: component.tokens,  # type: ignore[arg-type,return-value]
        reverse=True,
    )
    top = ", ".join(f"{component.label}~{component.tokens}" for component in ranked[:3])
    return (
        f"context budget exceeded: estimated ~{report.total_estimate} tokens "
        f"(input ~{report.input_estimate} + output reserve {report.output_reserve} "
        f"+ safety margin {report.safety_margin}) exceeds limit {report.limit} (num_ctx); "
        f"largest components: {top}; no request was sent"
    )
