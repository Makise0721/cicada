"""运行摘要: 由 RunResult 直接生成, 不依赖 observer 计数.

契约 (P3 计划 §7): 没有报告值显示 unknown; 0 计入已知, None 不转成 0;
provider duration 与本地 elapsed 分别标注不相加; input token 跨调用求和是
重复处理在内的计量, 不是唯一上下文 token 数; 摘要不输出任务验收结论.
"""

from __future__ import annotations

from cicada.core.agent import RunResult
from cicada.core.messages import ToolResultMessage


def _sum_known_int(values: list[int | None]) -> tuple[str, int]:
    """(总和显示或 unknown, 已知个数); 0 是合法已知值."""
    known = [value for value in values if value is not None]
    return (str(sum(known)), len(known)) if known else ("unknown", 0)


def _sum_known_float(values: list[float | None]) -> tuple[str, int]:
    known = [value for value in values if value is not None]
    return (f"{sum(known):.2f}", len(known)) if known else ("unknown", 0)


def build_run_summary(result: RunResult) -> str:
    """生成三行运行摘要: 运行状态 / 耗时 / token 计量与覆盖率."""
    calls = result.model_calls
    total = len(calls)
    tool_results = [m.result for m in result.messages if isinstance(m, ToolResultMessage)]
    tool_errors = sum(1 for item in tool_results if item.is_error)

    model_time = f"{sum(call.elapsed_s for call in calls):.2f}" if calls else "unknown"
    provider_time, provider_known = _sum_known_float(
        [call.metrics.provider_duration_s if call.metrics else None for call in calls]
    )
    input_sum, input_known = _sum_known_int(
        [call.metrics.input_tokens if call.metrics else None for call in calls]
    )
    output_sum, output_known = _sum_known_int(
        [call.metrics.output_tokens if call.metrics else None for call in calls]
    )

    return "\n".join(
        [
            f"[run_summary run_id={result.run_id} stop_reason={result.stop_reason} "
            f"model_calls={total} tools={len(tool_results)} tool_errors={tool_errors}]",
            # provider_duration 是服务端自报, 与本地 elapsed 分别标注, 不相加当总运行时间
            f"[model_time_s={model_time} provider_time_s={provider_time} "
            f"provider_reported_calls={provider_known}/{total}]",
            f"[input_tokens_known={input_sum} input_reported_calls={input_known}/{total} "
            f"output_tokens_known={output_sum} output_reported_calls={output_known}/{total}]",
        ]
    )
