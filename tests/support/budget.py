"""共享预算测试构造: 准入总需求复算与小窗口策略配置.

admission_total 与 assemble_report / 请求前预估同一口径 (raw + 输出预留 +
安全余量), 原 test_compaction / test_compaction_review / test_auto_trigger
三份同型私有夹具的单一来源.
"""

from __future__ import annotations

import math

from cicada.budget import BudgetConfig, estimate_input_components


def admission_total(system: str, messages, config: BudgetConfig) -> int:
    """请求前预估口的总需求复算 (与 assemble_report 同构; 无工具/无计划)."""
    raw, _components = estimate_input_components(system, (), tuple(messages), config)
    return raw + config.effective_output_reserve + max(
        config.safety_margin_min, math.ceil(config.safety_margin_ratio * raw)
    )


def policy_config(**overrides) -> BudgetConfig:
    """小窗口策略测试配置: 宽松 num_ctx + 每消息 1 token 开销, 便于精确卡窗口."""
    defaults = dict(
        num_ctx=100_000, num_predict=1,
        safety_margin_min=1, safety_margin_ratio=0, message_template_overhead=1,
    )
    defaults.update(overrides)
    return BudgetConfig(**defaults)
