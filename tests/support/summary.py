"""共享摘要通道替身与预算构造。

原 test_compaction / test_auto_trigger / test_compact_command 三份同型夹具的单一来源。
metrics 语义是行为差: 默认 None (计量不可得路径); 需要恒定计量时显式传
DEFAULT_SUMMARY_METRICS。
"""

from __future__ import annotations

import asyncio

from cicada.budget import BudgetConfig
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta

DEFAULT_SUMMARY_METRICS = ModelMetrics(input_tokens=9, output_tokens=2)


def budget(num_ctx: int = 1_000_000) -> BudgetConfig:
    return BudgetConfig(num_ctx=num_ctx, num_predict=2048)


class ScriptedSummary:
    """无门控摘要通道: 恒定正文 + 可选计量 (正常路径用)。"""

    def __init__(self, text: str = "较早请求的摘要正文", metrics=None) -> None:
        self._text = text
        self._metrics = metrics
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        yield TextDelta(self._text)
        yield StreamDone("stop", metrics=self._metrics)


class GatedSummary:
    """门控摘要通道: 放行前挂起, 可选忽略协作取消 (PA05 反例用)。"""

    def __init__(
        self, *, text: str = "门控摘要正文", ignore_cancel: bool = False,
        metrics=DEFAULT_SUMMARY_METRICS,
    ) -> None:
        self.gate = asyncio.Event()
        self.cancel_seen = False
        self.requests = []
        self._text = text
        self._ignore_cancel = ignore_cancel
        self._metrics = metrics

    async def stream(self, request, cancel):
        self.requests.append(request)
        await self.gate.wait()
        if cancel.cancelled:
            self.cancel_seen = True
            if not self._ignore_cancel:
                yield StreamDone("aborted")
                return
        yield TextDelta(self._text)
        yield StreamDone("stop", metrics=self._metrics)
