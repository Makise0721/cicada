"""剧本驱动的假模型: 与真实适配器实现同一 ModelPort."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, TypeAlias, Union

from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

ScriptEntry: TypeAlias = Union[list[StreamEvent], BaseException]


class FakeModel:
    """按剧本逐次响应; 记录每次请求供断言; 剧本用尽回退 stop."""

    def __init__(self, script: Sequence[ScriptEntry]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        entry = self._script.pop(0) if self._script else [StreamDone("stop")]
        if isinstance(entry, BaseException):
            raise entry
        for event in entry:
            if cancel.cancelled:
                yield StreamDone("aborted")
                return
            await asyncio.sleep(0)
            yield event


def fake_model_plugin(model: FakeModel) -> PluginDefinition:
    def setup(ctx: PluginContext) -> None:
        ctx.provide("model", model)

    return PluginDefinition(name="fake-model", setup=setup, provides=frozenset({"model"}))
