"""DeepSeek thinking Chat Completion 能力。"""

from typing import TYPE_CHECKING

from cicada.plugins.deepseek.plugin import deepseek_plugin
from cicada.plugins.deepseek.protocol import DeepSeekConfig

if TYPE_CHECKING:
    from cicada.plugins.deepseek.model import DeepSeekModel


def __getattr__(name: str):
    if name == "DeepSeekModel":
        from cicada.plugins.deepseek.model import DeepSeekModel

        return DeepSeekModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DeepSeekConfig", "DeepSeekModel", "deepseek_plugin"]
