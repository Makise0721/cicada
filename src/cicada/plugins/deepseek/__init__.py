"""DeepSeek thinking Chat Completion 能力。"""

from cicada.plugins.deepseek.model import DeepSeekModel
from cicada.plugins.deepseek.plugin import deepseek_plugin
from cicada.plugins.deepseek.protocol import DeepSeekConfig

__all__ = ["DeepSeekConfig", "DeepSeekModel", "deepseek_plugin"]
