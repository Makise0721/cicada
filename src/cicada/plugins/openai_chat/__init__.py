"""有界 OpenAI Chat Completions 兼容协议；不承诺供应商私有扩展。"""

from .model import ChatModel
from .protocol import ChatConfig

__all__ = ["ChatConfig", "ChatModel"]
