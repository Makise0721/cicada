"""Ollama 真实模型插件: 冻结 ModelPort 适配本地 Ollama /api/chat (NDJSON 流式)."""

from cicada.plugins.ollama.model import DEFAULT_TIMEOUT, OllamaModel
from cicada.plugins.ollama.plugin import ollama_plugin
from cicada.plugins.ollama.protocol import (
    OllamaConfig,
    ParsedChunk,
    ToolCallChunk,
    build_request,
    parse_line,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "OllamaConfig",
    "OllamaModel",
    "ParsedChunk",
    "ToolCallChunk",
    "build_request",
    "ollama_plugin",
    "parse_line",
]
