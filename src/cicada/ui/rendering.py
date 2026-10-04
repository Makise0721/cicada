"""显示清理 (T13): 不可信文本进入显示副本前剥离终端控制序列.

旧渲染链 (record/preview/status fragments, Rich->prompt_toolkit 排版) 随
Pi 主屏迁移退场 (票据 05); 本模块保留被 02/03/04 复用的 sanitize_display
与其正则。外部 CSI/OSC 不作为可信控制序列执行: 良构序列移除后, 未终结的
截断段 payload 不可与后续正文区分, 一并移除是截断的代价; 孤立 ESC 不再
构成执行通道。只作用于显示路径; 规范消息、会话历史与证据保持原始内容.
"""

from __future__ import annotations

import re

# 显示清理: CSI 序列、OSC 序列、未终结的截断段、除 \n \t 外的 C0/C1 控制字符
_CSI_RE = re.compile(r"\x1b\[[0-9;:?<=>!]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_UNTERMINATED_OSC_RE = re.compile(r"\x1b\][^\x1b]*")
_TRUNCATED_CSI_RE = re.compile(r"\x1b\[[0-9;:?<=>!]*[ -/]*$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")


def sanitize_display(text: str) -> str:
    """显示副本清理: 移除 CSI/OSC/未终结截断段与控制字符 (保留 \n 与 \t); 只用于渲染输入."""
    text = _UNTERMINATED_OSC_RE.sub("", _OSC_RE.sub("", text))
    text = _TRUNCATED_CSI_RE.sub("", _CSI_RE.sub("", text))
    return _CONTROL_RE.sub("", text)
