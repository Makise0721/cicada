"""系统提示构造: 内置提示 + 显式指令文件的有界加载与组装.

纯函数模块: 只返回文本与来源信息, 不读写 Agent 状态, 不导入内核/运行时/插件.
契约 (P3 计划 §4.2): 内置提示固定版本标识; 指令文件 16 KiB 原始字节上限, 先有界读取
limit+1 再按 UTF-8-sig 解码; 相对路径以 workspace 为根; 稳定顺序 内置 → 项目指令;
raw bytes hash 与组装文本 hash 分别标注.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPT_VERSION = "builtin:p3-v1"
MAX_INSTRUCTIONS_BYTES = 16 * 1024

DEFAULT_TOOLS = ("read", "edit", "write", "powershell")


class InstructionsError(ValueError):
    """指令文件不可用: 不存在/是目录/不可读/无效 UTF-8/超过 16 KiB."""


@dataclass(frozen=True)
class PromptInfo:
    """启动显示用的来源信息; 不含提示全文."""

    version: str
    workspace: Path
    instructions_path: Path | None
    instructions_sha256: str | None  # 源文件原始字节
    prompt_sha256: str  # 组装后文本的 UTF-8


def build_builtin_prompt(workspace: Path) -> str:
    """内置默认提示; 简洁英文, 含实际 workspace 与默认工具, 无时间/动态状态."""
    root = workspace.resolve()
    tools = "\n".join(f"- {name}" for name in DEFAULT_TOOLS)
    return (
        "You are Cicada, a coding agent running on Windows.\n"
        f"Your workspace directory is: {root}\n"
        "\n"
        "Available tools:\n"
        f"{tools}\n"
        "\n"
        "How to work:\n"
        "- Use the read tool to inspect actual files before changing anything; never guess at code.\n"
        "- Read and edit results carry the source path and line range; when truncated, continue with "
        "the given next_offset (or the suggested command for a partially read long line) instead of "
        "assuming you saw the whole file.\n"
        "- Use edit with exact, unique old_text matches; keep changes minimal and precise.\n"
        "- Use the powershell tool for shell commands (Windows, PowerShell 7 syntax).\n"
        "- After changing files, verify the result (re-read the file or run a check) before reporting.\n"
        "- Report what you actually did and observed; do not claim success without evidence.\n"
        "- Respond in the user's language.\n"
    )


def load_instructions(path_str: str, workspace: Path) -> tuple[str, Path, str]:
    """有界加载指令文件: 返回 (文本, 解析后路径, 原字节 sha256 hex); 失败抛 InstructionsError.

    相对路径以 workspace 为根解析; 绝对路径只读加载。空/纯空白文本是可接受输入
    (由组装方决定是否跳过), 本函数不拒绝。
    """
    candidate = Path(path_str)
    resolved = candidate if candidate.is_absolute() else workspace / candidate
    try:
        with open(resolved, "rb") as handle:
            raw = handle.read(MAX_INSTRUCTIONS_BYTES + 1)
    except OSError as exc:
        raise InstructionsError(f"cannot read instructions file {resolved}: {exc}") from exc
    if len(raw) > MAX_INSTRUCTIONS_BYTES:
        raise InstructionsError(
            f"instructions file {resolved} exceeds {MAX_INSTRUCTIONS_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InstructionsError(f"instructions file {resolved} is not valid UTF-8: {exc}") from exc
    return text, resolved, hashlib.sha256(raw).hexdigest()


def compose_system_prompt(
    workspace: Path, instructions_path: str | None = None
) -> tuple[str, PromptInfo]:
    """组装系统提示: 内置提示在前, 项目指令以明确来源标记追加在后.

    空/纯空白指令文件不追加空 section。返回 (提示文本, 来源信息)。
    """
    prompt = build_builtin_prompt(workspace)
    resolved: Path | None = None
    raw_hash: str | None = None
    if instructions_path is not None:
        text, resolved, raw_hash = load_instructions(instructions_path, workspace)
        if text.strip():
            prompt += (
                f"\n--- Project instructions (source: {resolved}, sha256:{raw_hash}) ---\n"
                f"{text.rstrip()}\n"
            )
    info = PromptInfo(
        version=PROMPT_VERSION,
        workspace=workspace.resolve(),
        instructions_path=resolved,
        instructions_sha256=raw_hash,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )
    return prompt, info
