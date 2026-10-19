"""系统提示构造: 内置提示 + 指令文件的有界加载与组装.

纯函数模块: 只返回文本与来源信息, 不读写 Agent 状态, 不导入内核/运行时/插件.
契约 (P3 计划 §4.2): 内置提示固定版本标识; 指令文件 16 KiB 原始字节上限, 先有界读取
limit+1 再按 UTF-8-sig 解码; 相对路径以 workspace 为根; 稳定顺序 内置 → 用户指令 → 项目指令;
raw bytes hash 与组装文本 hash 分别标注.

P4 §1/§6: 工具名单来自启动层的实际注册配置 (默认含 glob/grep), 指定检查模式把
启动者声明的 check-id/命令/timeout 如实写进提示, 但检查结果状态由运行时投影提供,
不在提示里伪造. 本模块仍不导入插件层, 检查声明用本地 CheckPrompt 记录承载.

select_instructions_file 只做启动来源选择 (显式 > 关闭 > 用户级与项目级
AGENTS.md), 不读正文; mode 仅供显示, 不进入提示文本或持久身份.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from cicada.skills import (
    AUTO_SKILLS_PROMPT_VERSION, SKILLS_PROMPT_VERSION, SkillCatalogSnapshot,
    catalog_prompt_section,
)

PROMPT_VERSION = "builtin:p4-v1"
MAX_INSTRUCTIONS_BYTES = 16 * 1024

DEFAULT_TOOLS = ("read", "edit", "write", "powershell", "glob", "grep")
CHECK_TOOL = "check"

# project-context-v1 C1: 启动层指令来源选择的固定 mode 词表; 仅供显示,
# 不进入提示文本、prompt hash 或 LaunchIdentity 的可比对字段。
INSTRUCTIONS_MODES = ("explicit", "automatic", "disabled", "absent")
INSTRUCTIONS_FILENAME = "AGENTS.md"
INSTRUCTION_MODE_LABELS = {"explicit": "显式", "automatic": "自动"}


class InstructionsError(ValueError):
    """指令文件不可用: 不存在/是目录/不可读/无效 UTF-8/超过 16 KiB."""


@dataclass(frozen=True)
class CheckPrompt:
    """提示里声明的指定检查; 值由启动层提供, 模型只能选择 check_id."""

    check_id: str
    command: str
    timeout_s: float


@dataclass(frozen=True)
class PromptInfo:
    """启动显示用的来源信息; 不含提示全文."""

    version: str
    workspace: Path
    instructions_path: Path | None
    instructions_sha256: str | None  # 单文件原始字节；双来源为有序来源摘要
    prompt_sha256: str  # 组装后文本的 UTF-8
    instructions_mode: str | None = None  # 仅供显示; None = 未传 mode 的旧构造
    instruction_sources: tuple[tuple[str, Path, str], ...] = ()  # scope, path, raw sha256


@dataclass(frozen=True)
class InstructionSelection:
    """启动层选出的指令来源 (C1); 正文仍由 load_instructions 有界读取一次.

    mode 只用于显示, 不参与提示构造、hash 或持久身份; path 与 mode 的组合
    必须一致 (explicit/automatic 有 path, disabled/absent 无 path).
    """

    path: str | None
    mode: str
    sources: tuple[tuple[str, str], ...] = ()  # scope, path; 自动模式为 user → project

    def __post_init__(self) -> None:
        if self.mode not in INSTRUCTIONS_MODES:
            raise ValueError(f"unknown instructions mode: {self.mode!r}")
        if (self.path is not None) != (self.mode in ("explicit", "automatic")):
            raise ValueError(
                f"contradictory instructions selection: path={self.path!r} mode={self.mode!r}"
            )


def select_instructions_file(
    workspace: Path, *, explicit_path: str | None = None, auto_enabled: bool = True,
    user_dir: Path | None = None,
) -> InstructionSelection:
    """显式文件全替自动来源；自动来源按用户级、项目级顺序选择。

    只选择来源, 不读正文。自动候选用 lstat 区分: 缺失是正常 absent;
    reparse point、目录或不可访问抛 InstructionsError, 不以 exists() 吞掉
    权限错误。显式路径沿用旧加载规则, 在此不做额外检查。
    """
    if explicit_path is not None:
        return InstructionSelection(explicit_path, "explicit", (("explicit", explicit_path),))
    if not auto_enabled:
        return InstructionSelection(None, "disabled")
    sources = []
    for scope, candidate in (
        ("user", (user_dir if user_dir is not None else Path.home() / ".cicada") / INSTRUCTIONS_FILENAME),
        ("project", workspace / INSTRUCTIONS_FILENAME),
    ):
        try:
            st = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise InstructionsError(f"cannot inspect instructions file {candidate}: {exc}") from exc
        if getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise InstructionsError(f"automatic instructions file {candidate} is a reparse point")
        if stat.S_ISDIR(st.st_mode):
            raise InstructionsError(f"automatic instructions file {candidate} is a directory")
        sources.append((scope, str(candidate)))
    return InstructionSelection(sources[-1][1], "automatic", tuple(sources)) if sources else InstructionSelection(None, "absent")


def _resolve_instructions_mode(instructions_path: str | None, mode: str | None) -> str:
    """组装前的 mode 一致性核对 (C1); None 按旧约定由是否传文件推导."""
    if mode is None:
        return "explicit" if instructions_path is not None else "absent"
    if mode not in INSTRUCTIONS_MODES:
        raise ValueError(f"unknown instructions mode: {mode!r}")
    if (instructions_path is not None) != (mode in ("explicit", "automatic")):
        raise ValueError(
            f"contradictory instructions input: path={instructions_path!r} mode={mode!r}"
        )
    return mode


def build_builtin_prompt(
    workspace: Path,
    tools: tuple[str, ...] = DEFAULT_TOOLS,
    checks: tuple[CheckPrompt, ...] = (),
) -> str:
    """内置默认提示; 简洁英文, 含实际 workspace 与实际注册工具, 无时间/动态状态.

    指定检查模式下 `checks` 非空: 工具名单应包含 check, 各声明的 check_id/命令/timeout
    原样列出; 检查的当前状态不在此处 (由运行时逐轮投影), 这里只声明固定契约.
    """
    root = workspace.resolve()
    tool_lines = "\n".join(f"- {name}" for name in tools)
    lines = [
        "You are Cicada, a coding agent running on Windows.\n"
        f"Your workspace directory is: {root}\n",
        "Available tools:",
        tool_lines,
        "",
        "How to work:",
        "- Use the read tool to inspect actual files before changing anything; never guess at code.",
        "- Read and edit results carry the source path and line range; when truncated, continue with "
        "the given next_offset (or the suggested command for a partially read long line) instead of "
        "assuming you saw the whole file.",
        "- Use edit with exact, unique old_text matches; keep changes minimal and precise.",
        "- Use glob to find files by name pattern and grep to search literal text (not regex); "
        "their results report real paths and line numbers you can pass to read.",
        "- Use the powershell tool for shell commands (Windows, PowerShell 7 syntax).",
        "- After changing files, verify the result (re-read the file or run a check) before reporting.",
        "- Report what you actually did and observed; do not claim success without evidence.",
        "- Respond in the user's language.",
    ]
    if checks:
        lines.append("")
        lines.append("Required checks (declared by the launcher; you cannot change them):")
        for check in checks:
            command = " ".join(check.command.split())
            lines.append(f"- {check.check_id}: command={command} timeout={check.timeout_s:g}s")
        lines.append("")
        lines.append("How checks work:")
        lines.append(
            "- Run a declared check with the check tool (action=run, check_id); you cannot "
            "substitute your own command, working directory or timeout."
        )
        lines.append(
            "- A passing check becomes stale as soon as any file changes afterwards; re-run the "
            "checks after every change so the receipts match the final code."
        )
        lines.append(
            "- Delivery is judged by the program from the actual final code state and the check "
            "receipts, not from your summary; report what you did, but the receipts decide."
        )
    return "\n".join(lines) + "\n"



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
    workspace: Path,
    instructions_path: str | None = None,
    *,
    tools: tuple[str, ...] = DEFAULT_TOOLS,
    checks: tuple[CheckPrompt, ...] = (),
    skills: SkillCatalogSnapshot | None = None,
    instructions_mode: str | None = None,
    instruction_sources: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str, PromptInfo]:
    """组装系统提示: 内置提示在前, Skills 目录段紧随, 项目指令以明确来源标记追加在后.

    空/纯空白指令文件不追加空 section。返回 (提示文本, 来源信息)。
    `tools`/`checks` 来自启动层实际注册的插件配置; 默认等于普通模式。
    `skills` 为启用 Skills 的冻结快照: 版本切换为 SKILLS_PROMPT_VERSION, 目录段
    由快照纯构造 (超限抛 SkillPromptError); None 时输出与未启用逐位一致。
    `instructions_mode` 是 C1 的仅信息关键字: 不改变提示文本/hash/版本, 只进
    PromptInfo 供显示; 矛盾组合 (有路径但 disabled/absent, 无路径但
    explicit/automatic) 在组装前拒绝。None 按旧约定推导 (有路径=explicit)。
    """
    mode = _resolve_instructions_mode(instructions_path, instructions_mode)
    prompt = build_builtin_prompt(workspace, tools, checks)
    version = PROMPT_VERSION
    if skills is not None:
        prompt += "\n" + catalog_prompt_section(skills)
        version = AUTO_SKILLS_PROMPT_VERSION if skills.roots else SKILLS_PROMPT_VERSION
    resolved: Path | None = None
    raw_hash: str | None = None
    sources = instruction_sources if instruction_sources is not None else (
        (("project", instructions_path),) if instructions_path is not None else ()
    )
    if instruction_sources is not None and (sources[-1][1] if sources else None) != instructions_path:
        raise ValueError("instruction sources do not match selected path")
    observed = []
    for scope, path in sources:
        text, source_path, source_hash = load_instructions(path, workspace)
        observed.append((scope, source_path, source_hash))
        if text.strip():
            label = "User" if scope == "user" else "Project"
            prompt += (
                f"\n--- {label} instructions (source: {source_path}, sha256:{source_hash}) ---\n"
                f"{text.rstrip()}\n"
            )
    if observed:
        resolved = observed[-1][1]
        raw_hash = observed[-1][2]
        if len(observed) > 1:
            payload = json.dumps(
                [(scope, str(path), digest) for scope, path, digest in observed],
                ensure_ascii=False, separators=(",", ":"),
            )
            raw_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    info = PromptInfo(
        version=version,
        workspace=workspace.resolve(),
        instructions_path=resolved,
        instructions_sha256=raw_hash,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        instructions_mode=mode,
        instruction_sources=tuple(observed),
    )
    return prompt, info
