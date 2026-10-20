"""CLI 私有启动准备：输入校验和配置转换，不打开应用或构造模型插件。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cicada.costs import CostError, CostLedger, CostProfile, positive_amount

from cicada.persistence.contracts import LaunchIdentity, PersistenceError, StoredIdentity
from cicada.plugins.deepseek import DeepSeekConfig
from cicada.plugins.deepseek.protocol import BASE_URL as DEEPSEEK_BASE_URL, REQUEST_LIMIT
from cicada.plugins.coding.verification_contracts import CheckDefinition
from cicada.plugins.session_store.store import validate_session_id
from cicada.skills import SKILLS_PROMPT_VERSION
from cicada.state_paths import workspace_state_dir
from cicada.user_config import (
    ResolvedLaunchConfig, UserConfigSnapshot, UserDefaults, resolve_launch_config,
)

# P4 §5/§8 检查模式的固定 profile 与输入边界
MAX_CHECK_COMMANDS = 8
CHECK_COMMAND_MAX_BYTES = 4096
CHECK_TIMEOUT_DEFAULT_S = 120.0
CHECK_TIMEOUT_MIN_S = 0.0
CHECK_TIMEOUT_MAX_S = 300.0
CHECK_MODE_MIN_NUM_CTX = 32768
CHECK_MODE_MAX_TURNS = 25
NORMAL_MODE_MAX_TURNS = 50

# 05: --tui --resume 不带 ID = 启动选择器 (哨兵; 单次模式明确拒绝)
_RESUME_PICK = object()


def parse_check_commands(
    commands: list[str] | None, timeout: float | None
) -> tuple[tuple[CheckDefinition, ...], str | None]:
    """解析 --check-command/--check-timeout; 返回 (检查定义, 错误消息).

    全部输入边界在此给出明确错误: 条数、空命令、命令字节上限、timeout 范围、
    无检查时使用 --check-timeout。
    """
    commands = list(commands or [])
    if not commands and timeout is not None:
        return (), "--check-timeout 只能与 --check-command 一起使用"
    if not commands:
        return (), None
    if len(commands) > MAX_CHECK_COMMANDS:
        return (), f"--check-command 最多 {MAX_CHECK_COMMANDS} 条, 收到 {len(commands)} 条"
    timeout_s = timeout if timeout is not None else CHECK_TIMEOUT_DEFAULT_S
    if not CHECK_TIMEOUT_MIN_S < timeout_s <= CHECK_TIMEOUT_MAX_S:
        return (), (
            f"--check-timeout 必须在 ({CHECK_TIMEOUT_MIN_S:g}, {CHECK_TIMEOUT_MAX_S:g}] 秒内, "
            f"收到 {timeout_s:g}"
        )
    definitions: list[CheckDefinition] = []
    for index, command in enumerate(commands, start=1):
        if not command or not command.strip():
            return (), f"--check-command 第 {index} 条为空命令"
        size = len(command.encode("utf-8"))
        if size > CHECK_COMMAND_MAX_BYTES:
            return (), (
                f"--check-command 第 {index} 条超过 {CHECK_COMMAND_MAX_BYTES} UTF-8 bytes "
                f"(收到 {size} bytes)"
            )
        definitions.append(CheckDefinition(check_id=f"check-{index}", command=command, timeout_s=timeout_s))
    return tuple(definitions), None


def validate_session_args(args: argparse.Namespace) -> str | None:
    """先于 TUI 导入、TTY 判断和配置读取，按既有顺序返回首个输入错误。"""
    if args.list_cursor is not None:
        return "--list-cursor 只能与 --list-sessions 一起使用"
    if args.save_session and args.resume is not None:
        return "--save-session 与 --resume 互斥: 保存新会话与恢复已有会话只能二选一"
    if args.resume_with_current_config and args.resume is None:
        return (
            "--resume-with-current-config 只能与 --resume 一起使用: "
            "显式采用当前配置仅对恢复有意义"
        )
    if args.resume is _RESUME_PICK and not args.tui:
        return (
            "--resume 不带会话 ID 仅在 --tui 下可用 (启动选择器); "
            "单次模式请给出精确的会话 ID"
        )
    if isinstance(args.resume, str):
        try:
            validate_session_id(args.resume)
        except PersistenceError as exc:
            return str(exc)
    if not args.tui and args.prompt is None and args.skill is None:
        return "缺少 prompt: 单次模式需要位置参数 prompt (交互入口用 --tui)"
    return None


def validate_model_args(
    args: argparse.Namespace,
) -> tuple[tuple[CheckDefinition, ...], str | None]:
    """TUI 环境检查后、任何工作区读取前的模型与检查旗标校验。"""
    if args.script is not None and any(
        getattr(args, name) is not None for name in (
            "model", "think", "num_ctx", "provider", "max_output_tokens", "reasoning_effort"
        )
    ):
        return (), "--script 与真实模型旗标互斥, 请只选一种模型来源"
    if args.num_ctx is not None and args.num_ctx <= 0:
        return (), f"--num-ctx 必须是正整数, 收到 {args.num_ctx}"
    checks, error = parse_check_commands(args.check_command, args.check_timeout)
    if error is not None:
        return (), error
    if checks and args.num_ctx is not None and args.num_ctx < CHECK_MODE_MIN_NUM_CTX:
        return (), f"检查模式要求 --num-ctx >= {CHECK_MODE_MIN_NUM_CTX}, 收到 {args.num_ctx}"
    return checks, None


def prepare_cost_control(args: argparse.Namespace, workspace: Path) -> CostLedger | None:
    """显式进程限额；金额上限与价格输入不从旧会话或用户默认值继承。"""
    amount = getattr(args, 'cost_limit_cny', None)
    calls = getattr(args, 'max_model_calls', None)
    profile_path = getattr(args, 'cost_profile', None)
    if amount is None and calls is None and profile_path is None:
        return None
    if args.script is not None:
        raise CostError('cost controls require a real model; --script is incompatible')
    if profile_path is not None and amount is None:
        raise CostError('--cost-profile requires --cost-limit-cny')
    if amount is not None and profile_path is None:
        raise CostError('--cost-limit-cny requires an explicit --cost-profile')
    profile = CostProfile.load(workspace / profile_path) if profile_path is not None else None
    return CostLedger(workspace_state_dir(workspace) / 'cost', limit=positive_amount(amount) if amount is not None else None,
                      max_calls=calls, profile=profile)


def prepare_restored_launch(
    stored: StoredIdentity, args: argparse.Namespace, checks: tuple[CheckDefinition, ...],
) -> LaunchIdentity | None:
    """普通恢复的继承身份；fake 无 DeepSeek 配置，存档读取和租约仍由入口负责。"""
    launch = stored.launch
    if launch.checks_signature is not None and not checks:
        raise ValueError(
            "存档为检查模式会话, 检查定义不保存在存档中, 无法省略继承; "
            "请显式给出与保存时相同的 --check-command, "
            "或用 --resume-with-current-config 以当前配置恢复"
        )
    if launch.prompt_version == SKILLS_PROMPT_VERSION and args.skills_dir is None:
        raise ValueError(
            "存档为 Skills 会话, --skills-dir 不从存档继承; "
            "请重传与保存时相同的 --skills-dir, "
            "或用 --resume-with-current-config 以当前配置恢复"
        )
    if launch.provider not in ("deepseek", "fake"):
        raise ValueError("存档 provider 已不受支持; 使用 --resume-with-current-config 显式采用当前配置")
    if launch.provider == "fake":
        return None
    options = launch.options
    if set(options) != {"num_ctx", "max_tokens", "reasoning_effort"}:
        raise ValueError("存档 DeepSeek options 白名单不匹配")
    if launch.base_url != DEEPSEEK_BASE_URL:
        raise ValueError("存档 DeepSeek endpoint 必须为官方 origin")
    DeepSeekConfig(
        model=launch.model, think=launch.think, num_ctx=options["num_ctx"],
        max_tokens=options["max_tokens"], reasoning_effort=options["reasoning_effort"],
        max_request_bytes=launch.max_request_bytes,
    )
    return launch


def resolve_deepseek_config(
    snapshot: UserConfigSnapshot, *, explicit: dict[str, object],
    archive: UserDefaults | None, saved_launch: LaunchIdentity | None,
) -> tuple[ResolvedLaunchConfig, DeepSeekConfig, dict[str, str]]:
    """初次准备与登录后重读共用相同配置转换，不读取凭据或创建插件。"""
    resolved = resolve_launch_config(snapshot, explicit=explicit, archive=archive)
    defaults = resolved.defaults
    config = DeepSeekConfig(
        model=defaults.model, think=defaults.think, num_ctx=defaults.num_ctx,
        max_tokens=defaults.max_output_tokens, reasoning_effort=defaults.reasoning_effort,
        max_request_bytes=saved_launch.max_request_bytes if saved_launch else REQUEST_LIMIT,
    )
    # 用户配置采用 max_output_tokens，会话身份沿用 max_tokens。
    sources = dict(resolved.sources)
    sources['max_tokens'] = sources.pop('max_output_tokens')
    return resolved, config, sources
