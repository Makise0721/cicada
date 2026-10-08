"""非交互入口: uv run python -m cicada --workspace <dir> [--script <script.json> | --model <name> --ollama-url <url> --think --num-ctx N] [--instructions-file <path>] [--check-command <PowerShell> ...] [--check-timeout <s>] "<prompt>"
交互入口: uv run python -m cicada --tui --workspace <dir> [同上模型/检查旗标] [首问]
持久会话 (05): --tui 默认自动保存新会话; --resume <ID> 恢复, --resume 不带 ID
在 --tui 下启动会话选择器 (只读列表, Esc 取消退出码 0); --list-sessions 列出
本工作区会话 (只读分页, 无需 prompt/TUI/模型探测)。恢复显示从只读快照重建
(历史回答/工具卡/旧检查归属/累计计量/中断标记), 不重放旧执行事件。

模型来源二选一:
- --script: fake model 剧本 (见 parse_script), 用于确定性回归;
- 缺省: 真实模型默认 DeepSeek Flash thinking; 显式 --provider ollama 保留本地 preflight。
--script 与真实模型旗标 (--model/--ollama-url/--think/--num-ctx) 互斥, 混用退出码 2.
两种模式均使用内置系统提示 (PROMPT_VERSION); --instructions-file 在其后追加项目指令,
文件错误 (缺失/超限/非 UTF-8) 先于 preflight 失败, 退出码 2.

--tui 进入交互终端 (生产组合: 共享应用 + Pi 主屏前端, 见 cicada.ui; 正常启动
安静, /status 提供有界可追溯诊断, 检查事实由 RequestOutcome 投影); UI 可选依赖
缺失或 stdin/stdout 非 TTY 退出码 2; 位置 prompt 可省略 (空启动), 给出则作为
首问提交; 进程退出码: 启动失败 2, UI 不可恢复/关闭错误优先 1, 正常关闭取最后
一条请求的退出码 (无请求 0). 单次模式要求位置 prompt; 普通 UI 导入要求为零 (T14).

持久恢复配置 (04): --resume 默认继承存档的模型/服务地址/think/窗口请求值
(省略参数取存档, 显式参数与存档比对, 冲突点名拒绝且零模型探测); 检查定义只存
签名, 检查模式存档恢复需显式给出相同 --check-command. --resume-with-current-config
与 --resume 组合, 显式采用当前默认与显式输入, 与存档的差异记入新启动代次;
两者都不换工作区、不解除 view-only/process_uncertain 恢复阻断.

P4 检查模式: 给出 --check-command (可重复, 最多 8 条) 即进入. 该模式要求 workspace
是 Git 工作树根; 启动时 initialize 建立基线快照, 失败退出码 2; 最多 25 轮; 真实模型
Ollama profile 固定 num_ctx>=32768、num_predict=2048、请求体上限 65536 bytes;
DeepSeek 普通/检查使用同一 max_tokens 与本地预留，摘要独立关闭 thinking。 运行结束后
程序独立扫描终结事实、刷新检查视图、生成变更证据并判定交付 (不读模型文字):
- 0: 模型 stop 且全部指定检查的最近回执 passed/current 且证据完整;
- 3: 模型 stop 但检查或证据条件不满足;
- 1: 模型 error/aborted/length; 2: 输入/启动错误 (普通模式保持 0=stop, 1/2 同上).

启动/执行/验证收尾/关闭编排统一走 cicada.application (单次 CLI 与 TUI 的共享
seam); 本模块保留参数解析、fake JSON 转换、preflight 与打印适配, 不复制检查
模式装配. 订阅适配忽略新增文本增量与阶段通知, 旧 CLI 打印语义保持.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

from cicada.application import (
    AgentEventReceived,
    ApplicationConfig,
    ApplicationOpenError,
    PersistenceConfig,
    RequestCompleted,
    RequestPhaseChanged,
    open_application,
    verification_checks_signature,
)
from cicada.budget import BudgetConfig
from cicada.core.events import (
    AssistantCompleted,
    Event,
    RunFinished,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.ports import StreamDone, StreamEvent, TextDelta, ToolCallEvent
from cicada.persistence.contracts import (
    LIST_SESSIONS_PAGE_SIZE,
    LaunchIdentity,
    PersistenceError,
    SessionPage,
    StoredIdentity,
    conflicting_launch_fields,
)
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.ollama import OllamaConfig, ollama_plugin
from cicada.plugins.deepseek import DeepSeekConfig, deepseek_plugin
from cicada.plugins.deepseek.protocol import BASE_URL as DEEPSEEK_BASE_URL, REQUEST_LIMIT
from cicada.plugins.session_store import list_sessions, read_session_identity
from cicada.plugins.session_store.store import validate_session_id
from cicada.plugins.skills import load_skill_catalog
from cicada.plugins.user_config import FileUserConfigStore
from cicada.user_config import (
    UserConfigError, UserConfigSnapshot, UserConfigStore, UserDefaults, resolve_launch_config,
)
from cicada.prompting import (
    CheckPrompt,
    DEFAULT_TOOLS,
    InstructionsError,
    PromptInfo,
    compose_system_prompt,
)
from cicada.reporting import build_delivery_section, build_run_summary
from cicada.skills import (
    SKILLS_PROMPT_VERSION,
    SkillCatalogError,
    SkillsError,
    build_skill_request,
    find_skill,
)

DEFAULT_CONFIG = OllamaConfig()
DEFAULT_NUM_CTX = 32768  # CLI 请求级默认值; 消除服务端默认 4K 依赖. 仅是请求值, 非实际分配保证

# P4 §5/§8 检查模式的固定 profile 与输入边界
MAX_CHECK_COMMANDS = 8
CHECK_COMMAND_MAX_BYTES = 4096
CHECK_TIMEOUT_DEFAULT_S = 120.0
CHECK_TIMEOUT_MIN_S = 0.0
CHECK_TIMEOUT_MAX_S = 300.0
CHECK_MODE_MIN_NUM_CTX = 32768
CHECK_MODE_NUM_PREDICT = 2048
CHECK_MODE_MAX_TURNS = 25
CHECK_MODE_MAX_REQUEST_BYTES = 65536
NORMAL_MODE_MAX_TURNS = 50

# 04: 存档 options 的已知键 (ollama 透传); 继承前拒绝未知键与非法值
KNOWN_OLLAMA_OPTION_KEYS = ("num_ctx", "num_predict")

# 05: --tui --resume 不带 ID = 启动选择器 (哨兵; 单次模式明确拒绝)
_RESUME_PICK = object()


async def _read_saved_identity(workspace: Path, session_id: str) -> StoredIdentity | None:
    """恢复前的存档身份预读; 失败返回 None (错误消息已打印, 调用方退出码 2).

    省略参数的继承需要存档值才能构造模型插件; 顺带把格式/workspace 校验提前到
    模型插件构造与可达性探测之前 (契约 §1 次序)。预读不迁移存档;
    写租约与原子迁移由 open_application 完成。
    """
    try:
        identity = read_session_identity(
            directory=Path(workspace) / ".cicada" / "sessions",
            session_id=session_id,
            workspace=Path(workspace),
        )
    except PersistenceError as exc:
        print(f"session store open failed: {exc}", file=sys.stderr)
        return None
    if identity is None:
        # 零 epoch 存档 (如保存进程在 bootstrap 前崩溃遗留): 与恢复准入同一拒绝
        print(
            "--resume: the session archive has no saved launch identity to compare against",
            file=sys.stderr,
        )
        return None
    return identity


def _validate_saved_options(options: dict[str, object]) -> str | None:
    """存档 options 键/值校验; 返回错误消息或 None (契约 §6: 未知配置键不执行)."""
    unknown = sorted(str(key) for key in options if key not in KNOWN_OLLAMA_OPTION_KEYS)
    if unknown:
        return (
            f"存档配置包含未知配置键: {', '.join(unknown)} "
            f"(当前程序仅支持: {', '.join(KNOWN_OLLAMA_OPTION_KEYS)})"
        )
    for key in KNOWN_OLLAMA_OPTION_KEYS:
        value = options.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return f"存档配置 {key} 必须是正整数, 存档记录 {value!r}"
    return None


_LIST_TITLE_MAX_CHARS = 30
_LIST_ERROR_MAX_CHARS = 80


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def render_session_page(page: SessionPage) -> list[str]:
    """--list-sessions 的有界输出行 (纯函数, 单测覆盖翻页与标注)。"""
    lines = [f"会话列表 共 {len(page.entries)} 条 (目录 {page.directory})"]
    if not page.entries:
        lines.append("（本工作区暂无会话存档）")
        return lines
    for entry in page.entries:
        updated = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.updated_at))
            if entry.updated_at is not None else "无记录"
        )
        line = (
            f"{entry.session_id}  {_clip(entry.short_title, _LIST_TITLE_MAX_CHARS)}  "
            f"更新={updated}  请求={entry.committed_requests} 中断={entry.interrupted_requests}"
        )
        if entry.occupied:
            line += " [占用]"
        if entry.blocked:
            line += (
                f" [阻断: 未证终结活动×{entry.unproven_activities};"
                f" 不可信回执×{entry.untrusted_receipts}]"
            )
        if entry.error is not None:
            line += f" [坏档: {_clip(entry.error, _LIST_ERROR_MAX_CHARS)}]"
        lines.append(line)
    if page.next_cursor is not None:
        lines.append(
            f"第 {LIST_SESSIONS_PAGE_SIZE} 条之后仍有会话, 下一页: "
            f"--list-cursor {page.next_cursor} (每页 {LIST_SESSIONS_PAGE_SIZE} 条)"
        )
    return lines


def _run_list_sessions(args: argparse.Namespace) -> int:
    """--list-sessions: 本 workspace 只读分页列表; 无 prompt/TUI/模型探测 (契约 §6)."""
    if (
        args.save_session or args.resume is not None or args.resume_with_current_config
        or args.tui or args.skill is not None
    ):
        print(
            "--list-sessions 是只读列表入口, 不能与 --save-session/--resume/--tui/--skill 组合",
            file=sys.stderr,
        )
        return 2
    if args.prompt is not None:
        print("--list-sessions 不执行请求, 不接受位置 prompt", file=sys.stderr)
        return 2
    directory = Path(args.workspace) / ".cicada" / "sessions"
    try:
        page = list_sessions(directory=directory, cursor=args.list_cursor)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    for line in render_session_page(page):
        print(line)
    return 0


def parse_script(raw: str) -> list[list[StreamEvent]]:
    """把剧本 JSON 转换为 FakeModel 剧本; 非法输入一律抛 ValueError."""
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"script is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError("script must be a JSON list of entries")
    script: list[list[StreamEvent]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"script entry {index} must be an object")
        events: list[StreamEvent] = []
        text = entry.get("text")
        if text:
            events.append(TextDelta(str(text)))
        tool_calls = entry.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise ValueError(f"script entry {index}: tool_calls must be a list")
        for call in tool_calls:
            if not isinstance(call, dict) or "id" not in call or "name" not in call:
                raise ValueError(
                    f"script entry {index}: each tool call must be an object with id and name"
                )
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"script entry {index}: tool call arguments must be an object"
                )
            events.append(ToolCallEvent(str(call["id"]), str(call["name"]), json.dumps(arguments)))
        if "error" in entry:
            events.append(StreamDone("error", str(entry["error"])))
        elif entry.get("length"):
            events.append(StreamDone("length"))
        elif entry.get("stop") or not tool_calls:
            events.append(StreamDone("stop"))
        else:
            events.append(StreamDone("tool_use"))
        script.append(events)
    return script


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


def print_event(event: Event) -> None:
    """旧 CLI 的内核事件打印; 新增事件类型 (文本增量等) 自然静默."""
    if isinstance(event, RunStarted):
        print(f"=== {event.run_id}")
    elif isinstance(event, TurnStarted):
        print(f"--- turn {event.turn_index}")
    elif isinstance(event, AssistantCompleted):
        if event.message.text:
            print(event.message.text)
    elif isinstance(event, ToolStarted):
        print(f">> {event.name} ({event.call_id})")
    elif isinstance(event, ToolCompleted):
        marker = " [error]" if event.result.is_error else ""
        print(f"<< {event.result.name} ({event.result.call_id}){marker}")
        if event.result.content:
            print(event.result.content)
    elif isinstance(event, RunFinished):
        suffix = f" ({event.error})" if event.error else ""
        print(f"=== finished: {event.stop_reason}{suffix}")


def _print_prompt_info(info: PromptInfo) -> None:
    """启动显示来源与 hash; 不打印提示全文."""
    line = f"[prompt version={info.version} workspace={info.workspace}"
    if info.instructions_path is not None:
        line += f" instructions={info.instructions_path} instructions_sha256={info.instructions_sha256}"
    line += f" prompt_sha256={info.prompt_sha256}]"
    print(line)


async def _ollama_reachable(base_url: str) -> bool:
    """preflight: GET /api/version (2s 超时); 连接失败或非 2xx 均视为不可达."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/version")
    except httpx.HTTPError:
        return False
    return response.is_success


async def _launch_session_selector(workspace: Path) -> str | None:
    """--tui 无 ID 恢复的选择器入口; 返回选中的 session_id (取消为 None)。

    独立成函数: 进程内测试可替换 (真实终端构造依赖 Windows 控制台)。
    """
    from cicada.ui.selector import run_session_selector
    from cicada.ui.terminal_io import PromptToolkitTerminal

    terminal = PromptToolkitTerminal()
    session_id, restore_errors = await run_session_selector(
        terminal, workspace / ".cicada" / "sessions"
    )
    if restore_errors:
        print("terminal restore errors: " + "; ".join(restore_errors), file=sys.stderr)
    return session_id


async def _run(args: argparse.Namespace, *, user_config_store: UserConfigStore | None = None) -> int:
    if args.list_sessions:
        return _run_list_sessions(args)
    if args.list_cursor is not None:
        # 审查 05-F6: 游标只属于列表入口; 误用时拒绝而非静默忽略并执行请求
        print("--list-cursor 只能与 --list-sessions 一起使用", file=sys.stderr)
        return 2
    if args.save_session and args.resume is not None:
        print("--save-session 与 --resume 互斥: 保存新会话与恢复已有会话只能二选一", file=sys.stderr)
        return 2
    if args.resume_with_current_config and args.resume is None:
        print(
            "--resume-with-current-config 只能与 --resume 一起使用: "
            "显式采用当前配置仅对恢复有意义",
            file=sys.stderr,
        )
        return 2
    if args.resume is _RESUME_PICK and not args.tui:
        print(
            "--resume 不带会话 ID 仅在 --tui 下可用 (启动选择器); "
            "单次模式请给出精确的会话 ID",
            file=sys.stderr,
        )
        return 2
    if isinstance(args.resume, str):
        # nargs="?" 会把紧随其后的词当 ID: 提前按 UUIDv4 校验, 错误不延迟到开库
        try:
            validate_session_id(args.resume)
        except PersistenceError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
    # 05: --tui 默认持久保存; --save-session 允许组合但不改变行为 (契约 §6)
    # 03: --skill 使位置 prompt 成为可选任务 (省略时用固定指令)
    if not args.tui and args.prompt is None and args.skill is None:
        print("缺少 prompt: 单次模式需要位置参数 prompt (交互入口用 --tui)", file=sys.stderr)
        return 2
    run_tui_session_fn = None
    if args.tui:
        # 依赖缺失先于 TTY 判断与配置组装, 给出明确退出 (T14)
        try:
            from cicada.ui import run_tui_session as run_tui_session_fn
        except ImportError as exc:
            print(
                f"TUI 依赖未安装: {exc} (请安装 UI 可选组, 如 pip install -e .[tui])",
                file=sys.stderr,
            )
            return 2
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            print(
                "TUI 需要交互式终端 (stdin/stdout 均为 TTY); 管道或重定向请用单次模式",
                file=sys.stderr,
            )
            return 2
    if args.script is not None and (
        args.model is not None
        or args.ollama_url is not None
        or args.think is not None
        or args.num_ctx is not None
        or args.provider is not None
        or args.max_output_tokens is not None
        or args.reasoning_effort is not None
    ):
        print(
            "--script 与真实模型旗标互斥, 请只选一种模型来源",
            file=sys.stderr,
        )
        return 2
    if args.num_ctx is not None and args.num_ctx <= 0:
        print(f"--num-ctx 必须是正整数, 收到 {args.num_ctx}", file=sys.stderr)
        return 2
    checks, check_error = parse_check_commands(args.check_command, args.check_timeout)
    if check_error is not None:
        print(check_error, file=sys.stderr)
        return 2
    if checks and args.num_ctx is not None and args.num_ctx < CHECK_MODE_MIN_NUM_CTX:
        print(
            f"检查模式要求 --num-ctx >= {CHECK_MODE_MIN_NUM_CTX}, 收到 {args.num_ctx}",
            file=sys.stderr,
        )
        return 2
    workspace = Path(args.workspace)
    # 指令文件错误先于 preflight/bootstrap (P3 §4.2); Skills 目录错误同为启动输入
    # 错误, 在存档身份预读与 provider 探测之前 (D8)
    skills_snapshot = None
    if args.skills_dir is not None:
        try:
            skills_snapshot = load_skill_catalog(args.skills_dir, workspace)
        except SkillCatalogError as exc:
            print(f"invalid skills directory: {exc}", file=sys.stderr)
            return 2
    if args.skill is not None:
        # 03 (D8): --skill 需要 --skills-dir; 名称/限额错误在 provider 探测与
        # 应用 open 之前退出码 2。位置 prompt 是任务, 省略时 helper 填固定指令
        if skills_snapshot is None:
            print(
                "--skill 需要 --skills-dir: 显式选择从启用目录的冻结快照取正文",
                file=sys.stderr,
            )
            return 2
        try:
            args.prompt = build_skill_request(
                find_skill(skills_snapshot, args.skill, allow_manual=True),
                args.prompt or "",
            )
        except SkillsError as exc:
            print(f"invalid --skill: {exc}", file=sys.stderr)
            return 2
    check_prompts = tuple(
        CheckPrompt(check.check_id, check.command, check.timeout_s) for check in checks
    )
    prompt_tools = (DEFAULT_TOOLS + ("check",)) if checks else DEFAULT_TOOLS
    if skills_snapshot is not None:
        prompt_tools = prompt_tools + ("load_skill",)
    try:
        prompt, prompt_info = compose_system_prompt(
            workspace,
            args.instructions_file,
            tools=prompt_tools,
            checks=check_prompts,
            skills=skills_snapshot,
        )
    except InstructionsError as exc:
        print(f"invalid instructions file: {exc}", file=sys.stderr)
        return 2
    except SkillsError as exc:
        print(f"invalid skills input: {exc}", file=sys.stderr)
        return 2
    if skills_snapshot is not None and not args.tui:
        print(
            f"[skills root={skills_snapshot.root} skills={len(skills_snapshot.documents)} "
            f"manifest_sha256={skills_snapshot.manifest_sha256[:12]}]"
        )
        if skills_snapshot.diagnostics:
            for diagnostic in skills_snapshot.diagnostics:
                print(f"[skills diagnostic: {diagnostic}]")
    if not args.tui:  # TUI 静默: prompt 身份进 /status (P12)
        _print_prompt_info(prompt_info)
    plan: VerificationPlan | None = None
    if checks:
        root = Path(os.path.realpath(Path(workspace).expanduser()))
        plan = VerificationPlan(
            verification_run_id=generate_verification_run_id(), root=root, checks=checks
        )
        if not args.tui:
            timeout_s = checks[0].timeout_s
            print(
                f"[check_mode checks={len(checks)} "
                f"ids={','.join(check.check_id for check in checks)} "
                f"timeout={timeout_s:g}s verification_run_id={plan.verification_run_id} "
                f"max_turns={CHECK_MODE_MAX_TURNS}]"
            )
    max_turns = CHECK_MODE_MAX_TURNS if checks else NORMAL_MODE_MAX_TURNS
    # 05: --tui --resume 不带 ID = 启动选择器 (选择期间无模型/工具活动; Esc 退出码 0)。
    # 选定后回写 args.resume, 使下方继承预读与按 ID 路径完全同路 (审查 05-H1);
    # 终端构造/启动失败为启动环境错误, 单行 stderr + 退出码 2 (审查 05-F5)
    if args.tui and args.resume is _RESUME_PICK:
        try:
            picked = await _launch_session_selector(Path(args.workspace))
        except Exception as exc:
            print(f"session selector failed: {exc!r}", file=sys.stderr)
            return 2
        if picked is None:
            return 0
        args.resume = picked
    # 04: 默认恢复预读存档身份——省略参数继承存档, 显式参数与存档比对;
    # 检查定义只存签名不存命令, 检查模式存档遗漏 --check-command 时明确拒绝
    saved_launch: LaunchIdentity | None = None
    if isinstance(args.resume, str) and args.resume_with_current_config:
        # 采用当前配置仍须先核对存档/workspace；不能由缺key掩盖坏档。
        if await _read_saved_identity(workspace, args.resume) is None:
            return 2
    if isinstance(args.resume, str) and not args.resume_with_current_config:
        saved_stored = await _read_saved_identity(workspace, args.resume)
        if saved_stored is None:
            return 2
        if saved_stored.launch.checks_signature is not None and not checks:
            print(
                "--resume: 存档为检查模式会话, 检查定义不保存在存档中, 无法省略继承; "
                "请显式给出与保存时相同的 --check-command, "
                "或用 --resume-with-current-config 以当前配置恢复",
                file=sys.stderr,
            )
            return 2
        if saved_stored.launch.prompt_version == SKILLS_PROMPT_VERSION and args.skills_dir is None:
            print(
                "--resume: 存档为 Skills 会话, --skills-dir 不从存档继承; "
                "请重传与保存时相同的 --skills-dir, "
                "或用 --resume-with-current-config 以当前配置恢复",
                file=sys.stderr,
            )
            return 2
        if saved_stored.launch.provider not in ("ollama", "deepseek", "fake"):
            print("--resume: 存档 provider 不受支持", file=sys.stderr)
            return 2
        if saved_stored.launch.provider == "deepseek":
            saved_launch = saved_stored.launch
            options = saved_launch.options
            try:
                if set(options) != {"num_ctx", "max_tokens", "reasoning_effort"}:
                    raise ValueError("存档 DeepSeek options 白名单不匹配")
                if saved_launch.base_url != DEEPSEEK_BASE_URL:
                    raise ValueError("存档 DeepSeek endpoint 必须为官方 origin")
                DeepSeekConfig(model=saved_launch.model, think=saved_launch.think,
                    num_ctx=options["num_ctx"], max_tokens=options["max_tokens"],
                    reasoning_effort=options["reasoning_effort"],
                    max_request_bytes=saved_launch.max_request_bytes)
            except ValueError as exc:
                print(f"--resume: {exc}", file=sys.stderr)
                return 2
        elif saved_stored.launch.provider == "ollama":
            saved_launch = saved_stored.launch
            option_error = _validate_saved_options(dict(saved_launch.options))
            if option_error is not None:
                print(f"--resume: {option_error}", file=sys.stderr)
                return 2
    # 01: 预算配置只由 ollama 路径实填; fake 默认 None = 预算层不安装 (D9)
    budget_config: BudgetConfig | None = None
    if args.script is not None:
        try:
            script = parse_script(Path(args.script).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"invalid script: {exc}", file=sys.stderr)
            return 2
        model_plugin = fake_model_plugin(FakeModel(script))
        model_label = "fake"
        identity = LaunchIdentity(
            provider="fake", model=model_label, base_url=None, think=None,
            options={}, max_request_bytes=None, max_turns=max_turns,
            prompt_version=prompt_info.version, prompt_sha256=prompt_info.prompt_sha256,
            instructions_sha256=prompt_info.instructions_sha256,
            checks_signature=(
                verification_checks_signature(plan.checks) if plan is not None else None
            ),
            # 04: 后继剧本是本次启动的显式输入; 来源记录进 epoch 身份 (不参与比对,
            # 不当作 live 模型行为或自动执行历史的授权)
            sources={"script": str(Path(args.script))},
        )
        preflight = None
    else:
        provider = args.provider or (saved_launch.provider if saved_launch is not None else "deepseek")
        if saved_launch is not None and provider != saved_launch.provider:
            print("--resume: provider 与存档不匹配; 使用 --resume-with-current-config 显式采用当前配置", file=sys.stderr)
            return 2
        if provider == "deepseek":
            if args.ollama_url is not None:
                print("--ollama-url 只对 --provider ollama 合法", file=sys.stderr)
                return 2
            saved_options = saved_launch.options if saved_launch is not None else {}
            explicit = {name: getattr(args, name) for name in (
                'provider', 'model', 'think', 'reasoning_effort', 'num_ctx', 'max_output_tokens')}
            archive = (UserDefaults(model=saved_launch.model, think=saved_launch.think,
                num_ctx=saved_options['num_ctx'], max_output_tokens=saved_options['max_tokens'],
                reasoning_effort=saved_options['reasoning_effort']) if saved_launch else None)
            if saved_launch is not None:
                # 与应用打开时的完整核对并存；在读取凭据及02向导前拒绝显式冲突。
                provisional = resolve_launch_config(
                    UserConfigSnapshot(UserDefaults(), frozenset(), None, 'missing', ''),
                    explicit=explicit, archive=archive).defaults
                from dataclasses import replace

                proposed = replace(saved_launch, model=provisional.model, think=provisional.think,
                    options={'num_ctx': provisional.num_ctx, 'max_tokens': provisional.max_output_tokens,
                             'reasoning_effort': provisional.reasoning_effort}, max_turns=max_turns,
                    prompt_version=prompt_info.version, prompt_sha256=prompt_info.prompt_sha256,
                    instructions_sha256=prompt_info.instructions_sha256,
                    checks_signature=verification_checks_signature(plan.checks) if plan else None)
                conflicts = conflicting_launch_fields(saved_launch, proposed)
                if conflicts:
                    print('--resume: config conflicts: ' + ', '.join(conflicts), file=sys.stderr)
                    return 2
            try:
                store = user_config_store if user_config_store is not None else FileUserConfigStore(Path.home() / '.cicada')
                snapshot = store.load(os.environ.get('DEEPSEEK_API_KEY'))
                resolved = resolve_launch_config(snapshot, explicit=explicit, archive=archive)
                defaults = resolved.defaults
                sources = dict(resolved.sources)
                # 会话options沿用既有max_tokens命名；用户文件采用max_output_tokens。
                sources['max_tokens'] = sources.pop('max_output_tokens')
                config = DeepSeekConfig(
                    model=defaults.model, think=defaults.think, num_ctx=defaults.num_ctx,
                    max_tokens=defaults.max_output_tokens, reasoning_effort=defaults.reasoning_effort,
                    max_request_bytes=saved_launch.max_request_bytes if saved_launch else REQUEST_LIMIT,
                )
            except UserConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except ValueError as exc:
                print(f"invalid DeepSeek config: {exc}", file=sys.stderr)
                return 2
            if checks and config.num_ctx < CHECK_MODE_MIN_NUM_CTX:
                print(f'检查模式要求有效 num_ctx >= {CHECK_MODE_MIN_NUM_CTX}', file=sys.stderr)
                return 2
            if resolved.credential is None:
                print('DEEPSEEK_API_KEY is missing: 请设置 DEEPSEEK_API_KEY 或配置引用的文件凭据后重试；'
                      '交互入口为 --tui', file=sys.stderr)
                return 2
            safe = config.identity
            identity = LaunchIdentity(
                provider=safe.provider, model=safe.model, base_url=safe.base_url, think=safe.think,
                options=dict(safe.options), max_request_bytes=safe.max_request_bytes, max_turns=max_turns,
                prompt_version=prompt_info.version, prompt_sha256=prompt_info.prompt_sha256,
                instructions_sha256=prompt_info.instructions_sha256,
                checks_signature=verification_checks_signature(plan.checks) if plan is not None else None,
                sources=sources,
            )
            budget_config = BudgetConfig(config.num_ctx, config.max_tokens)
            model_plugin = deepseek_plugin(config, api_key=resolved.credential.reveal())
            model_label = config.model
            preflight = None
            if not args.tui:
                print(f"[config provider=deepseek model={config.model} context_requested={config.num_ctx} "
                      f"source={sources['num_ctx']} think={str(config.think).lower()} "
                      f"max_tokens={config.max_tokens} reasoning_effort={config.reasoning_effort}]")
        else:
            if args.max_output_tokens is not None or args.reasoning_effort is not None:
                print("--max-output-tokens/--reasoning-effort 只对 DeepSeek 合法", file=sys.stderr)
                return 2
            # 04: 省略的模型/服务地址/think/窗口继承存档 (来源记 archive); 显式值保持
            # 原语义, 与存档的差异由恢复准入点名拒绝
            sources: dict[str, str] = {}
            eff_model = args.model or DEFAULT_CONFIG.model
            eff_base_url = args.ollama_url or DEFAULT_CONFIG.base_url
            eff_think: bool | None = bool(args.think)
            num_ctx_source = "cli" if args.num_ctx is not None else "default"
            if args.num_ctx is not None:
                num_ctx = args.num_ctx
            elif saved_launch is not None and saved_launch.options.get("num_ctx") is not None:
                num_ctx = saved_launch.options["num_ctx"]
                num_ctx_source = "archive"
            else:
                num_ctx = DEFAULT_NUM_CTX
            if saved_launch is not None:
                if args.model is None:
                    eff_model = saved_launch.model
                    sources["model"] = "archive"
                if args.ollama_url is None and saved_launch.base_url is not None:
                    eff_base_url = saved_launch.base_url
                    sources["base_url"] = "archive"
                if args.think is None:
                    eff_think = saved_launch.think
                    sources["think"] = "archive"
            sources["num_ctx"] = num_ctx_source
            options: dict[str, object] = {"num_ctx": num_ctx}
            request_limit: int | None = None
            if checks:
                options["num_predict"] = CHECK_MODE_NUM_PREDICT
                request_limit = CHECK_MODE_MAX_REQUEST_BYTES
            # 01: 预算由有效 num_ctx 实填 (D9, 含 archive 继承来源); 输出预留取配置的
            # num_predict (检查模式 2048), 否则用 BudgetConfig 缺省。fake 路径保持 None
            num_predict = options.get("num_predict")
            budget_config = BudgetConfig(
                num_ctx=num_ctx,
                num_predict=(
                    num_predict
                    if isinstance(num_predict, int) and not isinstance(num_predict, bool) and num_predict > 0
                    else None
                ),
            )
            config = OllamaConfig(
                base_url=eff_base_url,
                model=eff_model,
                think=eff_think,
                options=options,
                max_request_bytes=request_limit,
            )
            if not args.tui:
                policy_line = (
                    f" num_predict={CHECK_MODE_NUM_PREDICT} request_bytes_limit={CHECK_MODE_MAX_REQUEST_BYTES}"
                    if checks else ""
                )
                print(
                    f"[config model={config.model} context_requested={num_ctx} source={num_ctx_source} "
                    f"think={str(config.think).lower()}{policy_line} workspace={prompt_info.workspace}]"
                )
            identity = LaunchIdentity(
                provider="ollama", model=config.model, base_url=config.base_url, think=config.think,
                options=dict(config.options), max_request_bytes=config.max_request_bytes,
                max_turns=max_turns,
                prompt_version=prompt_info.version, prompt_sha256=prompt_info.prompt_sha256,
                instructions_sha256=prompt_info.instructions_sha256,
                checks_signature=(
                    verification_checks_signature(plan.checks) if plan is not None else None
                ),
                sources=sources,
            )
            if args.save_session or args.resume is not None or args.tui:
                # 持久路径 (TUI 默认持久): 可达性探测在存档格式/配置预检之后执行 (契约 §1 次序)
                async def preflight() -> None:
                    if not await _ollama_reachable(config.base_url):
                        raise ApplicationOpenError(
                            f"Ollama 不可达: {config.base_url} "
                            "(请先启动 Ollama, 或用 --script 走 fake model)"
                        )
            else:
                preflight = None
                if not await _ollama_reachable(config.base_url):
                    print(
                        f"Ollama 不可达: {config.base_url} (请先启动 Ollama, 或用 --script 走 fake model)",
                        file=sys.stderr,
                    )
                    return 2
            model_plugin = ollama_plugin(config)
            model_label = config.model

    resume_id = args.resume if isinstance(args.resume, str) else None
    if args.tui or args.save_session or args.resume is not None:
        persistence_config = PersistenceConfig(
            session_id=resume_id,
            identity=identity,
            adopt_current_config=args.resume_with_current_config,
        )
    else:
        persistence_config = None

    app_config = ApplicationConfig(
        workspace=workspace,
        system_prompt=prompt,
        prompt_info=prompt_info,
        model_plugin=model_plugin,
        plan=plan,
        max_turns=max_turns,
        model_label=model_label,
        persistence=persistence_config,
        budget=budget_config,
        skills=skills_snapshot,
    )
    if args.tui:
        assert run_tui_session_fn is not None
        return await _run_tui(
            run_tui_session_fn, app_config, initial_prompt=args.prompt, preflight=preflight
        )
    try:
        session = await open_application(app_config, preflight=preflight)
    except ApplicationOpenError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    if session.persistence is not None:
        state = session.persistence
        print(
            f"[session id={state.session_id} mode={state.mode} "
            f"restored_requests={state.restored_request_count}]"
        )
        if session.restored_interrupted:
            print(f"[session interrupted_requests={len(session.restored_interrupted)}]")
        if session.view_only_reason is not None:
            # A15: 未证终结活动的恢复仅可查看; CLI 新 prompt 零执行并退出码 3 (契约 §6)
            print(f"[session view-only: {session.view_only_reason}]", file=sys.stderr)
            await session.aclose()
            return 3

    def on_update(update) -> None:
        # 旧 CLI 打印适配: 只转发内核事件; 忽略文本增量与阶段/完成通知, 避免重复输出
        if isinstance(update, AgentEventReceived):
            print_event(update.event)
        elif isinstance(update, (RequestPhaseChanged, RequestCompleted)):
            return

    unsubscribe = session.subscribe(on_update)
    try:
        try:
            outcome = await session.execute(args.prompt)
        except ValueError as exc:
            print(f"invalid prompt: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            # 持久路径: 开始记录失败/存储锁存等拒绝; 请求可能已消耗 (退出码 1)
            print(f"{exc}", file=sys.stderr)
            return 1
    finally:
        unsubscribe()
        await session.aclose()
    print(build_run_summary(outcome.result))
    if outcome.decision is not None:
        print(build_delivery_section(outcome.decision, outcome.view, outcome.evidence, plan))
    if outcome.persistence is not None:
        facts = outcome.persistence
        if facts.saved:
            print(f"[session saved request_id={facts.request_id} seq={facts.seq}]")
        else:
            print(f"[session save failed: {facts.error}]", file=sys.stderr)
    return outcome.exit_code


async def _run_tui(
    run_tui_session_fn, app_config: ApplicationConfig, *, initial_prompt: str | None,
    preflight: Callable[[], object] | None = None,
) -> int:
    """--tui 分支: 依赖与 TTY 已在 _run 校验; 打开共享应用并运行生产组合."""
    try:
        session = await open_application(app_config, preflight=preflight)
    except ApplicationOpenError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    # /status 的只读诊断: CLI 组装不可变快照后注入, UI 不读取 provider 闭包
    from functools import partial

    from cicada.ui.diagnostics import UIDiagnostics, diagnostics_lines

    plan = app_config.plan
    info = app_config.prompt_info
    diagnostics = UIDiagnostics(
        workspace=str(app_config.workspace),
        model_label=app_config.model_label or None,
        prompt_version=info.version,
        prompt_sha256=info.prompt_sha256,
        instructions_path=(
            str(info.instructions_path) if info.instructions_path is not None else None
        ),
        instructions_sha256=info.instructions_sha256,
        verification_run_id=plan.verification_run_id if plan is not None else None,
        check_ids=tuple(check.check_id for check in plan.checks) if plan is not None else (),
        check_timeout_s=plan.checks[0].timeout_s if plan is not None else None,
        max_turns=app_config.max_turns,
    )
    # 恢复阻断: UI 可查看 (新请求被应用拒绝); 无新请求关闭后退出码 3 (契约 §6),
    # 关闭错误/不可恢复失败优先 1
    view_only = session.view_only_reason is not None
    code = await run_tui_session_fn(
        session, initial_prompt=initial_prompt, diagnostics=partial(diagnostics_lines, diagnostics)
    )
    if code == 1:
        return 1
    return 3 if view_only else code


def main(*, user_config_store: UserConfigStore | None = None) -> int:
    # 与 powershell 工具一致: 入口输出统一 UTF-8, 不随控制台代码页变化
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="cicada", description="Cicada coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''DeepSeek 用户默认配置：读取当前用户 ~/.cicada/config.toml，不会读取工作区配置。
新建会话及 --resume-with-current-config 每字段按 CLI > 用户文件 > 内置默认。
普通 --resume 继承存档模型配置；凭据每次启动按有效 DEEPSEEK_API_KEY > 文件引用。
help、list、fake 与显式 Ollama 不读取这些文件。文件存在但无效时明确失败。

config.toml 示例（普通字段均可省略；不接受 API key 或 max_request_bytes）：
version = 1
[defaults]
provider = "deepseek"
model = "deepseek-flash"
think = true
reasoning_effort = "high"
num_ctx = 131072
max_output_tokens = 32768

文件凭据：config.toml 可添加 [credentials.deepseek] 和 ref = "<UUIDv4>"。
credentials.json 格式为 {"version":1,"entries":{"<同一UUIDv4>":
{"provider":"deepseek","api_key":"<key>"}}}。仅按 ref 选择，不搜索孤立条目。
两个文件须为 UTF-8，各不超过 64 KiB；key 去首尾空白后须非空、无内部空白或
控制字符，最多 4096 UTF-8 bytes。当前入口只读这些文件。''',
    )
    parser.add_argument("--workspace", default=".", help="工作区根目录 (默认当前目录)")
    parser.add_argument("--script", help="fake model 剧本 JSON 文件; 与真实模型旗标互斥")
    parser.add_argument("--provider", choices=("deepseek", "ollama"), help="真实 provider (新会话默认 deepseek; 恢复继承存档)")
    parser.add_argument("--model", help="DeepSeek: deepseek-flash (默认)/deepseek-v4-pro; Ollama: 本地模型名")
    parser.add_argument("--ollama-url", help=f"Ollama 服务地址 (默认 {DEFAULT_CONFIG.base_url})")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="开启/关闭思考 (DeepSeek 默认开启; Ollama 默认关闭; 思考不展示)")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), help="DeepSeek 思考强度 (默认 high)")
    parser.add_argument("--max-output-tokens", type=int, help="DeepSeek 思考+答案总输出上限 (默认 32768)")
    parser.add_argument(
        "--num-ctx", type=int, help="本地窗口上限 (DeepSeek 默认 131072; Ollama 默认 32768)"
    )
    parser.add_argument(
        "--instructions-file", help="追加到系统提示的项目指令文件 (≤16 KiB UTF-8; 两种模式可用)"
    )
    parser.add_argument(
        "--skills-dir",
        help=(
            "启用本地 Skills 的目录 (相对 workspace; 直接子目录各含一份 SKILL.md)。"
            "恢复会话不从存档继承, 须显式重传同一目录"
        ),
    )
    parser.add_argument(
        "--skill",
        help=(
            "显式发送一个 Skill 的正文作为首条请求 (名称须在 --skills-dir 目录中; "
            "位置 prompt 作为任务, 省略时用固定指令; 手动专用 Skill 也允许显式选择)"
        ),
    )
    parser.add_argument(
        "--check-command", action="append", default=None, metavar="POWERSHELL",
        help=(
            f"指定检查命令 (可重复, 最多 {MAX_CHECK_COMMANDS} 条, 每条 ≤{CHECK_COMMAND_MAX_BYTES} bytes); "
            "给出即进入检查模式, 按顺序分配 check-1..check-N"
        ),
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="进入交互终端 (连续对话; 位置 prompt 可省略, 给出则作为首问)",
    )
    parser.add_argument(
        "--check-timeout", type=float, default=None, metavar="SECONDS",
        help=(
            f"检查命令超时秒数 (默认 {CHECK_TIMEOUT_DEFAULT_S:g}; 范围 "
            f"({CHECK_TIMEOUT_MIN_S:g}, {CHECK_TIMEOUT_MAX_S:g}]; 仅检查模式)"
        ),
    )
    parser.add_argument(
        "--save-session", action="store_true",
        help="单次模式创建持久会话并保存本次请求 (默认不落盘); 检查模式可组合; --tui 已默认保存, 该旗标组合允许但不改变行为",
    )
    parser.add_argument(
        "--resume", nargs="?", const=_RESUME_PICK, default=None, metavar="SESSION_ID",
        help=(
            "按 UUIDv4 会话 ID 恢复持久会话, 新请求追加保存; 与 --save-session 互斥。"
            "--tui 下可省略 ID: 启动选择器 (Esc 退出)"
        ),
    )
    parser.add_argument(
        "--resume-with-current-config", action="store_true",
        help=(
            "与 --resume 一起使用: 显式采用当前默认与显式参数恢复 (省略项取当前默认而非存档), "
            "与存档的差异记入新启动代次; 不换工作区、不解除恢复阻断"
        ),
    )
    parser.add_argument(
        "--list-sessions", action="store_true",
        help=(
            f"列出本工作区的持久会话 (只读分页, 每页 {LIST_SESSIONS_PAGE_SIZE} 条; "
            "无需 prompt/TUI/模型探测; 不能与 --save-session/--resume/--tui 组合)"
        ),
    )
    parser.add_argument(
        "--list-cursor", default=None, metavar="CURSOR",
        help="--list-sessions 的翻页游标 (上一页输出末行给出)",
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="交给 agent 的 prompt (单次模式必填; --tui 可作首问)",
    )
    return asyncio.run(_run(parser.parse_args(), user_config_store=user_config_store))


if __name__ == "__main__":
    raise SystemExit(main())
