"""非交互入口: uv run python -m cicada --workspace <dir> [--script <script.json> | --model <name> --think --num-ctx N] [--instructions-file <path>] [--check-command <PowerShell> ...] [--check-timeout <s>] "<prompt>"
交互入口: uv run python -m cicada --tui --workspace <dir> [同上模型/检查旗标] [首问]
持久会话 (05): --tui 默认自动保存新会话; --resume <ID> 恢复, --resume 不带 ID
在 --tui 下启动会话选择器 (只读列表, Esc 取消退出码 0); --list-sessions 列出
本工作区会话 (只读分页, 无需 prompt/TUI/模型探测)。恢复显示从只读快照重建
(历史回答/工具卡/旧检查归属/累计计量/中断标记), 不重放旧执行事件。

模型来源二选一:
- --script: fake model 剧本 (见 parse_script), 用于确定性回归;
- 缺省: 真实模型默认 DeepSeek Flash thinking; 可用 --model 选择 API 模型。
--script 与真实模型旗标 (--model/--think/--num-ctx) 互斥, 混用退出码 2.
两种模式均使用内置系统提示 (PROMPT_VERSION); --instructions-file 在其后追加项目指令,
文件错误 (缺失/超限/非 UTF-8) 先于 preflight 失败, 退出码 2. 无显式文件时自动选择
workspace 根目录 AGENTS.md (C1); --no-auto-instructions 关闭自动选择, 自动候选
不可用同样先于 preflight 报错。指令来源 mode 仅供显示, 不进入提示身份。

--tui 进入交互终端 (生产组合: 共享应用 + Pi 主屏前端, 见 cicada.ui; 正常启动
安静, 检查事实由 RequestOutcome 投影); UI 可选依赖
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
窗口至少 32768;
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
from dataclasses import replace
from pathlib import Path
from typing import Callable

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
from cicada.cli_startup import (
    CHECK_COMMAND_MAX_BYTES,
    CHECK_MODE_MAX_TURNS,
    CHECK_MODE_MIN_NUM_CTX,
    CHECK_TIMEOUT_DEFAULT_S,
    CHECK_TIMEOUT_MAX_S,
    CHECK_TIMEOUT_MIN_S,
    MAX_CHECK_COMMANDS,
    NORMAL_MODE_MAX_TURNS,
    _RESUME_PICK,
    parse_check_commands,
    prepare_cost_control,
    prepare_restored_launch,
    resolve_deepseek_config,
    validate_model_args,
    validate_session_args,
)
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
from cicada.costs import CostError
from cicada.diagnostic_export import ReportError, validate_report_path
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
    VerificationPlan,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.deepseek import deepseek_plugin
from cicada.plugins.session_store import list_sessions, read_session_identity
from cicada.plugins.skills import load_skill_catalog
from cicada.plugins.user_config import FileUserConfigStore
from cicada.user_config import (
    LoginResult, UserConfigError, UserConfigSnapshot, UserConfigStore,
    UserDefaults, resolve_launch_config,
)
from cicada.prompting import (
    CheckPrompt,
    DEFAULT_TOOLS,
    InstructionsError,
    PromptInfo,
    compose_system_prompt,
    select_instructions_file,
)
from cicada.reporting import build_delivery_section, build_run_summary
from cicada.skills import (
    SkillCatalogError,
    SkillsError,
    build_skill_request,
    find_skill,
)

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
            f"{entry.session_id}  {_clip(entry.user_name or entry.short_title or '未命名', _LIST_TITLE_MAX_CHARS)}  "
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
        or any(getattr(args, field, None) is not None for field in
               ('cost_limit_cny', 'cost_profile', 'max_model_calls', 'export_report'))
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


async def _launch_session_selector(workspace: Path) -> str | None:
    """--tui 无 ID 恢复的选择器入口; 返回选中的 session_id (取消为 None)。

    独立成函数: 进程内测试可替换 (真实终端构造依赖 Windows 控制台)。
    """
    from cicada.ui.selector import run_session_selector
    from cicada.ui.terminal_io import PromptToolkitTerminal

    terminal = PromptToolkitTerminal()
    from cicada.application import make_session_catalog_reader

    directory = workspace / ".cicada" / "sessions"
    reader = make_session_catalog_reader(directory=directory, workspace=str(workspace))
    try:
        session_id, restore_errors = await run_session_selector(
            terminal, directory, observe=reader.observe,
        )
    finally:
        await reader.aclose()
    if restore_errors:
        print("terminal restore errors: " + "; ".join(restore_errors), file=sys.stderr)
    return session_id


async def _launch_first_run(
    store: UserConfigStore, snapshot: UserConfigSnapshot, defaults: UserDefaults, *,
    explicit: dict[str, object], credentials_only: bool,
) -> LoginResult | None:
    """missing-only 接缝；安全视图重新观察变化时拒绝覆盖旧启动配置。"""
    from cicada.ui.login_wizard import run_startup_login
    from cicada.ui.terminal_io import PromptToolkitTerminal

    view = store.read_view()
    observed = UserDefaults(view.provider, view.model, view.think, view.reasoning_effort,
                            view.num_ctx, view.max_output_tokens)
    if (observed != replace(snapshot.defaults, credential_ref=None)
            or view.credential_present or view.credential_source != 'missing'):
        raise UserConfigError('config', 'configuration-changing')
    view = replace(view, provider=defaults.provider, model=defaults.model, think=defaults.think,
        reasoning_effort=defaults.reasoning_effort, num_ctx=defaults.num_ctx,
        max_output_tokens=defaults.max_output_tokens, credentials_only=credentials_only,
        effect=('只补凭据，本次仍继承存档' if credentials_only else
                '保存下次默认值；本次显式 CLI 参数仍优先'))
    result, errors = await run_startup_login(
        PromptToolkitTerminal(), store, view,
        save_defaults=snapshot.defaults if credentials_only else defaults, explicit=explicit)
    if errors:
        # 终端底层异常不经通用展示，以免包含秘密；保留恢复失败事实。
        print('login terminal restore errors', file=sys.stderr)
        return LoginResult(bool(result and result.defaults_saved), False,
                           'not-requested', 'terminal-failed')
    return result


async def _run(args: argparse.Namespace, *, user_config_store: UserConfigStore | None = None) -> int:
    if args.list_sessions:
        return _run_list_sessions(args)
    argument_error = validate_session_args(args)
    if argument_error is not None:
        print(argument_error, file=sys.stderr)
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
    checks, argument_error = validate_model_args(args)
    if argument_error is not None:
        print(argument_error, file=sys.stderr)
        return 2
    workspace = Path(args.workspace)
    if not workspace.expanduser().is_dir():
        print('workspace 必须为存在的目录', file=sys.stderr)
        return 2
    try:
        cost_control = prepare_cost_control(args, workspace.expanduser().resolve())
        if getattr(args, 'export_report', None) is not None:
            if args.tui:
                raise ReportError('--export-report is for single requests; use /export in the TUI')
            validate_report_path(workspace.expanduser().resolve() / Path(args.export_report).expanduser())
    except (CostError, ReportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
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
        # C1: 启动层来源选择 (显式 > --no-auto-instructions > 根目录 AGENTS.md);
        # 自动候选的 reparse/目录/不可读在此报错, 正文仍由组装链有界读取一次
        selection = select_instructions_file(
            workspace,
            explicit_path=args.instructions_file,
            auto_enabled=not args.no_auto_instructions,
        )
        prompt, prompt_info = compose_system_prompt(
            workspace,
            selection.path,
            tools=prompt_tools,
            checks=check_prompts,
            skills=skills_snapshot,
            instructions_mode=selection.mode,
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
    if not args.tui:  # TUI 静默: 启动身份不打印 (P12; 原 /status 页已删除)
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
        try:
            saved_launch = prepare_restored_launch(saved_stored, args, checks)
        except ValueError as exc:
            print(f"--resume: {exc}", file=sys.stderr)
            return 2
    # API 路径启用预算; fake 默认不安装预算层。
    budget_config: BudgetConfig | None = None
    defaults_saved = False
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
            resolved, config, sources = resolve_deepseek_config(
                snapshot, explicit=explicit, archive=archive, saved_launch=saved_launch,
            )
            defaults = resolved.defaults
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
            if not args.tui:
                print('DEEPSEEK_API_KEY is missing: 请设置 DEEPSEEK_API_KEY 或配置引用的文件凭据后重试；'
                      '交互入口为 --tui', file=sys.stderr)
                return 2
            if checks:
                # 沿生产只读 inventory 核对 Git 根，不创建 Workspace 输出目录。
                from cicada.core.cancel import CancelToken
                from cicada.plugins.coding.inventory import GitInventory, InventoryError
                from cicada.plugins.coding.workspace import Workspace

                root = Path(os.path.realpath(workspace.expanduser()))
                try:
                    await GitInventory(Workspace(root, root / '.cicada' / 'outputs')).list_files(CancelToken())
                except InventoryError:
                    print('检查模式工作区 Git 预检失败', file=sys.stderr)
                    return 2
            try:
                result = await _launch_first_run(store, snapshot, defaults,
                    explicit=explicit, credentials_only=archive is not None)
                if result is None:
                    return 0
                defaults_saved = result.defaults_saved
                if not defaults_saved or result.error_code is not None:
                    print(('用户默认值已保存；' if defaults_saved else '') +
                          'login failed: ' + (result.error_code or 'save-failed'), file=sys.stderr)
                    return 2
                snapshot = store.load(os.environ.get('DEEPSEEK_API_KEY'))
                resolved, config, sources = resolve_deepseek_config(
                    snapshot, explicit=explicit, archive=archive, saved_launch=saved_launch,
                )
                defaults = resolved.defaults
                if resolved.credential is None:
                    raise UserConfigError('credential', 'missing-credential')
            except UserConfigError as exc:
                print(('用户默认值已保存；' if defaults_saved else '') + str(exc), file=sys.stderr)
                return 2
            except Exception:
                print(('用户默认值已保存；' if defaults_saved else '') +
                      'login startup failed', file=sys.stderr)
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
        restore_capture=not args.no_restore_capture,
        cost_control=cost_control,
    )
    if args.tui:
        assert run_tui_session_fn is not None
        return await _run_tui(
            run_tui_session_fn, app_config, initial_prompt=args.prompt, preflight=preflight,
            defaults_saved=defaults_saved,
            login_store_factory=lambda: user_config_store if user_config_store is not None else FileUserConfigStore(Path.home() / '.cicada'),
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
        # C3: 启动机会式清理的告警事实; 清理失败只留痕, 不阻断启动
        for warning in session.restore_warnings:
            print(f"[restore maintenance: {warning}]", file=sys.stderr)
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
    if session.cost is not None:
        from cicada.costs import cost_lines
        print('\n'.join(cost_lines(session.cost)))
    if getattr(args, 'export_report', None) is not None:
        try:
            destination = session.export_diagnostics(Path(args.export_report))
            print(f'[report exported: {destination}]')
        except ReportError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return outcome.exit_code


async def _run_tui(
    run_tui_session_fn, app_config: ApplicationConfig, *, initial_prompt: str | None,
    preflight: Callable[[], object] | None = None,
    defaults_saved: bool = False,
    login_store_factory: Callable[[], UserConfigStore] | None = None,
) -> int:
    """--tui 分支: 依赖与 TTY 已在 _run 校验; 打开共享应用并运行生产组合."""
    try:
        session = await open_application(app_config, preflight=preflight,
            login_store_factory=login_store_factory,
            login_environment=lambda: os.environ.get('DEEPSEEK_API_KEY'))
    except ApplicationOpenError as exc:
        if defaults_saved:
            print('用户默认值已保存；应用打开失败', file=sys.stderr)
        print(f"{exc}", file=sys.stderr)
        return 2
    # 恢复阻断: UI 可查看 (新请求被应用拒绝); 无新请求关闭后退出码 3 (契约 §6),
    # 关闭错误/不可恢复失败优先 1
    view_only = session.view_only_reason is not None
    code = await run_tui_session_fn(session, initial_prompt=initial_prompt)
    if code == 1:
        return 1
    return 3 if view_only else code


def main(*, user_config_store: UserConfigStore | None = None,
         interactive_default: bool = False) -> int:
    # 与 powershell 工具一致: 入口输出统一 UTF-8, 不随控制台代码页变化
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="cicada", description=(
            "Cicada coding agent：无任务参数时启动交互终端，工作区默认为当前目录；"
            "位置 prompt 用于单次任务，--tui 可将其作为首问。"
            if interactive_default else "Cicada coding agent"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''DeepSeek 用户默认配置：读取当前用户 ~/.cicada/config.toml，不会读取工作区配置。
新建会话及 --resume-with-current-config 每字段按 CLI > 用户文件 > 内置默认。
普通 --resume 继承存档模型配置；凭据每次启动按有效 DEEPSEEK_API_KEY > 文件引用。
help、list 与 fake 不读取这些文件。文件存在但无效时明确失败。

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
控制字符，最多 4096 UTF-8 bytes。--tui 缺凭据时在应用打开前启动遮蔽输入向导，
确认后可靠保存；取消退出且不创建会话。普通恢复只补凭据并保留存档设置。
非交互缺凭据时退出码 2：设置 DEEPSEEK_API_KEY，或先用 --tui 完成配置。

主屏 /login 不接受参数；↑/↓ 选择、Enter 确认、Tab 返回、Esc/Ctrl+C 取消。
仅保存：下次新建会话及采用当前配置恢复生效，当前会话不变。
保存并应用：正常空闲 DeepSeek 会话可选，仅影响后继普通请求与摘要；
之后 /model 保留新凭据和思考设置。其他提供商或受阻断会话只能保存默认值。
保存与应用分别报告结果；默认已保存而应用失败时，当前绑定保留。
保留凭据不会复制环境 key；替换后未来启动仍优先环境变量。
普通恢复继续继承存档配置；登录不验证账号、额度或远端模型可用性。

主屏 /rewind 查看当前持久会话各请求的文件改动，点击查看 diff、计划并确认还原，
Esc/Ctrl+C 返回。仅捕获 edit/write；PowerShell/check/进程外改动不在还原范围。
还原前自动保存备份点（可选它撤销还原），确认页列出新建文件删除与后继影响。
还原只改变文件，不回滚对话或检查记录；交付需重新检查。捕获失败拒绝本次编辑。
数据在工作区 .cicada/restore/；保留每会话最近 100 点、工作区 blob 512 MiB，
启动机会式清理 30 天前点；已清理点不可还原。--no-restore-capture 关闭新捕获，
已有点仍可查看；非持久会话不捕获，单次 CLI 无还原入口。

费用控制默认关闭。--max-model-calls 单独限制本进程发送次数；金额上限须同时给
--cost-limit-cny 与 --cost-profile。普通/摘要共用额度，clear/new/resume/model/login
不重置，重启新建额度。发送前按1,100,000输入tokens和本次输出cap预留，超限零发送；
完整成功计量及响应关闭后结算，缺计量/错误/取消保留预留并封口。/cost 查看详情。
金额与单价最大10亿元、最多8位小数。价格文件为UTF-8 JSON（≤64KiB），示例（非现价）：
{"schema_version":1,"source":"https://example.invalid/pricing","checked_at":"2026-10-15",
 "models":{"deepseek-flash":{"input_cny_per_million":"1.00","output_cny_per_million":"1.00"}}}
按需增加deepseek-v4-pro。价格应覆盖所用模型/时段的上界；配置单价及输入上界冻结，
不是实时价格或账户账单。账本在.cicada/cost/，不从旧存档恢复消费额度。''',
    )
    parser.add_argument("--workspace", default=".", help="工作区根目录 (默认当前目录)")
    parser.add_argument("--script", help="fake model 剧本 JSON 文件; 与真实模型旗标互斥")
    parser.add_argument("--provider", choices=("deepseek",), help="真实 provider (新会话默认 deepseek; 恢复继承存档)")
    parser.add_argument("--model", help="deepseek-flash (默认)/deepseek-v4-pro")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="开启/关闭思考 (默认开启; 思考可在 TUI 展开查看)")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), help="DeepSeek 思考强度 (默认 high)")
    parser.add_argument("--max-output-tokens", type=int, help="DeepSeek 思考+答案总输出上限 (默认 32768)")
    parser.add_argument('--cost-limit-cny', help='本次进程费用上界（元），须配 --cost-profile；重启新建额度')
    parser.add_argument('--cost-profile', help='本地价格 JSON（相对 workspace），记录来源、核对日期和模型单价')
    parser.add_argument('--max-model-calls', type=int, help='本次进程真实模型发送次数上限；普通与摘要共用')
    parser.add_argument('--export-report', metavar='PATH', help='单次请求结束后导出 .md/.json 诊断报告；拒绝覆盖已有文件；TUI 用 /export <路径>')
    parser.add_argument(
        "--num-ctx", type=int, help="本地窗口上限 (默认 131072)"
    )
    parser.add_argument(
        "--instructions-file", help="追加到系统提示的项目指令文件 (≤16 KiB UTF-8; 两种模式可用)"
    )
    parser.add_argument(
        "--no-auto-instructions", action="store_true",
        help=(
            "关闭根目录 AGENTS.md 自动加载 (无项目指令启动); "
            "显式 --instructions-file 不受影响, 仍照常加载"
        ),
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
        "--no-restore-capture", action="store_true",
        help=(
            "关闭请求还原点捕获 (默认开启): 关闭后 edit/write 照常执行, "
            "但本会话不产生新还原点; 仅持久会话 (--tui/--save-session/--resume) 受影响"
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
    args = parser.parse_args()
    if interactive_default and args.prompt is None and args.skill is None and not args.list_sessions:
        args.tui = True
    return asyncio.run(_run(args, user_config_store=user_config_store))


def console_main() -> int:
    """已安装的 cicada 命令：无任务时进入 TUI，沿用模块入口的全部启动校验。"""
    return main(interactive_default=True)


if __name__ == "__main__":
    raise SystemExit(main())
