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
两种模式均使用内置系统提示 (PROMPT_VERSION); --instructions-file 在其后追加显式指令,
文件错误 (缺失/超限/非 UTF-8) 先于 preflight 失败, 退出码 2. 无显式文件时自动选择
用户级与 workspace 根目录 AGENTS.md; --no-auto-instructions 关闭自动选择, 自动候选
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
from cicada.plugins.model_catalog import model_catalog_plugin, option_id, profile_identity
from cicada.plugins.session_store import list_sessions, read_session_identity
from cicada.plugins.session_store.migration import migrate_legacy_session
from cicada.plugins.skills import load_auto_skill_catalog, load_skill_catalog
from cicada.plugins.user_config import FileUserConfigStore
from cicada.state_paths import legacy_session_directory, session_directory, workspace_state_dir
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
        migrate_legacy_session(workspace, session_id)
        identity = read_session_identity(
            directory=session_directory(workspace),
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
    try:
        page = _list_workspace_sessions(Path(args.workspace), args.list_cursor)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    for line in render_session_page(page):
        print(line)
    return 0


def _list_workspace_sessions(workspace: Path, cursor: str | None) -> SessionPage:
    """合并用户级与旧目录的只读列表；同 ID 以用户级副本为准。"""
    directory = session_directory(workspace)
    if cursor is None or cursor == "":
        offset = 0
    else:
        try:
            offset = int(cursor)
        except ValueError as exc:
            raise ValueError(f"invalid list cursor: {cursor!r}") from exc
        if offset < 0:
            raise ValueError(f"invalid list cursor: {cursor!r}")
    current = list_sessions(directory=directory, limit=2**31 - 1).entries
    legacy = list_sessions(directory=legacy_session_directory(workspace), limit=2**31 - 1).entries
    entries = {row.session_id: row for row in legacy}
    entries.update({row.session_id: row for row in current})
    ordered = sorted(entries.values(), key=lambda row: (
        -row.updated_at if row.updated_at is not None else float("inf"), row.session_id,
    ))
    page = ordered[offset:offset + LIST_SESSIONS_PAGE_SIZE]
    next_cursor = str(offset + LIST_SESSIONS_PAGE_SIZE) if offset + LIST_SESSIONS_PAGE_SIZE < len(ordered) else None
    return SessionPage(str(directory), tuple(page), next_cursor)


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
    if len(info.instruction_sources) > 1:
        line += " instruction_sources=" + ",".join(
            f"{scope}:{path}" for scope, path, _ in info.instruction_sources
        )
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

    directory = session_directory(workspace)
    reader = make_session_catalog_reader(
        directory=directory, workspace=str(workspace),
        legacy_directory=legacy_session_directory(workspace),
    )
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
    explicit: dict[str, object], credentials_only: bool, first_run: bool = False,
) -> LoginResult | None:
    """missing-only 接缝；安全视图重新观察变化时拒绝覆盖旧启动配置。"""
    from cicada.ui.login_wizard import run_startup_login
    from cicada.ui.terminal_io import PromptToolkitTerminal

    view = (store.read_view(provider_id=snapshot.defaults.provider, model_key=snapshot.defaults.model)
            if snapshot.providers else store.read_view())
    observed = UserDefaults(view.provider, view.model, view.think, view.reasoning_effort,
                            view.num_ctx, view.max_output_tokens)
    safe_profiles = tuple(replace(p, credential_ref=None) for p in snapshot.providers)
    if (observed != replace(snapshot.defaults, credential_ref=None)
            or (snapshot.providers and view.providers != safe_profiles)
            or view.credential_present or view.credential_source != 'missing'):
        raise UserConfigError('config', 'configuration-changing')
    view = replace(view, provider=defaults.provider, model=defaults.model, think=defaults.think,
        reasoning_effort=defaults.reasoning_effort, num_ctx=defaults.num_ctx,
        max_output_tokens=defaults.max_output_tokens, credentials_only=credentials_only,
        effect=('只补凭据，本次仍继承存档' if credentials_only else
                '保存连接配置；本次显式 CLI 参数仍优先'))
    result, errors = await run_startup_login(
        PromptToolkitTerminal(), store, view,
        save_defaults=snapshot.defaults if credentials_only else defaults, explicit=explicit,
        initial_set_default=first_run)
    if errors:
        # 终端底层异常不经通用展示，以免包含秘密；保留恢复失败事实。
        print('login terminal restore errors', file=sys.stderr)
        return LoginResult(bool(result and result.defaults_saved), False,
                           'not-requested', 'terminal-failed')
    return result


def _startup_selection_args(args, result):
    """启动向导选择供本次使用；显式 CLI 参数仍优先，不依赖是否修改默认值。"""
    selected = argparse.Namespace(**vars(args))
    if selected.provider is None and result.selected_provider is not None:
        selected.provider = result.selected_provider
    if (selected.model is None and selected.provider == result.selected_provider
            and result.selected_model is not None):
        selected.model = result.selected_model
    return selected


async def _prepare_catalog_launch(args, store, saved_launch, prompt_info, plan, max_turns, *, allow_login=True):
    """先核对安全连接，再解析所选凭据；存档设置不被当前默认值覆盖。"""
    profiles = store.read_profiles()
    provider_id = saved_launch.provider if saved_launch else args.provider
    model_key = (dict(saved_launch.options).get('model_key', saved_launch.model)
                 if saved_launch else args.model)
    if saved_launch:
        provider = next((p for p in profiles if p.id == provider_id), None)
        model = next((m for m in provider.models if m.key == model_key), None) if provider else None
        protocol = dict(saved_launch.options).get('protocol', 'deepseek-chat')
        if (provider is None or model is None or provider.protocol != protocol
                or provider.base_url != saved_launch.base_url or model.model_id != saved_launch.model):
            raise UserConfigError('config', 'archive-connection-changed')
        if ((args.provider is not None and args.provider != provider_id)
                or (args.model is not None and args.model != model_key)):
            raise UserConfigError('config', 'archive-config-conflict')
        for name, previous in (
            ('num_ctx', saved_launch.options['num_ctx']),
            ('max_output_tokens', saved_launch.options['max_tokens']),
            ('think', saved_launch.think),
            ('reasoning_effort', saved_launch.options.get('reasoning_effort', 'high')),
        ):
            if getattr(args, name) is not None and getattr(args, name) != previous:
                raise UserConfigError('config', 'archive-config-conflict')
    snapshot = store.load(provider_id=provider_id, model_key=model_key)
    defaults = snapshot.defaults
    provider = next(p for p in snapshot.providers if p.id == defaults.provider)
    model = next(m for m in provider.models if m.key == defaults.model)
    if saved_launch:
        values = dict(saved_launch.options)
        model = replace(model, context_window=values['num_ctx'], max_output_tokens=values['max_tokens'],
            think=bool(saved_launch.think), reasoning_effort=values.get('reasoning_effort', 'high'),
            stream_usage=values.get('stream_usage', True),
            output_token_field=values.get('output_token_field', 'max_tokens'))
    else:
        from cicada.user_config import supports_thinking

        if (not supports_thinking(provider.protocol, provider.base_url, model.model_id)
                and (args.think is True or args.reasoning_effort is not None)):
            raise UserConfigError('config', 'unsupported-thinking')
        model = replace(model,
            context_window=args.num_ctx if args.num_ctx is not None else model.context_window,
            max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else model.max_output_tokens,
            think=args.think if args.think is not None else model.think,
            reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else model.reasoning_effort)
    provider = replace(provider, models=tuple(model if m.key == model.key else m for m in provider.models))
    profiles = tuple(provider if p.id == provider.id else p for p in snapshot.providers)
    safe = profile_identity(provider, model)
    if plan is not None and model.context_window < CHECK_MODE_MIN_NUM_CTX:
        raise UserConfigError('config', 'check-context-too-small')
    sources = {name: ('archive' if saved_launch else 'cli' if getattr(args, arg) is not None else 'user')
        for name, arg in (('provider', 'provider'), ('model', 'model'), ('num_ctx', 'num_ctx'),
                          ('max_tokens', 'max_output_tokens'), ('think', 'think'),
                          ('reasoning_effort', 'reasoning_effort'))}
    identity = LaunchIdentity(provider=safe.provider, model=safe.model, base_url=safe.base_url,
        think=safe.think, options=dict(safe.options), max_request_bytes=safe.max_request_bytes,
        max_turns=max_turns, prompt_version=prompt_info.version,
        prompt_sha256=prompt_info.prompt_sha256, instructions_sha256=prompt_info.instructions_sha256,
        checks_signature=verification_checks_signature(plan.checks) if plan else None, sources=sources)
    if saved_launch and conflicting_launch_fields(saved_launch, identity):
        raise UserConfigError('config', 'archive-config-conflict')
    if snapshot.credential is None and provider.auth != 'none':
        if not args.tui or not allow_login:
            raise UserConfigError('credential', 'missing-credential')
        result = await _launch_first_run(store, snapshot, defaults,
            explicit={}, credentials_only=saved_launch is not None)
        if result is None:
            return None
        if not result.defaults_saved or result.error_code:
            raise UserConfigError('login', 'save-failed')
        # 配置向导可以选择另一供应商；以保存后的选择重走同一解析链。
        launch_args = args if saved_launch else _startup_selection_args(args, result)
        return await _prepare_catalog_launch(launch_args, store, saved_launch, prompt_info, plan, max_turns,
                                             allow_login=False)
    selected = option_id(provider.id, model.key)
    initial_plugin = None
    if provider.protocol == 'deepseek-chat' and provider.id == 'deepseek' and model.key == model.model_id:
        from cicada.plugins.deepseek import DeepSeekConfig
        initial_plugin = deepseek_plugin(DeepSeekConfig(model=model.model_id, think=model.think,
            reasoning_effort=model.reasoning_effort, num_ctx=model.context_window,
            max_tokens=model.max_output_tokens), api_key=snapshot.credential.reveal())
    plugin = model_catalog_plugin(profiles, selected,
        credential=lambda pid, key: store.load(provider_id=pid, model_key=key),
        default_id=option_id(*store.read_default_reference()), initial_plugin=initial_plugin)
    return plugin, f'{provider.name} / {model.name or model.model_id}', identity, BudgetConfig(
        model.context_window, model.max_output_tokens)


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
    try:
        if args.skills_dir is not None:
            skills_snapshot = load_skill_catalog(args.skills_dir, workspace)
        elif not args.no_auto_skills:
            skills_snapshot = load_auto_skill_catalog(workspace)
    except SkillCatalogError as exc:
        print(f"invalid skills directory: {exc}", file=sys.stderr)
        return 2
    if args.skill is not None:
        # 03 (D8): --skill 需要有效目录; 名称/限额错误在 provider 探测与
        # 应用 open 之前退出码 2。位置 prompt 是任务, 省略时 helper 填固定指令
        if skills_snapshot is None:
            print(
                "--skill 需要 --skills-dir 或自动发现的 Skill 目录",
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
        # 启动层来源选择 (显式 > --no-auto-instructions > 用户级与项目级 AGENTS.md);
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
            instruction_sources=selection.sources if selection.mode == "automatic" else None,
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
    catalog_store = None
    catalog_launch = False
    if args.script is None:
        catalog_store = user_config_store if user_config_store is not None else FileUserConfigStore(Path.home() / '.cicada')
        if hasattr(catalog_store, 'read_profiles'):
            try:
                profiles = catalog_store.read_profiles()
                catalog_launch = (args.provider not in (None, 'deepseek')
                    or any(p.id != 'deepseek' or p.protocol != 'deepseek-chat'
                           or any(m.key != m.model_id for m in p.models) for p in profiles)
                    or bool(saved_launch and 'protocol' in saved_launch.options))
            except UserConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 2
    # API 路径启用预算; fake 默认不安装预算层。
    budget_config: BudgetConfig | None = None
    defaults_saved = False
    first_run = False
    catalog_reprepared = None
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
    elif catalog_launch:
        try:
            prepared = await _prepare_catalog_launch(args, catalog_store, saved_launch, prompt_info, plan, max_turns)
            if prepared is None:
                return 0
            model_plugin, model_label, identity, budget_config = prepared
        except (UserConfigError, ValueError):
            print('模型配置、凭据或恢复身份不匹配；请检查 /login，恢复可显式使用 --resume-with-current-config', file=sys.stderr)
            return 2
        preflight = None
        if not args.tui:
            print(f'[config provider={identity.provider} model={identity.model} context_requested={budget_config.num_ctx}]')
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
            environment_key = os.environ.get('DEEPSEEK_API_KEY')
            snapshot = store.load(environment_key)
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
        if isinstance(store, FileUserConfigStore):
            try:
                if store.ensure_default_config():
                    first_run = True
                    snapshot = store.load(environment_key)
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
            # 取消过首次向导后，自动生成的默认文件仍在；首次成功配置应保留默认勾选。
            first_run = first_run or (
                archive is None and snapshot.defaults.credential_ref is None
                and all(value is None for value in explicit.values())
                and snapshot.defaults == UserDefaults())
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
                    await GitInventory(Workspace(root, workspace_state_dir(root) / 'outputs')).list_files(CancelToken())
                except InventoryError:
                    print('检查模式工作区 Git 预检失败', file=sys.stderr)
                    return 2
            try:
                result = await _launch_first_run(store, snapshot, defaults,
                    explicit=explicit, credentials_only=archive is not None, first_run=first_run)
                if result is None:
                    return 0
                defaults_saved = result.defaults_saved
                if not defaults_saved or result.error_code is not None:
                    print(('用户默认值已保存；' if defaults_saved else '') +
                          'login failed: ' + (result.error_code or 'save-failed'), file=sys.stderr)
                    return 2
                if hasattr(store, 'read_profiles'):
                    launch_args = args if saved_launch else _startup_selection_args(args, result)
                    catalog_reprepared = await _prepare_catalog_launch(
                        launch_args, store, saved_launch, prompt_info, plan, max_turns, allow_login=False)
                    if catalog_reprepared is None:
                        return 0
                else:
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
        if catalog_reprepared is not None:
            model_plugin, model_label, identity, budget_config = catalog_reprepared
            preflight = None
        else:
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
            if snapshot.providers:
                model_plugin = model_catalog_plugin(snapshot.providers,
                    option_id(defaults.provider, defaults.model),
                    credential=lambda pid, key: store.load(provider_id=pid, model_key=key),
                    default_id=option_id(snapshot.defaults.provider, snapshot.defaults.model),
                    initial_plugin=model_plugin)
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
        runtime_dir=workspace_state_dir(workspace),
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
        epilog='''用户配置：真实模型读取 ~/.cicada/config.toml，缺失时生成 DeepSeek 默认文件。
内置 DeepSeek 专用协议与 OpenAI Chat Completions 兼容子集；自定义连接不启用私有思考。
新建/采用当前配置：先选供应商实例和模型，再以 CLI 覆盖该模型设置。
--provider 是保存的实例 ID，--model 是该实例内的本地模型 key，远端 ID 可以包含斜线。
普通 --resume 继承存档的非秘密配置；连接改变时须 --resume-with-current-config 显式采用。
help、list 与 fake 不读取用户连接或凭据。运行状态在 ~/.cicada/workspaces/<路径摘要>/。

config v2 示例（数值为示例，能力由用户确认；凭据不放入配置）：
version = 2
[defaults]
provider = "work"
model = "coder"
[providers.work]
name = "工作网关"
protocol = "openai-chat-completions"
base_url = "https://gateway.example.com/v1"
default_model = "coder"
api_key_env = "WORK_LLM_API_KEY"
[providers.work.models.coder]
id = "org/model-name"
context_window = 131072
max_output_tokens = 8192
tools = true

Base URL 使用完整 API 根路径，程序追加 /chat/completions，不自动修改 /v1。
高级模型项：stream_usage=true/false，output_token_field="max_tokens"/"max_completion_tokens"。
支持旧版 DeepSeek config v1，首次保存到多供应商格式时保留原设置和凭据引用。
文件 Key 用遮蔽输入保存到独立 credentials.json；环境变量仅读取目标实例的显式声明。
被声明环境变量存在但无效时拒绝，不借用其他实例的 Key。两个文件各限64KiB。

/login 配置已有或新连接及多个模型；↑/↓选择、Enter确认、Tab返回、Esc/Ctrl+C取消。
保存配置和保存并应用分别选择，设置下次默认另行确认。保存成功但应用受阻分别报告。
/model 统一目录可直接输入过滤，Enter切换当前对话，Ctrl+D设默认、Ctrl+O关闭目标思考切换、
Ctrl+L配置所选实例；Esc保留草稿。切换不清空对话或验证事实，不自动发摘要或探测账号。
默认值与当前模型分别生效；打开菜单/保存配置不验证远端模型可用性。

主屏 /rewind 查看当前持久会话各请求的文件改动，点击查看 diff、计划并确认还原，
Esc/Ctrl+C 返回。仅捕获 edit/write；PowerShell/check/进程外改动不在还原范围。
还原前自动保存备份点（可选它撤销还原），确认页列出新建文件删除与后继影响。
还原只改变文件，不回滚对话或检查记录；交付需重新检查。捕获失败拒绝本次编辑。
数据在 ~/.cicada/workspaces/<路径摘要>/restore/；保留每会话最近 100 点、工作区 blob 512 MiB，
启动机会式清理 30 天前点；已清理点不可还原。--no-restore-capture 关闭新捕获，
已有点仍可查看；非持久会话不捕获，单次 CLI 无还原入口。

费用控制默认关闭。--max-model-calls 单独限制本进程发送次数；金额上限须同时给
--cost-limit-cny 与 --cost-profile。普通/摘要共用额度，clear/new/resume/model/login
不重置，重启新建额度。发送前按显式输入上界和本次输出cap预留，超限零发送；
完整成功计量及响应关闭后结算，缺计量/错误/取消保留预留并封口。/cost 查看详情。
金额与单价最大10亿元、最多8位小数。价格文件为UTF-8 JSON（≤64KiB），示例（非现价）：
{"schema_version":1,"source":"https://example.invalid/pricing","checked_at":"2026-10-15",
 "models":{"deepseek-flash":{"input_cny_per_million":"1.00","output_cny_per_million":"1.00"}}}
v1仅适用于DeepSeek官方，输入上界1,100,000；其他实例必须使用价格v2，例如：
{"schema_version":2,"source":"https://example.invalid/pricing","checked_at":"2026-10-21",
 "models":[{"provider":"work","protocol":"openai-chat-completions",
 "base_url":"https://gateway.example.com/v1","model":"org/model-name","input_token_bound":131072,
 "input_cny_per_million":"1.00","output_cny_per_million":"1.00"}]}
实例/协议/端点/远端模型必须完全匹配。价格应覆盖所用模型/时段的上界；配置冻结，
不是实时价格或账户账单。账本在用户级工作区状态目录的 cost/，不从旧存档恢复消费额度。''',
    )
    parser.add_argument("--workspace", default=".", help="工作区根目录 (默认当前目录)")
    parser.add_argument("--script", help="fake model 剧本 JSON 文件; 与真实模型旗标互斥")
    parser.add_argument("--provider", help="已配置的供应商实例 ID (缺省使用用户默认值；恢复继承存档)")
    parser.add_argument("--model", help="所选供应商下的模型配置 key；内置 deepseek-flash/deepseek-v4-pro")
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
        "--instructions-file", help="替换自动指令的指定文件 (≤16 KiB UTF-8; 两种模式可用)"
    )
    parser.add_argument(
        "--no-auto-instructions", action="store_true",
        help=(
            "关闭用户级和工作区根 AGENTS.md 自动加载; "
            "显式 --instructions-file 不受影响, 仍照常加载"
        ),
    )
    parser.add_argument(
        "--skills-dir",
        help=(
            "只使用指定的本地 Skills 目录 (相对 workspace; 直接子目录各含 SKILL.md)。"
            "旧版显式 Skills 会话恢复时须重传同一目录"
        ),
    )
    parser.add_argument(
        "--no-auto-skills", action="store_true",
        help="关闭用户级和项目级 Skills 自动发现；显式 --skills-dir 仍生效",
    )
    parser.add_argument(
        "--skill",
        help=(
            "显式发送一个 Skill 的正文作为首条请求 (名称须在有效 Skill 目录中; "
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
