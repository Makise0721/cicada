"""非交互入口: uv run python -m cicada --workspace <dir> [--script <script.json> | --model <name> --ollama-url <url> --think --num-ctx N] [--instructions-file <path>] [--check-command <PowerShell> ...] [--check-timeout <s>] "<prompt>"
交互入口: uv run python -m cicada --tui --workspace <dir> [同上模型/检查旗标] [首问]

模型来源二选一:
- --script: fake model 剧本 (见 parse_script), 用于确定性回归;
- 缺省: 真实模型 (本地 Ollama /api/chat), preflight GET /api/version 不可达即退出码 2.
--script 与真实模型旗标 (--model/--ollama-url/--think/--num-ctx) 互斥, 混用退出码 2.
两种模式均使用内置系统提示 (PROMPT_VERSION); --instructions-file 在其后追加项目指令,
文件错误 (缺失/超限/非 UTF-8) 先于 preflight 失败, 退出码 2.

--tui 进入交互终端 (生产组合: 共享应用 + Pi 主屏前端, 见 cicada.ui; 正常启动
安静, /status 提供有界可追溯诊断, 检查事实由 RequestOutcome 投影); UI 可选依赖
缺失或 stdin/stdout 非 TTY 退出码 2; 位置 prompt 可省略 (空启动), 给出则作为
首问提交; 进程退出码: 启动失败 2, UI 不可恢复/关闭错误优先 1, 正常关闭取最后
一条请求的退出码 (无请求 0). 单次模式要求位置 prompt; 普通 UI 导入要求为零 (T14).

P4 检查模式: 给出 --check-command (可重复, 最多 8 条) 即进入. 该模式要求 workspace
是 Git 工作树根; 启动时 initialize 建立基线快照, 失败退出码 2; 最多 25 轮; 真实模型
profile 固定 num_ctx>=32768、num_predict=2048、请求体上限 65536 bytes. 运行结束后
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
from pathlib import Path

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
from cicada.persistence.contracts import LaunchIdentity, PersistenceError, StoredIdentity
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    VerificationPlan,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.ollama import OllamaConfig, ollama_plugin
from cicada.plugins.session_store import open_session_store
from cicada.prompting import (
    CheckPrompt,
    DEFAULT_TOOLS,
    InstructionsError,
    PromptInfo,
    compose_system_prompt,
)
from cicada.reporting import build_delivery_section, build_run_summary

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


async def _read_saved_identity(workspace: Path, session_id: str) -> StoredIdentity | None:
    """恢复前的存档身份预读; 失败返回 None (错误消息已打印, 调用方退出码 2).

    省略参数的继承需要存档值才能构造模型插件; 顺带把格式/workspace 校验提前到
    模型插件构造与可达性探测之前 (契约 §1 次序)。store 打开后立即关闭, 存档的
    完整校验与租约由 open_application 再次完成。
    """
    try:
        store = open_session_store(
            directory=Path(workspace) / ".cicada" / "sessions",
            session_id=session_id,
            workspace=Path(workspace),
        )
    except PersistenceError as exc:
        print(f"session store open failed: {exc}", file=sys.stderr)
        return None
    try:
        return store.recovery_snapshot().identity
    finally:
        await store.aclose()


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


async def _run(args: argparse.Namespace) -> int:
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
    if (args.save_session or args.resume is not None) and args.tui:
        print("持久会话的 TUI 入口 (自动保存/恢复) 在阶段 05 开放; 当前仅单次 CLI 支持", file=sys.stderr)
        return 2
    if not args.tui and args.prompt is None:
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
        or args.think
        or args.num_ctx is not None
    ):
        print(
            "--script 与 --model/--ollama-url/--think/--num-ctx 互斥, 请只选一种模型来源",
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
    # 指令文件错误先于 preflight/bootstrap (P3 §4.2)
    check_prompts = tuple(
        CheckPrompt(check.check_id, check.command, check.timeout_s) for check in checks
    )
    try:
        prompt, prompt_info = compose_system_prompt(
            workspace,
            args.instructions_file,
            tools=DEFAULT_TOOLS + ("check",) if checks else DEFAULT_TOOLS,
            checks=check_prompts,
        )
    except InstructionsError as exc:
        print(f"invalid instructions file: {exc}", file=sys.stderr)
        return 2
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
    # 04: 默认恢复预读存档身份——省略参数继承存档, 显式参数与存档比对;
    # 检查定义只存签名不存命令, 检查模式存档遗漏 --check-command 时明确拒绝
    saved_launch: LaunchIdentity | None = None
    if args.resume is not None and not args.resume_with_current_config:
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
        if saved_stored.launch.provider == "ollama":
            saved_launch = saved_stored.launch
            option_error = _validate_saved_options(dict(saved_launch.options))
            if option_error is not None:
                print(f"--resume: {option_error}", file=sys.stderr)
                return 2
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
            sources={},
        )
        preflight = None
    else:
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
            if not args.think:
                eff_think = saved_launch.think
                sources["think"] = "archive"
        sources["num_ctx"] = num_ctx_source
        options: dict[str, object] = {"num_ctx": num_ctx}
        request_limit: int | None = None
        if checks:
            options["num_predict"] = CHECK_MODE_NUM_PREDICT
            request_limit = CHECK_MODE_MAX_REQUEST_BYTES
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
        if args.save_session or args.resume is not None:
            # 持久路径: 可达性探测在存档格式/配置预检之后执行 (契约 §1 次序)
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

    persistence_config = None
    if args.save_session or args.resume is not None:
        persistence_config = PersistenceConfig(
            session_id=args.resume,
            identity=identity,
            adopt_current_config=args.resume_with_current_config,
        )

    app_config = ApplicationConfig(
        workspace=workspace,
        system_prompt=prompt,
        prompt_info=prompt_info,
        model_plugin=model_plugin,
        plan=plan,
        max_turns=max_turns,
        model_label=model_label,
        persistence=persistence_config,
    )
    if args.tui:
        assert run_tui_session_fn is not None
        return await _run_tui(run_tui_session_fn, app_config, initial_prompt=args.prompt)
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
    run_tui_session_fn, app_config: ApplicationConfig, *, initial_prompt: str | None
) -> int:
    """--tui 分支: 依赖与 TTY 已在 _run 校验; 打开共享应用并运行生产组合."""
    try:
        session = await open_application(app_config)
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
    return await run_tui_session_fn(
        session, initial_prompt=initial_prompt, diagnostics=partial(diagnostics_lines, diagnostics)
    )


def main() -> int:
    # 与 powershell 工具一致: 入口输出统一 UTF-8, 不随控制台代码页变化
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="cicada", description="Cicada coding agent")
    parser.add_argument("--workspace", default=".", help="工作区根目录 (默认当前目录)")
    parser.add_argument("--script", help="fake model 剧本 JSON 文件; 与真实模型旗标互斥")
    parser.add_argument("--model", help=f"Ollama 模型名 (默认 {DEFAULT_CONFIG.model})")
    parser.add_argument("--ollama-url", help=f"Ollama 服务地址 (默认 {DEFAULT_CONFIG.base_url})")
    parser.add_argument("--think", action="store_true", help="开启模型思考 (思考增量不展示)")
    parser.add_argument(
        "--num-ctx", type=int, help=f"Ollama 上下文窗口请求值 (默认 {DEFAULT_NUM_CTX}; 仅真实模型)"
    )
    parser.add_argument(
        "--instructions-file", help="追加到系统提示的项目指令文件 (≤16 KiB UTF-8; 两种模式可用)"
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
        help="单次模式创建持久会话并保存本次请求 (默认不落盘); 检查模式可组合, 与 --tui 组合暂不支持",
    )
    parser.add_argument(
        "--resume", metavar="SESSION_ID",
        help="按 UUIDv4 会话 ID 恢复持久会话, 新请求追加保存; 与 --save-session 互斥",
    )
    parser.add_argument(
        "--resume-with-current-config", action="store_true",
        help=(
            "与 --resume 一起使用: 显式采用当前默认与显式参数恢复 (省略项取当前默认而非存档), "
            "与存档的差异记入新启动代次; 不换工作区、不解除恢复阻断"
        ),
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="交给 agent 的 prompt (单次模式必填; --tui 可作首问)",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
