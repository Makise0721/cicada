"""非交互入口: uv run python -m cicada --workspace <dir> [--script <script.json> | --model <name> --ollama-url <url> --think --num-ctx N] [--instructions-file <path>] [--check-command <PowerShell> ...] [--check-timeout <s>] "<prompt>".

模型来源二选一:
- --script: fake model 剧本 (见 parse_script), 用于确定性回归;
- 缺省: 真实模型 (本地 Ollama /api/chat), preflight GET /api/version 不可达即退出码 2.
--script 与真实模型旗标 (--model/--ollama-url/--think/--num-ctx) 互斥, 混用退出码 2.
两种模式均使用内置系统提示 (PROMPT_VERSION); --instructions-file 在其后追加项目指令,
文件错误 (缺失/超限/非 UTF-8) 先于 preflight 失败, 退出码 2.

P4 检查模式: 给出 --check-command (可重复, 最多 8 条) 即进入. 该模式要求 workspace
是 Git 工作树根; 启动时 initialize 建立基线快照, 失败退出码 2; 最多 25 轮; 真实模型
profile 固定 num_ctx>=32768、num_predict=2048、请求体上限 65536 bytes. 运行结束后
程序独立扫描终结事实、刷新检查视图、生成变更证据并判定交付 (不读模型文字):
- 0: 模型 stop 且全部指定检查的最近回执 passed/current 且证据完整;
- 3: 模型 stop 但检查或证据条件不满足;
- 1: 模型 error/aborted/length; 2: 输入/启动错误 (普通模式保持 0=stop, 1/2 同上).

剧本 JSON 为条目列表, 每条目是一轮模型响应:
  {"text": "...", "tool_calls": [{"id", "name", "arguments": {...}}],
   "stop": true | "error": "..." | "length": true}
终结缺省: 有 tool_calls 视为 tool_use, 否则视为 stop.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

from cicada.boot import App, BootError, bootstrap
from cicada.core.cancel import CancelToken
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
from cicada.plugins.coding.inventory import inventory_plugin
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.tool_check import check_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_glob import glob_plugin
from cicada.plugins.coding.tool_grep import grep_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.verification import (
    Verifier,
    generate_verification_run_id,
    verification_plugin,
)
from cicada.plugins.coding.verification_contracts import (
    VERIFICATION_CAPABILITY,
    CheckDefinition,
    VerificationPlan,
    VerificationView,
)
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.ollama import OllamaConfig, ollama_plugin
from cicada.prompting import (
    CheckPrompt,
    DEFAULT_TOOLS,
    InstructionsError,
    PromptInfo,
    compose_system_prompt,
)
from cicada.reporting import (
    DeliveryDecision,
    build_delivery_section,
    build_run_summary,
    decide_delivery,
)
from cicada.run_policy import RunPolicy, verification_policy
from cicada.runtime.plugin import PluginDefinition

TOOL_CAPABILITIES = (
    "tool.read", "tool.edit", "tool.write", "tool.powershell", "tool.glob", "tool.grep",
)
CHECK_TOOL_CAPABILITIES = TOOL_CAPABILITIES + ("tool.check",)
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


def default_definitions(
    workspace: Path, model_plugin: PluginDefinition, plan: VerificationPlan | None = None
) -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = [
        workspace_plugin(workspace),
        inventory_plugin(),
        process_plugin(),
        read_plugin(),
        edit_plugin(),
        write_plugin(),
        powershell_plugin(),
        glob_plugin(),
        grep_plugin(),
    ]
    if plan is not None:
        definitions.append(verification_plugin(plan))
        definitions.append(check_plugin())
    definitions.append(model_plugin)
    return definitions


def print_event(event: Event) -> None:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_receipt_artifacts(view: VerificationView) -> tuple[str, ...]:
    """核对最近回执引用的输出工件与写入时记录的 hash; 缺失/篡改都是阻断原因.

    只核对交付判定实际依赖的最近回执; 被取代的历史回执工件是历史证据, 不阻断。
    """
    problems: list[str] = []
    for state in view.checks:
        receipt = state.receipt
        if receipt is None or receipt.output_artifact_path is None:
            continue
        path = receipt.output_artifact_path
        try:
            observed = _sha256_file(path)
        except OSError as exc:
            problems.append(
                f"check {state.check_id} output artifact is missing or unreadable ({path}): {exc}")
            continue
        if receipt.output_artifact_sha256 is not None and observed != receipt.output_artifact_sha256:
            problems.append(
                f"check {state.check_id} output artifact no longer matches the hash recorded "
                f"when it was written ({path})")
    return tuple(problems)


async def _run(args: argparse.Namespace) -> int:
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
    _print_prompt_info(prompt_info)
    plan: VerificationPlan | None = None
    if checks:
        root = Path(os.path.realpath(Path(workspace).expanduser()))
        plan = VerificationPlan(
            verification_run_id=generate_verification_run_id(), root=root, checks=checks
        )
        timeout_s = checks[0].timeout_s
        print(
            f"[check_mode checks={len(checks)} "
            f"ids={','.join(check.check_id for check in checks)} "
            f"timeout={timeout_s:g}s verification_run_id={plan.verification_run_id} "
            f"max_turns={CHECK_MODE_MAX_TURNS}]"
        )
    if args.script is not None:
        try:
            script = parse_script(Path(args.script).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"invalid script: {exc}", file=sys.stderr)
            return 2
        model_plugin = fake_model_plugin(FakeModel(script))
    else:
        num_ctx = args.num_ctx if args.num_ctx is not None else DEFAULT_NUM_CTX
        source = "cli" if args.num_ctx is not None else "default"
        options: dict[str, object] = {"num_ctx": num_ctx}
        request_limit: int | None = None
        if checks:
            options["num_predict"] = CHECK_MODE_NUM_PREDICT
            request_limit = CHECK_MODE_MAX_REQUEST_BYTES
        config = OllamaConfig(
            base_url=args.ollama_url or DEFAULT_CONFIG.base_url,
            model=args.model or DEFAULT_CONFIG.model,
            think=args.think,
            options=options,
            max_request_bytes=request_limit,
        )
        policy_line = (
            f" num_predict={CHECK_MODE_NUM_PREDICT} request_bytes_limit={CHECK_MODE_MAX_REQUEST_BYTES}"
            if checks else ""
        )
        print(
            f"[config model={config.model} context_requested={num_ctx} source={source} "
            f"think={str(config.think).lower()}{policy_line} workspace={prompt_info.workspace}]"
        )
        if not await _ollama_reachable(config.base_url):
            print(
                f"Ollama 不可达: {config.base_url} (请先启动 Ollama, 或用 --script 走 fake model)",
                file=sys.stderr,
            )
            return 2
        model_plugin = ollama_plugin(config)

    # 检查模式: bootstrap 里同步工厂取得验证服务并包装 ModelPort; CLI 保留策略引用,
    # 运行结束后用同一套扫描逻辑消费末次工具结果。
    policy_state: dict[str, object] = {}

    def _model_policy(model, runtime):
        service = runtime.capability(VERIFICATION_CAPABILITY)
        wrapped = verification_policy(model, service)
        policy_state["policy"] = wrapped
        policy_state["service"] = service
        return wrapped

    try:
        app: App = await bootstrap(
            default_definitions(workspace, model_plugin, plan),
            tool_capabilities=CHECK_TOOL_CAPABILITIES if checks else TOOL_CAPABILITIES,
            system_prompt=prompt,
            model_policy=_model_policy if checks else None,
            max_turns=CHECK_MODE_MAX_TURNS if checks else 50,
        )
    except BootError as exc:
        print(f"boot failed: {exc}", file=sys.stderr)
        return 2
    app.agent.subscribe(print_event)
    delivery: tuple[DeliveryDecision, VerificationView, object, Verifier] | None = None
    try:
        if checks:
            service: Verifier = policy_state["service"]  # type: ignore[assignment]
            capture = await service.initialize(CancelToken())
            if not capture.available:
                detail = f" ({capture.failure_kind}): {capture.error}" if capture.error else ""
                print(f"verification initialize failed{detail}", file=sys.stderr)
                return 2
        result = await app.agent.run(args.prompt)
        if checks:
            policy: RunPolicy = policy_state["policy"]  # type: ignore[assignment]
            service = policy_state["service"]  # type: ignore[assignment]
            # 末次工具结果 (含最后一轮的 check/powershell 错误) 先锁存, 再取最终视图与证据
            policy._scan_terminal_facts(result.messages)
            view = await service.refresh(CancelToken())
            evidence = await service.finalize(CancelToken())
            delivery = (
                decide_delivery(result, view, evidence, _verify_receipt_artifacts(view)),
                view,
                evidence,
                service,
            )
    finally:
        await app.aclose()
    print(build_run_summary(result))
    if delivery is not None:
        decision, view, evidence, service = delivery
        print(build_delivery_section(decision, view, evidence, service.plan))
        if not decision.model_stopped:
            return 1
        return 0 if decision.can_deliver else 3
    return 0 if result.stop_reason == "stop" else 1


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
        "--check-timeout", type=float, default=None, metavar="SECONDS",
        help=(
            f"检查命令超时秒数 (默认 {CHECK_TIMEOUT_DEFAULT_S:g}; 范围 "
            f"({CHECK_TIMEOUT_MIN_S:g}, {CHECK_TIMEOUT_MAX_S:g}]; 仅检查模式)"
        ),
    )
    parser.add_argument("prompt", help="交给 agent 的 prompt")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
