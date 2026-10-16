"""PV02 评估方监督 + 受限模型子进程。默认只允许离线演练。

真实凭据、总账、官方检查与墙钟均由 Medium 父进程拥有。Low 子进程只经过
本地代理发送生产模型请求、上报生产终局、触发固定 check-1；不能读私料或凭据。
代理不接受 URL、命令、路径或模型选择。离线与 live 复用同一代理和真实子进程。
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from cicada.core.ports import ModelMetrics, StreamDone
from cicada.costs import CostLedger, CostProfile
from cicada.plugins.coding.verification_contracts import CheckDefinition
from cicada.prompting import CheckPrompt, DEFAULT_TOOLS, compose_system_prompt
from tests.support import swebench_bridge as bridge
from tests.support import swebench_isolation as isolation
from tests.support import swebench_lifecycle as lifecycle
from tests.support.swebench_budget import AttemptBudget, AttemptBudgetError, PROCESS_1, PROCESS_2
from tests.support.swebench_tasks import load_json, sha256_file, spec_from_manifest, write_json

PROJECT = Path(__file__).resolve().parents[2]
WIRE_LIMIT = 2 * 1024 * 1024

# watcher 是**存活检查**, 不是命令通道: 子进程自己的长时同步工作 (逐轮工作区快照扫描)
# 会让在途轮询超时, 而父进程只要还活着就一定会立刻回 stopped。首题 live 实测:
# 单次轮询失败就把整轮请求取消, 生产账本因此留下未决预留而根因不可见。
WATCH_TIMEOUT_S = 30.0
WATCH_INTERVAL_S = 0.1
WATCH_MAX_FAILURES = 5


def archive_manifest(workspace: Path) -> dict:
    """已关闭存档的内容身份, 在第二个 OS 进程启动前重核。"""
    root = workspace / ".cicada" / "sessions"
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _public_check(summary: dict) -> dict:
    return {key: summary.get(key) for key in (
        "result", "required_passed", "required_total", "patch_empty", "grade_id", "cleanup")}


class Broker:
    """固定权限的本地代理; 所有账本操作在父进程同一个 asyncio loop 串行执行。

    ``audit`` 给出审计文件路径时, 逐路由记一行 (决策/原因/状态), 只记事实不记正文;
    审计失败不得影响请求本身 (live 首题就是因为没有这份轨迹而无法判定根因)。
    """

    def __init__(self, *, gate, check, deadline, credential=None, admission=None, audit=None):
        self.gate, self.check = gate, check
        self.deadline, self.credential, self.admission = deadline, credential, admission
        self.token = uuid.uuid4().hex
        self.stopped = False
        self.stop_reason = None
        self.audit = Path(audit) if audit is not None else None
        # record 会被 loop 线程与 Handler 线程同时调用; Windows 的追加写不是原子的,
        # 没有这把锁就可能丢行/串行 (审计本身就成了不可信的证物)。
        self._audit_lock = threading.Lock()
        self.rejected_tokens = 0
        self.checks = []
        self.inflight = None
        self.checking = False
        self.server = None

    def record(self, event: str, **fields) -> None:
        """审计一行; 任何失败都吞掉 —— 审计不得成为新的失败面。

        record 会被多个 `except BaseException` 处理器调用, 所以连序列化异常也要吞,
        否则它会把原异常换掉, 自己变成新的失败面。
        """
        if self.audit is None:
            return
        try:
            line = json.dumps({"at": time.time(), "event": event, **fields},
                              ensure_ascii=False, sort_keys=True) + "\n"
            with self._audit_lock:
                self.audit.parent.mkdir(parents=True, exist_ok=True)
                with self.audit.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
        except BaseException:  # noqa: BLE001 - 审计永不改变请求结果
            pass

    async def dispatch(self, route: str, body: bytes) -> tuple[int, dict, bytes]:
        if route == "/state":
            return 200, {}, json.dumps({"stopped": self.stopped}).encode()
        if self.stopped or time.monotonic() >= self.deadline:
            self.stop("run-timeout-or-stopped")
            raise AttemptBudgetError("production deadline reached")
        if route == "/model":
            if self.checking:
                self.record("model-refused", reason="check-in-flight")
                raise AttemptBudgetError("model send overlaps a check")
            if self.admission:
                try:
                    self.admission()
                except BaseException as exc:
                    self.record("model-refused", reason="admission",
                                detail=type(exc).__name__ + ": " + str(exc)[:200])
                    raise
            if any(not item.resolved for item in self.gate.observations.values()):
                self.stop("unconfirmed-previous-send")
                raise AttemptBudgetError("previous send lacks production confirmation")
            self.inflight = asyncio.current_task()
            response = None
            try:
                request = httpx.Request("POST", "https://api.deepseek.com/chat/completions",
                    content=body, headers={"Content-Type": "application/json",
                    "Authorization": "Bearer " + (self.credential or lifecycle.OFFLINE_CREDENTIAL)})
                response = await self.gate.handle_async_request(request)
                content = bytearray()
                async for chunk in response.aiter_raw():
                    content.extend(chunk)
                    if len(content) > WIRE_LIMIT:
                        raise AttemptBudgetError("response exceeds frozen wire cap")
                call_id = next(reversed(self.gate.observations))
                self.record("model-ok", status=response.status_code, call_id=call_id,
                            request_bytes=len(body), response_bytes=len(content),
                            handoffs=self.gate.handoffs)
                return response.status_code, {"X-Call-Id": call_id}, bytes(content)
            except BaseException as exc:
                self.record("model-failed", detail=type(exc).__name__ + ": " + str(exc)[:200],
                            gate_sealed=self.gate.network_sealed_reason)
                self.stop("model-transport-failed")
                raise
            finally:
                if response is not None:
                    await response.aclose()
                self.inflight = None
        if route == "/confirm":
            data = json.loads(body)
            metrics = data.get("metrics")
            terminal = None if data.get("stop_reason") is None else StreamDone(
                data["stop_reason"], error="production-error" if data.get("error") else None,
                metrics=ModelMetrics(**metrics) if metrics else None)
            if data.get("call_id") not in self.gate.observations:
                self.stop("unknown-production-confirmation")
                raise AttemptBudgetError("unknown production confirmation")
            self.gate.confirm(data["call_id"], terminal)
            self.record("confirm", call_id=data.get("call_id"),
                        stop_reason=data.get("stop_reason"), error=bool(data.get("error")))
            return 200, {}, b"{}"
        if route == "/seal":
            self.record("seal")
            self.stop("child-sealed")
            return 200, {}, b"{}"
        if route == "/check":
            if self.checking or self.gate.active_call_id is not None:
                self.record("check-refused", reason="overlapping")
                raise AttemptBudgetError("overlapping check refused")
            self.checking = True
            try:
                code, summary = await asyncio.to_thread(self.check)
                self.checks.append({"code": code, **summary})
                self.record("check-ok", code=code)
                if code == 2:
                    self.stop("evaluation-anomaly")
                return 200, {}, json.dumps({"code": code, "summary": _public_check(summary)}).encode()
            except BaseException as exc:
                self.record("check-failed", detail=type(exc).__name__ + ": " + str(exc)[:200])
                self.stop("evaluation-error")
                raise
            finally:
                self.checking = False
        raise AttemptBudgetError("unknown broker operation")

    def stop(self, reason):
        first = self.stop_reason is None
        self.stop_reason = self.stop_reason or reason
        self.stopped = True
        if first:
            # 只记首个原因: 封口原因首因优先, 后续 stop 再来一行会让人把最后一行当根因。
            self.record("stop", reason=reason)
        self.gate.seal_network(reason)
        self.gate.budget.seal(reason)

    def start(self):
        loop = asyncio.get_running_loop()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get("X-PV-Token") != owner.token:
                    # 只记第一次: Low 子进程能刷这个端口, 无界记录会把封印目录撑大。
                    owner.rejected_tokens += 1
                    if owner.rejected_tokens == 1:
                        owner.record("handler-rejected", path=self.path, reason="token")
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= WIRE_LIMIT:
                        raise ValueError("bad content length")
                    body = self.rfile.read(length)
                    future = asyncio.run_coroutine_threadsafe(owner.dispatch(self.path, body), loop)
                    code, headers, result = future.result()
                except BaseException as exc:
                    # 这里的 detail 是"为什么没成"的唯一落痕: 首题 live 缺的就是这一行。
                    owner.record("handler-refused", path=self.path,
                                 detail=type(exc).__name__ + ": " + str(exc)[:200])
                    code, headers, result = 503, {}, b'{"error":"broker refused"}'
                try:
                    self.send_response(code)
                    self.send_header("Content-Length", str(len(result)))
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(result)
                except OSError:
                    if self.path in ("/model", "/confirm", "/check"):
                        loop.call_soon_threadsafe(owner.stop, "child-disconnected")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    async def close(self):
        if self.inflight is not None:
            self.inflight.cancel()
            await asyncio.gather(self.inflight, return_exceptions=True)
        # Docker 评估 worker 有自己的截止与清理; 取消 asyncio 等待不证明线程已退出。
        while self.checking:
            await asyncio.sleep(0.05)
        if self.server:
            await asyncio.to_thread(self.server.shutdown)
            self.server.server_close()
            self.thread.join(timeout=2)


class RemoteGate(httpx.AsyncBaseTransport):
    """Low 子进程的传输适配器; 真实 key 不进入此进程。"""

    def __init__(self, endpoint, token):
        self.endpoint, self.token = endpoint, token
        self.inner = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        self.observations = {}
        self.network_sealed_reason = None
        self.budget = self

    def _post(self, route, data):
        with httpx.Client(trust_env=False, timeout=30) as client:
            response = client.post(self.endpoint + route,
                                  headers={"X-PV-Token": self.token}, json=data)
            response.raise_for_status()
            return response.json()

    def seal_network(self, reason):
        self.network_sealed_reason = reason

    def seal(self, reason):
        self.seal_network(reason)
        self._post("/seal", {})

    def confirm(self, call_id, terminal):
        self._post("/confirm", {"call_id": call_id,
            "stop_reason": terminal.stop_reason if terminal else None,
            "error": bool(terminal.error) if terminal else True,
            "metrics": asdict(terminal.metrics) if terminal and terminal.metrics else None})

    async def handle_async_request(self, request):
        body = await request.aread()
        local = httpx.Request("POST", self.endpoint + "/model", content=body,
                              headers={"X-PV-Token": self.token})
        response = await self.inner.handle_async_request(local)
        call_id = response.headers.get("X-Call-Id")
        if call_id:
            self.observations[call_id] = True
        return response

    async def aclose(self):
        await self.inner.aclose()


async def watch_parent(endpoint: str, token: str, task, diagnostics: dict | None = None) -> None:
    """轮询父进程停止信号; **单次失败不取消**整轮请求。

    为什么: 轮询是存活检查, 而子进程自己的长时同步工作 (逐轮工作区快照扫描) 会让在途
    轮询超时。首题 live 因此在第 5 次发送前被取消, 生产账本留下未决预留, 父进程却只
    看得到 "child-sealed"。只有明确的 ``stopped`` 或连续失败 ``WATCH_MAX_FAILURES``
    次才取消; 失败形态与恢复都进诊断事实 (首题"不知道是超时还是别的 httpx 错误"的盲点)。
    """
    failures = 0
    total_failures = 0
    async with httpx.AsyncClient(trust_env=False, timeout=WATCH_TIMEOUT_S) as client:
        while not task.done():
            try:
                response = await client.post(endpoint + "/state",
                                             headers={"X-PV-Token": token})
                stopped = bool(response.json()["stopped"])
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                failures += 1
                total_failures += 1
                if diagnostics is not None:
                    diagnostics["watch"] = {
                        "stop_seen": False, "failures": failures,
                        "total_failures": total_failures,
                        "last_failure": type(exc).__name__ + ": " + str(exc)[:200],
                        "at": round(time.time(), 3)}
                if failures >= WATCH_MAX_FAILURES:
                    task.cancel()
                    return
            else:
                if failures and diagnostics is not None:
                    diagnostics["watch"] = {**(diagnostics.get("watch") or {}),
                                            "failures": 0, "recovered": True}
                failures = 0
                if stopped:
                    if diagnostics is not None:
                        diagnostics["watch"] = {"stop_seen": True, "failures": 0,
                                                "total_failures": total_failures}
                    task.cancel()
                    return
            await asyncio.sleep(WATCH_INTERVAL_S)


async def _child(config_path: Path, diagnostics: dict | None = None) -> dict:
    config = load_json(config_path)
    workspace = Path(config["workspace"])
    gate = RemoteGate(config["endpoint"], config["token"])
    check_command = config["check_command"]
    timeout = config["check_timeout_s"]
    prompt, info = compose_system_prompt(workspace, None, tools=DEFAULT_TOOLS + ("check",),
        checks=(CheckPrompt("check-1", check_command, timeout),))
    profile = CostProfile.load(Path(config["profile"]))
    ledger = CostLedger(Path(config["child_root"]) / "cost" / config["phase"],
                       limit=Decimal("4.50"), max_calls=24, profile=profile)
    # 诊断容器由 main() 持有: 异常退出时也要落盘 (取消/异常路径才是最需要证据的路径)。
    diagnostics = diagnostics if diagnostics is not None else {
        "watch": {"stop_seen": False, "failures": 0, "total_failures": 0}, "streams": []}
    spec = lifecycle.LifecycleSpec(workspace=workspace, instruction=config["instruction"],
        model_doc=config["model"], checks=(CheckDefinition("check-1", check_command, timeout),),
        cost_control=ledger)
    # 只通过生产 close 发协作取消; 父进程先封发送, 等待退出后才核对存档。
    task = asyncio.create_task(
        lifecycle.run_process_coding(spec, gate, system_prompt=prompt, prompt_info=info,
                                     diagnostics=diagnostics)
        if config["scenario"] == "coding" else
        lifecycle.run_process_1(spec, gate, system_prompt=prompt, prompt_info=info,
                                diagnostics=diagnostics)
        if config["phase"] == PROCESS_1 else
        lifecycle.run_process_2(spec, gate, None, session_id=config["session_id"],
                                system_prompt=prompt, prompt_info=info,
                                diagnostics=diagnostics))
    watcher = asyncio.create_task(watch_parent(config["endpoint"], config["token"], task,
                                               diagnostics))
    try:
        result = await task
        result["process_ledger"] = lifecycle._ledger_facts(ledger)
        result["pid"] = os.getpid()
        result["diagnostics"] = diagnostics
        return result
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await gate.aclose()


def _child_env(child_root: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"}}
    env.update({"PYTHONPATH": str(PROJECT / "src") + os.pathsep + str(PROJECT),
        "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
        "TMP": str(child_root / "temp"), "TEMP": str(child_root / "temp"),
        "UV_CACHE_DIR": str(child_root / "uv-cache"),
        "XDG_CACHE_HOME": str(child_root / "cache"),
        "APPDATA": str(child_root / "appdata"), "LOCALAPPDATA": str(child_root / "localappdata")})
    return env


async def _supervise_child(config, config_path, broker, deadline):
    write_json(config_path, config)
    child = isolation.LowProcess([sys.executable, "-m", "tests.support.swebench_live",
        "child", "--config", str(config_path)], PROJECT, _child_env(Path(config["child_root"])))
    timeout = False
    try:
        while child.poll() is None:
            if time.monotonic() >= deadline or broker.stopped:
                timeout = time.monotonic() >= deadline
                broker.stop("run-timeout" if timeout else "attempt-stopped")
                if broker.inflight:
                    broker.inflight.cancel()
                grace = time.monotonic() + 10
                while child.poll() is None and time.monotonic() < grace:
                    await asyncio.sleep(0.05)
                if child.poll() is None:
                    child.terminate_tree()
                break
            await asyncio.sleep(0.05)
        code = child.poll()
        if code is None:
            raise AttemptBudgetError("child termination is unconfirmed")
        facts_path = Path(config["child_root"]) / (config["phase"] + "-facts.json")
        facts = load_json(facts_path) if facts_path.is_file() else {}
        facts.update({"pid": child.pid, "os_exit_code": code,
                      "os_exit_confirmed": True, "timed_out": timeout})
        return facts
    finally:
        if child.poll() is None:
            broker.stop("supervisor-exit")
            child.terminate_tree()
        child.close()


def _make_budget(root, manifest_path, prepared, profile, manifest):
    return AttemptBudget(root, attempt_id="pv02-" + manifest["task_key"],
        manifest_hash=sha256_file(manifest_path), prepared_lock_hash=sha256_file(prepared / "lock.json"),
        profile=profile, model_identity={"provider": "deepseek", "model": manifest["model"]["normal"]["model"]})


def _diagnosis(child_facts: dict, broker, budget) -> str:
    """把"为什么没成"从事实里抽出来, 附在父进程的失败报错上。

    首题 live 的教训: 父进程只报一句 "first child did not complete normally",
    真正的原因 (子进程被谁取消、哪次发送没相关上、账本是否封口) 全在事实里没被带出来。
    """
    diagnostics = (child_facts or {}).get("diagnostics") or {}
    streams = diagnostics.get("streams") or []
    production = (child_facts or {}).get("process_ledger") or {}
    parts = [
        f"broker={broker.stop_reason or 'running'}",
        f"gate_sealed={getattr(broker.gate, 'network_sealed_reason', None)}",
        f"attempt_sealed={budget.sealed_reason}",
        f"os_exit={child_facts.get('os_exit_code')}",
        f"timed_out={child_facts.get('timed_out')}",
        f"closed_ok={child_facts.get('closed_ok')}",
        f"watch={diagnostics.get('watch')}",
        f"streams={streams[-3:]}",
        "production=" + str({key: production.get(key) for key in (
            "calls", "settled_upper_cny", "pending_upper_cny", "sealed_reason")}),
    ]
    return "; ".join(parts)


def check_gate(checks: list[dict], *, offline: bool) -> str | None:
    """检查准入: 返回拒绝原因, None 表示通过。

    离线演练由 fixture 保证"先 read+check、再编辑"的顺序, 所以**离线**要求第一次观察到的
    检查必须是空补丁失败 —— 那是我们自己的契约, 也证明检查确实落在未修改的工作区上。
    **live** 下模型有行为自由(本题 prompt 要求先检后改, 但模型可以先改后检): 那次检查会落在
    已修改的工作区上, `patch_empty=false` 却是真实的 11/11 通过。因此 live 只要求
    "存在一次覆盖达标的通过检查", 而把"首次是否为空的失败检查"作为**事实**记录下来
    (见 facts["initial_check"]), 不再当作门禁 —— 门禁不应该把"做对了题"判成失败。

    调用点位置同样是契约的一部分(见 `_run`): live 判必须在**两个进程都跑完**之后执行,
    因为 restore-compact-continue 的通过检查出现在第二个进程的 R3。
    """
    if offline:
        if not checks or checks[0].get("code") != 1 or checks[0].get("patch_empty") is not True:
            return "initial check did not observe an empty failing patch"
        return None
    if not any(check.get("code") == 0 and check.get("result") == "passed"
               and check.get("required_passed") == check.get("required_total")
               for check in checks):
        return "no passing check with full required coverage was observed"
    return None


async def _run(*, prepared_root, manifest_path, attempt_root, profile, profile_raw, credential=None,
               admission=None, offline=True):
    from tests.e2e import preview_runner as runner

    prepared, manifest_path, attempt_root = map(Path, (prepared_root, manifest_path, attempt_root))
    spec = spec_from_manifest(manifest_path)
    manifest, lock = load_json(manifest_path), runner.load_lock(prepared)
    problems = runner.verify_frozen_identity(spec, lock, prepared)
    if problems:
        raise AttemptBudgetError("frozen identity does not match: " + "; ".join(problems))
    prepared_workspace = prepared / "workspace"
    if bridge.workspace_manifest(prepared_workspace, bridge.GitCLI(prepared_workspace)).to_json() != lock["workspace"]["manifest"]:
        raise AttemptBudgetError("workspace differs from prepared baseline; refusing reset")
    if attempt_root.exists() and any(attempt_root.iterdir()):
        raise AttemptBudgetError("attempt root is not empty; refusing to overwrite evidence")
    child_root, private = attempt_root / "child", attempt_root / "private"
    private.mkdir(parents=True)
    write_json(private / "controller.json", {"task_key": spec.task_key, "supervisor_pid": os.getpid()})
    protection = isolation.seal_private(private)
    # 每个 attempt 都从只读基线创建独立工作区, 不重置演练补丁或复用旧存档。
    workspace = child_root / "workspace"
    child_root.mkdir(parents=True)
    clone_log = []
    runner.build_workspace(prepared_workspace / ".git", spec.base_commit, workspace, clone_log,
                           local_clone=True)
    owned_baseline = bridge.workspace_manifest(workspace, bridge.GitCLI(workspace)).to_json()
    write_json(private / "workspace-clone.json", {"workspace": str(workspace),
        "baseline": owned_baseline, "commands": [record.to_json() for record in clone_log]})
    if owned_baseline != lock["workspace"]["manifest"]:
        raise AttemptBudgetError("attempt clone differs from frozen baseline")
    # 评分原始材料也保存在 private 下, Low 模型只收到批准的公共摘要。
    for directory in (workspace, child_root, *(child_root / name for name in (
            "temp", "cache", "uv-cache", "appdata", "localappdata"))):
        isolation.allow_low_write(directory)
    if isolation.isolation_problems(private, protection):
        raise AttemptBudgetError("controller private materials are reachable")
    # 旧 prepare 原始日志含评分私料, 与新父进程账本一样封印, 内容不改写。
    evidence_protection = isolation.seal_private(prepared / "evidence")
    if isolation.isolation_problems(prepared / "evidence", evidence_protection):
        raise AttemptBudgetError("prepared evaluator logs are reachable")
    profile_path = child_root / "profile.json"
    # 用原 profile 原始字节, 避免重编码产生另一份价格身份。
    # 调用者传入路径只用于复制, 不把真实凭据放到模型进程。
    profile_path.write_bytes(profile_raw)
    edits = lifecycle.learned_edits(workspace, spec.base_commit,
        prepared / "private" / "gold_patch.diff", spec.allowed_paths) if offline else None
    started = time.monotonic()
    deadline = started + manifest["limits"]["run_timeout_s"]
    budget_root = private / "budget"
    budget = _make_budget(budget_root, manifest_path, prepared, profile, manifest)
    budget.acquire(PROCESS_1)
    fixture = lifecycle.offline_fixture(edits=edits, allowed_paths=spec.allowed_paths,
        sequence=lifecycle._CODING_SEQUENCE if spec.scenario == "coding" else lifecycle._FIXTURE_SEQUENCE
        ) if offline else httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    gate = lifecycle.AttemptGateTransport(fixture, budget, phase=PROCESS_1, deadline_monotonic=deadline)

    def check():
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.evaluate_workspace_patch(spec, prepared, workspace, lock, private / "checks")

    broker = Broker(gate=gate, check=check, deadline=deadline, credential=credential,
                    admission=admission, audit=private / "broker-audit.jsonl")
    endpoint = broker.start()
    # 端口在整个 attempt 内不变, 因而恢复前后的 checks_signature 相同。
    config = {"workspace": str(workspace), "endpoint": endpoint, "token": broker.token,
        "check_command": f"& '{sys.executable}' -m tests.support.swebench_live check "
                         f"--endpoint '{endpoint}' --token '{broker.token}'",
        "check_timeout_s": manifest["limits"]["check_timeout_s"],
        "profile": str(profile_path), "child_root": str(child_root),
        "phase": PROCESS_1, "scenario": spec.scenario, "model": manifest["model"],
        "instruction": (prepared / "public" / "instruction.md").read_text(encoding="utf-8")}
    facts = {"task_key": spec.task_key, "scenario": spec.scenario, "live": "not-run" if offline else "run",
        "prompt_sha256": lock["prompt_sha256"], "runtime_sha256": lifecycle.runtime_identity(),
        "manifest_sha256": sha256_file(manifest_path), "profile_sha256": profile.sha256,
        "prepared_lock_sha256": sha256_file(prepared / "lock.json"),
        "workspace": str(workspace), "baseline_manifest": owned_baseline,
        "process_ledgers": {}, "writable_directories": [str(workspace), str(child_root)],
        "problems": [], "model_requests": 0 if offline else None}
    try:
        p1 = await _supervise_child(config, child_root / "config.json", broker, deadline)
        facts["process_1"] = p1
        facts["process_ledgers"][PROCESS_1] = p1.get("process_ledger")
        if p1["os_exit_code"] != 0 or p1["timed_out"] or broker.stopped or p1.get("closed_ok") is not True:
            raise AttemptBudgetError("first child did not complete normally: " + _diagnosis(
                p1, broker, budget))
        # 首次检查的形态是**事实**而不是门禁: live 下模型可以先改后检 (见 check_gate 说明)。
        first_check = broker.checks[0] if broker.checks else None
        facts["initial_check"] = None if first_check is None else {
            key: first_check.get(key) for key in
            ("code", "result", "patch_empty", "required_passed", "required_total")}
        facts["initial_check_was_empty_failure"] = bool(
            first_check is not None and first_check.get("code") == 1
            and first_check.get("patch_empty") is True)
        if offline:
            # 离线严格判就留在 p1 之后: 首检不是空补丁失败立刻失败, 不先跑 p2 (与既有离线证据语义一致)。
            gate_problem = check_gate(broker.checks, offline=True)
            if gate_problem is not None:
                raise AttemptBudgetError(gate_problem + ": " + _diagnosis(p1, broker, budget))
        if spec.scenario == "restore-compact-continue":
            requests = p1.get("requests") or []
            if len(requests) != 2 or p1.get("manifest_unchanged") is not True or any(
                    not r.get("saved") or r.get("save_error") or r.get("stop_reason") != "stop" for r in requests):
                raise AttemptBudgetError("R1/R2 did not both commit without edits")
            checkpoint = budget.close_checkpoint(process_identity={
                "pid": p1["pid"], "os_exit_confirmed": True, "session_id": p1["session_id"],
                "request_ids": [r["request_id"] for r in requests],
                "workspace_manifest": bridge.workspace_manifest(workspace, bridge.GitCLI(workspace)).to_json(),
                "archive_manifest": archive_manifest(workspace),
                "elapsed_s": time.monotonic() - started})
            facts["checkpoint_after_process_1"] = checkpoint
            write_json(private / "process-1-checkpoint.json", checkpoint)
            budget.release()
            second = _make_budget(budget_root, manifest_path, prepared, profile, manifest)
            second.acquire(PROCESS_2)
            continuation = second.verify_continuation(expect_phase=PROCESS_1)
            saved = continuation["checkpoint"]["process_identity"]
            if (saved["archive_manifest"] != archive_manifest(workspace)
                    or saved["workspace_manifest"] != bridge.workspace_manifest(workspace, bridge.GitCLI(workspace)).to_json()
                    or saved["session_id"] != p1["session_id"]
                    or saved["elapsed_s"] >= manifest["limits"]["run_timeout_s"]):
                raise AttemptBudgetError("continuation archive/workspace/time identity differs")
            budget = second
            gate = lifecycle.AttemptGateTransport(fixture, second, phase=PROCESS_2,
                session=p1["session_id"], deadline_monotonic=deadline)
            broker.gate = gate
            config.update(phase=PROCESS_2, session_id=saved["session_id"])
            if admission:
                admission()
            p2 = await _supervise_child(config, child_root / "config.json", broker, deadline)
            facts["process_2"] = p2
            facts["process_ledgers"][PROCESS_2] = p2.get("process_ledger")
            if (p2["os_exit_code"] != 0 or p2["timed_out"] or p2.get("restored_requests") != 2
                    or p2.get("closed_ok") is not True):
                raise AttemptBudgetError("second child did not restore and complete: " + _diagnosis(
                    p2, broker, budget))
            if p2.get("restored_request_ids") != saved["request_ids"]:
                raise AttemptBudgetError("restored request identities differ")
            if p2.get("context_compacted_events") != 1 or not p2.get("compacted"):
                raise AttemptBudgetError("summary was not committed")
            if not any(call["kind"] == "summary" and call["state"] == "settled" for call in budget.recorded_calls()):
                raise AttemptBudgetError("summary send did not settle")
            final_process = p2
        else:
            final_process = p1
        if not offline:
            # live 判必须在 p2 之后: restore-compact-continue 的**通过检查发生在 p2 的 R3**
            # (R1 只读且只跑一次空补丁检查), 在 p1 处判会必然误拒生命周期题 —— 这与首题
            # "只在 live 暴露" 同型, 离线分支(严格顺序)覆盖不到, 所以不要把这段前移。
            gate_problem = check_gate(broker.checks, offline=False)
            if gate_problem is not None:
                raise AttemptBudgetError(gate_problem + ": " + _diagnosis(
                    final_process, broker, budget))
        if not final_process.get("requests") or final_process["requests"][-1].get("exit_code") != 0:
            raise AttemptBudgetError("final request has no current passing check: " + _diagnosis(
                final_process, broker, budget))
        if broker.stopped or gate.network_sealed_reason or budget.sealed_reason or budget.totals()["pending_cent"]:
            raise AttemptBudgetError("attempt sealed or has a pending reservation: " + _diagnosis(
                final_process, broker, budget))
        for ledger in facts["process_ledgers"].values():
            if not ledger or ledger["sealed_reason"] or Decimal(ledger["pending_upper_cny"] or "0"):
                raise AttemptBudgetError("production process ledger is not settled")
        facts["production_elapsed_s"] = time.monotonic() - started
        # 独立最终评分在模型进程退出后另起干净容器, 单独记时。
        grade_started = time.monotonic()
        code, grade = await asyncio.to_thread(check)
        facts["grading_elapsed_s"] = time.monotonic() - grade_started
        facts["final_grade"] = grade
        if code != 0 or broker.stopped:
            raise AttemptBudgetError("independent final grade did not pass")
        facts["outcome"] = "offline-passed-review-pending" if offline else "live-passed"
    except BaseException as exc:
        broker.stop("attempt-failed")
        facts["outcome"] = "offline-failed" if offline else "live-failed"
        if isinstance(exc, AttemptBudgetError):
            # problems 保持有界 (UI/日志一行读); 完整诊断单独落盘 —— 它是"为什么没成"的
            # 权威文本, 被 200 字截断就等于这一轮的补诊断没有到达证据面。
            facts["problems"].append(type(exc).__name__ + ": " + str(exc)[:200])
            facts["failure_diagnosis"] = str(exc)
        else:
            facts["problems"].append(type(exc).__name__)
            facts["failure_diagnosis"] = f"{type(exc).__name__}: {exc}"
    finally:
        facts.setdefault("production_elapsed_s", time.monotonic() - started)
        cleanup_started = time.monotonic()
        await broker.close()
        try:
            await fixture.aclose()
        except BaseException:
            broker.stop("parent-transport-close-failed")
            facts["problems"].append("parent transport close failed")
            facts["outcome"] = "offline-failed" if offline else "live-failed"
        facts["cleanup_elapsed_s"] = time.monotonic() - cleanup_started
        facts["checks"] = broker.checks
        facts["broker_stop_reason"] = broker.stop_reason
        facts["broker_audit"] = str(broker.audit) if broker.audit is not None else None
        facts["attempt_ledger"] = budget.totals()
        facts["attempt_calls"] = budget.recorded_calls()
        if not offline:
            facts["model_requests"] = sum(c["event"] == "reserve" for c in budget.events())
        facts["sealed_reason"] = budget.sealed_reason
        if (broker.stopped or budget.sealed_reason) and facts["outcome"].endswith(("pending", "passed")):
            facts["outcome"] = "offline-failed" if offline else "live-failed"
            facts["problems"].append("attempt stopped before cleanup completed")
        budget.release()
        write_json(attempt_root / ("offline-admission.json" if offline else "live-result.json"), facts)
    return facts


def run_supervised(*, prepared_root, manifest_path, attempt_root, profile_path,
                   authorization_path=None, admission_path=None, review_path=None, offline=True):
    from tests.support.swebench_authorization import candidate_binding, execution_binding, validate_authorization

    profile = CostProfile.load(Path(profile_path))
    # frozen dataclass: 原始 bytes 单独传给内部, 不修改生产 profile 对象。
    def binding(current_profile):
        args = dict(manifest_path=Path(manifest_path), prepared_root=Path(prepared_root),
                    attempt_root=Path(attempt_root), profile=current_profile)
        return (candidate_binding(**args) if offline else execution_binding(**args,
                admission_path=admission_path, review_path=review_path))

    expected = binding(profile)
    credential = None
    admission = None
    if not offline:
        def admission():
            current_profile = CostProfile.load(Path(profile_path))
            current = binding(current_profile)
            if current != expected:
                raise AttemptBudgetError("candidate identity changed during execution")
            return validate_authorization(authorization_path, current)
        admission()
        credential = os.environ.get("DEEPSEEK_API_KEY")
        if not credential:
            raise AttemptBudgetError("DEEPSEEK_API_KEY is missing from evaluator environment")
    return asyncio.run(_run(prepared_root=Path(prepared_root).resolve(),
        manifest_path=Path(manifest_path).resolve(), attempt_root=Path(attempt_root).resolve(),
        profile=profile, profile_raw=Path(profile_path).read_bytes(),
        credential=credential, admission=admission, offline=offline))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("child")
    child.add_argument("--config", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--endpoint", required=True)
    check.add_argument("--token", required=True)
    args = parser.parse_args()
    if args.action == "check":
        with httpx.Client(trust_env=False, timeout=300) as client:
            result = client.post(args.endpoint + "/check", headers={"X-PV-Token": args.token})
            result.raise_for_status()
            data = result.json()
            print(json.dumps(data["summary"], ensure_ascii=False))
            return data["code"]
    config = load_json(args.config)
    child_root = Path(config["child_root"])
    diagnostics: dict = {"watch": {"stop_seen": False, "failures": 0, "total_failures": 0},
                         "streams": []}
    # Low 子进程没有 std 句柄管道 (继承被父进程关闭): 自身落盘才能留下 traceback 与
    # 库里的杂散输出。日志不是关键路径 —— 打不开也必须照常写 facts。
    try:
        handle = (child_root / (config["phase"] + "-child.log")).open(
            "w", encoding="utf-8", newline="\n")
    except OSError:
        handle = None
    with contextlib.ExitStack() as stack:
        if handle is not None:
            stack.enter_context(handle)
            stack.enter_context(contextlib.redirect_stdout(handle))
            stack.enter_context(contextlib.redirect_stderr(handle))
        print(f"pid={os.getpid()} phase={config['phase']} scenario={config['scenario']}",
              flush=True)
        try:
            facts = asyncio.run(_child(args.config, diagnostics))
            code = 0
        except BaseException as exc:
            facts, code = {"error_type": type(exc).__name__, "error_detail": str(exc)[:300]}, 2
            traceback.print_exc()
    facts.setdefault("diagnostics", diagnostics)
    write_json(child_root / (config["phase"] + "-facts.json"), facts)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
