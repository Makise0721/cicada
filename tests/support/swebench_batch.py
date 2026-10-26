"""Bounded Go baseline: Windows agent, Linux verifier, no implicit live sends."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from string import Formatter
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import httpx

from tests.e2e import preview_runner as preview
from tests.support import swebench_bridge as bridge
from tests.support import swebench_isolation as isolation
from tests.support import swebench_lifecycle as lifecycle
from tests.support.swebench_budget import AttemptBudgetError, _fsync_json
from tests.support.swebench_live import Broker, RemoteGate, _supervise_child, watch_parent
from tests.support.swebench_tasks import load_json, sha256_file, write_json

PROJECT = Path(__file__).resolve().parents[2]
MODEL = {"provider": "opencode-go", "model_key": "deepseek-v4.1-flash",
         "model": "deepseek-v4.1-flash", "base_url": "https://opencode.ai/inference/go/openai/v1",
         "num_ctx": 1_000_000, "max_tokens": 8192, "think": False,
         "reasoning_effort": "high"}
LIMITS = {"calls_per_task": 24, "calls_batch": 48, "seconds_per_task": 1200,
          "seconds_batch": 2400, "grader_seconds": 240, "check_seconds": 300}
SOURCE_ROOTS = {"ansible": "lib/ansible/", "qutebrowser": "qutebrowser/"}
# 单次 write 的整文件内容超过该字节数时, 离线 fixture 改用 edit 精确替换对
# (SSE 解析器对 tool arguments 有 64 KiB 上限, 真实模型同样流不出超限 write)。
WRITE_TO_ARGUMENT_LIMIT_BYTES = 48 * 1024


def large_file_edit_pairs(base_text: str, new_text: str) -> list[dict]:
    """行级 diff 出 edit 工具替换对; old_text 带上下文扩展到原文唯一。"""
    import difflib

    base_lines = base_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    opcodes = difflib.SequenceMatcher(a=base_lines, b=new_lines, autojunk=False).get_opcodes()
    pairs = []
    prev_hi = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        lo, hi = i1, i2
        for _ in range(64):
            old = "".join(base_lines[lo:hi])
            if base_text.count(old) == 1:
                break
            if lo <= prev_hi and hi >= len(base_lines):
                raise ValueError("fixture edit old_text not unique within bound")
            lo = max(prev_hi, lo - 1)
            hi = min(len(base_lines), hi + 1)
        else:
            raise ValueError("fixture edit old_text not unique within bound")
        before, after = i1 - lo, hi - i2
        pairs.append({"old_text": old,
                      "new_text": "".join(new_lines[j1 - before:j2 + after])})
        prev_hi = hi
    return pairs


class CallBudget:
    """Durable physical-send count; deliberately makes no monetary claim."""

    def __init__(self, path: Path, max_calls: int):
        self.path, self.max_calls = path, max_calls
        self.calls: list[dict] = []
        self.sealed_reason = None

    def persist(self):
        _fsync_json(self.path, {"calls": self.calls, "max_calls": self.max_calls,
                              "sealed_reason": self.sealed_reason, "cost_cny": None})

    def reserve(self, *, call_id, phase, kind, output_cap, session=None):
        if self.sealed_reason or len(self.calls) >= self.max_calls:
            self.seal(self.sealed_reason or "physical-call-limit")
            raise AttemptBudgetError("physical call budget refused; not sent")
        self.calls.append({"call_id": call_id, "kind": kind, "output_cap": output_cap,
                           "state": "pending", "input_tokens": None, "output_tokens": None})
        self.persist()

    def settle(self, call_id, metrics, *, complete, closed=True):
        call = next(c for c in self.calls if c["call_id"] == call_id)
        valid = (complete and closed and metrics is not None
                 and type(metrics.input_tokens) is int and metrics.input_tokens >= 0
                 and type(metrics.output_tokens) is int
                 and 0 <= metrics.output_tokens <= call["output_cap"])
        if valid:
            call.update(state="complete", input_tokens=metrics.input_tokens,
                        output_tokens=metrics.output_tokens)
            self.persist()
        else:
            self.seal("incomplete-call-evidence")
        return bool(valid)

    def seal(self, reason):
        self.sealed_reason = self.sealed_reason or reason
        self.persist()


def classify_go(body_bytes):
    body = json.loads(body_bytes)
    cap = body.get("max_tokens")
    if (body.get("model") != MODEL["model"] or body.get("stream") is not True
            or body.get("thinking") != {"type": "disabled"}
            or body.get("stream_options") != {"include_usage": True}
            or type(cap) is not int or cap not in (MODEL["max_tokens"], 4096)
            or "reasoning_effort" in body):
        raise ValueError("request differs from frozen Go profile")
    return ("normal" if cap == MODEL["max_tokens"] else "summary"), cap, MODEL["model"]


def runtime_identity():
    return {**lifecycle.runtime_identity(),
            "tests/support/swebench_batch.py": lifecycle.source_sha256(Path(__file__)),
            "uv.lock": sha256_file(PROJECT / "uv.lock"),
            "pyproject.toml": lifecycle.source_sha256(PROJECT / "pyproject.toml")}


def prompt_profile(template: str) -> dict:
    parsed = list(Formatter().parse(template))
    fields = [field for _, field, _, _ in parsed
              if field is not None]
    if (sorted(fields) != ["tool_lines", "workspace"]
            or len(template.encode("utf-8")) > 16 * 1024):
        raise ValueError("prompt template requires workspace and tool_lines exactly once (16 KiB max)")
    for _, field, spec, conversion in parsed:
        if field is not None and (spec or conversion):
            raise ValueError("prompt template fields cannot have format specs or conversions")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return {"template": template, "sha256": digest}


def compose_batch_prompt(workspace, command, profile=None):
    from cicada.prompting import CheckPrompt, DEFAULT_TOOLS, build_builtin_prompt, compose_system_prompt

    tools = DEFAULT_TOOLS + ("check",)
    prompt, info = compose_system_prompt(workspace, None, tools=tools,
        checks=(CheckPrompt("check-1", command, LIMITS["check_seconds"]),))
    if profile is not None:
        base = build_builtin_prompt(workspace, tools)
        candidate = profile["template"].format(workspace=workspace.resolve(),
            tool_lines="\n".join(f"- {name}" for name in tools)).rstrip() + "\n"
        prompt = candidate + prompt[len(base):]
        info = replace(info, version="experimental:template:" + profile["sha256"],
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return prompt, info


def prepare(calibration_root: Path, package: Path, check_feedback="counts-only", prompt_template=None):
    profile = None if prompt_template is None else prompt_profile(prompt_template.read_text(encoding="utf-8"))
    package.mkdir(parents=True, exist_ok=False)
    selection = load_json(calibration_root / "selection.json")
    tasks = []
    hashes = {}
    for key, source_root in SOURCE_ROOTS.items():
        selected = next(t for t in selection["tasks"] if t["key"] == key)
        receipt = load_json(calibration_root / selected["prepare_receipt"])
        if receipt["status"] != "offline-qualified":
            raise ValueError("candidate is not offline-qualified")
        inputs = [calibration_root / "public" / key / "instruction.md",
                  calibration_root / selected["prepare_receipt"],
                  *(calibration_root / "private" / key / "tests" / n for n in bridge.VERIFIER_FILES)]
        for path in inputs:
            hashes[str(path)] = sha256_file(path)
        tasks.append({"key": key, "instance_id": selected["instance_id"],
                      "base_commit": receipt["base_commit"], "image_digest": receipt["image"]["digest"],
                      "source_root": source_root, "network_mode": receipt["verifier_network"],
                      "allowed_extra_paths": selected.get("allowed_extra_paths", [])})
    manifest = {"schema_version": 1, "model": MODEL, "limits": LIMITS, "tasks": tasks,
                "check_feedback": check_feedback, "trace_policy": "private-request-sse-v1",
                "calibration_root": str(calibration_root), "input_hashes": hashes,
                "candidate_coverage": selection.get("candidate_coverage",
                    {"qualified": 2, "original_candidates": 3}),
                "billing": "unknown; no monetary ceiling asserted", "source_runtime": runtime_identity()}
    if profile is not None:
        manifest["system_prompt"] = profile
    write_json(package / "manifest.json", manifest)
    write_json(package / "lock.json", {"manifest_sha256": sha256_file(package / "manifest.json")})
    write_json(package / "authorization-proposal.json", {
        "authorized": False, "scope": "two-tasks-once-stop-on-first-failure",
        "manifest_sha256": sha256_file(package / "manifest.json"), "limits": LIMITS,
        "model": MODEL, "check_feedback": check_feedback, "cost_cny": None,
        "offline_admission": "pending", "review": "pending"})


def verify(package: Path):
    manifest = load_json(package / "manifest.json")
    if sha256_file(package / "manifest.json") != load_json(package / "lock.json")["manifest_sha256"]:
        raise ValueError("manifest changed after freeze")
    if (manifest["model"] != MODEL or manifest["limits"] != LIMITS
            or manifest.get("check_feedback") not in ("counts-only", "required-tests-v1")
            or manifest.get("trace_policy") != "private-request-sse-v1"
            or [(t["key"], t["source_root"]) for t in manifest["tasks"]] != list(SOURCE_ROOTS.items())
            or manifest["source_runtime"] != runtime_identity()):
        raise ValueError("execution profile or source runtime changed after freeze")
    if "system_prompt" in manifest:
        profile = manifest["system_prompt"]
        if prompt_profile(profile["template"]) != profile:
            raise ValueError("frozen prompt template identity mismatch")
    for name, digest in manifest["input_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("frozen task input changed")
    return manifest


def live_admission(package, run_root, authorization, admission, review):
    manifest = verify(package)
    if None in (authorization, admission, review):
        raise ValueError("live requires explicit authorization, offline admission and review")
    auth, offline, reviewed = map(load_json, (authorization, admission, review))
    binding = {"manifest_sha256": sha256_file(package / "manifest.json"),
               "run_root": str(run_root), "admission_sha256": sha256_file(admission),
               "review_sha256": sha256_file(review)}
    if (auth.get("authorized") is not True or auth.get("binding") != binding
            or auth.get("scope") != "two-tasks-once-stop-on-first-failure"
            or offline.get("status") != "offline-passed" or offline.get("manifest_sha256") != binding["manifest_sha256"]
            or reviewed.get("status") != "passed" or reviewed.get("manifest_sha256") != binding["manifest_sha256"]):
        raise ValueError("authorization or offline evidence does not bind this execution")
    return manifest


async def child(config_path):
    from cicada.application import ApplicationConfig, open_application
    from cicada.budget import BudgetConfig
    from cicada.plugins.coding.verification import generate_verification_run_id
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
    from cicada.plugins.openai_chat.opencode_go import GoConfig, GoModel
    from cicada.runtime.plugin import PluginDefinition

    config = load_json(config_path)
    root, workspace = Path(config["child_root"]), Path(config["workspace"])
    remote = RemoteGate(config["endpoint"], config["token"])
    model_config = GoConfig(**MODEL)

    def setup(ctx):
        client = httpx.AsyncClient(transport=remote, trust_env=False, follow_redirects=False)
        ctx.defer(client.aclose)
        ctx.provide("model", GoModel(client, model_config, lifecycle.OFFLINE_CREDENTIAL,
            session_id_provider=lambda: config["routing_session"]))

    command = config["check_command"]
    prompt, info = compose_batch_prompt(workspace, command, config.get("system_prompt"))
    plan = VerificationPlan(generate_verification_run_id(), workspace,
        (CheckDefinition("check-1", command, LIMITS["check_seconds"]),))
    session = await open_application(ApplicationConfig(workspace, prompt, info,
        PluginDefinition(name="batch-go", setup=setup, provides=frozenset({"model"})),
        plan=plan, max_turns=50, model_label=MODEL["model"], persistence=None,
        budget=BudgetConfig(MODEL["num_ctx"], MODEL["max_tokens"]),
        runtime_dir=root / "runtime"))
    log = lifecycle._EventLog()
    session.subscribe(log)
    diagnostics = {"watch": {}, "streams": []}
    extra = config.get("allowed_extra_paths") or ()
    extra_note = f"此外还可修改这些配置数据文件：{'、'.join(extra)}。" if extra else ""
    text = config["instruction"] + (
        f"\n/app 是当前 Windows 工作区。自行读取并定位，只修改 {config['source_root']} 下的 Python 源码，"
        "允许新增 Python 文件；不得修改测试、依赖或项目指令。禁止联网、查找参考答案、提交。" + extra_note +
        "宿主为 Windows/PowerShell，不在宿主运行或导入工作区代码；"
        "先用 check-1 观察初始失败，修改后再次 check-1，检查在固定 Linux 容器执行。")
    facts = {}
    try:
        with patch.object(GoModel, "stream", lifecycle.observe_model_stream(GoModel.stream, remote, diagnostics)):
            task = asyncio.create_task(session.execute(text))
            watcher = asyncio.create_task(watch_parent(config["endpoint"], config["token"], task, diagnostics))
            try:
                outcome = await task
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
        facts.update(exit_code=outcome.exit_code, stop_reason=outcome.result.stop_reason,
                     delivery=None if outcome.decision is None else asdict(outcome.decision),
                     completed_events=log.completed, diagnostics=diagnostics,
                     model_elapsed_s=sum(c.elapsed_s for c in outcome.result.model_calls),
                     prompt_sha256=info.prompt_sha256, prompt_version=info.version)
    finally:
        await lifecycle.close_process(session, log, facts)
        write_json(root / "process-1-facts.json", facts)
    return facts


def in_business_scope(task, path: str) -> bool:
    """业务范围: source_root 下 Python 源码, 加每题冻结的精确白名单路径 (如 .yml 配置数据)。"""
    return (path.startswith(task["source_root"]) and path.endswith(".py")
            or path in task.get("allowed_extra_paths", ()))


def export(workspace, task, temp):
    git = bridge.GitCLI(workspace)
    if git.run("rev-parse", "HEAD", check=True).stdout.strip() != task["base_commit"]:
        raise ValueError("workspace HEAD changed")
    status = git.run("status", "--porcelain", "-z", "--untracked-files=all", "--ignored").stdout
    paths = bridge.porcelain_paths(status)
    allowed = tuple(p for p in paths if in_business_scope(task, p))
    return bridge.export_patch(workspace, task["base_commit"], allowed, temp, git)


async def run_task(manifest, task, root, *, offline, credential, admission, max_calls, batch_deadline):
    calibration = Path(manifest["calibration_root"])
    private, child_root = root / "private", root / "child"
    private.mkdir(parents=True)
    child_root.mkdir()
    write_json(private / "controller.json", {"task": task["key"], "pid": os.getpid()})
    facts = {"task": task["key"], "live": not offline, "status": "failed"}
    attempt_started = time.monotonic()
    try:
        protection = isolation.seal_private(private)
        workspace = child_root / "workspace"
        preview.build_workspace(calibration / "prepared" / task["key"] / "workspace" / ".git",
                                task["base_commit"], workspace, [], local_clone=True)
        for path in [child_root, workspace, *(child_root / p for p in ("temp", "cache", "uv-cache", "appdata", "localappdata"))]:
            isolation.allow_low_write(path)
        problems = isolation.isolation_problems(private, protection)
        if problems:
            raise ValueError("attempt private isolation failed: " + "; ".join(problems))
        started = time.monotonic()
        deadline = min(batch_deadline, started + LIMITS["seconds_per_task"])
        budget = CallBudget(private / "calls.json", min(LIMITS["calls_per_task"], max_calls))
        if offline:
            gold = calibration / "private" / task["key"] / "solution/gold_patch.diff"
            names = subprocess.check_output(["git", "apply", "--numstat", str(gold)], encoding="utf-8").splitlines()
            paths = tuple(line.split("\t", 2)[2] for line in names)
            # Official gold may touch changelogs/docs outside the agent protocol; the
            # rehearsal must stay inside it (in_business_scope). If those extra files were
            # required to pass, the fixture check below fails and admission is refused.
            business = tuple(p for p in paths if in_business_scope(task, p))
            if not business:
                raise ValueError("offline reference has no in-scope python source files")
            fixed = lifecycle.learned_solution(workspace, task["base_commit"], gold, business)
            # A single write whose whole-file content exceeds the SSE tool-argument byte
            # limit is not a shape the live model could stream either; large files use
            # precise edit pairs derived from the base text (frame 0 already read them).
            large_edits = {name: large_file_edit_pairs((workspace / name).read_text(encoding="utf-8"), text)
                           for name, text in fixed.items()
                           if len(text.encode("utf-8")) > WRITE_TO_ARGUMENT_LIMIT_BYTES}
            change_calls = [("write", {"path": name, "content": text})
                            for name, text in fixed.items() if name not in large_edits]
            change_calls += [("edit", {"path": name, "edits": pairs})
                             for name, pairs in large_edits.items()]
            frames = [
                lifecycle._call_frame(0, [("read", {"path": p}) for p in business]
                    + [("check", {"action": "run", "check_id": "check-1"})], finish="tool_calls"),
                lifecycle._call_frame(1, change_calls
                    + [("check", {"action": "run", "check_id": "check-1"})], finish="tool_calls"),
                lifecycle._text_frame(2, "offline rehearsal complete."),
            ]
            for frame in frames:
                frame["choices"][0]["delta"].pop("reasoning_content", None)
            fixture = iter(frames)
            inner = httpx.MockTransport(lambda request: lifecycle._sse([next(fixture)]))
        else:
            inner = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        gate = lifecycle.AttemptGateTransport(inner, budget, phase="process-1",
            deadline_monotonic=deadline, request_classifier=classify_go)
        docker = bridge.DockerCLI([])
        supervisor = bridge.ContainerSupervisor(private, docker)
        f2p, p2p = bridge.parse_required_tests((calibration / "private" / task["key"] / "tests/config.json").read_bytes())
        grades = []

        def check():
            check_started = time.monotonic()
            exported = export(workspace, task, private)
            result = bridge.ContainerEvaluator(docker, supervisor).evaluate(bridge.ContainerSpec(
                image=task["image_digest"], image_digest=task["image_digest"],
                verifier_dir=calibration / "private" / task["key"] / "tests",
                owner_label=supervisor.owner_label("batch-check"), network_mode=task["network_mode"]),
                exported.patch_bytes, private / f"grade-{len(grades) + 1}", f2p, p2p,
                timeout_s=min(LIMITS["grader_seconds"], max(0.01, deadline - time.monotonic())))
            # Re-export detects mutations while the container was checking the captured patch.
            fresh = export(workspace, task, private).source_manifest == exported.source_manifest
            official = result.output_json or {}
            anomaly = (result.timed_out or result.apply_failed or result.cleanup != "confirmed"
                       or not official.get("artifacts_complete") or not fresh)
            passed = bool(official.get("grader_pass")) and not anomaly
            code = 2 if anomaly else 0 if passed else 1
            summary = {"result": "anomaly" if anomaly else "passed" if passed else "failed",
                       "required_passed": result.required_passed, "required_total": result.required_total,
                       "patch_empty": exported.empty, "grade_id": str(len(grades) + 1),
                       "cleanup": result.cleanup, "elapsed_s": time.monotonic() - check_started,
                       "not_passed_tests": result.missing_required,
                       "snapshot": exported.source_manifest}
            grades.append(summary)
            write_json(private / f"grade-{len(grades)}.json", result.to_json())
            return code, summary

        routing = uuid4().hex
        broker = Broker(gate=gate, check=check, deadline=deadline, credential=credential,
            admission=admission, audit=private / "broker.jsonl",
            model_url=MODEL["base_url"] + "/chat/completions",
            model_headers={"User-Agent": "cicada/" + version("cicada"), "x-opencode-session": routing},
            check_feedback=manifest["check_feedback"], trace_dir=private / "model-trace")
        endpoint = broker.start()
        config = {"workspace": str(workspace), "child_root": str(child_root),
                  "endpoint": endpoint, "token": broker.token, "routing_session": routing,
                  "phase": "process-1", "child_module": "tests.support.swebench_batch",
                  "source_root": task["source_root"],
                  "allowed_extra_paths": task.get("allowed_extra_paths", []),
                  "instruction": (calibration / "public" / task["key"] / "instruction.md").read_text(encoding="utf-8"),
                  "check_command": f"& '{sys.executable}' -m tests.support.swebench_batch check --endpoint '{endpoint}' --token '{broker.token}'"}
        if "system_prompt" in manifest:
            config["system_prompt"] = manifest["system_prompt"]
    except (Exception, asyncio.CancelledError) as exc:
        # 准备段 (封印/克隆/参考解/评分装配) 失败也必须留任务级失败档案; 运行段由下方 finally 落盘。
        facts.update(failure_type=type(exc).__name__, elapsed_s=time.monotonic() - attempt_started)
        write_json(private / "result.json", facts)
        raise
    try:
        facts["child"] = await _supervise_child(config, child_root / "config.json", broker, deadline)
        final = export(workspace, task, private)
        current_pass = any(g["result"] == "passed" and g["snapshot"] == final.source_manifest for g in grades)
        complete = all(c["state"] == "complete" for c in budget.calls) and bool(budget.calls)
        child_facts = facts["child"]
        passed = (child_facts.get("os_exit_code") == 0 and child_facts.get("closed_ok") is True
                  and child_facts.get("exit_code") == 0 and current_pass and complete
                  and not broker.stopped and not budget.sealed_reason)
        facts.update(status="passed" if passed else "failed", patch_sha256=final.sha256,
                     changed_paths=final.changed_paths, token_usage_complete=complete,
                     official_patch_passed=current_pass,
                     engineering_delivered=bool((child_facts.get("delivery") or {}).get("can_deliver")))
    except (Exception, asyncio.CancelledError) as exc:
        broker.stop("attempt-exception")
        facts["failure_type"] = type(exc).__name__
    finally:
        try:
            await broker.close()
        except Exception as exc:
            facts.update(status="close-unconfirmed", broker_close_error=type(exc).__name__)
        try:
            await inner.aclose()
        except Exception as exc:
            facts.update(status="close-unconfirmed", transport_close_error=type(exc).__name__)
        facts.update(container_cleanup=supervisor.cleanup_registered(), calls=budget.calls,
                     checks=grades, physical_sends=gate.handoffs, cost_cny=None,
                     elapsed_s=time.monotonic() - started, stop_reason=broker.stop_reason,
                     sealed_reason=budget.sealed_reason,
                     transport_observations=[asdict(o) for o in gate.observations.values()])
        if any(c["outcome"] == "uncertain" for c in facts["container_cleanup"]):
            facts["status"] = "cleanup-unconfirmed"
        write_json(private / "result.json", facts)
    return facts


def seal_calibration(root, keys):
    directories = [root / "private",
                   *(root / "prepared" / k / "evidence" for k in keys)]
    if (root / "history").is_dir():
        directories.append(root / "history")
    records = []
    for directory in directories:
        protection = isolation.seal_private(directory)
        problems = isolation.isolation_problems(directory, protection)
        records.append({"directory": str(directory), "problems": problems})
        if problems:
            raise ValueError("calibration materials are reachable by low process")
    # These summaries embed verifier logs; stamp their actual files as well.
    summaries = [root / "offline-preflight.json", *(root / "prepared" / k / "prepare-receipt.json"
                 for k in keys)]
    for path in summaries:
        isolation._icacls(str(path), "/grant", f"*{isolation.PROTECTED_PACKAGE_SID}:(R)")
        if isolation.low_child_exit_code(f'type "{path}" >nul 2>&1', root) == 0:
            raise ValueError("calibration summary is readable by low process")
    if isolation.low_child_exit_code("docker version >nul 2>&1", root) == 0:
        raise ValueError("low child can access Docker")
    return records


async def run(package, run_root, *, live=False, authorization=None, admission=None, review=None):
    manifest = verify(package)
    admit = None
    credential = None
    if live:
        admit = lambda: live_admission(package, run_root, authorization, admission, review)
        admit()  # Must precede credential lookup and any real transport construction.
        credential = os.environ.get("CICADA_SWEBENCH_GO_KEY")
        if not credential:
            raise ValueError("CICADA_SWEBENCH_GO_KEY is missing in evaluator environment")
    run_root.mkdir(parents=True, exist_ok=False)
    controller = run_root / "private"
    controller.mkdir()
    write_json(controller / "controller.json", {"pid": os.getpid(), "live": live})
    isolation.seal_private(controller)
    protection = seal_calibration(Path(manifest["calibration_root"]), [t["key"] for t in manifest["tasks"]])
    deadline = time.monotonic() + LIMITS["seconds_batch"]
    result = {"status": "running", "live": live, "manifest_sha256": sha256_file(package / "manifest.json"),
              "model_requests": 0, "tasks": [], "candidate_coverage": manifest["candidate_coverage"],
              "isolation": protection, "cost_cny": None}
    for task in manifest["tasks"]:
        try:
            facts = await run_task(manifest, task, run_root / task["key"], offline=not live,
                credential=credential, admission=admit,
                max_calls=LIMITS["calls_batch"] - sum(t["physical_sends"] for t in result["tasks"]),
                batch_deadline=deadline)
        except (Exception, asyncio.CancelledError) as exc:
            # 控制器级异常/取消也先落任务失败事实与公开摘要再上抛, 不依赖 stdout 留证。
            result["tasks"].append({"task": task["key"], "status": "failed",
                                    "failure_type": type(exc).__name__})
            result["status"] = "failed"
            write_json(controller / "result.json", result)
            write_json(run_root / "summary.json", {k: result[k] for k in (
                "status", "live", "model_requests", "manifest_sha256", "candidate_coverage", "cost_cny")})
            raise
        result["tasks"].append(facts)
        result["model_requests"] += facts["physical_sends"] if live else 0
        write_json(controller / "result.json", result)
        if facts["status"] != "passed":
            break
    result["status"] = ("live-passed" if live else "offline-passed") if (
        len(result["tasks"]) == len(manifest["tasks"]) and all(t["status"] == "passed" for t in result["tasks"])) else "failed"
    result["input_tokens"] = sum(c["input_tokens"] for t in result["tasks"] for c in t["calls"]) if all(
        c["state"] == "complete" for t in result["tasks"] for c in t["calls"]) else None
    result["output_tokens"] = sum(c["output_tokens"] for t in result["tasks"] for c in t["calls"]) if result["input_tokens"] is not None else None
    write_json(controller / "result.json", result)
    write_json(run_root / "summary.json", {k: result[k] for k in (
        "status", "live", "model_requests", "manifest_sha256", "candidate_coverage", "cost_cny")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--calibration-root", type=Path, required=True)
    prep.add_argument("--package", type=Path, required=True)
    prep.add_argument("--check-feedback", choices=("counts-only", "required-tests-v1"), default="counts-only")
    prep.add_argument("--prompt-template", type=Path,
                      help="Freeze an experimental replacement for the built-in base prompt")
    launch = sub.add_parser("run")
    launch.add_argument("--package", type=Path, required=True)
    launch.add_argument("--run-root", type=Path, required=True)
    launch.add_argument("--live", action="store_true")
    for name in ("authorization", "admission", "review"):
        launch.add_argument("--" + name, type=Path)
    worker = sub.add_parser("child")
    worker.add_argument("--config", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--endpoint", required=True)
    check.add_argument("--token", required=True)
    args = parser.parse_args()
    if args.action == "check":
        with httpx.Client(trust_env=False, timeout=LIMITS["grader_seconds"] + 30) as client:
            response = client.post(args.endpoint + "/check", headers={"X-PV-Token": args.token})
            response.raise_for_status()
            data = response.json()
            print(json.dumps(data["summary"], ensure_ascii=False))
            return data["code"]
    if args.action == "child":
        root = Path(load_json(args.config)["child_root"])
        # Low processes have no inherited output handles; preserve errors locally.
        import contextlib
        import traceback
        with (root / "child.log").open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                asyncio.run(child(args.config))
            except BaseException:
                traceback.print_exc()
                return 2
        return 0
    try:
        if args.action == "prepare":
            prepare(args.calibration_root.resolve(), args.package.resolve(), args.check_feedback, args.prompt_template)
            print(json.dumps({"status": "prepared", "model_requests": 0}))
        else:
            result = asyncio.run(run(args.package.resolve(), args.run_root.resolve(), live=args.live,
                authorization=args.authorization, admission=args.admission, review=args.review))
            print(json.dumps({"status": result["status"], "model_requests": result["model_requests"]}))
            return 0 if result["status"].endswith("passed") else 1
    except (OSError, ValueError, AttemptBudgetError, bridge.BridgeError, isolation.IsolationError) as exc:
        print(json.dumps({"status": "refused-or-failed", "error": str(exc)[:300]}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
