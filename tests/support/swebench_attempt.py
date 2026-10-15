"""一次性 attempt 生命周期、真实 CLI 与独立最终评分的组合。"""
from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import importlib.metadata
import platform
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx

from tests.support.swebench_bridge import GitCLI, workspace_manifest
from tests.support.swebench_cost import CostLedger, CostTransport, PRICING_IDENTITY
from tests.support.swebench_run import run_cli


DISCIPLINE = ("上述 /app 指本次 Windows 工作区，源码与官方基础版本一致。请自行读取与定位问题，"
    "只修改 lib/ansible/utils/vars.py 和 lib/ansible/vars/manager.py。先用 check-1 确认初始未通过，"
    "修改后用 check-1 重检；检查在固定 Linux 容器运行。不得改测试、样本、依赖或项目指令，"
    "不查找参考修复，不联网、不提交。不要在宿主机对工作区执行 python 或导入其代码，验证只用 check-1。"
    "普通环境是 Windows/PowerShell。最后如实说明修改及指定检查结果；"
    "未解决就报告未完成。")


class PreflightRejected(RuntimeError):
    """只在构造/交接模型 transport 之前产生，保证模型 HTTP 为零。"""


def runtime_identity():
    project = Path(__file__).resolve().parents[2]
    paths = list((project / "src").rglob("*.py")) + [project / "pyproject.toml", project / "uv.lock",
        project / "tests/e2e/swebench_runner.py", project / "tests/e2e/p4_observer.py",
        project / "tests/support/swebench_offline.py",
        *(project / "tests/support" / name for name in (
            "swebench_bridge.py", "swebench_cost.py", "swebench_owner.py", "swebench_run.py", "swebench_attempt.py"))]
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def environment_identity():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version)
                                     for d in importlib.metadata.distributions()))}


def verify_live_pricing(prepared):
    """本次运行前人工核对的公开价格快照；旧日期/未知字段拒绝，零 API 调用。"""
    proof = json.loads((prepared / "live-pricing.json").read_text(encoding="utf-8"))
    expected = {k: v for k, v in PRICING_IDENTITY.items() if k != "checked"}
    if proof.get("pricing") != expected or proof.get("model") != "deepseek-flash":
        raise ValueError("current pricing identity differs from frozen contract")
    checked = datetime.fromisoformat(proof["checked_at"])
    if checked.tzinfo is None:
        raise ValueError("live pricing proof needs an explicit timezone")
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if not 0 <= age <= 24 * 3600 or proof.get("server_context_tokens") != 1_000_000:
        raise ValueError("live pricing proof is stale or server contract differs")
    source = prepared / "live-pricing-source.html"
    if hashlib.sha256(source.read_bytes()).hexdigest() != proof.get("source_sha256"):
        raise ValueError("live pricing source snapshot differs")
    rate = proof.get("cache_hit_input_cny_per_million")
    if rate is not None:
        amount = Decimal(str(rate))
        if not amount.is_finite() or not 0 <= amount <= Decimal("2"):
            raise ValueError("cache pricing contradicts the uncached upper bound")
    return proof


def prepare_attempt(prepared, root, *, live):
    from tests.e2e import swebench_runner as h

    if not root.is_relative_to(h.PROJECT) or root == h.PROJECT or root.exists() or root.is_relative_to(prepared):
        raise ValueError("attempt must be a new directory inside the project")
    receipt = h.load_json(prepared / h.RECEIPT_FILENAME)
    if receipt.get("outcome") != "prepared":
        raise ValueError("predecessor preparation is not complete")
    lock = h.load_lock(prepared)
    problems = h.verify_frozen_identity(lock, prepared / "evidence")
    if problems:
        raise ValueError("prepared frozen identity changed")
    before = workspace_manifest(prepared / "workspace", GitCLI(prepared / "workspace"))
    if before.to_json() != lock["workspace"]["manifest"] or before.dirty_paths:
        raise ValueError("prepared workspace is no longer the clean fixed base")
    pricing = None
    if live:
        admission = h.load_json(prepared / "offline-admission.json")
        if (admission.get("outcome") != "admitted" or admission.get("review_status") != "passed"
                or admission.get("runtime_sha256") != runtime_identity()
                or admission.get("lock_sha256") != h.sha256_file(prepared / h.LOCK_FILENAME)
                or admission.get("environment") != environment_identity()):
            raise ValueError("offline admission is absent or stale")
        offline_root = Path(admission["attempt_root"])
        if (not offline_root.is_relative_to(h.PROJECT)
                or h.sha256_file(offline_root / "result.json") != admission.get("result_sha256")
                or len(admission.get("reviews", {})) < 2):
            raise ValueError("offline result or independent review evidence is absent or changed")
        for path, digest in admission["reviews"].items():
            review = Path(path)
            if not review.is_relative_to(h.PROJECT) or h.sha256_file(review) != digest:
                raise ValueError("independent review evidence changed")
        from tests.support.swebench_offline import report
        if report(offline_root)["outcome"] != "passed":
            raise ValueError("offline raw evidence is absent, changed, or no longer current")
        pricing = verify_live_pricing(prepared)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("public", "private", "evidence", "temp"):
        (root / name).mkdir()
    commands = []
    h.build_workspace(prepared / "workspace", lock["instance"]["base_commit"], root / "workspace", commands)
    h.write_json(root / "evidence/checkout-commands.json", [r.to_json() for r in commands])
    manifest = workspace_manifest(root / "workspace", GitCLI(root / "workspace"))
    if manifest.to_json() != lock["workspace"]["manifest"]:
        raise ValueError("fresh attempt does not match the complete fixed base")
    instruction = (prepared / "public/instruction.md").read_bytes()
    if hashlib.sha256(instruction).hexdigest() != lock["instruction_sha256"]:
        raise ValueError("public instruction identity changed")
    (root / "public/instruction.md").write_bytes(instruction)
    h.verifier_dir_for(root / "private")  # 不复制参考修复到真实 attempt。
    h.write_json(root / h.LOCK_FILENAME, copy.deepcopy(lock))
    h.write_json(root / "evidence/provenance.json", {"prepared_root": str(prepared),
        "prepared_lock_sha256": h.sha256_file(prepared / h.LOCK_FILENAME),
        "runtime_sha256": runtime_identity(), "live_pricing": pricing})
    prompt = instruction.decode("utf-8") + "\n\n" + DISCIPLINE
    (root / "public/prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    return lock, prompt


def classify_result(production, transport, grade, cleanup, current_snapshot, identity_problems):
    delivery = production.get("delivery") or {}
    reason = transport.stop_reason
    if reason in ("budget-limit", "call-limit", "run-timeout"):
        return {"budget-limit": "stopped-budget", "call-limit": "stopped-call-limit",
                "run-timeout": "stopped-timeout"}[reason]
    if identity_problems or cleanup != "confirmed" or production["recording_errors"]:
        return "inconclusive-evidence"
    if transport.ledger.sealed or reason:
        return "failed-provider"
    if grade.get("result") != "passed":
        return "failed-behavior" if grade.get("result") == "failed" else "inconclusive-evidence"
    view = delivery.get("view") or {}
    checks = view.get("checks", [])
    if (production["main_exit_code"] != 0 or not production["application_closed"]
            or (delivery.get("result") or {}).get("stop_reason") != "stop"
            or not (delivery.get("decision") or {}).get("can_deliver")
            or not (delivery.get("evidence") or {}).get("complete")
            or not current_snapshot or current_snapshot != view.get("snapshot_ref")
            or not checks or any(c.get("freshness") != "current" or
                (c.get("receipt") or {}).get("verification_status") != "passed" for c in checks)):
        return "failed-delivery"
    return "passed"


def pricing_estimates(calls, cache_hit_rate=None):
    uncached = Decimal("0")
    cached = Decimal("0")
    complete = True
    hit_total = miss_total = covered = 0
    rate = Decimal(str(cache_hit_rate)) if cache_hit_rate is not None else None
    for call in calls:
        usage = call.get("usage")
        if call["state"] != "settled" or usage is None:
            complete = False
            continue
        prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
        uncached += (Decimal(prompt) * 2 + Decimal(output) * 8) / Decimal(1_000_000)
        hit, miss = usage.get("cache_hit_input_tokens"), usage.get("cache_miss_input_tokens")
        if hit is None or miss is None or hit + miss != prompt:
            complete = False
            continue
        hit_total += hit
        miss_total += miss
        if hit and rate is None:
            complete = False
            continue
        cached += (Decimal(hit) * (rate or Decimal("0")) + Decimal(miss) * 2 + Decimal(output) * 8) / Decimal(1_000_000)
        covered += 1
    return str(uncached), {"status": "available" if complete else "unavailable",
        "cny": str(cached) if complete else None, "known_part_cny": str(cached),
        "cache_hit_input_cny_per_million": str(rate) if rate is not None else None,
        "cache_hit_tokens_known": hit_total, "cache_miss_tokens_known": miss_total,
        "covered_calls": covered, "total_calls": len(calls),
        "reason": None if complete else "cache usage or frozen cache price is incomplete"}


def execute_attempt(prepared, root, *, inner=None, offline=False, timeout_s=900):
    from tests.e2e import swebench_runner as h

    started = time.time()
    try:
        lock, prompt = prepare_attempt(prepared, root, live=not offline)
        if offline:
            if inner is None:
                raise ValueError("offline mode requires a local transport")
            key = "offline-benchmark-key"
        else:
            if inner is not None or timeout_s != 900:
                raise ValueError("live transport and runtime overrides are forbidden")
            from cicada.plugins.user_config import FileUserConfigStore
            credential = FileUserConfigStore(Path.home() / ".cicada").load(os.environ.get("DEEPSEEK_API_KEY")).credential
            if credential is None:
                raise ValueError("live credential is missing")
            key = credential.reveal()
            inner = httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1), trust_env=False)
    except Exception as exc:
        raise PreflightRejected(str(exc)) from exc
    ledger = CostLedger(root / "evidence/cost")
    cleanup = "pending"
    production = {"main_exit_code": None, "application_closed": False,
                  "recording_errors": [], "delivery": None}
    grade = {}
    current_snapshot = None
    identity_problems = []
    transport = CostTransport(inner, ledger)
    initial_runtime = runtime_identity()
    initial_environment = environment_identity()
    grade_evidence = {}
    final_patch = {}
    try:
        ledger.acquire({"attempt": root.name, "mode": "offline" if offline else "live",
            "design": lock["design"], "prepared_lock_sha256": h.sha256_file(prepared / h.LOCK_FILENAME)})
        with h.evaluation_owner(root):
            production = run_cli(root, transport, key, prompt, timeout_s=timeout_s)
            if not production["application_closed"] or transport.active_call_id is not None:
                transport.stop("production-close-unconfirmed")
            ledger.finalize()
            _, grade = h.evaluate_workspace_patch(root, root / "workspace", lock, root / "evidence", require_external_owner=False)
            h.write_json(root / "evidence/final-grade.json", grade)
            if grade.get("grade_id"):
                grade_dir = root / "evidence/checks" / grade["grade_id"]
                grade_evidence = h.load_json(grade_dir / "evaluation.json")
                final_patch = h.load_json(grade_dir / "patch.json")
            qualification = h.eligibility_from_production(root / "workspace", root / "temp")
            current_snapshot = qualification["snapshot_ref"]
            h.write_json(root / "evidence/final-qualification.json", qualification)
            identity_problems = h.verify_frozen_identity(lock, root / "evidence")
            if environment_identity() != initial_environment:
                identity_problems.append("runtime environment changed during attempt")
            if runtime_identity() != initial_runtime:
                identity_problems.append("harness source changed during attempt")
        cleanup = "confirmed"
    except BaseException as exc:
        transport.stop("attempt-exception")
        identity_problems.append(type(exc).__name__)
    finally:
        if not transport.client_closed:
            try:
                asyncio.run(transport.aclose())
            except BaseException as exc:
                identity_problems.append(type(exc).__name__)
        if ledger.budget:
            h.write_json(root / "evidence/cost/calls-final.json", ledger.calls)
        ledger.release_lock()
    outcome = classify_result(production, transport, grade, cleanup, current_snapshot, identity_problems)
    budget = ledger.budget
    pricing_proof = h.load_json(root / "evidence/provenance.json").get("live_pricing") or {}
    estimate, cache_estimate = pricing_estimates(ledger.calls, pricing_proof.get("cache_hit_input_cny_per_million"))
    result = {"outcome": outcome, "attempt": root.name, "design": lock["design"],
        "mode": "offline-mock" if offline else "live", "live": "not-run" if offline else "run-once",
        "started": started, "ended": time.time(), "production": production, "grade": grade,
        "grade_evidence": grade_evidence, "final_patch": final_patch,
        "current_snapshot_ref": current_snapshot, "cleanup": cleanup, "identity_problems": identity_problems,
        "http_handoffs": transport.handoffs, "summary_handoffs": sum(c["kind"] == "summary" and c.get("handed_off", False) for c in ledger.calls),
        "usage_coverage": sum(c["state"] == "settled" for c in ledger.calls), "budget": budget,
        "stop_reason": transport.stop_reason, "known_pricing_estimate": estimate,
        "known_pricing_estimate_basis": "uncached equivalent of known settled calls; unrounded",
        "cache_aware_pricing_estimate": cache_estimate,
        "account_actual": "unavailable",
        "fee_bound_holds": budget.get("seal_reason") not in ("prompt-over-server-bound", "completion-over-cap")
            and Decimal(budget.get("total_upper", "0")) <= Decimal("4.50"),
        "runtime_sha256": initial_runtime, "environment": initial_environment, "runtime_artifacts": str(root / "workspace/.cicada"),
        "evidence_index": [p.relative_to(root).as_posix() for p in sorted((root / "evidence").rglob("*")) if p.is_file()]}
    result["evidence_sha256"] = {relative: h.sha256_file(root / relative) for relative in result["evidence_index"]}
    h.write_json(root / "result.json", result)
    return result
