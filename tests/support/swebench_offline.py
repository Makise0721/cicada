"""同一固定题目的离线响应 fixture、报告与准入证据；不调用真实模型。"""
from __future__ import annotations

import json
import shutil
import stat
import time
import uuid
from pathlib import Path

import httpx

from tests.support.swebench_attempt import execute_attempt, runtime_identity, environment_identity
from tests.support.swebench_cost import _atomic_json
from tests.support.swebench_bridge import GitCLI, ALLOWED_PATHS, workspace_manifest


def fixture_transport(prepared):
    from tests.e2e import swebench_runner as h

    h.verify_downloaded_materials()
    if (prepared / "private/gold_patch.diff").read_bytes() != (h.downloads_dir() / "upstream/solution/gold_patch.diff").read_bytes():
        raise ValueError("private calibration patch identity differs")
    fixture = prepared / "private" / ("offline-fixture-" + uuid.uuid4().hex)
    h.build_workspace(prepared / "workspace", h.load_lock(prepared)["instance"]["base_commit"], fixture, [])
    try:
        GitCLI(fixture).run("apply", str(prepared / "private/gold_patch.diff"), check=True)
        replacements = {name: (fixture / name).read_text(encoding="utf-8") for name in ALLOWED_PATHS}
    finally:
        if not fixture.resolve().is_relative_to((prepared / "private").resolve()):
            raise ValueError("private fixture cleanup escaped its root")
        def writable_retry(function, path, exc_info):
            Path(path).chmod(stat.S_IWRITE)
            function(path)
        shutil.rmtree(fixture, onerror=writable_retry)
    number = 0

    def respond(request):
        nonlocal number
        number += 1
        if number == 1:
            calls = [("read", {"path": name}) for name in ALLOWED_PATHS]
            calls.append(("check", {"action": "run", "check_id": "check-1"}))
        elif number == 2:
            calls = [("write", {"path": name, "content": contents}) for name, contents in replacements.items()]
        elif number == 3:
            calls = [("check", {"action": "run", "check_id": "check-1"})]
        elif number == 4:
            calls = []
        else:
            raise ValueError("offline script exhausted")
        delta = {"reasoning_content": "offline calibration fixture"}
        if calls:
            delta["tool_calls"] = [{"index": i, "id": f"fixture-{number}-{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
                for i, (name, arguments) in enumerate(calls)]
        else:
            delta["content"] = "Offline fixture complete; evaluate recorded check facts."
        payload = {"choices": [{"index": 0, "delta": delta,
                   "finish_reason": "tool_calls" if calls else "stop"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        stream = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode("utf-8")
        return httpx.Response(200, stream=httpx.ByteStream(stream))

    return httpx.MockTransport(respond)


def report(root):
    from tests.e2e import swebench_runner as h

    result = h.load_json(root / "result.json")
    problems = []
    if result.get("runtime_sha256") != runtime_identity():
        problems.append("runtime source differs from recorded attempt")
    if result.get("environment") != environment_identity():
        problems.append("runtime environment differs from recorded attempt")
    for relative, digest in result.get("evidence_sha256", {}).items():
        path = root / relative
        if not path.is_file() or h.sha256_file(path) != digest:
            problems.append("attempt evidence is missing or changed: " + relative)
    final_patch = result.get("final_patch") or {}
    if final_patch.get("source_manifest") != workspace_manifest(root / "workspace", GitCLI(root / "workspace")).to_json():
        problems.append("workspace differs from the independently graded source")
    summary = {key: result.get(key) for key in (
        "outcome", "attempt", "design", "mode", "live", "cleanup", "http_handoffs", "summary_handoffs",
        "usage_coverage", "known_pricing_estimate", "cache_aware_pricing_estimate", "account_actual", "fee_bound_holds", "stop_reason")}
    summary["historical_outcome"] = result["outcome"]
    summary["report_problems"] = problems
    if problems:
        summary["outcome"] = "inconclusive-evidence"
    summary["budget"] = {key: result.get("budget", {}).get(key) for key in (
        "state", "settled_upper", "unresolved_reserve", "total_upper", "seal_reason")}
    summary["production_exit_code"] = result.get("production", {}).get("main_exit_code")
    summary["can_deliver"] = ((result.get("production", {}).get("delivery") or {}).get("decision") or {}).get("can_deliver")
    summary["official_grade"] = {key: result.get("grade_evidence", {}).get(key) for key in (
        "verifier_exit_code", "reward", "required_passed", "required_total", "cleanup")}
    h.write_json(root / "report.json", summary)
    (root / "report.md").write_text(
        f"Mode: {summary['mode']} / Live: {summary['live']}\n\n"
        f"Outcome: {summary['outcome']}; production exit={summary['production_exit_code']}; "
        f"can_deliver={summary['can_deliver']}; official grade={summary['official_grade']}.\n\n"
        f"HTTP handoffs={summary['http_handoffs']}; usage coverage={summary['usage_coverage']}; "
        f"budget={summary['budget']}; account actual=unavailable.\n\n"
        "Offline mock usage and calibration responses do not establish model ability or real service charges.\n",
        encoding="utf-8", newline="\n")
    return summary


def dry_run(prepared):
    from tests.e2e import swebench_runner as h

    root = prepared.parent / ("offline-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    inner = fixture_transport(prepared)
    result = execute_attempt(prepared, root, inner=inner, offline=True)
    summary = report(root)
    receipts = (((result.get("production") or {}).get("delivery") or {}).get("view") or {}).get("receipts", [])
    initial_grades = []
    for path in sorted((root / "evidence/checks").glob("*/evaluation.json")):
        evaluation = h.load_json(path)
        patch = h.load_json(path.parent / "patch.json")
        grade = evaluation.get("output_json") or {}
        if patch.get("empty") and grade.get("f2p_failed") and grade.get("p2p_passed") and grade.get("artifacts_complete"):
            initial_grades.append(path.relative_to(root).as_posix())
    messages = (((result.get("production") or {}).get("delivery") or {}).get("result") or {}).get("messages", [])
    tools = [m["result"] for m in messages if m.get("type") == "ToolResultMessage"]
    successful_tools = [t["name"] for t in tools if not t.get("is_error")]
    tool_path_observed = successful_tools.count("read") >= 2 and successful_tools.count("write") >= 2 and successful_tools.count("check") >= 1
    initial_failed = bool(initial_grades and receipts and receipts[0].get("verification_status") == "failed"
                          and receipts[0].get("exit_code") == 1 and receipts[0].get("output_complete"))
    eligible = (summary["outcome"] == "passed" and initial_failed and tool_path_observed
                and result["http_handoffs"] == result["usage_coverage"] == 4
                and result["budget"]["unresolved_reserve"] == "0.00")
    admission = {"outcome": "offline-passed-review-pending" if eligible else "offline-failed",
        "review_status": "pending", "live": "not-run", "attempt_root": str(root),
        "runtime_sha256": runtime_identity(), "environment": environment_identity(),
        "lock_sha256": h.sha256_file(prepared / h.LOCK_FILENAME),
        "result_sha256": h.sha256_file(root / "result.json"), "initial_check_failed": initial_failed,
        "initial_grades": initial_grades, "production_tool_path_observed": tool_path_observed,
        "report": summary}
    _atomic_json(prepared / "offline-admission.json", admission)
    return admission


def finalize_admission(prepared, reviews):
    """交付者核对独立审查闭环后记录；不发送任何模型请求。"""
    from tests.e2e import swebench_runner as h

    admission = h.load_json(prepared / "offline-admission.json")
    root = Path(admission["attempt_root"])
    if (admission["outcome"] != "offline-passed-review-pending"
            or admission["runtime_sha256"] != runtime_identity()
            or admission["environment"] != environment_identity()
            or admission["lock_sha256"] != h.sha256_file(prepared / h.LOCK_FILENAME)
            or admission["result_sha256"] != h.sha256_file(root / "result.json")
            or report(root)["outcome"] != "passed" or len(reviews) < 2):
        raise ValueError("offline evidence no longer supports admission")
    admission["reviews"] = {str(path.resolve()): h.sha256_file(path) for path in reviews}
    admission["review_status"] = "passed"
    admission["outcome"] = "admitted"
    _atomic_json(prepared / "offline-admission.json", admission)
    return admission
