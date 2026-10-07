"""本地 Skills v1 票 01: 固定离线编码场景与执行基线.

三个固定小场景驱动共享 ApplicationSession + 真实 Coding 工具 + 真实检查执行 +
真实临时 SQLite 存储, 模型每轮由剧本 FakeModel 替身驱动 (零真实模型调用、零服务启动):

- A 多文件定位修改: script 经过 glob/grep/read/edit/check, 验证真实 diff 与有效回执;
- B 检查失败后修复: 先真实 failed receipt, 再正确改动与当前 passed receipt,
  随后改动使旧 PASS 失效 (stale), 旧事实保留;
- C 压缩/恢复后续写: 两个完整请求 -> 替身摘要 (scripted, 不评价语义质量) ->
  重启恢复 -> 继续读取/修改/检查; 核对历史工具不重放、旧回执历史化、
  摘要来源与实际新请求。

每次 run 位于 pytest 私有临时目录且以独占方式新建, 不覆盖历史工件与既有 P4 证据;
fixture/script/配置/预期输出的身份按 run 写入 manifest JSON 并绑定主仓库 HEAD。
离线通过只证明接缝、来源、历史、预算和验证归属; 真实模型效果未测, 不推导 (D9)。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

FIXTURE_VERSION = "local-skills-offline-baseline-1"
DECISION_VERSION = "local-skills-v1-20261007-r1"
SYSTEM_PROMPT = "You are the offline skills-baseline coding assistant (no skills configured)."
PROMPT_VERSION = "skills-baseline-1"
CHECK_ID = "check-1"
MAX_TURNS = 24

SUMMARY_TEXT = "请求一读取了 src/thermo/convert.py；请求二运行 check-1 并通过。"
SUMMARY_METRICS = {"input_tokens": 64, "output_tokens": 16}

# --- 固定 fixture 源码 -------------------------------------------------------------------

CHECKER_SOURCE = '''\
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from thermo.convert import celsius_to_fahrenheit

cases = [(0.0, 32.0), (37.0, 98.6), (100.0, 212.0), (-40.0, -40.0)]
failures = [
    f"{celsius}->{celsius_to_fahrenheit(celsius)}!={expected}"
    for celsius, expected in cases
    if abs(celsius_to_fahrenheit(celsius) - expected) > 1e-9
]
if failures:
    print("CHECK_FAIL", failures)
    raise SystemExit(1)
print("CHECK_PASS: celsius_to_fahrenheit matches the fixed reference points")
'''

INIT_SOURCE = '"""Thermo fixture package."""\n'

UNITS_SOURCE = '''\
"""Fixed conversion constants shared by the thermo fixture modules."""

FAHRENHEIT_SCALE = 9 / 5
FAHRENHEIT_OFFSET = 32
'''

CONVERT_HEADER = '"""Temperature conversion helpers checked by the fixture checker."""\n\n'
CONVERT_REVIEWED_HEADER = (
    '"""Temperature conversion helpers checked by the fixture checker (reviewed)."""\n\n'
)
CONVERT_IMPORT_DEFECT = "from thermo.units import FAHRENHEIT_SCALE\n"
CONVERT_IMPORT_FIXED = "from thermo.units import FAHRENHEIT_OFFSET, FAHRENHEIT_SCALE\n"
CONVERT_BODY_DEFECT = (
    "\n\ndef celsius_to_fahrenheit(temp_c: float) -> float:\n"
    "    # Known fixture defect: the Fahrenheit offset is not applied.\n"
    "    return temp_c * FAHRENHEIT_SCALE\n"
)
CONVERT_BODY_FIXED = (
    "\n\ndef celsius_to_fahrenheit(temp_c: float) -> float:\n"
    "    return temp_c * FAHRENHEIT_SCALE + FAHRENHEIT_OFFSET\n"
)

DEFECT_CONVERT_SOURCE = CONVERT_HEADER + CONVERT_IMPORT_DEFECT + CONVERT_BODY_DEFECT
FIXED_CONVERT_SOURCE = CONVERT_HEADER + CONVERT_IMPORT_FIXED + CONVERT_BODY_FIXED

CANONICAL_FIX_EDITS = (
    {"old_text": CONVERT_IMPORT_DEFECT + CONVERT_BODY_DEFECT,
     "new_text": CONVERT_IMPORT_FIXED + CONVERT_BODY_FIXED},
)
REVIEW_NOTE_EDITS = ({"old_text": CONVERT_HEADER, "new_text": CONVERT_REVIEWED_HEADER},)

CONVERT_PATH = "src/thermo/convert.py"
UNITS_PATH = "src/thermo/units.py"


def thermo_files(*, fixed: bool) -> dict[str, str]:
    """场景 A/B 用缺陷基线, 场景 C 从正确实现开始; checker 三场景一致."""
    return {
        "src/thermo/__init__.py": INIT_SOURCE,
        UNITS_PATH: UNITS_SOURCE,
        CONVERT_PATH: FIXED_CONVERT_SOURCE if fixed else DEFECT_CONVERT_SOURCE,
        "check_task.py": CHECKER_SOURCE,
    }


# --- 场景剧本 (parse_script JSON 条目; FakeModel 逐次消费) ----------------------------------

SCENARIO_A_SCRIPT = [
    {"tool_calls": [{"id": "a1", "name": "glob", "arguments": {"pattern": "**/*.py"}}]},
    {"tool_calls": [{"id": "a2", "name": "grep", "arguments": {"pattern": "FAHRENHEIT"}}]},
    {"tool_calls": [{"id": "a3", "name": "read", "arguments": {"path": UNITS_PATH}}]},
    {"tool_calls": [{"id": "a4", "name": "read", "arguments": {"path": CONVERT_PATH}}]},
    {"tool_calls": [{"id": "a5", "name": "edit",
                     "arguments": {"path": CONVERT_PATH, "edits": list(CANONICAL_FIX_EDITS)}}]},
    {"tool_calls": [{"id": "a6", "name": "check",
                     "arguments": {"action": "run", "check_id": CHECK_ID}}]},
    {"text": "Located the offset defect across modules, fixed it, check-1 passed.", "stop": True},
]

SCENARIO_B_SCRIPT = [
    {"tool_calls": [{"id": "b1", "name": "check",
                     "arguments": {"action": "run", "check_id": CHECK_ID}}]},
    {"tool_calls": [{"id": "b2", "name": "edit",
                     "arguments": {"path": CONVERT_PATH, "edits": list(CANONICAL_FIX_EDITS)}}]},
    {"tool_calls": [{"id": "b3", "name": "check",
                     "arguments": {"action": "run", "check_id": CHECK_ID}}]},
    {"tool_calls": [{"id": "b4", "name": "edit",
                     "arguments": {"path": CONVERT_PATH, "edits": list(REVIEW_NOTE_EDITS)}}]},
    {"text": "The failed receipt is superseded by a passing one; the newest edit "
             "invalidates that pass.", "stop": True},
]

# 场景 C 第一进程: 请求一 read、请求二 check, 最后一个条目是压缩请求消费的替身摘要。
SCENARIO_C_FIRST_SCRIPT = [
    {"tool_calls": [{"id": "c1", "name": "read", "arguments": {"path": CONVERT_PATH}}]},
    {"text": "Read the current implementation.", "stop": True},
    {"tool_calls": [{"id": "c2", "name": "check",
                     "arguments": {"action": "run", "check_id": CHECK_ID}}]},
    {"text": "check-1 passed on the baseline implementation.", "stop": True},
    {"text": SUMMARY_TEXT, "stop": True},
]

# 场景 C 恢复进程: 继续读取/修改/检查。
SCENARIO_C_RESUME_SCRIPT = [
    {"tool_calls": [{"id": "c3", "name": "read", "arguments": {"path": CONVERT_PATH}}]},
    {"tool_calls": [{"id": "c4", "name": "edit",
                     "arguments": {"path": CONVERT_PATH, "edits": list(REVIEW_NOTE_EDITS)}}]},
    {"tool_calls": [{"id": "c5", "name": "check",
                     "arguments": {"action": "run", "check_id": CHECK_ID}}]},
    {"text": "Continued after restore: read, edited and re-checked.", "stop": True},
]

# --- 预期结果 (fixture 版本身份的一部分; 测试断言 observed == expected) ----------------------

SCENARIO_A_EXPECTED = {
    "tool_sequence": ["glob", "grep", "read", "read", "edit", "check"],
    "tool_error_count": 0,
    "changed_paths": [CONVERT_PATH],
    "receipt_statuses": ["passed"],
    "current_freshness": "current",
    "can_deliver": True,
    "checker_baseline_exit": 1,
    "checker_after_exit": 0,
}

SCENARIO_B_EXPECTED = {
    "tool_sequence": ["check", "edit", "check", "edit"],
    "tool_error_count": 0,
    "receipt_statuses": ["failed", "passed"],
    "current_receipt_status": "passed",
    "final_freshness": "stale",
    "can_deliver": False,
    "checker_after_exit": 0,
}

SCENARIO_C_EXPECTED = {
    "request_tool_sequences": [["read"], ["check"], ["read", "edit", "check"]],
    "tool_error_count": 0,
    "summary_reason": "manual",
    "summary_text": SUMMARY_TEXT,
    "summary_metrics_input_tokens": SUMMARY_METRICS["input_tokens"],
    "archive_receipt_count": 2,
    "restored_receipt_count": 1,
    "current_freshness": "current",
    "changed_paths": [CONVERT_PATH],
    "can_deliver": True,
    "checker_after_exit": 0,
}

# 无 Skills 基线 (A01 准备): 启用 Skills 前模型可见的系统提示与完整工具声明集合
EXPECTED_NO_SKILLS_TOOLS = sorted(
    ["read", "edit", "write", "powershell", "glob", "grep", "check"]
)


def declared_tools(model) -> list[str]:
    """模型实际收到的工具声明名单 (首个请求); FakeModel 记录每次 ModelRequest."""
    return sorted(spec.name for spec in model.requests[0].tools)

# --- 构建与身份 ---------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.email=fixture@local",
         "-c", "user.name=fixture", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def git_head(project: Path = PROJECT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True,
        encoding="utf-8", timeout=60,
    ).stdout.strip()


def build_workspace(dest: Path, *, fixed: bool) -> dict:
    """新私有目录写入固定包 + checker 并提交基线; 返回文件身份与基线 commit."""
    dest.mkdir(parents=True, exist_ok=False)
    files = thermo_files(fixed=fixed)
    for relative, content in files.items():
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    assert _git(dest, "init", "-q").returncode == 0
    assert _git(dest, "add", "-A").returncode == 0
    assert _git(dest, "commit", "-q", "-m", "fixture baseline").returncode == 0
    commit = _git(dest, "rev-parse", "HEAD").stdout.strip()
    return {
        "files_sha256": {relative: sha256_text(content) for relative, content in files.items()},
        "baseline_commit": commit,
    }


def run_checker(root: Path) -> subprocess.CompletedProcess:
    """会话之外独立复跑 checker, 证明检查器自身可执行且结果确定."""
    return subprocess.run(
        [sys.executable, "check_task.py"], cwd=root, capture_output=True, text=True,
        encoding="utf-8", timeout=120,
    )


def check_command() -> str:
    """VerificationPlan 固定命令: 显式解释器路径, 不依赖 PATH."""
    return f"& '{sys.executable}' check_task.py"


def script_text(entries: list) -> str:
    return json.dumps(entries, ensure_ascii=False, indent=2)


def parse_script_text(text: str) -> list:
    from cicada.__main__ import parse_script

    return parse_script(text)


# --- 应用配置构造 ---------------------------------------------------------------------


def prompt_info(root: Path):
    from cicada.prompting import PromptInfo

    return PromptInfo(
        version=PROMPT_VERSION,
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=sha256_text(SYSTEM_PROMPT),
    )


def launch_identity(checks):
    from cicada.application import verification_checks_signature
    from cicada.persistence.contracts import LaunchIdentity

    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=MAX_TURNS, prompt_version=PROMPT_VERSION,
        prompt_sha256=sha256_text(SYSTEM_PROMPT), instructions_sha256=None,
        checks_signature=verification_checks_signature(checks) if checks else None,
        sources={},
    )


def check_definitions() -> tuple:
    from cicada.plugins.coding.verification_contracts import CheckDefinition

    return (CheckDefinition(CHECK_ID, check_command(), timeout_s=120.0),)


def application_config(root: Path, entries: list, *, checks: tuple | None = None,
                       session_id: str | None = None, identity=None, budget=None):
    """按 fixture 固定配置构造 ApplicationConfig; 返回 (config, model) 供请求断言.

    session_id=None 且 identity=None 为非持久; identity 缺省按当前 checks 派生。
    跨进程恢复须显式传同一 identity (默认恢复要求身份逐位一致)。
    """
    from cicada.application import ApplicationConfig, PersistenceConfig
    from cicada.plugins.coding.verification import generate_verification_run_id
    from cicada.plugins.coding.verification_contracts import VerificationPlan
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin

    if checks is None:
        checks = check_definitions()
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=checks,
    )
    model = FakeModel(entries)
    persistence = None
    if session_id is not None or identity is not None:
        persistence = PersistenceConfig(
            session_id=session_id, identity=identity or launch_identity(plan.checks)
        )
    config = ApplicationConfig(
        workspace=root,
        system_prompt=SYSTEM_PROMPT,
        prompt_info=prompt_info(root),
        model_plugin=fake_model_plugin(model),
        plan=plan,
        max_turns=MAX_TURNS,
        model_label="fake",
        persistence=persistence,
        budget=budget,
    )
    return config, model


# --- 工件记录 ---------------------------------------------------------------------


def artifact_dir(tmp_path: Path, scenario: str) -> Path:
    """run 内独占新建; 跨 run 由 pytest 私有 tmp_path 隔离, 不覆盖任何旧工件."""
    run_dir = tmp_path / "skills-baseline-artifacts" / scenario
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8", newline="\n",
    )


def git_facts(root: Path) -> dict:
    status = _git(root, "status", "--porcelain")
    diff = _git(root, "diff")
    assert status.returncode == 0 and diff.returncode == 0, status.stderr + diff.stderr
    return {"status": status.stdout, "diff": diff.stdout}


def receipt_facts(receipt) -> dict:
    return {
        "receipt_id": receipt.receipt_id,
        "verification_status": receipt.verification_status,
        "exit_code": receipt.exit_code,
        "failure_kind": receipt.failure_kind,
        "snapshot_after": receipt.snapshot_after,
        "output_artifact_sha256": receipt.output_artifact_sha256,
    }


def outcome_facts(outcome) -> dict:
    from cicada.core.messages import ToolResultMessage

    view, evidence, decision = outcome.view, outcome.evidence, outcome.decision
    return {
        "stop_reason": outcome.result.stop_reason,
        "error": outcome.result.error,
        "exit_code": outcome.exit_code,
        "tool_sequence": [
            m.result.name for m in outcome.result.messages if isinstance(m, ToolResultMessage)
        ],
        "tool_error_contents": [
            m.result.content for m in outcome.result.messages
            if isinstance(m, ToolResultMessage) and m.result.is_error
        ],
        "checks": [
            {
                "check_id": state.check_id,
                "status": state.status,
                "freshness": state.freshness,
                "receipt": receipt_facts(state.receipt) if state.receipt else None,
            }
            for state in (view.checks if view is not None else ())
        ],
        "receipts": [
            receipt_facts(receipt) for receipt in (view.receipts if view is not None else ())
        ],
        "changes": [
            {
                "relative_path": change.relative_path,
                "kind": change.kind,
                "before_sha256": change.before_sha256,
                "after_sha256": change.after_sha256,
            }
            for change in (evidence.changes if evidence is not None else ())
        ],
        "evidence_complete": evidence.complete if evidence is not None else None,
        "decision": (
            {
                "model_stopped": decision.model_stopped,
                "can_deliver": decision.can_deliver,
                "blocking_reasons": list(decision.blocking_reasons),
            }
            if decision is not None else None
        ),
        "persistence": (
            {"request_id": outcome.persistence.request_id, "saved": outcome.persistence.saved,
             "seq": outcome.persistence.seq}
            if outcome.persistence is not None else None
        ),
    }


def write_messages_artifact(path: Path, messages) -> None:
    """完整消息按持久化 codec 逐条规范编码落盘 (完整消息记录)."""
    from cicada.persistence.codec import encode_messages

    lines = [
        json.dumps({"index": index, "message_type": type(message).__name__,
                    "encoded": encode_messages((message,))}, ensure_ascii=False)
        for index, message in enumerate(messages)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_run_artifacts(run_dir: Path, *, scenario: str, expected: dict,
                        workspace_facts: dict, script: str, outcomes: list,
                        root: Path, declared_tools: list[list[str]] | None = None) -> dict:
    """写入 outcome/messages/diff 工件与 run manifest (身份 + 预期 + 工件 hash)."""
    facts = [outcome_facts(outcome) for outcome in outcomes]
    write_json(run_dir / "outcomes.json", facts)
    message_files = []
    for index, outcome in enumerate(outcomes):
        message_path = run_dir / f"messages-{index:02d}.jsonl"
        write_messages_artifact(message_path, outcome.result.messages)
        message_files.append({"path": message_path.name, "sha256": sha256_file(message_path)})
    git = git_facts(root)
    (run_dir / "git-status.txt").write_text(git["status"], encoding="utf-8", newline="\n")
    (run_dir / "git-diff.patch").write_text(git["diff"], encoding="utf-8", newline="\n")
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "decision_version": DECISION_VERSION,
        "scenario": scenario,
        "main_head": git_head(),
        "python": sys.version,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "prompt_version": PROMPT_VERSION,
        "check_command": check_command(),
        "declared_tools": declared_tools,
        "fixture_files_sha256": workspace_facts["files_sha256"],
        "fixture_baseline_commit": workspace_facts["baseline_commit"],
        "script_sha256": sha256_text(script),
        "expected": expected,
        "artifacts": {
            "outcomes_sha256": sha256_file(run_dir / "outcomes.json"),
            "messages": message_files,
            "git_status_sha256": sha256_file(run_dir / "git-status.txt"),
            "git_diff_sha256": sha256_file(run_dir / "git-diff.patch"),
        },
        "model": {"kind": "scripted_fake", "real_model_calls": 0, "services_started": 0},
        "real_model_effect": "untested (spec D9)",
    }
    write_json(run_dir / "run-manifest.json", manifest)
    return manifest
