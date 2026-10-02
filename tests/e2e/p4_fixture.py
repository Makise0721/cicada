"""P4 07 固定验收 fixture 与 live harness.

固定三件套 (checker / prompt / launcher) 与 profile, 在独立初始化的 Git fixture 中
让实施后的 Cicada CLI (真实 Ollama 模型) 完成"给 build_run_summary 增加 tools_ok"
的小改动。新活动0607-main-v2保留旧活动, 最多两次完整尝试。

- fixture: 复制主仓库当前 `src/` 的全部 .py 到独立 Git 仓库, 附四行为 checker;
  checker 用 sys.path.insert(0, <fixture>/src) 优先导入 fixture 源码。
- 构建时证明: baseline 上 checker exit 1 且 stderr 为空 (唯一失败原因是缺
  tools_ok, import 未失败), 并用探针证明导入路径绑定 fixture/src。
- 请求/响应抓取: 本地转发代理 tee 原始 HTTP 请求体与 NDJSON 响应, 不改动
  Ollama 适配器; /api/version /api/tags /api/ps 直连真实服务核对。
- 600 秒 episode 预算由本 harness 的 subprocess timeout 控制, 不是 CLI 产品能力。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"
VENV_PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
REAL_OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
NUM_CTX = 32768
NUM_PREDICT = 2048
MAX_TURNS = 25
ATTEMPT_BUDGET = 2
EPISODE_TIMEOUT_S = 600.0
PROMPT_EVAL_PEAK_LIMIT = 24576

CAMPAIGN_ID = "0607-main-v2"
EVIDENCE_ROOT = PROJECT / "docs" / "reviews" / "p4-evidence" / CAMPAIGN_ID / "live"
FIXTURE_RELATIVE = Path(".scratch/p4-implementation/temp/0607-main-v2-live")
OBSERVER = Path(__file__).with_name("p4_observer.py")

# 固定三件套; 跨尝试不变, hash 记入证据。
CHECKER_NAME = "check_p4.py"
CHECKER_SOURCE = r'''from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cicada.core.agent import RunResult
from cicada.core.messages import ToolResult, ToolResultMessage
from cicada.reporting import build_run_summary
cases = [("empty", [], 0), ("all_success", [False, False], 2),
         ("mixed", [False, True, False], 2), ("all_error", [True, True], 0)]
failures = []
for name, errors, expected_ok in cases:
    messages = tuple(ToolResultMessage(ToolResult(str(i), "read", "probe", is_error=e))
                     for i, e in enumerate(errors))
    summary = build_run_summary(RunResult("probe", "stop", None, messages))
    first = summary.splitlines()[0]
    required = {"tools_ok": expected_ok, "tools": len(errors), "tool_errors": sum(errors)}
    missing = [f"{key}={value}" for key, value in required.items()
               if not re.search(rf"(?: |\[){key}={value}(?: |\])", first)]
    if missing:
        failures.append((name, missing, first))
    if len(summary.splitlines()) != 3 or "input_tokens_known=unknown" not in summary:
        failures.append((name, "existing summary fields/shape changed", summary))
if failures:
    print("CHECK_FAIL", failures)
    raise SystemExit(1)
print("CHECK_PASS: 4 behavior cases; existing counters, shape and unknown metrics preserved")
'''
# 检查命令只引用相对路径; python 解释器由 harness 的 PATH 提供 (launcher 事实)。
CHECK_COMMAND = "& python check_p4.py"
TASK_PROMPT = (
    "这个工作区是 Cicada 项目的一份源码副本。请完成一个小改动: 给 build_run_summary 函数"
    "生成的摘要第一行增加 tools_ok=N, 表示 is_error=False 的工具结果数。"
    "保留原有计数、三行结构和未知计量语义。"
    "请先用 grep 工具搜索 \"def build_run_summary\" 定位实现文件, 再用 read 读取相关源码, "
    "先用 check 工具运行指定检查 check-1 确认初始失败, 再完成最小修改并重检通过。"
    "不要修改检查器, 最后汇报实际 diff 和检查退出状态。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def profile() -> dict:
    return {
        "model": MODEL, "think": False, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
        "max_turns": MAX_TURNS, "episode_timeout_s": EPISODE_TIMEOUT_S,
        "check_command": CHECK_COMMAND, "check_timeout_s": 120.0,
        "max_request_bytes": 65536,
    }


def freeze_campaign() -> dict:
    """冻结活动输入; 第二次尝试拒绝换源码、prompt、checker或launcher."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    expected = {
        "campaign_id": CAMPAIGN_ID, "attempt_budget": ATTEMPT_BUDGET, "main_head": head,
        "profile": profile(), "prompt_sha256": sha256_text(TASK_PROMPT),
        "checker_sha256": sha256_text(CHECKER_SOURCE),
        "prompt": TASK_PROMPT, "checker_text": CHECKER_SOURCE,
        "launcher_sha256": sha256_file(Path(__file__)), "observer_sha256": sha256_file(OBSERVER),
        "src_manifest": {
            path.relative_to(PROJECT).as_posix(): sha256_file(path)
            for path in sorted(SRC.rglob("*.py")) if "__pycache__" not in path.parts
        },
    }
    campaign = EVIDENCE_ROOT.parent
    campaign.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign / "campaign-manifest.json"
    try:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(expected, indent=2, ensure_ascii=False))
    except FileExistsError:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("campaign inputs changed; a new owner decision is required")
    return expected


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "user.email=p4@local", "-c", "user.name=p4", *args],
        cwd=repo, capture_output=True, text=True, encoding="utf-8",
    )


def manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    }


def build_fixture(dest: Path) -> dict:
    """独立 Git fixture: 复制实施后 src/ + 固定 checker, 并完成 baseline 反证.

    返回 fixture 事实 (commit/manifest/checker 反证/导入优先探针), 全部进入证据。
    """
    dest.mkdir(parents=True, exist_ok=False)
    copied = 0
    for source in sorted(SRC.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        target = dest / "src" / source.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    checker_path = dest / CHECKER_NAME
    checker_path.write_text(CHECKER_SOURCE, encoding="utf-8", newline="\n")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()
    commit = _git(dest, "init", "-q")
    assert commit.returncode == 0, commit.stderr
    assert _git(dest, "add", "-A").returncode == 0
    assert _git(dest, "commit", "-q", "-m", "P4 验收副本基线").returncode == 0
    fixture_commit = _git(dest, "rev-parse", "HEAD").stdout.strip()
    baseline_manifest = manifest(dest)

    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    # baseline 反证 1: checker 对原始 fixture exit 1 且无 traceback -> 唯一失败分支是 tools_ok
    baseline_run = subprocess.run(
        [str(VENV_PYTHON), CHECKER_NAME], cwd=dest, capture_output=True, text=True,
        encoding="utf-8", env=env, timeout=60,
    )
    assert baseline_run.returncode == 1, (
        f"checker must fail on baseline, got {baseline_run.returncode}: {baseline_run.stderr}"
    )
    assert baseline_run.stderr == "", f"baseline failure must not be an import/traceback: {baseline_run.stderr}"
    assert baseline_run.stdout.startswith("CHECK_FAIL "), baseline_run.stdout
    failures = ast.literal_eval(baseline_run.stdout.removeprefix("CHECK_FAIL ").strip())
    expected_missing = {
        "empty": ["tools_ok=0"], "all_success": ["tools_ok=2"],
        "mixed": ["tools_ok=2"], "all_error": ["tools_ok=0"],
    }
    assert len(failures) == 4 and {item[0]: item[1] for item in failures} == expected_missing, (
        f"baseline must fail only for missing tools_ok counts: {failures}"
    )
    # baseline 反证 2: 导入优先绑定 fixture/src (相同 PYTHONPATH 环境下)
    probe = subprocess.run(
        [str(VENV_PYTHON), "-c",
         f"import sys; sys.path.insert(0, r'{(dest / 'src').resolve()}'); import cicada; print(cicada.__file__)"],
        cwd=dest, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    expected_init = (dest.resolve() / "src" / "cicada" / "__init__.py")
    assert probe.stdout.strip().lower() == str(expected_init).lower(), probe.stdout

    return {
        "fixture_root": str(dest),
        "fixture_commit": fixture_commit,
        "copied_py_files": copied,
        "checker_sha256": sha256_text(CHECKER_SOURCE),
        "checker_baseline_exit": baseline_run.returncode,
        "checker_baseline_stderr_empty": baseline_run.stderr == "",
        "checker_baseline_stdout": baseline_run.stdout,
        "import_probe": probe.stdout.strip(),
        "main_head": head,
        "baseline_manifest": baseline_manifest,
    }


class _TeeProxy:
    """转发本地代理: tee 原始请求体与 NDJSON 响应到内存, 结束后落盘."""

    def __init__(self, record_dir: Path) -> None:
        self.record_dir = record_dir
        self.requests: list[dict] = []
        self.responses: list[dict] = []
        self.errors: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # 静默默认日志
                pass

            def do_GET(self) -> None:
                self._relay(b"")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self._relay(self.rfile.read(length) if length else b"")

            def _relay(self, body: bytes) -> None:
                try:
                    method = "POST" if body else "GET"
                    headers = {"content-type": self.headers.get("content-type", "application/json")}
                    with httpx.stream(method, f"{REAL_OLLAMA}{self.path}", content=body or None,
                                      headers=headers, timeout=600.0) as resp:
                        outer.requests.append({"path": self.path, "method": method, "body_b64": _b64(body)})
                        self.send_response(resp.status_code)
                        content_type = resp.headers.get("content-type")
                        if content_type:
                            self.send_header("Content-Type", content_type)
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        collected = bytearray()
                        for chunk in resp.iter_raw():
                            collected.extend(chunk)
                            self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        outer.responses.append({"path": self.path, "body_b64": _b64(bytes(collected))})
                except Exception as exc:  # 代理故障按证据记录并回 502
                    outer.errors.append(f"{self.path}: {exc}")
                    try:
                        self.send_error(502, str(exc))
                    except Exception:
                        pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def save(self) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        (self.record_dir / "requests.jsonl").write_text(
            "\n".join(json.dumps(item) for item in self.requests), encoding="utf-8", newline="\n")
        (self.record_dir / "responses.jsonl").write_text(
            "\n".join(json.dumps(item) for item in self.responses), encoding="utf-8", newline="\n")
        (self.record_dir / "errors.json").write_text(
            json.dumps(self.errors, indent=2), encoding="utf-8", newline="\n")
        for index, item in enumerate(self.responses, start=1):
            (self.record_dir / f"response-{index:02d}.ndjson").write_bytes(_unb64(item["body_b64"]))

    def prompt_eval_counts(self) -> list[dict]:
        """逐响应解析 done:true 行的计量 (prompt_eval_count/eval_count/total_duration)."""
        counts: list[dict] = []
        for item in self.responses:
            for line in _unb64(item["body_b64"]).decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("done"):
                    counts.append({
                        "prompt_eval_count": record.get("prompt_eval_count"),
                        "eval_count": record.get("eval_count"),
                        "total_duration": record.get("total_duration"),
                    })
        return counts


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64

    return base64.b64decode(text.encode("ascii"))


def gate() -> str | None:
    """live gate: 真实服务 /api/version + /api/tags 中模型存在; 不可达返回原因."""
    try:
        with httpx.Client(timeout=5.0) as client:
            version = client.get(f"{REAL_OLLAMA}/api/version")
            version.raise_for_status()
            tags = client.get(f"{REAL_OLLAMA}/api/tags")
            tags.raise_for_status()
    except httpx.HTTPError as exc:
        return f"ollama unreachable at {REAL_OLLAMA}: {exc}"
    names = {entry.get("name") for entry in tags.json().get("models", [])}
    if MODEL not in names:
        return f"model {MODEL!r} not present in /api/tags"
    return None


def ollama_digest() -> dict:
    with httpx.Client(timeout=5.0) as client:
        version = client.get(f"{REAL_OLLAMA}/api/version").json()
        tags = client.get(f"{REAL_OLLAMA}/api/tags").json()
    tag_digest = next(
        (entry.get("digest") for entry in tags.get("models", []) if entry.get("name") == MODEL), None
    )
    return {"api_version": version, "tags": tags, "model_digest_in_tags": tag_digest}


def loaded_state() -> dict:
    with httpx.Client(timeout=5.0) as client:
        return client.get(f"{REAL_OLLAMA}/api/ps").json()


def run_attempt(attempt: int) -> dict:
    """保留本活动原始尝试; 任何失败都先落结果, 不覆盖旧slot."""
    if type(attempt) is not int or not 1 <= attempt <= ATTEMPT_BUDGET:
        raise ValueError(f"attempt must be an integer from 1 to {ATTEMPT_BUDGET}")
    attempt_dir = EVIDENCE_ROOT / f"attempt-{attempt}"
    fixture_root = PROJECT / FIXTURE_RELATIVE / f"attempt-{attempt}" / "fixture"
    if fixture_root.exists():
        raise FileExistsError(f"live fixture already exists; refusing to overwrite: {fixture_root}")
    if attempt_dir.exists():
        raise FileExistsError(f"live evidence already exists; refusing to overwrite: {attempt_dir}")
    frozen = freeze_campaign()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        return _execute_attempt(attempt, attempt_dir, fixture_root, frozen, started)
    except BaseException as exc:
        _write_json(attempt_dir / "attempt-error.json", {
            "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_s": time.monotonic() - started, "phase": "harness failure",
        })
        raise


def _remaining(started: float, cap: float) -> float:
    remaining = EPISODE_TIMEOUT_S - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("600s episode budget exhausted")
    return min(cap, remaining)


def _execute_attempt(attempt: int, attempt_dir: Path, fixture_root: Path,
                     frozen: dict, started: float) -> dict:
    identity = {
        "campaign_id": CAMPAIGN_ID, "attempt": attempt,
        "campaign_manifest_sha256": sha256_file(EVIDENCE_ROOT.parent / "campaign-manifest.json"),
        "launcher": {"file": "tests/e2e/p4_fixture.py", "sha256": frozen["launcher_sha256"]},
        "observer": {"file": "tests/e2e/p4_observer.py", "sha256": frozen["observer_sha256"]},
        "profile": profile(), "python": sys.version,
    }
    _write_json(attempt_dir / "identity.json", identity)
    facts = build_fixture(fixture_root)
    assert facts["main_head"] == frozen["main_head"], "HEAD changed after campaign freeze"
    actual_src = {p: h for p, h in facts["baseline_manifest"].items() if p.startswith("src/")}
    assert actual_src == frozen["src_manifest"], "source bytes changed after campaign freeze"
    identity.update({
        "main_head": facts["main_head"], "fixture_commit": facts["fixture_commit"],
        "fixture_copied_py_files": facts["copied_py_files"],
        "checker_sha256": facts["checker_sha256"], "prompt_sha256": frozen["prompt_sha256"],
        "checker_baseline_exit": facts["checker_baseline_exit"],
        "checker_baseline_stderr_empty": facts["checker_baseline_stderr_empty"],
        "checker_baseline_stdout": facts["checker_baseline_stdout"],
        "fixture_import_probe": facts["import_probe"],
    })
    identity.update(ollama_digest())
    _write_json(attempt_dir / "identity.json", identity)
    _write_json(attempt_dir / "fixture-manifest-before.json", facts["baseline_manifest"])
    proxy = _TeeProxy(attempt_dir / "proxy")
    cli_args = [
        "--workspace", str(fixture_root), "--ollama-url", proxy.url,
        "--model", MODEL, "--num-ctx", str(NUM_CTX),
        "--check-command", CHECK_COMMAND, TASK_PROMPT,
    ]
    argv = [str(VENV_PYTHON), str(OBSERVER), "--evidence-dir", str(attempt_dir), "--", *cli_args]
    env = {
        **os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{PROJECT / '.venv' / 'Scripts'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    cli_started = time.monotonic()
    timed_out = False
    try:
        identity["argv"] = argv
        identity["cli_args"] = cli_args
        _write_json(attempt_dir / "identity.json", identity)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", env=env,
                                  timeout=_remaining(started, EPISODE_TIMEOUT_S))
            returncode: int | str = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = "episode-timeout"
            stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
    finally:
        proxy.stop()
        proxy.save()
    cli_elapsed = time.monotonic() - cli_started
    # 先保存CLI事实, 后续服务/独立checker取证故障不会抹掉原始运行.
    (attempt_dir / "cli-stdout.txt").write_text(stdout, encoding="utf-8", newline="\n")
    (attempt_dir / "cli-stderr.txt").write_text(stderr, encoding="utf-8", newline="\n")
    _write_json(attempt_dir / "cli-result.json", {
        "returncode": returncode, "timed_out": timed_out, "cli_wall_time_s": cli_elapsed,
    })
    ps_after = loaded_state()
    _write_json(attempt_dir / "ps-after.json", ps_after)
    checker_run = subprocess.run(
        [str(VENV_PYTHON), CHECKER_NAME], cwd=fixture_root, capture_output=True, text=True,
        encoding="utf-8", env=env, timeout=_remaining(started, 60),
    )
    _write_json(attempt_dir / "checker-run.txt", {
        "exit_code": checker_run.returncode, "stdout": checker_run.stdout, "stderr": checker_run.stderr,
    })
    status = _git(fixture_root, "status", "--porcelain")
    diff = _git(fixture_root, "diff")
    assert status.returncode == 0 and diff.returncode == 0, status.stderr + diff.stderr
    (attempt_dir / "git-status.txt").write_text(status.stdout, encoding="utf-8", newline="\n")
    (attempt_dir / "git-diff.patch").write_text(diff.stdout, encoding="utf-8", newline="\n")
    _write_json(attempt_dir / "fixture-manifest-after.json", manifest(fixture_root))
    outputs = fixture_root / ".cicada" / "outputs"
    if outputs.exists():
        shutil.copytree(outputs, attempt_dir / "artifacts")
    counts = proxy.prompt_eval_counts()
    peaks = [item["prompt_eval_count"] for item in counts if item["prompt_eval_count"] is not None]
    metrics = {
        "per_call": counts, "prompt_eval_count_peak": max(peaks) if peaks else None,
        "prompt_eval_count_sum": sum(peaks) if peaks else None,
        "prompt_eval_sum_note": "跨调用求和含重复处理, 不是唯一上下文 token 数",
        "eval_count_sum": sum(item["eval_count"] for item in counts if item["eval_count"] is not None),
        "proxy_errors": proxy.errors,
    }
    _write_json(attempt_dir / "proxy-metrics.json", metrics)
    checker_after = sha256_file(fixture_root / CHECKER_NAME)
    elapsed = time.monotonic() - started
    result = {
        "campaign_id": CAMPAIGN_ID, "attempt": attempt, "returncode": returncode,
        "timed_out": timed_out, "elapsed_s": elapsed, "cli_elapsed_s": cli_elapsed,
        "episode_within_budget": elapsed <= EPISODE_TIMEOUT_S,
        "stdout": stdout, "stderr": stderr, "checker_exit": checker_run.returncode,
        "checker_sha256_after": checker_after, "checker_unchanged": checker_after == facts["checker_sha256"],
        "changed_git_lines": [line for line in status.stdout.splitlines() if line.strip()],
        "git_diff": diff.stdout, "prompt_eval_count_peak": metrics["prompt_eval_count_peak"],
        "prompt_eval_count_sum": metrics["prompt_eval_count_sum"], "proxy_errors": proxy.errors,
        "evidence_dir": str(attempt_dir), "identity": identity,
    }
    _write_json(attempt_dir / "attempt-outcome.json", result)
    return result


def validate_attempt(result: dict) -> None:
    """同时核对真实行为、观察留证、请求profile与交付身份; 结果落盘后才返回."""
    evidence_dir = Path(result["evidence_dir"])
    try:
        _assert_attempt(result, evidence_dir)
    except BaseException as exc:
        _write_json(evidence_dir / "acceptance.json", {
            "accepted": False, "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    _write_json(evidence_dir / "acceptance.json", {"accepted": True})


def _assert_attempt(result: dict, evidence_dir: Path) -> None:
    assert not result["timed_out"] and result["episode_within_budget"], "600s episode budget exceeded"
    assert not result["proxy_errors"], result["proxy_errors"]
    assert result["returncode"] == 0, f"CLI exit {result['returncode']}"
    observer = json.loads((evidence_dir / "observer-result.json").read_text(encoding="utf-8"))
    assert observer["evidence_complete"] and not observer["recording_errors"], observer
    assert observer["main_exit_code"] == result["returncode"]
    run = json.loads((evidence_dir / "run-result.json").read_text(encoding="utf-8"))
    assert run["stop_reason"] == "stop" and run["error"] is None
    delivery = json.loads((evidence_dir / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["result"] == run
    assert delivery["decision"]["can_deliver"] and delivery["decision"]["model_stopped"]
    view, changes = delivery["view"], delivery["evidence"]
    assert not view["process_uncertain"] and not view["blocking_reasons"]
    assert changes["complete"] and changes["snapshot_ref"] == view["snapshot_ref"]
    assert len(view["checks"]) == 1
    state = view["checks"][0]
    assert state["check_id"] == "check-1" and state["freshness"] == "current"
    assert state["receipt"]["verification_status"] == "passed" and state["receipt"]["exit_code"] == 0
    assert state["receipt"]["snapshot_after"] == view["snapshot_ref"]
    # 初始红、修改后绿都保留真实回执; 不能只看模型汇报.
    assert any(r["verification_status"] == "failed" and r["exit_code"] == 1 for r in view["receipts"])
    assert view["receipts"][-1] == state["receipt"]
    for receipt in view["receipts"]:
        if receipt["output_artifact_path"] is not None:
            saved = evidence_dir / "artifacts" / Path(receipt["output_artifact_path"]).name
            assert sha256_file(saved) == receipt["output_artifact_sha256"]
    assert sha256_file(evidence_dir / "artifacts" / Path(changes["artifact_path"]).name) == changes["artifact_sha256"]
    assert [c["relative_path"] for c in changes["changes"]] == ["src/cicada/reporting.py"]
    snapshots = [json.loads(line)["value"] for line in (evidence_dir / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    snapshot = snapshots[-1]["snapshot"]
    assert snapshot["snapshot_ref"] == view["snapshot_ref"]
    entry = next(e for e in snapshot["entries"] if e["relative_path"] == "src/cicada/reporting.py")
    assert entry["sha256"] == changes["changes"][0]["after_sha256"]
    events = [json.loads(line) for line in (evidence_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    names = [event["name"] for event in events if event["type"] == "ToolStarted"]
    search = next(i for i, name in enumerate(names) if name in ("grep", "glob"))
    read = names.index("read", search + 1)
    edit = next(i for i in range(read + 1, len(names)) if names[i] in ("edit", "write"))
    assert "check" in names[edit + 1:], names
    assert result["checker_exit"] == 0 and result["checker_unchanged"]
    changed = [line for line in result["changed_git_lines"] if line != "?? .cicada/"]
    assert changed == [" M src/cicada/reporting.py"], changed
    assert "tools_ok" in result["git_diff"]
    assert "[delivery can_deliver=true model_stopped=true]" in result["stdout"]
    peak = result["prompt_eval_count_peak"]
    assert peak is not None and peak <= PROMPT_EVAL_PEAK_LIMIT
    requests = [json.loads(line) for line in (evidence_dir / "proxy/requests.jsonl").read_text(encoding="utf-8").splitlines()]
    chats = [_unb64(item["body_b64"]) for item in requests if item["path"] == "/api/chat"]
    assert len(chats) == len(run["model_calls"]) and chats
    for body in chats:
        assert len(body) <= 65536
        payload = json.loads(body)
        assert payload["model"] == MODEL and payload["think"] is False
        assert payload["options"]["num_ctx"] == NUM_CTX and payload["options"]["num_predict"] == NUM_PREDICT
        assert all(m["content"] == TASK_PROMPT for m in payload["messages"] if m["role"] == "user")
    metrics = json.loads((evidence_dir / "proxy-metrics.json").read_text(encoding="utf-8"))
    assert [c["metrics"]["input_tokens"] for c in run["model_calls"]] == [c["prompt_eval_count"] for c in metrics["per_call"]]
    ps = json.loads((evidence_dir / "ps-after.json").read_text(encoding="utf-8"))
    digest = result["identity"]["model_digest_in_tags"]
    assert digest and any(m.get("name") == MODEL and m.get("digest") == digest and m.get("context_length") == NUM_CTX for m in ps["models"])
