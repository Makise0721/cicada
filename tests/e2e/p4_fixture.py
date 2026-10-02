"""P4 07 固定验收 fixture 与 live harness.

固定三件套 (checker / prompt / launcher) 与 profile, 在独立初始化的 Git fixture 中
让实施后的 Cicada CLI (真实 Ollama 模型) 完成"给 build_run_summary 增加 tools_ok"
的小改动。全部证据写入项目内 `docs/reviews/p4-evidence/0607-I/live/<attempt>/`。

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

EVIDENCE_ROOT = PROJECT / "docs" / "reviews" / "p4-evidence" / "0607-I" / "live"

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
    assert _git(dest, "commit", "-q", "-m", "p4 fixture baseline").returncode == 0
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
    """执行一次完整 live 尝试并保存全部证据; 断言结果由调用方在证据落盘后判定."""
    if type(attempt) is not int or not 1 <= attempt <= ATTEMPT_BUDGET:
        raise ValueError(f"attempt must be an integer from 1 to {ATTEMPT_BUDGET}")
    attempt_dir = EVIDENCE_ROOT / f"attempt-{attempt}"
    fixture_root = PROJECT / ".scratch" / "p4-implementation" / "temp" / "I-0607-live" / f"attempt-{attempt}" / "fixture"
    if fixture_root.exists():
        raise FileExistsError(f"live fixture already exists; refusing to overwrite: {fixture_root}")
    attempt_dir.mkdir(parents=True, exist_ok=False)

    identity = {
        "attempt": attempt,
        "launcher": {"file": "tests/e2e/p4_fixture.py", "sha256": sha256_file(Path(__file__))},
        "profile": {
            "model": MODEL, "think": False, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
            "max_turns": MAX_TURNS, "episode_timeout_s": EPISODE_TIMEOUT_S,
            "check_command": CHECK_COMMAND, "check_timeout_s": 120.0,
        },
        "python": sys.version,
        "argv_preview": {
            "module": "cicada", "flags": [
                "--workspace <fixture>", f"--model {MODEL}", "--num-ctx 32768",
                "--ollama-url <local-tee-proxy>", f"--check-command {CHECK_COMMAND}",
            ],
        },
    }
    facts = build_fixture(fixture_root)
    identity.update({
        "main_head": facts["main_head"],
        "fixture_commit": facts["fixture_commit"],
        "fixture_copied_py_files": facts["copied_py_files"],
        "checker_sha256": facts["checker_sha256"],
        "prompt_sha256": sha256_text(TASK_PROMPT),
        "checker_baseline_exit": facts["checker_baseline_exit"],
        "checker_baseline_stderr_empty": facts["checker_baseline_stderr_empty"],
        "checker_baseline_stdout": facts["checker_baseline_stdout"],
        "fixture_import_probe": facts["import_probe"],
    })
    identity.update(ollama_digest())
    (attempt_dir / "identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8", newline="\n")
    (attempt_dir / "fixture-manifest-before.json").write_text(
        json.dumps(facts["baseline_manifest"], indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    proxy = _TeeProxy(attempt_dir / "proxy")
    argv = [
        str(VENV_PYTHON), "-m", "cicada",
        "--workspace", str(fixture_root),
        "--ollama-url", proxy.url,
        "--model", MODEL,
        "--num-ctx", str(NUM_CTX),
        "--check-command", CHECK_COMMAND,
        TASK_PROMPT,
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONIOENCODING": "utf-8",
        # launcher 卫生策略: checker/CLI 不写字节码, fixture 只留下真实源码变更
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{PROJECT / '.venv' / 'Scripts'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", env=env,
            timeout=EPISODE_TIMEOUT_S,
        )
        returncode: int | str = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = "episode-timeout"
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
    elapsed = time.monotonic() - started

    proxy.stop()
    proxy.save()
    ps_after = loaded_state()
    (attempt_dir / "ps-after.json").write_text(
        json.dumps(ps_after, indent=2), encoding="utf-8", newline="\n")
    (attempt_dir / "cli-stdout.txt").write_text(stdout, encoding="utf-8", newline="\n")
    (attempt_dir / "cli-stderr.txt").write_text(stderr, encoding="utf-8", newline="\n")

    # 独立 checker (与模型报告无关) 与 fixture 终态
    checker_run = subprocess.run(
        [str(VENV_PYTHON), CHECKER_NAME], cwd=fixture_root, capture_output=True, text=True,
        encoding="utf-8", env=env, timeout=60,
    )
    (attempt_dir / "checker-run.txt").write_text(
        json.dumps({"exit_code": checker_run.returncode, "stdout": checker_run.stdout,
                    "stderr": checker_run.stderr}, indent=2), encoding="utf-8", newline="\n")
    status = _git(fixture_root, "status", "--porcelain")
    diff = _git(fixture_root, "diff")
    (attempt_dir / "git-status.txt").write_text(status.stdout, encoding="utf-8", newline="\n")
    (attempt_dir / "git-diff.patch").write_text(diff.stdout, encoding="utf-8", newline="\n")
    (attempt_dir / "fixture-manifest-after.json").write_text(
        json.dumps(manifest(fixture_root), indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    artifacts_dir = attempt_dir / "artifacts"
    outputs = fixture_root / ".cicada" / "outputs"
    if outputs.exists():
        shutil.copytree(outputs, artifacts_dir)

    counts = proxy.prompt_eval_counts()
    peaks = [item["prompt_eval_count"] for item in counts if item["prompt_eval_count"] is not None]
    metrics = {
        "per_call": counts,
        "prompt_eval_count_peak": max(peaks) if peaks else None,
        "prompt_eval_count_sum": sum(peaks) if peaks else None,
        "prompt_eval_sum_note": "跨调用求和含重复处理, 不是唯一上下文 token 数",
        "eval_count_sum": sum(
            item["eval_count"] for item in counts if item["eval_count"] is not None),
        "proxy_errors": proxy.errors,
    }
    (attempt_dir / "proxy-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")

    checker_after = sha256_file(fixture_root / CHECKER_NAME)
    changed_git = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "attempt": attempt,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 2),
        "stdout": stdout,
        "stderr": stderr,
        "checker_exit": checker_run.returncode,
        "checker_sha256_after": checker_after,
        "checker_unchanged": checker_after == facts["checker_sha256"],
        "changed_git_lines": changed_git,
        "git_diff": diff.stdout,
        "prompt_eval_count_peak": metrics["prompt_eval_count_peak"],
        "prompt_eval_count_sum": metrics["prompt_eval_count_sum"],
        "proxy_errors": proxy.errors,
        "evidence_dir": str(attempt_dir),
        "identity": identity,
    }
