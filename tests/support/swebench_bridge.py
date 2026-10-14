"""SWE-bench Pro V2 单题桥接: Git workspace/补丁导出与本地官方容器评测 (票 01)。

职责边界 (契约 C2–C4):
- 只服务 tests/e2e/swebench_runner.py 的 harness 入口, 不注册生产插件、不改 core/runtime。
- 冻结身份常量来自 selection.md r2; base_commit/F2P/P2P 等在 prepare 时从固定数据集
  行与镜像实测取得, 不用 instance 名猜测。
- 每次评测一个固定 digest 新容器; 创建前登记 intent + owner label, 创建后登记容器 ID,
  由 ContainerSupervisor 查询/停止/确认退出, 不依赖 checker 子进程存活。
- 补丁导出用临时 index (read-tree base → add 冻结范围 → diff --cached --binary),
  真实 index 与工作区不被改写; 范围外改动拒绝而不静默过滤。

容器与 docker CLI 交互全部经 run() 记录命令与结果; 证据写入由调用方提供的目录。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# --- 冻结身份 (selection.md r2; 票 01 实测补齐其余字段) ---

DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_VERSION = "2.0.0"
DATASET_PARQUET_LFS_SHA256 = "1067f72c1f43bbc578276888625db3b836cebb535546c9eaf824c5d5dd104a7f"

UPSTREAM_REPO = "scaleapi/SWE-bench_Pro-os"
UPSTREAM_REVISION = "31a20c751b0bd56db6510ce7d4d3c269c5b7c72b"
UPSTREAM_TREE = "beb15336e37995a2dc2e5908e2bde931d59cef1e"

INSTANCE_ID = (
    "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-"
    "v30a923fb5c164d6cd18280c02422f75e611e8fb2"
)
IMAGE_TAG = f"ghcr.io/scaleapi/swe-bench_pro-v2:{INSTANCE_ID}"
IMAGE_PLATFORM = "linux/amd64"
INSTANCE_REPO = "ansible/ansible"

# 模型允许修改的两个生产源码文件 (selection.md); 其余路径一律拒绝导出。
ALLOWED_PATHS = ("lib/ansible/utils/vars.py", "lib/ansible/vars/manager.py")

# 官方 verifier 文件 (v2/tasks/<instance>/tests/, 相对上游 revision)。
VERIFIER_FILES = (
    "test.sh",
    "run_script.sh",
    "parser.py",
    "config.json",
    "test_patch.patch",
)

# 上游 v2/SHA256SUMS 文件名, 用于核对下载材料; 本身 hash 一并记录。
UPSTREAM_SHA256SUMS_PATH = "v2/SHA256SUMS"

# 官方评测入口与工件 (契约 C4; 不改 test.sh/run_script.sh/parser.py/config.json/test_patch.patch)。
VERIFIER_COMMAND = ["bash", "/tests/test.sh"]
REWARD_PATH = "/logs/verifier/reward.txt"
OUTPUT_JSON_PATH = "/logs/verifier/output.json"
VERIFIER_ARTIFACTS = (
    "reward.txt",
    "output.json",
    "run-script-stdout.txt",
    "run-script-stderr.txt",
)

# 评测监督上限 (spec D3); check-timeout 300s 留清理余量。
GRADER_TIMEOUT_S = 240.0
# 单次校准评测的耗时上限 (spec D3: 空/参考补丁各一次且 <=120s)。
CALIBRATION_TIMEOUT_S = 120.0

DOCKER_LABEL_OWNER = "cicada.swebench.owner"
SUPERVISOR_DIRNAME = "supervisor"


class BridgeError(RuntimeError):
    """携带固定分类的桥接失败; message 不含凭据或模型文字。"""


class PatchScopeError(BridgeError):
    """工作区改动超出冻结 allowlist; 拒绝导出而非静默过滤。"""

    def __init__(self, offending: list[str]):
        self.offending = offending
        super().__init__(f"workspace changes outside allowlist: {len(offending)} path(s)")


@dataclass
class CommandRecord:
    """一条实际执行的命令与有界结果; 回执与证据共用。"""

    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float
    truncated: bool = False

    def to_json(self) -> dict:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "seconds": round(self.seconds, 3),
            "truncated": self.truncated,
        }


OUTPUT_TAIL_BYTES = 20_000


def _tail(text: str) -> tuple[str, bool]:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= OUTPUT_TAIL_BYTES:
        return text, False
    return data[-OUTPUT_TAIL_BYTES:].decode("utf-8", errors="replace"), True


class DockerCLI:
    """docker CLI 只读封装; 每条命令记录进调用方命令日志。"""

    def __init__(self, command_log: list[CommandRecord] | None = None):
        self.command_log = command_log if command_log is not None else []
        self.deadline: float | None = None
        self.cancel = None

    def run(self, *args: str, timeout_s: float = 120.0,
            check: bool = False) -> CommandRecord:
        started = time.monotonic()
        if self.cancel is not None and self.cancel.is_set():
            return CommandRecord(["docker", *args], None, "", "evaluation cancelled before command", 0)
        if self.deadline is not None:
            timeout_s = min(timeout_s, max(0.01, self.deadline - started))
        try:
            proc = subprocess.run(
                ["docker", *args], capture_output=True, timeout=timeout_s)
            code: int | None = proc.returncode
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            code = None
            out = (exc.stdout or b"").decode("utf-8", errors="replace")
            err = (exc.stderr or b"").decode("utf-8", errors="replace")
        record = CommandRecord(
            command=["docker", *args], exit_code=code,
            stdout=out, stderr=err,
            seconds=time.monotonic() - started, truncated=False)
        self.command_log.append(record)
        if check and (code is None or code != 0):
            raise BridgeError(
                f"docker command failed ({args[0]}): exit={code} stderr={record.stderr[:400]}")
        return record


def docker_available() -> tuple[bool, str]:
    """现有引擎是否可用; 不可用只报告事实, 不安装或启动服务。"""
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker CLI unreachable: {type(exc).__name__}"
    if proc.returncode != 0:
        return False, proc.stderr.decode("utf-8", errors="replace").strip()[:400]
    return True, proc.stdout.decode("utf-8", errors="replace").strip()


# --- 容器监督者 (契约 C4) ---


class ContainerSupervisor:
    """本票容器生命周期的拥有者; 状态持久登记在 evidence/supervisor 下。

    - 创建前: intent 记录 (fsync) 唯一 owner label;
    - 创建后: container ID 记录; 查不到创建结果时按 intent/精确 label 核对;
    - 结束: stopped/removed 确认记录; 无法确认时保留 process_uncertain 事实。
    生命周期属于主 runner 进程, 独立于可能被终止的 checker 子进程。
    """

    def __init__(self, evidence_dir: Path, docker: DockerCLI):
        self.dir = evidence_dir / SUPERVISOR_DIRNAME
        self.dir.mkdir(parents=True, exist_ok=True)
        self.docker = docker
        self._intents: list[dict] = []

    @property
    def intents_file(self) -> Path:
        return self.dir / "intents.jsonl"

    @property
    def healthy(self) -> bool:
        """owner 健康即可创建容器; 进程内监督者始终拥有本线程。"""
        return True

    def _append(self, filename: str, record: dict) -> None:
        path = self.dir / filename
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def owner_label(self, purpose: str) -> str:
        return f"cicada-sb-{purpose}-{uuid.uuid4().hex[:12]}"

    def register_intent(self, label: str, purpose: str, image: str, pid: int) -> None:
        record = {
            "kind": "create-intent", "owner_label": label, "purpose": purpose,
            "image": image, "pid": pid, "time": time.time(),
        }
        self._append("intents.jsonl", record)
        self._intents.append(record)

    def register_container(self, label: str, container_id: str) -> None:
        self._append("intents.jsonl", {
            "kind": "created", "owner_label": label, "container_id": container_id,
            "time": time.time(),
        })

    def register_exit(self, label: str, container_id: str | None, outcome: str) -> None:
        self._append("intents.jsonl", {
            "kind": "exit", "owner_label": label, "container_id": container_id,
            "outcome": outcome, "time": time.time(),
        })

    def query_by_label(self, label: str) -> str | None:
        """按精确 owner label 查询现存容器; 不按名称猜测归属。"""
        record = self.docker.run(
            "ps", "-a", "--filter", f"label={DOCKER_LABEL_OWNER}={label}", "--format", "{{.ID}}", timeout_s=5)
        if record.exit_code != 0:
            raise BridgeError("cannot query owned containers")
        ids = [line for line in record.stdout.splitlines() if line.strip()]
        return ids[0] if ids else None

    def stop_and_remove(self, label: str, container_id: str | None) -> str:
        """停止并删除本票容器; 返回 confirmed / uncertain。"""
        cid = container_id or self.query_by_label(label)
        if cid is None:
            self.register_exit(label, None, "not-found")
            return "not-found"
        stop = self.docker.run("stop", "-t", "1", cid, timeout_s=10)
        rm = self.docker.run("rm", "-f", cid, timeout_s=5)
        gone = self.query_by_label(label) is None
        confirmed = rm.exit_code == 0 and gone
        self.register_exit(label, cid, "confirmed" if confirmed else "uncertain")
        return "confirmed" if confirmed else "uncertain"

    def confirm_orphans(self, labels: list[str]) -> list[dict]:
        """checker 死亡后由主 runner 调用: 按 intent 核对并收尾未确认容器。"""
        results = []
        for label in labels:
            cid = self.query_by_label(label)
            if cid is None:
                results.append({"owner_label": label, "outcome": "already-gone"})
                continue
            outcome = self.stop_and_remove(label, cid)
            results.append({"owner_label": label, "outcome": outcome})
        return results

    def cleanup_registered(self) -> list[dict]:
        if not self.intents_file.exists():
            return []
        records = [json.loads(line) for line in self.intents_file.read_text(encoding="utf-8").splitlines()]
        labels = {r["owner_label"] for r in records if r["kind"] == "create-intent"}
        return self.confirm_orphans(sorted(labels))


@dataclass
class ContainerSpec:
    """一次评测容器的固定参数 (契约 C4)。"""

    image: str
    image_digest: str
    verifier_dir: Path
    owner_label: str


@dataclass
class EvaluationResult:
    """一次官方 verifier 运行的完整事实。"""

    owner_label: str
    container_id: str | None
    applied_patch: bool
    patch_empty: bool
    verifier_exit_code: int | None
    reward: int | None
    output_json: dict | None
    required_total: int
    required_passed: int
    missing_required: list[str]
    f2p_passed: bool
    p2p_passed: bool
    timed_out: bool
    apply_failed: bool
    cleanup: str
    wall_time_s: float
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "owner_label": self.owner_label,
            "container_id": self.container_id,
            "applied_patch": self.applied_patch,
            "patch_empty": self.patch_empty,
            "verifier_exit_code": self.verifier_exit_code,
            "reward": self.reward,
            "output_json": self.output_json,
            "required_total": self.required_total,
            "required_passed": self.required_passed,
            "missing_required": self.missing_required,
            "f2p_passed": self.f2p_passed,
            "p2p_passed": self.p2p_passed,
            "timed_out": self.timed_out,
            "apply_failed": self.apply_failed,
            "cleanup": self.cleanup,
            "wall_time_s": round(self.wall_time_s, 3),
            "artifacts": self.artifacts,
        }


def parse_required_tests(config_bytes: bytes) -> tuple[list[str], list[str]]:
    """官方 config.json 的 F2P/P2P; 字段是 JSON 编码的字符串列表 (上游格式)。"""
    import ast

    config = json.loads(config_bytes.decode("utf-8"))

    def parse(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return ast.literal_eval(value)
        return value

    f2p = list(parse(config.get("fail_to_pass", [])))
    p2p = list(parse(config.get("pass_to_pass", [])))
    return f2p, p2p


def grade_from_artifacts(
    verifier_exit_code: int | None,
    reward_bytes: bytes | None,
    output_json_bytes: bytes | None,
    f2p: list[str],
    p2p: list[str],
) -> dict:
    """官方 exit + reward + output.json 的联合判定 (契约 C4: 缺用例/空集/解析失败不得 false pass)。"""
    reward = None
    if reward_bytes is not None:
        text = reward_bytes.decode("utf-8", errors="replace").strip()
        if text in ("0", "1"):
            reward = int(text)
    tests: dict[str, str] = {}
    parse_failed = output_json_bytes is None
    if output_json_bytes is not None:
        try:
            payload = json.loads(output_json_bytes.decode("utf-8"))
            for entry in payload.get("tests", []):
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    if entry["name"] in tests:
                        parse_failed = True
                    tests[entry["name"]] = str(entry.get("status", ""))
        except (ValueError, AttributeError):
            parse_failed = True
    required = list(dict.fromkeys([*f2p, *p2p]))
    passed = {name for name, status in tests.items() if status == "PASSED"}
    missing = [name for name in required if name not in passed]
    result = {
        "reward": reward,
        "required_total": len(required),
        "required_passed": len(required) - len(missing),
        "missing_required": missing,
        "required_observed": sum(name in tests for name in required),
        "f2p_failed": bool(f2p) and all(tests.get(name) == "FAILED" for name in f2p),
        "f2p_passed": all(name in passed for name in f2p) and bool(f2p),
        "p2p_passed": all(name in passed for name in p2p),
        "parse_failed": parse_failed,
        "required_empty": not required,
    }
    result["grader_pass"] = (
        verifier_exit_code == 0
        and reward == 1
        and bool(f2p)
        and not result["required_empty"]
        and not result["parse_failed"]
        and not missing
    )
    return result


class ContainerEvaluator:
    """创建固定 digest 新容器, 应用补丁并运行原样官方 verifier (契约 C2/D2)。"""

    def __init__(self, docker: DockerCLI, supervisor: ContainerSupervisor):
        self.docker = docker
        self.supervisor = supervisor

    def evaluate(
        self,
        spec: ContainerSpec,
        patch_bytes: bytes | None,
        evidence_dir: Path,
        f2p: list[str],
        p2p: list[str],
        timeout_s: float = GRADER_TIMEOUT_S,
    ) -> EvaluationResult:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        name = spec.owner_label
        started = time.monotonic()
        self.docker.deadline = started + max(1, timeout_s - 25)
        result = EvaluationResult(
            owner_label=spec.owner_label, container_id=None,
            applied_patch=False, patch_empty=patch_bytes is None or not patch_bytes.strip(),
            verifier_exit_code=None, reward=None, output_json=None,
            required_total=len({*f2p, *p2p}), required_passed=0,
            missing_required=[*f2p, *p2p], f2p_passed=False, p2p_passed=False,
            timed_out=False, apply_failed=False, cleanup="pending", wall_time_s=0.0,
        )
        # 创建前登记 intent; 精确 label 后续核对, 不按名称猜测。
        self.supervisor.register_intent(
            spec.owner_label, "evaluate", spec.image, os.getpid())
        create = self.docker.run(
            "create", "--name", name,
            "--platform", IMAGE_PLATFORM,
            "--label", f"{DOCKER_LABEL_OWNER}={spec.owner_label}",
            "--volume", f"{spec.verifier_dir.resolve()}:/tests:ro",
            "--workdir", "/app",
            "--network", "none",
            spec.image_digest, "sleep", "infinity",
            timeout_s=10,
        )
        if create.exit_code != 0:
            self.docker.deadline = None
            result.cleanup = self.supervisor.stop_and_remove(spec.owner_label, None)
            result.wall_time_s = time.monotonic() - started
            return result
        cid = create.stdout.strip()
        result.container_id = cid
        try:
            self.supervisor.register_container(spec.owner_label, cid)
            start = self.docker.run("start", cid, timeout_s=60)
            if start.exit_code != 0:
                return result
            if not result.patch_empty:
                # 补丁按字节原样复制进容器, 再 git apply; 失败即异常, 不静默跳过。
                self._copy_patch(cid, patch_bytes or b"", evidence_dir)
                apply_rec = self.docker.run(
                    "exec", "-w", "/app", cid, "git", "apply", "--verbose",
                    "/tmp/model.patch", timeout_s=60)
                (evidence_dir / "apply-patch.json").write_text(
                    json.dumps(apply_rec.to_json(), ensure_ascii=False, indent=1),
                    encoding="utf-8")
                if apply_rec.exit_code != 0:
                    result.apply_failed = True
                    return result
                result.applied_patch = True
            logs = self.docker.run("exec", cid, "mkdir", "-p", "/logs/verifier", timeout_s=10)
            if logs.exit_code != 0:
                return result
            exec_rec = self.docker.run(
                "exec", "-w", "/app", cid, *VERIFIER_COMMAND, timeout_s=timeout_s)
            result.verifier_exit_code = exec_rec.exit_code
            result.timed_out = exec_rec.exit_code is None
            (evidence_dir / "verifier-exec.json").write_text(
                json.dumps(exec_rec.to_json(), ensure_ascii=False, indent=1),
                encoding="utf-8")
            self._collect_artifacts(cid, evidence_dir, result)
            grade = grade_from_artifacts(
                result.verifier_exit_code,
                (evidence_dir / "verifier-logs" / "reward.txt").read_bytes()
                if (evidence_dir / "verifier-logs" / "reward.txt").exists() else None,
                (evidence_dir / "verifier-logs" / "output.json").read_bytes()
                if (evidence_dir / "verifier-logs" / "output.json").exists() else None,
                f2p, p2p)
            result.reward = grade["reward"]
            result.required_total = grade["required_total"]
            result.required_passed = grade["required_passed"]
            result.missing_required = grade["missing_required"]
            result.f2p_passed = grade["f2p_passed"]
            result.p2p_passed = grade["p2p_passed"]
            grade["artifacts_complete"] = all(name in result.artifacts for name in VERIFIER_ARTIFACTS)
            grade["grader_pass"] = grade["grader_pass"] and grade["artifacts_complete"]
            result.output_json = grade
        finally:
            self.docker.deadline = None
            result.cleanup = self.supervisor.stop_and_remove(spec.owner_label, cid)
            result.wall_time_s = time.monotonic() - started
            (evidence_dir / "docker-commands.json").write_text(
                json.dumps([r.to_json() for r in self.docker.command_log], ensure_ascii=False, indent=1), encoding="utf-8")
            (evidence_dir / "cleanup-commands.json").write_text(
                json.dumps([r.to_json() for r in self.supervisor.docker.command_log], ensure_ascii=False, indent=1), encoding="utf-8")
        return result

    def _copy_patch(self, cid: str, patch_bytes: bytes, evidence_dir: Path) -> None:
        """docker cp 只接受 tar 流或本地路径; 用本地临时文件按字节复制。"""
        tmp = evidence_dir / "model.patch"
        tmp.write_bytes(patch_bytes)
        rec = self.docker.run("cp", str(tmp), f"{cid}:/tmp/model.patch", timeout_s=60)
        (evidence_dir / "copy-patch.json").write_text(
            json.dumps(rec.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        if rec.exit_code != 0:
            raise BridgeError("docker cp patch failed")

    def _collect_artifacts(self, cid: str, evidence_dir: Path,
                           result: EvaluationResult) -> None:
        logs_dir = evidence_dir / "verifier-logs"
        rec = self.docker.run("cp", f"{cid}:/logs/verifier", str(logs_dir), timeout_s=60)
        (evidence_dir / "collect-logs.json").write_text(
            json.dumps(rec.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        if rec.exit_code != 0 or not logs_dir.is_dir():
            return
        for filename in VERIFIER_ARTIFACTS:
            path = logs_dir / filename
            if path.exists():
                result.artifacts[filename] = (
                    path.read_bytes().decode("utf-8", errors="replace"))


# --- Git workspace 与补丁导出 (契约 C2/C3) ---


class GitCLI:
    def __init__(self, workspace: Path, command_log: list[CommandRecord] | None = None):
        self.workspace = workspace
        self.command_log = command_log if command_log is not None else []

    def run(self, *args: str, timeout_s: float = 300.0, check: bool = False,
            env_extra: dict[str, str] | None = None) -> CommandRecord:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        if env_extra:
            env.update(env_extra)
        started = time.monotonic()
        proc = subprocess.run(
            ["git", "-C", str(self.workspace), *args],
            capture_output=True, timeout=timeout_s, env=env)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        record = CommandRecord(
            command=["git", "-C", str(self.workspace), *args], exit_code=proc.returncode,
            stdout=out, stderr=err,
            seconds=time.monotonic() - started, truncated=False)
        self.command_log.append(record)
        if check and proc.returncode != 0:
            raise BridgeError(f"git {args[0]} failed: {record.stderr[:400]}")
        return record


GIT_FREEZE_CONFIG = [
    ("core.longpaths", "true"),
    ("core.autocrlf", "false"),
    ("core.fileMode", "false"),
    ("core.symlinks", "false"),
    ("core.safecrlf", "false"),
    ("gc.auto", "0"),
    ("core.attributesFile", os.devnull),
]


@dataclass
class WorkspaceManifest:
    head: str
    path_count: int
    entries_sha256: str
    dirty_paths: list[str]
    file_count: int
    max_file_bytes: int
    total_bytes: int
    content_sha256: str

    def to_json(self) -> dict:
        return {
            "head": self.head, "path_count": self.path_count,
            "entries_sha256": self.entries_sha256, "dirty_paths": self.dirty_paths[:50],
            "dirty_count": len(self.dirty_paths), "file_count": self.file_count,
            "max_file_bytes": self.max_file_bytes, "total_bytes": self.total_bytes,
            "content_sha256": self.content_sha256,
        }


def porcelain_paths(status: str) -> list[str]:
    """git status --porcelain -z 的路径展开; rename/copy 条目同时含新路径与原始路径。"""
    tokens = [t for t in status.split("\0") if t]
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        paths.append(token[3:])
        if len(token) >= 2 and token[0] in "RC" or (len(token) >= 2 and token[1] in "RC"):
            if index < len(tokens):
                paths.append(tokens[index])
                index += 1
    return paths


def runtime_artifact(path: str) -> bool:
    """生产 policy 固定排除的根 .cicada 工件；不扩大其他业务路径范围。"""
    return path == ".cicada/" or path.startswith(".cicada/")


def workspace_manifest(workspace: Path, git: GitCLI) -> WorkspaceManifest:
    """HEAD + ls-files -s 指纹 + 工作区脏路径 + 大小资格 (只读)。"""
    head = git.run("rev-parse", "HEAD", check=True).stdout.strip()
    entries = git.run("ls-files", "-s", "-z", check=True).stdout
    digest = __import__("hashlib").sha256(entries.encode("utf-8")).hexdigest()
    status = git.run("status", "--porcelain", "-z", "--untracked-files=all").stdout
    dirty = [p for p in porcelain_paths(status) if not runtime_artifact(p)]
    file_count = 0
    total = 0
    max_file = 0
    content = __import__("hashlib").sha256()
    paths = []
    for directory, dirs, files in os.walk(workspace):
        if Path(directory) == workspace:
            dirs[:] = [name for name in dirs if name not in (".git", ".cicada")]
        paths.extend(Path(directory) / name for name in files)
    for path in sorted(paths):
        if path.is_file() and not path.is_symlink():
            file_count += 1
            size = path.stat().st_size
            total += size
            max_file = max(max_file, size)
            content.update(path.relative_to(workspace).as_posix().encode("utf-8") + b"\0")
            content.update(__import__("hashlib").sha256(path.read_bytes()).digest())
    return WorkspaceManifest(
        head=head, path_count=len([p for p in entries.split("\0") if p]),
        entries_sha256=digest, dirty_paths=dirty, file_count=file_count,
        max_file_bytes=max_file, total_bytes=total, content_sha256=content.hexdigest())


@dataclass
class PatchExport:
    patch_bytes: bytes
    sha256: str
    changed_paths: list[str]
    index_unchanged: bool
    empty: bool
    source_manifest: dict

    def to_json(self) -> dict:
        return {
            "sha256": self.sha256, "changed_paths": self.changed_paths,
            "index_unchanged": self.index_unchanged, "empty": self.empty,
            "bytes": len(self.patch_bytes), "source_manifest": self.source_manifest,
        }


def export_patch(workspace: Path, base_commit: str, allowed: tuple[str, ...],
                 temp_dir: Path, git: GitCLI) -> PatchExport:
    """相对固定 base 的完整补丁; 临时 index, 真实 index 不变 (契约 C3)。

    覆盖修改/删除/新增/模式变化; allowlist 之外的路径抛 PatchScopeError。
    """
    import hashlib

    filters = git.run("config", "--get-regexp", r"^filter\.")
    if filters.stdout.strip():
        raise BridgeError("repository Git filters are outside the frozen patch contract")
    before = workspace_manifest(workspace, git)
    status = git.run("status", "--porcelain", "-z", "--untracked-files=all", "--ignored").stdout
    changed: list[str] = []
    offending: list[str] = []
    for path in porcelain_paths(status):
        if runtime_artifact(path):
            continue
        if path not in allowed:
            offending.append(path)
        else:
            changed.append(path)
    if offending:
        raise PatchScopeError(sorted(offending))

    index_before = git.run("ls-files", "-s").stdout
    index_file = temp_dir.resolve() / f"index-{uuid.uuid4().hex}"
    env = {"GIT_INDEX_FILE": str(index_file)}
    git.run("read-tree", base_commit, check=True, env_extra=env)
    for path in changed:
        git.run("add", "--", path, check=True, env_extra=env)
    diff = git.run(
        "diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", base_commit,
        check=True, env_extra=env)
    index_file.unlink(missing_ok=True)
    index_after = git.run("ls-files", "-s").stdout
    after = workspace_manifest(workspace, git)
    if before != after:
        raise BridgeError("workspace changed during patch export")
    patch = diff.stdout.encode("utf-8")
    return PatchExport(
        patch_bytes=patch, sha256=hashlib.sha256(patch).hexdigest(),
        changed_paths=sorted(changed), index_unchanged=index_before == index_after,
        empty=not patch.strip(), source_manifest=after.to_json())


def capture_container_manifest(image: str, supervisor: ContainerSupervisor,
                               docker: DockerCLI, evidence_dir: Path) -> tuple[str, str]:
    """从镜像容器读取权威 git ls-files -s (模式/内容身份), 返回 (label, stdout)。"""
    label = supervisor.owner_label("manifest")
    supervisor.register_intent(label, "manifest", image, os.getpid())
    name = label
    create = docker.run(
        "create", "--name", name, "--platform", IMAGE_PLATFORM,
        "--label", f"{DOCKER_LABEL_OWNER}={label}", "--workdir", "/app",
        "--network", "none", image, "git", "ls-files", "-s")
    if create.exit_code != 0:
        supervisor.stop_and_remove(label, None)
        raise BridgeError("manifest container create failed")
    cid = create.stdout.strip()
    try:
        supervisor.register_container(label, cid)
        start = docker.run("start", cid, timeout_s=60)
        wait = docker.run("wait", cid, timeout_s=300)
        logs = docker.run("logs", cid, timeout_s=60)
        (evidence_dir / "container-manifest.json").write_text(json.dumps({
            "owner_label": label, "container_id": cid,
            "start": start.to_json(), "wait": wait.to_json(),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        if start.exit_code != 0 or wait.exit_code != 0:
            raise BridgeError("manifest container run failed")
        return label, logs.stdout
    finally:
        supervisor.stop_and_remove(label, cid)
