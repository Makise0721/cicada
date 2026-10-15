"""两题公开实测的任务身份接缝 (PV01)。

职责边界:
- 只服务 tests/e2e/preview_runner.py; 旧单题 tests/e2e/swebench_runner.py 与其
  模块级常量保持原样, 本模块不改写 bridge 的身份常量, 也不 monkeypatch 题目身份。
- 题目相关参数经不可变 TaskSpec 显式传入; 新增题只增 TaskSpec, 不改旧题行为。
- 只读公开材料 (固定上游 revision 的 instruction/task/verifier/参考补丁) 与固定数据集行;
  受保护评分材料按题冻结在本包 private/ 下, 不进入模型可见路径。

identity 交叉核对: manifest 声明的身份必须与 selection.json 及固定 parquet 行一致,
拒绝任何未冻结的题、不匹配映射或改写后的 base_commit。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from tests.support import swebench_bridge as bridge

PROJECT = Path(__file__).resolve().parents[2]
PUBLIC_VALIDATION = PROJECT / ".scratch" / "preview-readiness" / "public-validation"
SELECTION_PATH = PUBLIC_VALIDATION / "selection.json"
MANIFESTS_DIR = PUBLIC_VALIDATION / "manifests"
DOWNLOADS_ROOT = PUBLIC_VALIDATION / "downloads"
LEGACY_DOWNLOADS = PROJECT / ".scratch" / "live-coding-under-5-rmb" / "runs" / "downloads"

SCHEMA_VERSION = 1
PARQUET_NAME = "swebench-pro-default-test.parquet"
SHA256SUMS_NAME = "upstream-v2-SHA256SUMS"
COMMIT_META_NAME = "upstream-commit.json"

# 上游 v2/SHA256SUMS 的固定 Git blob 身份 (与旧单题流程同一 pin)。
SHA256SUMS_GIT_BLOB = "3015526e9ef6422e9c7712744fc5941387ecf513"

# 每个 TaskSpec 需要的上游公开材料 (相对 v2/tasks/<instance_id>/)。
UPSTREAM_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/gold_patch.diff",
    "solution/solve.sh",
    *(f"tests/{name}" for name in bridge.VERIFIER_FILES),
)

# 公开材料中可用于模型侧提示的文件 (其余留在 private/)。
PUBLIC_MATERIALS = ("instruction.md", "task.toml")

# 冻结的每题允许改动集合 (authorization-package.md); manifest 只能声明完全相同的集合,
# 不得增删。放宽 allowlist 等于放宽"模型只能改业务文件、不得改测试"的纪律, 必须由主 Agent
# 在授权包中显式裁决, 不能由本 harness 的 manifest 自行扩大。
FROZEN_ALLOWED_PATHS: dict[str, tuple[str, ...]] = {
    "repeat-lifecycle": ("lib/ansible/utils/vars.py", "lib/ansible/vars/manager.py"),
    "new-multifile": (
        "lib/ansible/module_utils/common/locale.py",
        "lib/ansible/module_utils/basic.py",
    ),
}

# manifest 中必须与 canonical manifest 逐字一致的字段 (除已对 selection.json 交叉核对的
# 身份字段外)。--manifest 是文档化的准入接口, 不许携带未授权限额/模型/设计号; 复制件内容
# 一致则允许, 篡改即拒绝。限额是真钱闸门, 不能让 harness 自己放宽。
FROZEN_MANIFEST_FIELDS = (
    "design_id", "scenario", "task_key", "instance_id", "repo", "base_commit",
    "upstream_revision", "allowed_paths", "dataset", "public_inputs",
    "limits", "model", "cost_profile_path",
)


class ManifestError(RuntimeError):
    """manifest/身份校验失败; 保留具体不足即停止, 不换题不削弱基准。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    """原子写 + fsync; 回执与证据共用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class TaskSpec:
    """一道固定公开任务的身份; 不可变, 由 manifest 与 selection.json 交叉核对后构造。"""

    task_key: str
    scenario: str
    instance_id: str
    repo: str
    base_commit: str
    image_tag: str
    allowed_paths: tuple[str, ...]
    upstream_revision: str
    parquet_sha256: str
    instruction_sha256: str
    task_toml_sha256: str

    def to_json(self) -> dict:
        return {
            "task_key": self.task_key, "scenario": self.scenario,
            "instance_id": self.instance_id, "repo": self.repo,
            "base_commit": self.base_commit, "image_tag": self.image_tag,
            "allowed_paths": list(self.allowed_paths),
            "upstream_revision": self.upstream_revision,
            "parquet_sha256": self.parquet_sha256,
            "instruction_sha256": self.instruction_sha256,
            "task_toml_sha256": self.task_toml_sha256,
        }


def load_selection() -> dict:
    selection = load_json(SELECTION_PATH)
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("selection.json schema_version is not 1")
    return selection


def manifest_path_for(task_key: str) -> Path:
    return MANIFESTS_DIR / f"{task_key}.json"


def downloads_dir(spec: TaskSpec) -> Path:
    return DOWNLOADS_ROOT / spec.task_key


def spec_from_manifest(manifest_path: Path) -> TaskSpec:
    """读取并校验 manifest: 身份必须与冻结 selection.json 完全一致。"""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("manifest schema_version is not 1")
    required = ("design_id", "dataset", "upstream_revision", "task_key", "instance_id",
                "repo", "base_commit", "allowed_paths", "scenario", "public_inputs",
                "limits", "model", "cost_profile_path")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ManifestError(f"manifest missing fields: {','.join(missing)}")

    selection = load_selection()
    task_key = manifest["task_key"]
    tasks = selection.get("tasks", {})
    if task_key not in tasks:
        raise ManifestError(f"task_key {task_key!r} is not one of the frozen selection tasks")
    frozen = tasks[task_key]
    meta = frozen["metadata"]

    scenario = manifest["scenario"]
    if scenario not in ("coding", "restore-compact-continue"):
        raise ManifestError(f"unsupported scenario {scenario!r}")
    # 冻结场景绑定: 生命周期题只做 restore-compact-continue, 新题只做 coding。
    expected_scenario = "restore-compact-continue" if task_key == "repeat-lifecycle" else "coding"
    if scenario != expected_scenario:
        raise ManifestError(f"scenario {scenario!r} does not match frozen binding for {task_key}")

    dataset = manifest["dataset"]
    if (dataset.get("id") != selection["dataset"] or dataset.get("config") != selection["config"]
            or dataset.get("split") != selection["split"]
            or dataset.get("revision") != selection["hf_revision"]
            or dataset.get("parquet_sha256") != selection["parquet_sha256"]):
        raise ManifestError("manifest dataset identity differs from frozen selection")
    if manifest["upstream_revision"] != selection["upstream_revision"]:
        raise ManifestError("manifest upstream_revision differs from frozen selection")
    if manifest["instance_id"] != meta["instance_id"] or manifest["repo"] != meta["repo"]:
        raise ManifestError("manifest instance identity differs from frozen selection")
    if manifest["base_commit"] != meta["base_commit"]:
        raise ManifestError("manifest base_commit differs from frozen selection")
    allowed = tuple(manifest["allowed_paths"])
    if not allowed or len(set(allowed)) != len(allowed):
        raise ManifestError("allowed_paths must be a non-empty unique list")
    frozen_allowed = FROZEN_ALLOWED_PATHS.get(task_key)
    if frozen_allowed is None:
        raise ManifestError(f"no frozen allowed_paths for task_key {task_key!r}")
    if set(allowed) != set(frozen_allowed) or len(allowed) != len(frozen_allowed):
        raise ManifestError(
            f"allowed_paths differ from frozen authorization contract for {task_key}: "
            f"expected {sorted(frozen_allowed)}")

    inputs = manifest["public_inputs"]
    for name in PUBLIC_MATERIALS:
        entry = inputs.get(name)
        if not isinstance(entry, dict) or "sha256" not in entry or "path" not in entry:
            raise ManifestError(f"public_inputs.{name} needs path and sha256")
        local = (PUBLIC_VALIDATION / entry["path"]).resolve()
        if not local.is_file():
            raise ManifestError(f"public input missing on disk: {entry['path']}")
        if sha256_file(local) != entry["sha256"]:
            raise ManifestError(f"public input {name} differs from manifest hash")
        if sha256_file(local) != frozen["sources"][name]["sha256"]:
            raise ManifestError(f"public input {name} differs from frozen selection hash")

    # --manifest 是准入接口, 复制件必须与 canonical manifest 逐字一致: 限额/模型/设计号
    # 都是真钱与授权边界, 不能由传入文件自行放宽。同一内容复制到别处仍可用。
    _require_canonical_manifest(manifest_path, manifest, task_key)

    return TaskSpec(
        task_key=task_key, scenario=scenario, instance_id=meta["instance_id"],
        repo=meta["repo"], base_commit=meta["base_commit"],
        image_tag=f"ghcr.io/scaleapi/swe-bench_pro-v2:{meta['instance_id']}",
        allowed_paths=allowed, upstream_revision=selection["upstream_revision"],
        parquet_sha256=selection["parquet_sha256"],
        instruction_sha256=inputs["instruction.md"]["sha256"],
        task_toml_sha256=inputs["task.toml"]["sha256"],
    )


# --- manifest 授权边界 ---


def _require_canonical_manifest(manifest_path: Path, manifest: dict, task_key: str) -> None:
    """传入 manifest 的授权字段必须与 canonical manifest 逐字一致。

    身份字段已对 selection.json 交叉核对; 这里补上 design_id / limits / model /
    cost_profile_path 等授权边界: 复制件内容一致仍可用, 篡改即拒绝。
    """
    canonical_path = manifest_path_for(task_key)
    if not canonical_path.is_file():
        raise ManifestError(f"canonical manifest missing: {canonical_path}")
    canonical = load_json(canonical_path)
    for field in FROZEN_MANIFEST_FIELDS:
        if manifest.get(field) != canonical.get(field):
            raise ManifestError(
                f"manifest field {field!r} differs from the frozen authorization contract")


# --- 公开材料下载与核验 ---


def _legacy_verified(name: str, expected_sha256: str | None) -> Path | None:
    """旧流程已缓存的共享冻结材料: 只在校验通过后复用, 不凭空采信。"""
    candidate = LEGACY_DOWNLOADS / name
    if not candidate.is_file():
        return None
    if expected_sha256 is not None and sha256_file(candidate) != expected_sha256:
        return None
    return candidate


DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_S = 120


def _download(url: str, target: Path) -> None:
    """下载固定 revision 的公开材料; 只对网络读取做有界重试。

    重试仅覆盖瞬时传输错误 (超时/连接重置); 本地写盘失败不在重试范围, 直接抛出,
    避免把磁盘故障误报成网络问题。内容 hash 仍由 verify_materials 独立核对。
    HTTP 状态错误不重试。
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen

    target.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    data: bytes | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                data = response.read()
            break
        except HTTPError:
            raise
        except (URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt == DOWNLOAD_ATTEMPTS:
                break
            time.sleep(2.0 * attempt)
    if data is None:
        raise ManifestError(
            f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {type(last).__name__}")
    temporary = target.with_suffix(target.suffix + ".download")
    temporary.write_bytes(data)
    os.replace(temporary, target)


def ensure_materials(spec: TaskSpec) -> None:
    """下载缺失的固定 revision 公开材料; 共享冻结材料优先复用已校验的本地副本。"""
    base = downloads_dir(spec)
    raw = f"https://raw.githubusercontent.com/{bridge.UPSTREAM_REPO}/{spec.upstream_revision}"

    parquet = base / PARQUET_NAME
    if not parquet.is_file():
        legacy = _legacy_verified(PARQUET_NAME, spec.parquet_sha256)
        if legacy is not None:
            parquet.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_bytes(legacy.read_bytes())
        else:
            _download(f"https://hf-mirror.com/datasets/{bridge.DATASET_ID}/resolve/"
                      f"{bridge.DATASET_REVISION}/data/default/test-00000-of-00001.parquet", parquet)

    sums = base / SHA256SUMS_NAME
    if not sums.is_file():
        legacy = _legacy_verified(SHA256SUMS_NAME, None)
        if legacy is not None and _git_blob_sha1(legacy.read_bytes()) == SHA256SUMS_GIT_BLOB:
            sums.parent.mkdir(parents=True, exist_ok=True)
            sums.write_bytes(legacy.read_bytes())
        else:
            _download(f"{raw}/v2/SHA256SUMS", sums)

    commit = base / COMMIT_META_NAME
    if not commit.is_file():
        _download(f"https://api.github.com/repos/{bridge.UPSTREAM_REPO}/commits/"
                  f"{spec.upstream_revision}", commit)

    for name in UPSTREAM_FILES:
        target = base / "upstream" / name
        if not target.is_file():
            _download(f"{raw}/v2/tasks/{spec.instance_id}/{name}", target)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def upstream_expected_hashes(spec: TaskSpec) -> dict[str, str]:
    sums = downloads_dir(spec) / SHA256SUMS_NAME
    if not sums.is_file():
        raise ManifestError(f"upstream SHA256SUMS missing: {sums}")
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        expected[name.strip().removeprefix("*")] = digest.strip()
    return expected


def verify_materials(spec: TaskSpec) -> dict:
    """核对已下载材料的 hash: parquet 对数据集 pin, 上游文件对固定 SHA256SUMS。"""
    base = downloads_dir(spec)
    parquet = base / PARQUET_NAME
    if not parquet.is_file():
        raise ManifestError(f"dataset parquet missing: {parquet}")
    actual = sha256_file(parquet)
    if actual != spec.parquet_sha256 or actual != bridge.DATASET_PARQUET_LFS_SHA256:
        raise ManifestError(f"dataset parquet sha256 mismatch: {actual}")

    sums = base / SHA256SUMS_NAME
    if not sums.is_file():
        raise ManifestError("upstream v2/SHA256SUMS missing")
    sums_bytes = sums.read_bytes()
    if _git_blob_sha1(sums_bytes) != SHA256SUMS_GIT_BLOB:
        raise ManifestError("SHA256SUMS differs from pinned upstream Git blob")
    expected = upstream_expected_hashes(spec)

    materials: dict[str, dict] = {
        "dataset_parquet": {"sha256": actual, "lfs_oid": bridge.DATASET_PARQUET_LFS_SHA256},
        "v2-SHA256SUMS": {"sha256": sha256_file(sums)},
    }
    for name in UPSTREAM_FILES:
        path = base / "upstream" / name
        if not path.is_file():
            raise ManifestError(f"upstream material missing: {name}")
        digest = sha256_file(path)
        recorded = expected.get(f"tasks/{spec.instance_id}/{name}")
        if recorded is None:
            raise ManifestError(f"tasks/{spec.instance_id}/{name} not listed in upstream SHA256SUMS")
        if digest != recorded:
            raise ManifestError(f"{name} sha256 mismatch against upstream SHA256SUMS")
        materials[name] = {"sha256": digest}

    commit_path = base / COMMIT_META_NAME
    if not commit_path.is_file():
        raise ManifestError("upstream commit metadata missing")
    commit = load_json(commit_path)
    if (commit.get("sha") != spec.upstream_revision
            or commit.get("commit", {}).get("tree", {}).get("sha") != bridge.UPSTREAM_TREE):
        raise ManifestError("upstream commit/tree identity mismatch")
    materials["upstream-commit.json"] = {
        "sha256": sha256_file(commit_path), "revision": spec.upstream_revision,
        "tree": bridge.UPSTREAM_TREE}
    return materials


def dataset_row(spec: TaskSpec) -> dict:
    """从固定 parquet 读取白名单元数据; base_commit 只来自数据集行。"""
    import pyarrow.parquet as pq

    table = pq.read_table(downloads_dir(spec) / PARQUET_NAME)
    data = table.to_pydict()
    if spec.instance_id not in data["instance_id"]:
        raise ManifestError("instance not present in pinned dataset revision")
    index = data["instance_id"].index(spec.instance_id)
    row = {key: data[key][index] for key in data}
    if row["repo"] != spec.repo or row["version"] != bridge.DATASET_VERSION:
        raise ManifestError("dataset row repo/version mismatch")
    if row["docker_image"] != spec.image_tag:
        raise ManifestError("dataset row docker_image mismatch")
    if row["base_commit"] != spec.base_commit:
        raise ManifestError("dataset row base_commit differs from frozen selection")
    f2p = json.loads(row["fail_to_pass"]) if isinstance(row["fail_to_pass"], str) else row["fail_to_pass"]
    p2p = json.loads(row["pass_to_pass"]) if isinstance(row["pass_to_pass"], str) else row["pass_to_pass"]
    if not f2p:
        raise ManifestError("fail_to_pass empty; admission refuses")
    return {
        "base_commit": row["base_commit"], "fail_to_pass": list(f2p),
        "pass_to_pass": list(p2p),
        "selected_test_files": json.loads(row["selected_test_files_to_run"]),
    }


def verifier_identity(spec: TaskSpec) -> dict:
    """官方 verifier 文件 hash 与 F2P/P2P (config.json 解析)。"""
    tests_dir = downloads_dir(spec) / "upstream" / "tests"
    hashes = {name: sha256_file(tests_dir / name) for name in bridge.VERIFIER_FILES}
    f2p, p2p = bridge.parse_required_tests((tests_dir / "config.json").read_bytes())
    return {"sha256": hashes, "fail_to_pass": f2p, "pass_to_pass": p2p,
            "config_bytes_sha256": sha256_file(tests_dir / "config.json")}


def verifier_dir_for(spec: TaskSpec, private_dir: Path) -> Path:
    """冻结 verifier 的挂载目录: 只含五个官方文件, hash 来自 lock。"""
    target = private_dir / "verifier"
    target.mkdir(parents=True, exist_ok=True)
    source = downloads_dir(spec) / "upstream" / "tests"
    for name in bridge.VERIFIER_FILES:
        data = (source / name).read_bytes()
        out = target / name
        if out.exists() and out.read_bytes() != data:
            raise ManifestError(f"private verifier changed: {name}")
        if not out.exists():
            out.write_bytes(data)
    return target


def gold_patch_bytes(spec: TaskSpec) -> bytes:
    return (downloads_dir(spec) / "upstream" / "solution" / "gold_patch.diff").read_bytes()


# --- 镜像与容器身份 ---


def image_identity(docker: bridge.DockerCLI, spec: TaskSpec) -> dict:
    record = docker.run("image", "inspect", spec.image_tag, check=True)
    payload = json.loads(record.stdout)[0]
    digests = [d for d in payload.get("RepoDigests", []) if d.startswith("ghcr.io/scaleapi/")]
    if not digests:
        raise ManifestError("image has no ghcr.io repo digest (not pulled by digest)")
    return {
        "tag": spec.image_tag, "digest": digests[0], "image_id": payload.get("Id"),
        "architecture": payload.get("Architecture"), "os": payload.get("Os"),
    }


def container_git_head(docker: bridge.DockerCLI, supervisor: bridge.ContainerSupervisor,
                       image: str, evidence_dir: Path) -> str:
    """镜像容器内 git rev-parse HEAD: 证明冻结源码基线就是声明的 base_commit。

    运行记录 (start/wait/logs) 写入 evidence_dir, 使容器身份可按证据复核, 而不只留返回值。
    """
    label = supervisor.owner_label("source-head")
    supervisor.register_intent(label, "source-head", image, os.getpid())
    create = docker.run(
        "create", "--name", label, "--platform", bridge.IMAGE_PLATFORM,
        "--label", f"{bridge.DOCKER_LABEL_OWNER}={label}", "--workdir", "/app",
        "--network", "none", image, "git", "rev-parse", "HEAD")
    if create.exit_code != 0:
        supervisor.stop_and_remove(label, None)
        raise ManifestError("source-head container create failed")
    cid = create.stdout.strip()
    try:
        supervisor.register_container(label, cid)
        start = docker.run("start", cid, timeout_s=60)
        wait = docker.run("wait", cid, timeout_s=300)
        logs = docker.run("logs", cid, timeout_s=60)
        write_json(evidence_dir / "source-head.json", {
            "owner_label": label, "container_id": cid, "image": image,
            "start": start.to_json(), "wait": wait.to_json()})
        if start.exit_code != 0 or wait.exit_code != 0:
            raise ManifestError("source-head container run failed")
        head = logs.stdout.strip().splitlines()[-1].strip() if logs.stdout.strip() else ""
        if not head:
            raise ManifestError("source-head container produced no HEAD")
        return head
    finally:
        supervisor.stop_and_remove(label, cid)
