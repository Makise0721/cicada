"""SWE-bench Pro V2 单题验证 runner (契约 C1–C4; 票 01 范围)。

统一入口 (runbook r2):
  uv run --frozen python tests/e2e/swebench_runner.py prepare --run-root <dir>
  ... check --run-root <attempt> --workspace <ws>   (由生产 check-1 调用)
  ... grade --run-root <attempt> [--workspace <ws>]

- check 退出码: 0=完整通过, 1=代码未解, 2=环境/身份/证据异常 (契约 C4)。
- stdout 只输出有界摘要 (pass/fail、覆盖数量、冻结工件标识), 不含用例名、
  官方日志或私有材料; 完整证据写 run_root/evidence。
- run/dry-run/report 属票 02/03, 在本文件中按票推进。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from tests.support import swebench_bridge as bridge  # noqa: E402
from tests.support.swebench_owner import EvaluationOwner, pid_alive, request_check  # noqa: E402
from tests.support.swebench_bridge import (  # noqa: E402
    ALLOWED_PATHS,
    CALIBRATION_TIMEOUT_S,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_PARQUET_LFS_SHA256,
    DATASET_REVISION,
    DATASET_SPLIT,
    DATASET_VERSION,
    GRADER_TIMEOUT_S,
    IMAGE_PLATFORM,
    IMAGE_TAG,
    INSTANCE_ID,
    INSTANCE_REPO,
    UPSTREAM_REPO,
    UPSTREAM_REVISION,
    UPSTREAM_TREE,
    UPSTREAM_SHA256SUMS_PATH,
    VERIFIER_FILES,
    BridgeError,
    ContainerEvaluator,
    ContainerSupervisor,
    DockerCLI,
    GitCLI,
    PatchScopeError,
    export_patch,
    parse_required_tests,
    porcelain_paths,
    workspace_manifest,
)

DESIGN_ID = "live-coding-under-5-rmb-20261014-r2"
DOWNLOADS_DIRNAME = "downloads"
PUBLIC_DIRNAME = "public"
PRIVATE_DIRNAME = "private"
WORKSPACE_DIRNAME = "workspace"
EVIDENCE_DIRNAME = "evidence"
TEMP_DIRNAME = "temp"
LOCK_FILENAME = "lock.json"
RECEIPT_FILENAME = "prepare-receipt.json"
SUPERVISOR_WINDOW = "window.json"


class PrepareFailed(RuntimeError):
    """准备失败: 保留具体不足即停止, 不换题不削弱基准 (spec D1)。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_root_dirs(run_root: Path) -> dict[str, Path]:
    return {
        "public": run_root / PUBLIC_DIRNAME,
        "private": run_root / PRIVATE_DIRNAME,
        "workspace": run_root / WORKSPACE_DIRNAME,
        "evidence": run_root / EVIDENCE_DIRNAME,
        "temp": run_root / TEMP_DIRNAME,
    }


def downloads_dir() -> Path:
    return PROJECT / ".scratch" / "live-coding-under-5-rmb" / "runs" / DOWNLOADS_DIRNAME


def ensure_materials() -> None:
    """仅下载固定 revision；下载后统一校验，绝不跟随 main。"""
    from urllib.request import urlopen

    base = downloads_dir()
    raw = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_REVISION}"
    sources = {
        "upstream-v2-SHA256SUMS": f"{raw}/v2/SHA256SUMS",
        "upstream-commit.json": f"https://api.github.com/repos/{UPSTREAM_REPO}/commits/{UPSTREAM_REVISION}",
        "swebench-pro-default-test.parquet":
            f"https://hf-mirror.com/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/data/default/test-00000-of-00001.parquet",
    }
    names = ["instruction.md", "task.toml", "solution/gold_patch.diff", "solution/solve.sh",
             "environment/Dockerfile", *(f"tests/{name}" for name in VERIFIER_FILES)]
    sources.update({f"upstream/{name}": f"{raw}/v2/tasks/{INSTANCE_ID}/{name}" for name in names})
    for name, url in sources.items():
        path = base / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(url, timeout=60) as response:
                data = response.read()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(data)
            os.replace(temporary, path)


# --- 监督者窗口 (契约 C4: 独立 check 须有健康 owner) ---


def write_supervisor_window(evidence_dir: Path, purpose: str) -> dict:
    window = {"pid": os.getpid(), "purpose": purpose, "started": time.time()}
    write_json(evidence_dir / "supervisor" / SUPERVISOR_WINDOW, window)
    return window


def owner_window_healthy(attempt_dir: Path, *, require_external: bool) -> tuple[bool, str]:
    """容器创建要求健康 owner (契约 C4); check 额外要求外部存活 owner。"""
    window_path = attempt_dir / EVIDENCE_DIRNAME / "supervisor" / SUPERVISOR_WINDOW
    if not window_path.exists():
        return False, "no supervisor window record"
    try:
        window = load_json(window_path)
    except ValueError:
        return False, "supervisor window unreadable"
    if window.get("purpose") not in ("run", "dry-run", "grade", "prepare"):
        return False, f"window purpose {window.get('purpose')!r} may not own check containers"
    if require_external:
        if window.get("pid") == os.getpid():
            return False, "check cannot own its own containers"
        if not pid_alive(int(window["pid"])):
            return False, f"owner pid {window['pid']} is gone"
    return True, "healthy"


# --- 冻结材料定位与核验 ---


def verify_downloaded_materials() -> dict:
    """核对下载材料 hash: parquet 对 HF LFS OID, 上游文件对 v2/SHA256SUMS。"""
    base = downloads_dir()
    parquet = base / "swebench-pro-default-test.parquet"
    if not parquet.exists():
        raise PrepareFailed(f"dataset parquet missing: {parquet}")
    actual = sha256_file(parquet)
    if actual != DATASET_PARQUET_LFS_SHA256:
        raise PrepareFailed(f"dataset parquet sha256 mismatch: {actual}")
    sums_path = base / "upstream-v2-SHA256SUMS"
    if not sums_path.exists():
        raise PrepareFailed("upstream v2/SHA256SUMS missing")
    sums_bytes = sums_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(sums_bytes)).encode() + b"\0" + sums_bytes).hexdigest()
    if blob != "3015526e9ef6422e9c7712744fc5941387ecf513":
        raise PrepareFailed("SHA256SUMS differs from pinned upstream Git blob")
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        expected[name.strip().removeprefix("*")] = digest.strip()
    materials: dict[str, dict] = {
        "dataset_parquet": {
            "sha256": actual,
            "source": f"https://hf-mirror.com/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/",
            "lfs_oid": DATASET_PARQUET_LFS_SHA256,
        }
    }
    upstream_files = {
        "instruction.md": f"v2/tasks/{INSTANCE_ID}/instruction.md",
        "task.toml": f"v2/tasks/{INSTANCE_ID}/task.toml",
        "tests/test.sh": f"v2/tasks/{INSTANCE_ID}/tests/test.sh",
        "tests/run_script.sh": f"v2/tasks/{INSTANCE_ID}/tests/run_script.sh",
        "tests/parser.py": f"v2/tasks/{INSTANCE_ID}/tests/parser.py",
        "tests/config.json": f"v2/tasks/{INSTANCE_ID}/tests/config.json",
        "tests/test_patch.patch": f"v2/tasks/{INSTANCE_ID}/tests/test_patch.patch",
        "solution/gold_patch.diff": f"v2/tasks/{INSTANCE_ID}/solution/gold_patch.diff",
        "solution/solve.sh": f"v2/tasks/{INSTANCE_ID}/solution/solve.sh",
        "environment/Dockerfile": f"v2/tasks/{INSTANCE_ID}/environment/Dockerfile",
    }
    for local, upstream in upstream_files.items():
        path = base / "upstream" / local
        if not path.exists():
            raise PrepareFailed(f"upstream material missing: {local}")
        digest = sha256_file(path)
        recorded = expected.get(upstream.removeprefix("v2/"))
        if recorded is None:
            raise PrepareFailed(f"{upstream} not listed in upstream SHA256SUMS")
        if digest != recorded:
            raise PrepareFailed(f"{local} sha256 mismatch against upstream SHA256SUMS")
        materials[local] = {"sha256": digest, "upstream_path": upstream}
    sums_digest = sha256_file(sums_path)
    materials["v2-SHA256SUMS"] = {"sha256": sums_digest, "upstream_path": UPSTREAM_SHA256SUMS_PATH}
    commit_path = base / "upstream-commit.json"
    if not commit_path.exists():
        raise PrepareFailed("upstream commit metadata missing")
    commit = load_json(commit_path)
    if commit.get("sha") != UPSTREAM_REVISION or commit.get("commit", {}).get("tree", {}).get("sha") != UPSTREAM_TREE:
        raise PrepareFailed("upstream commit/tree identity mismatch")
    materials["upstream-commit.json"] = {
        "sha256": sha256_file(commit_path), "revision": UPSTREAM_REVISION, "tree": UPSTREAM_TREE}
    return materials


def dataset_row() -> dict:
    """从固定 parquet 读取白名单元数据; base_commit 只来自数据集行 (spec D1)。"""
    import pyarrow.parquet as pq

    table = pq.read_table(downloads_dir() / "swebench-pro-default-test.parquet")
    data = table.to_pydict()
    if INSTANCE_ID not in data["instance_id"]:
        raise PrepareFailed("instance not present in pinned dataset revision")
    index = data["instance_id"].index(INSTANCE_ID)
    row = {key: data[key][index] for key in data}
    if row["repo"] != INSTANCE_REPO or row["version"] != DATASET_VERSION:
        raise PrepareFailed("dataset row repo/version mismatch")
    if row["docker_image"] != IMAGE_TAG:
        raise PrepareFailed("dataset row docker_image mismatch")
    f2p = json.loads(row["fail_to_pass"]) if isinstance(row["fail_to_pass"], str) else row["fail_to_pass"]
    p2p = json.loads(row["pass_to_pass"]) if isinstance(row["pass_to_pass"], str) else row["pass_to_pass"]
    if not f2p:
        raise PrepareFailed("fail_to_pass empty; admission refuses")
    return {
        "base_commit": row["base_commit"],
        "fail_to_pass": list(f2p),
        "pass_to_pass": list(p2p),
        "selected_test_files": json.loads(row["selected_test_files_to_run"]),
        "before_repo_set_cmd": row["before_repo_set_cmd"],
    }


def verifier_identity() -> dict:
    """官方 verifier 文件 hash 与 F2P/P2P (config.json 解析); 与数据集行一致性由调用方核对。"""
    tests_dir = downloads_dir() / "upstream" / "tests"
    hashes = {name: sha256_file(tests_dir / name) for name in VERIFIER_FILES}
    f2p, p2p = parse_required_tests((tests_dir / "config.json").read_bytes())
    return {"sha256": hashes, "fail_to_pass": f2p, "pass_to_pass": p2p,
            "config_bytes_sha256": sha256_file(tests_dir / "config.json")}


def verifier_dir_for(private_dir: Path) -> Path:
    """冻结 verifier 的挂载目录: 只含五个官方文件, hash 来自 lock。"""
    target = private_dir / "verifier"
    target.mkdir(parents=True, exist_ok=True)
    source = downloads_dir() / "upstream" / "tests"
    for name in VERIFIER_FILES:
        data = (source / name).read_bytes()
        out = target / name
        if out.exists() and out.read_bytes() != data:
            raise PrepareFailed(f"private verifier changed: {name}")
        if not out.exists():
            out.write_bytes(data)
    return target


# --- prepare ---


def image_identity(docker: DockerCLI) -> dict:
    record = docker.run("image", "inspect", IMAGE_TAG, check=True)
    payload = json.loads(record.stdout)[0]
    digests = [d for d in payload.get("RepoDigests", []) if d.startswith("ghcr.io/scaleapi/")]
    if not digests:
        raise PrepareFailed("image has no ghcr.io repo digest (not pulled by digest)")
    return {
        "tag": IMAGE_TAG,
        "digest": digests[0],
        "image_id": payload.get("Id"),
        "architecture": payload.get("Architecture"),
        "os": payload.get("Os"),
    }


def extract_base_git(docker: DockerCLI, supervisor: ContainerSupervisor,
                     temp_dir: Path, image: str = IMAGE_TAG) -> Path:
    """从镜像 created 容器复制 /app/.git (清洁基线, sanitized history)。"""
    label = supervisor.owner_label("extract")
    supervisor.register_intent(label, "extract-base-git", image, os.getpid())
    name = label
    create = docker.run("create", "--name", name, "--platform", IMAGE_PLATFORM,
                        "--label", f"{bridge.DOCKER_LABEL_OWNER}={label}", image, "true")
    if create.exit_code != 0:
        supervisor.stop_and_remove(label, None)
        raise PrepareFailed("extract container create failed")
    cid = create.stdout.strip()
    target = temp_dir / "base.git"
    try:
        supervisor.register_container(label, cid)
        if target.exists():
            raise PrepareFailed("temp base.git already exists")
        copy = docker.run("cp", f"{cid}:/app/.git", str(target), timeout_s=600)
        if copy.exit_code != 0:
            raise PrepareFailed(f"docker cp /app/.git failed: {copy.stderr[:200]}")
    finally:
        supervisor.stop_and_remove(label, cid)
    return target


def build_workspace(base_git: Path, base_commit: str, workspace_dir: Path,
                    command_log: list) -> None:
    if workspace_dir.exists():
        raise PrepareFailed("workspace directory already exists")
    clone = subprocess.run(
        ["git", "clone", "--no-checkout", "--quiet",
         base_git.as_uri(), str(workspace_dir)],
        capture_output=True, timeout=1800)
    command_log.append(bridge.CommandRecord(
        command=["git", "clone", "--no-checkout", base_git.as_uri(), str(workspace_dir)],
        exit_code=clone.returncode,
        stdout=clone.stdout.decode("utf-8", errors="replace")[-2000:],
        stderr=clone.stderr.decode("utf-8", errors="replace")[-2000:],
        seconds=0.0))
    if clone.returncode != 0:
        raise PrepareFailed("workspace clone failed")
    git = GitCLI(workspace_dir, command_log)
    for key, value in bridge.GIT_FREEZE_CONFIG:
        git.run("config", key, value, check=True)
    git.run("remote", "remove", "origin", check=False)
    git.run("checkout", "--quiet", base_commit, check=True)
    head = git.run("rev-parse", "HEAD", check=True).stdout.strip()
    if head != base_commit:
        raise PrepareFailed(f"workspace HEAD {head} != base {base_commit}")
    status = git.run("status", "--porcelain", "--untracked-files=all").stdout
    if status.strip():
        raise PrepareFailed("workspace not clean after checkout")


def eligibility_from_production(workspace_dir: Path, temp_dir: Path) -> dict:
    """现行生产 inventory/snapshot 限额资格 (spec: 不放宽限额)。

    Workspace 输出目录指向 run_root temp, 不向工作区写入任何文件。
    """
    from cicada.core.cancel import CancelToken
    from cicada.plugins.coding.inventory import GitInventory, InventoryError, MAX_PATHS
    from cicada.plugins.coding.snapshot import MAX_FILE_BYTES, MAX_TOTAL_BYTES, Snapshotter
    from cicada.plugins.coding.workspace import Workspace

    manifest = workspace_manifest(workspace_dir, GitCLI(workspace_dir))
    production_paths = None
    inventory_error = None
    snapshot = None
    output_dir = temp_dir / "inventory-outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        async def capture():
            ws = Workspace(root=workspace_dir, output_dir=output_dir)
            inv = GitInventory(ws)
            return await inv.list_files(CancelToken()), await Snapshotter(ws, inv).capture(CancelToken())
        inventory, snapshot = asyncio.run(capture())
        production_paths = len(inventory.paths)
    except (InventoryError, OSError, RuntimeError, ValueError) as exc:
        inventory_error = f"{type(exc).__name__}: {exc}"
    eligibility = {
        "git_path_count": manifest.path_count,
        "max_paths_limit": MAX_PATHS,
        "paths_within_limit": manifest.path_count <= MAX_PATHS,
        "production_inventory_paths": production_paths,
        "production_inventory_error": inventory_error,
        "max_file_bytes": manifest.max_file_bytes,
        "max_file_limit": MAX_FILE_BYTES,
        "total_bytes": manifest.total_bytes,
        "total_limit": MAX_TOTAL_BYTES,
        "snapshot_within_limits": (
            manifest.max_file_bytes <= MAX_FILE_BYTES and manifest.total_bytes <= MAX_TOTAL_BYTES),
        "snapshot_available": snapshot.available if snapshot is not None else False,
        "snapshot_failure_kind": snapshot.failure_kind if snapshot is not None else None,
        "snapshot_error": snapshot.error if snapshot is not None else None,
        "snapshot_ref": snapshot.snapshot.snapshot_ref if snapshot is not None and snapshot.available else None,
        "git_symlink_paths": [entry.relative_path for entry in inventory.files if "120000" in entry.git_modes] if production_paths is not None else [],
        "dirty_paths": manifest.dirty_paths,
        "entries_sha256": manifest.entries_sha256,
    }
    eligibility["qualified"] = (eligibility["paths_within_limit"] and eligibility["snapshot_within_limits"]
                                 and inventory_error is None and eligibility["snapshot_available"])
    return eligibility


def container_git_identity(docker: DockerCLI, supervisor: ContainerSupervisor,
                           evidence_dir: Path, image: str = IMAGE_TAG) -> str:
    """镜像容器内 git ls-files -s 输出 (模式/内容权威), 与 Windows checkout 核对。"""
    _, stdout = bridge.capture_container_manifest(
        image, supervisor, docker, evidence_dir)
    return stdout


def verify_workspace_identity(workspace_dir: Path, container_ls_files: str,
                              command_log: list) -> dict:
    local = GitCLI(workspace_dir, command_log).run("ls-files", "-s").stdout
    local_normalized = local.replace("\r\n", "\n")
    match = local_normalized == container_ls_files.replace("\r\n", "\n")
    return {
        "ls_files_match": match,
        "local_entries_sha256": sha256_bytes(local_normalized.encode("utf-8")),
        "container_entries_sha256": sha256_bytes(
            container_ls_files.replace("\r\n", "\n").encode("utf-8")),
    }


def evaluate_calibration(spec, patch_bytes: bytes | None, evidence_dir: Path,
                         f2p: list[str], p2p: list[str], timeout_s: float):
    def verify():
        verify_downloaded_materials()
        for name, digest in spec["verifier_sha256"].items():
            if sha256_file(spec["verifier_dir"] / name) != digest:
                raise PrepareFailed("calibration verifier identity changed")
    verify()
    docker = DockerCLI()
    supervisor = ContainerSupervisor(spec["evidence_dir"], docker)
    result = ContainerEvaluator(docker, supervisor).evaluate(
        bridge.ContainerSpec(
            image=IMAGE_TAG, image_digest=spec["image"]["digest"],
            verifier_dir=spec["verifier_dir"], owner_label=supervisor.owner_label(
                "calibration-empty" if patch_bytes is None else "calibration-gold")),
        patch_bytes, evidence_dir, f2p, p2p, timeout_s=timeout_s)
    verify()
    return result


def cmd_prepare(args) -> int:
    run_root = Path(args.run_root).resolve()
    dirs = run_root_dirs(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise PrepareFailed(f"run root not empty: {run_root}")
    for name, path in dirs.items():
        if name == "workspace":
            continue
        path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command_log: list[bridge.CommandRecord] = []
    write_json(dirs["evidence"] / "supervisor" / SUPERVISOR_WINDOW,
               {"pid": os.getpid(), "purpose": "prepare", "started": started})
    receipt: dict = {
        "design": DESIGN_ID, "instance_id": INSTANCE_ID, "started": started,
        "commands": [], "warnings": [], "failures": [],
    }

    def dump() -> None:
        receipt["commands"] = [record.to_json() for record in command_log]
        receipt["elapsed_s"] = round(time.time() - started, 3)
        write_json(run_root / RECEIPT_FILENAME, receipt)

    try:
        docker = DockerCLI(command_log)
        available, detail = bridge.docker_available()
        if not available:
            raise PrepareFailed(f"docker engine unavailable: {detail}")
        receipt["environment"] = {"docker": detail}
        ensure_materials()
        materials = verify_downloaded_materials()
        receipt["materials"] = materials
        row = dataset_row()
        verifier = verifier_identity()
        if (verifier["fail_to_pass"] != row["fail_to_pass"]
                or verifier["pass_to_pass"] != row["pass_to_pass"]):
            raise PrepareFailed("verifier config F2P/P2P differ from dataset row")
        receipt["dataset_row"] = {
            "base_commit": row["base_commit"],
            "f2p_count": len(row["fail_to_pass"]),
            "p2p_count": len(row["pass_to_pass"]),
            "selected_test_files": row["selected_test_files"],
        }
        receipt["verifier"] = {"sha256": verifier["sha256"],
                               "config_bytes_sha256": verifier["config_bytes_sha256"]}
        image = image_identity(docker)
        receipt["image"] = image
        if image["architecture"] != "amd64" or image["os"] != "linux":
            raise PrepareFailed("image platform is not linux/amd64")

        # Windows workspace: 镜像清洁基线 → 本地 clone (换行/模式冻结)。
        base_git = extract_base_git(docker, ContainerSupervisor(dirs["evidence"], docker),
                                    dirs["temp"], image["digest"])
        build_workspace(base_git, row["base_commit"], dirs["workspace"], command_log)
        git = GitCLI(dirs["workspace"], command_log)
        receipt["workspace"] = {
            "path": str(dirs["workspace"]),
            "base_commit": row["base_commit"],
            "git_config": {key: git.run("config", key).stdout.strip()
                           for key, _ in bridge.GIT_FREEZE_CONFIG},
        }
        eligibility = eligibility_from_production(dirs["workspace"], dirs["temp"])
        receipt["eligibility"] = eligibility
        if not eligibility["qualified"]:
            raise PrepareFailed(f"production workspace admission refused: {eligibility['snapshot_failure_kind']} / {eligibility['snapshot_error'] or eligibility['production_inventory_error']}")
        container_ls = container_git_identity(
            docker, ContainerSupervisor(dirs["evidence"], docker), dirs["evidence"], image["digest"])
        identity = verify_workspace_identity(dirs["workspace"], container_ls, command_log)
        if not identity["ls_files_match"]:
            raise PrepareFailed("workspace ls-files -s differs from image container")
        receipt["workspace_identity"] = identity

        # 公开 instruction → public/; 参考补丁与 verifier 留 private/。
        (dirs["public"] / "instruction.md").write_bytes(
            (downloads_dir() / "upstream" / "instruction.md").read_bytes())
        receipt["instruction"] = {
            "sha256": sha256_file(downloads_dir() / "upstream" / "instruction.md")}
        verifier_mount = verifier_dir_for(dirs["private"])
        gold_patch = (downloads_dir() / "upstream" / "solution" / "gold_patch.diff").read_bytes()
        (dirs["private"] / "gold_patch.diff").write_bytes(gold_patch)

        # 补丁导出能力代表 (C8): 新增/删除/修改/模式且真实 index 不变, 在临时 repo 验证。
        representative = patch_capability_representative(dirs["temp"], command_log)
        receipt["patch_capability_representative"] = representative
        rt = representative["round_trip"]
        mode = representative["mode"]
        if (not rt["content_round_trip"] or rt["apply_exit"] != 0
                or not rt["index_unchanged"] or rt["empty"]
                or mode["apply_exit"] != 0 or not mode["clone_mode_100755"]
                or not mode["patch_has_mode_line"] or not mode["source_index_unchanged"]):
            raise PrepareFailed("patch capability representative failed")
        empty_export = export_patch(
            dirs["workspace"], row["base_commit"], ALLOWED_PATHS, dirs["temp"], git)
        if not empty_export.empty or not empty_export.index_unchanged:
            raise PrepareFailed("clean workspace must export an empty patch with index unchanged")
        receipt["empty_export"] = empty_export.to_json()

        # 校准反证: 空补丁失败、参考补丁成功; 各一次, 各 <=120s (spec D3)。
        spec = {"evidence_dir": dirs["evidence"], "image": image, "verifier_dir": verifier_mount,
                "verifier_sha256": verifier["sha256"]}
        empty_eval = evaluate_calibration(spec, None, dirs["evidence"] / "calibration-empty",
                                          row["fail_to_pass"], row["pass_to_pass"],
                                          CALIBRATION_TIMEOUT_S)
        receipt["calibration_empty"] = empty_eval.to_json()
        if empty_eval.timed_out or empty_eval.apply_failed or empty_eval.cleanup != "confirmed":
            raise PrepareFailed("empty-patch calibration did not complete cleanly")
        empty_grade = empty_eval.output_json or {}
        if (empty_eval.verifier_exit_code in (None, 0) or empty_eval.reward != 0
                or not empty_grade.get("artifacts_complete") or empty_grade.get("parse_failed")
                or empty_grade.get("required_observed") != empty_eval.required_total
                or not empty_grade.get("f2p_failed") or not empty_grade.get("p2p_passed")):
            raise PrepareFailed("empty patch did not prove the required F2P failure with intact P2P")
        gold_eval = evaluate_calibration(spec, gold_patch, dirs["evidence"] / "calibration-gold",
                                         row["fail_to_pass"], row["pass_to_pass"],
                                         CALIBRATION_TIMEOUT_S)
        receipt["calibration_gold"] = gold_eval.to_json()
        if gold_eval.timed_out or gold_eval.apply_failed or gold_eval.cleanup != "confirmed":
            raise PrepareFailed("reference-patch calibration did not complete cleanly")
        if not gold_eval.output_json or not gold_eval.output_json.get("grader_pass"):
            raise PrepareFailed("reference patch failed official verifier")
        if empty_eval.wall_time_s > CALIBRATION_TIMEOUT_S or gold_eval.wall_time_s > CALIBRATION_TIMEOUT_S:
            raise PrepareFailed("calibration exceeded 120s")

        write_json(run_root / LOCK_FILENAME, build_lock(run_root, row, verifier, image, receipt))
        receipt["checker_cancellation"] = verify_checker_cancellation(run_root)
        receipt["outcome"] = "prepared"
        dump()
        print(json.dumps({"outcome": "prepared", "run_root": str(run_root),
                          "elapsed_s": receipt["elapsed_s"]}, ensure_ascii=False))
        return 0
    except (PrepareFailed, BridgeError, PatchScopeError, OSError, ValueError, RuntimeError) as exc:
        receipt["outcome"] = "prepare-failed"
        receipt["failures"].append({"type": type(exc).__name__, "message": str(exc)[:500]})
        dump()
        print(json.dumps({"outcome": "prepare-failed",
                          "reason": str(exc)[:500]}, ensure_ascii=False), file=sys.stderr)
        return 2


def build_lock(run_root: Path, row: dict, verifier: dict, image: dict, receipt: dict) -> dict:
    import platform

    from cicada.plugins.coding.process import resolve_pwsh

    git = GitCLI(run_root / WORKSPACE_DIRNAME)
    manifest = workspace_manifest(run_root / WORKSPACE_DIRNAME, git)
    uv_record = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=60)
    pwsh = subprocess.run(
        [resolve_pwsh(), "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
        capture_output=True, text=True, timeout=60)
    return {
        "design": DESIGN_ID,
        "dataset": {
            "id": DATASET_ID, "config": DATASET_CONFIG, "split": DATASET_SPLIT,
            "revision": DATASET_REVISION, "version": DATASET_VERSION,
            "parquet_sha256": DATASET_PARQUET_LFS_SHA256,
        },
        "upstream": {
            "repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "tree": UPSTREAM_TREE,
            "sha256sums_sha256": receipt["materials"]["v2-SHA256SUMS"]["sha256"],
        },
        "instance": {
            "instance_id": INSTANCE_ID, "repo": INSTANCE_REPO,
            "base_commit": row["base_commit"],
            "fail_to_pass": row["fail_to_pass"], "pass_to_pass": row["pass_to_pass"],
            "fail_to_pass_nonempty": bool(row["fail_to_pass"]),
            "selected_test_files": row["selected_test_files"],
        },
        "image": image,
        "instruction_sha256": receipt["instruction"]["sha256"],
        "verifier": verifier["sha256"],
        "verifier_config_sha256": verifier["config_bytes_sha256"],
        "workspace": {
            "path_name": WORKSPACE_DIRNAME,
            "head": row["base_commit"],
            "git_entries_sha256": manifest.entries_sha256,
            "git_path_count": manifest.path_count,
            "manifest": manifest.to_json(),
            "git_config": {key: git.run("config", key).stdout.strip()
                           for key, _ in bridge.GIT_FREEZE_CONFIG},
            "ls_files_match_container": receipt.get("workspace_identity", {}).get("ls_files_match"),
        },
        "cicada": {
            "source_sha256": {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256_file(p)
                               for p in sorted((PROJECT / "src").rglob("*.py"))},
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                text=True).stdout.strip(),
            "git_tree": subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=PROJECT, capture_output=True,
                text=True).stdout.strip(),
        },
        "environment": {
            "python": platform.python_version(),
            "uv": uv_record.stdout.strip() if uv_record.returncode == 0 else None,
            "pwsh": pwsh.stdout.strip(),
            "docker_server": receipt.get("environment", {}).get("docker"),
            "platform": platform.platform(),
        },
        "created": time.time(),
        "profile": {"model": "deepseek-flash", "num_ctx": 262144,
                    "max_tokens": 8192, "thinking": "enabled", "reasoning_effort": "low",
                    "http_limit": 12, "run_timeout": 600, "check_timeout": 300,
                    "grader_timeout": 240, "budget_cny": "4.50"},
        "pending_fields": [],
    }


def patch_capability_representative(temp_dir: Path, command_log: list) -> dict:
    """离线代表 (契约 C8): 新增/删除/修改导出往返 + 模式补丁 + 真实 index 不变。

    Windows worktree (core.fileMode=false) 不产生模式变化; 模式维度用临时
    index 显式构造 (update-index --chmod), 证明导出与应用完整表达 Git 模式。
    """
    temp_dir = temp_dir.resolve()
    repo = temp_dir / "capability-repo"
    if repo.exists():
        raise PrepareFailed("capability repo temp path already exists")
    repo.mkdir(parents=True)
    base_git = GitCLI(repo, command_log)
    base_git.run("init", "--quiet", check=True)
    for key, value in bridge.GIT_FREEZE_CONFIG:
        base_git.run("config", key, value, check=True)
    base_git.run("config", "user.email", "harness@example.invalid", check=True)
    base_git.run("config", "user.name", "harness", check=True)
    (repo / "keep.txt").write_bytes(b"keep\n")
    (repo / "drop.txt").write_bytes(b"drop\n")
    (repo / "mod.txt").write_bytes(b"one\n")
    base_git.run("add", "--", ".", check=True)
    base_git.run("commit", "--quiet", "-m", "base", check=True)
    base_commit = base_git.run("rev-parse", "HEAD", check=True).stdout.strip()
    # 工作区改动: 修改/删除/新增 (Windows 可表达的全部维度)。
    (repo / "mod.txt").write_bytes(b"one\ntwo\n")
    (repo / "drop.txt").unlink()
    (repo / "new.txt").write_bytes(b"new\n")
    index_before = base_git.run("ls-files", "-s").stdout
    export = export_patch(repo, base_commit, ("keep.txt", "mod.txt", "drop.txt", "new.txt"),
                          temp_dir, base_git)
    index_after = base_git.run("ls-files", "-s").stdout

    clone_dir = temp_dir / "capability-clone"
    subprocess.run(["git", "clone", "--quiet", repo.as_uri(), str(clone_dir)],
                   capture_output=True, timeout=300, check=True)
    clone_git = GitCLI(clone_dir, command_log)
    clone_git.run("checkout", "--quiet", base_commit, check=True)
    patch_file = temp_dir / "capability.patch"
    patch_file.write_bytes(export.patch_bytes)
    applied = subprocess.run(
        ["git", "-C", str(clone_dir), "apply", str(patch_file)],
        capture_output=True, text=True, timeout=120)
    round_trip = {
        "empty": export.empty,
        "changed_paths": export.changed_paths,
        "index_unchanged": export.index_unchanged and index_before == index_after,
        "apply_exit": applied.returncode,
        "apply_stderr_tail": applied.stderr[-500:],
        "bytes": len(export.patch_bytes),
        "content_round_trip": (
            (clone_dir / "mod.txt").read_bytes() == b"one\ntwo\n"
            and (clone_dir / "new.txt").read_bytes() == b"new\n"
            and not (clone_dir / "drop.txt").exists()
            and (clone_dir / "keep.txt").read_bytes() == b"keep\n"),
    }

    # 模式维度: 临时 index 显式 +x, 导出含 "new file mode"/"old mode" 行。
    mode_repo = temp_dir / "capability-mode"
    mode_repo.mkdir()
    mode_git = GitCLI(mode_repo, command_log)
    mode_git.run("init", "--quiet", check=True)
    for key, value in bridge.GIT_FREEZE_CONFIG:
        mode_git.run("config", key, value, check=True)
    mode_git.run("config", "user.email", "harness@example.invalid", check=True)
    mode_git.run("config", "user.name", "harness", check=True)
    (mode_repo / "script.sh").write_bytes(b"#!/bin/sh\necho hi\n")
    mode_git.run("add", "--", ".", check=True)
    mode_git.run("commit", "--quiet", "-m", "base", check=True)
    mode_base = mode_git.run("rev-parse", "HEAD", check=True).stdout.strip()
    mode_index = temp_dir / f"index-mode-{uuid.uuid4().hex}"
    env = {"GIT_INDEX_FILE": str(mode_index)}
    mode_git.run("read-tree", mode_base, check=True, env_extra=env)
    mode_git.run("update-index", "--chmod=+x", "--", "script.sh", check=True, env_extra=env)
    mode_diff = mode_git.run(
        "diff", "--cached", "--binary", "--full-index", mode_base,
        check=True, env_extra=env)
    mode_index.unlink(missing_ok=True)
    mode_real_before = mode_git.run("ls-files", "-s").stdout
    mode_patch = temp_dir / "capability-mode.patch"
    mode_patch.write_bytes(mode_diff.stdout.encode("utf-8"))
    mode_clone = temp_dir / "capability-mode-clone"
    subprocess.run(["git", "clone", "--quiet", mode_repo.as_uri(), str(mode_clone)],
                   capture_output=True, timeout=300, check=True)
    mode_applied = subprocess.run(
        ["git", "-C", str(mode_clone), "apply", "--index", str(mode_patch)],
        capture_output=True, text=True, timeout=120)
    mode_entry = GitCLI(mode_clone, command_log).run("ls-files", "-s", "script.sh").stdout.strip()
    return {
        "round_trip": round_trip,
        "mode": {
            "patch_has_mode_line": "mode 100755" in mode_diff.stdout,
            "apply_exit": mode_applied.returncode,
            "clone_entry": mode_entry,
            "clone_mode_100755": mode_entry.startswith("100755"),
            "source_index_unchanged": mode_real_before == mode_git.run("ls-files", "-s").stdout,
        },
    }


# --- check (生产 check-1 调用; 契约 C4) ---


def load_lock(run_root: Path) -> dict:
    lock_path = run_root / LOCK_FILENAME
    if not lock_path.exists():
        raise BridgeError(f"lock.json missing under {run_root}")
    return load_json(lock_path)


def evaluation_owner(run_root: Path):
    lock = load_lock(run_root)
    evidence = run_root / EVIDENCE_DIRNAME
    supervisor = ContainerSupervisor(evidence, DockerCLI())
    return EvaluationOwner(
        run_root,
        lambda cancel: evaluate_workspace_patch(run_root, run_root / WORKSPACE_DIRNAME, lock,
                                         evidence, require_external_owner=False, cancel=cancel),
        supervisor.cleanup_registered,
    )


def verify_checker_cancellation(run_root: Path) -> dict:
    """真实 checker 被强制结束后，主 owner 仍回收其已登记容器。"""
    intent_file = run_root / "evidence/supervisor/intents.jsonl"
    initial = intent_file.read_text(encoding="utf-8") if intent_file.exists() else ""
    with evaluation_owner(run_root) as owner:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "check",
                                  "--run-root", str(run_root), "--workspace", str(run_root / WORKSPACE_DIRNAME)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 90
        created = None
        try:
            while time.monotonic() < deadline and child.poll() is None:
                current = intent_file.read_text(encoding="utf-8") if intent_file.exists() else ""
                new = [json.loads(line) for line in current[len(initial):].splitlines()]
                created = next((r for r in new if r["kind"] == "created"), None)
                if created:
                    child.kill()
                    child.wait(timeout=10)
                    break
                time.sleep(0.05)
            if not created:
                raise PrepareFailed("checker cancellation probe never registered a container")
            deadline = time.monotonic() + 30
            while owner.requester is not None and time.monotonic() < deadline:
                time.sleep(0.1)
            if owner.requester is not None:
                raise PrepareFailed("owner did not finish orphan cleanup")
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=15)
    label = created["owner_label"]
    if ContainerSupervisor(run_root / EVIDENCE_DIRNAME, DockerCLI()).query_by_label(label) is not None:
        raise PrepareFailed("checker-owned container remains")
    return {"checker_killed": True, "container_id": created["container_id"],
            "owner_label": label, "cleanup": "confirmed"}


def verify_frozen_identity(lock: dict, evidence_dir: Path) -> list[str]:
    """每次评测前重验外部冻结输入 (spec D5); 返回异常列表。"""
    problems: list[str] = []
    if lock.get("design") != DESIGN_ID:
        problems.append("design differs from frozen contract")
    for relative, digest in lock.get("cicada", {}).get("source_sha256", {}).items():
        path = PROJECT / relative
        if not path.is_file() or sha256_file(path) != digest:
            problems.append("production source differs from lock")
            break
    try:
        materials = verify_downloaded_materials()
    except PrepareFailed as exc:
        return [f"materials: {exc}"]
    if materials["tests/test.sh"]["sha256"] != lock["verifier"]["test.sh"] \
            or materials["tests/config.json"]["sha256"] != lock["verifier"]["config.json"] \
            or materials["tests/parser.py"]["sha256"] != lock["verifier"]["parser.py"] \
            or materials["tests/run_script.sh"]["sha256"] != lock["verifier"]["run_script.sh"] \
            or materials["tests/test_patch.patch"]["sha256"] != lock["verifier"]["test_patch.patch"]:
        problems.append("verifier file hashes differ from lock")
    for name, digest in lock["verifier"].items():
        mounted = evidence_dir.parent / PRIVATE_DIRNAME / "verifier" / name
        if not mounted.is_file() or sha256_file(mounted) != digest:
            problems.append("mounted verifier differs from lock")
            break
    docker = DockerCLI()
    inspect = docker.run("image", "inspect", lock["image"]["tag"])
    if inspect.exit_code != 0:
        problems.append("pinned image not present")
    else:
        payload = json.loads(inspect.stdout)[0]
        digests = [d for d in payload.get("RepoDigests", []) if d.startswith("ghcr.io/scaleapi/")]
        if not digests or digests[0] != lock["image"]["digest"]:
            problems.append("image digest differs from lock")
    return problems


def evaluate_workspace_patch(run_root: Path, workspace: Path, lock: dict,
                             evidence_dir: Path, *, require_external_owner: bool, cancel=None) -> tuple[int, dict]:
    """导出当前完整补丁并评测; 返回 (check 退出码, 摘要)。"""
    docker = DockerCLI()
    docker.cancel = cancel
    supervisor = ContainerSupervisor(evidence_dir, DockerCLI())
    healthy, detail = owner_window_healthy(run_root, require_external=require_external_owner)
    if not healthy:
        summary = {"result": "anomaly", "reason": f"supervisor owner: {detail}"}
        print(json.dumps(summary, ensure_ascii=False))
        return 2, summary
    problems = verify_frozen_identity(lock, evidence_dir)
    git = GitCLI(workspace)
    manifest = workspace_manifest(workspace, git)
    if manifest.head != lock["instance"]["base_commit"]:
        problems.append(f"workspace HEAD {manifest.head} != base")
    if manifest.entries_sha256 != lock["workspace"]["git_entries_sha256"]:
        problems.append("git index fingerprint differs from lock; export refuses")
    for key, value in lock["workspace"]["git_config"].items():
        if git.run("config", key).stdout.strip() != value:
            problems.append("workspace Git config differs from lock")
    if problems:
        summary = {"result": "anomaly", "reasons": problems}
        print(json.dumps(summary, ensure_ascii=False))
        return 2, summary
    temp_dir = run_root / TEMP_DIRNAME / f"check-{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        exported = export_patch(workspace, lock["instance"]["base_commit"],
                                ALLOWED_PATHS, temp_dir, git)
    except PatchScopeError as exc:
        (evidence_dir / "scope-violation.json").write_text(
            json.dumps({"offending": exc.offending}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        summary = {"result": "anomaly", "reason": "changes outside allowlist",
                   "offending_count": len(exc.offending)}
        print(json.dumps(summary, ensure_ascii=False))
        return 2, summary
    if not exported.index_unchanged:
        summary = {"result": "anomaly", "reason": "git index changed during export"}
        print(json.dumps(summary, ensure_ascii=False))
        return 2, summary
    check_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    check_dir = evidence_dir / "checks" / check_id
    verifier_mount = verifier_dir_for(run_root / PRIVATE_DIRNAME)
    evaluator = ContainerEvaluator(docker, supervisor)
    result = evaluator.evaluate(
        bridge.ContainerSpec(
            image=lock["image"]["tag"], image_digest=lock["image"]["digest"],
            verifier_dir=verifier_mount, owner_label=supervisor.owner_label("check")),
        None if exported.empty else exported.patch_bytes,
        check_dir,
        lock["instance"]["fail_to_pass"], lock["instance"]["pass_to_pass"],
        timeout_s=GRADER_TIMEOUT_S)
    grade = result.output_json or {}
    identity_after = verify_frozen_identity(lock, evidence_dir)
    if workspace_manifest(workspace, git).to_json() != exported.source_manifest:
        identity_after.append("workspace changed during evaluation")
    grade["identity_after_problems"] = identity_after
    passed = bool(grade.get("grader_pass")) and result.cleanup == "confirmed" and not identity_after
    unresolved_code = (
        not result.timed_out and not result.apply_failed
        and result.cleanup == "confirmed"
        and result.verifier_exit_code is not None
        and grade.get("reward") == 0
        and grade.get("artifacts_complete") and not identity_after
        and grade.get("required_observed") == result.required_total
        and not grade.get("parse_failed") and not grade.get("required_empty"))
    (check_dir / "patch.json").write_text(
        json.dumps(exported.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    (check_dir / "evaluation.json").write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    if passed:
        outcome = "passed"
        code = 0
    elif unresolved_code:
        outcome = "failed"
        code = 1
    else:
        outcome = "anomaly"
        code = 2
    summary = {
        "result": outcome,
        "required_passed": grade.get("required_passed"),
        "required_total": grade.get("required_total"),
        "patch_sha256": exported.sha256,
        "patch_empty": exported.empty,
        "grade_id": check_id,
        "verifier_exit_code": result.verifier_exit_code,
        "cleanup": result.cleanup,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code, summary


def cmd_check(args) -> int:
    run_root = Path(args.run_root).resolve()
    workspace = Path(args.workspace).resolve()
    try:
        lock = load_lock(run_root)
    except (BridgeError, ValueError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2
    try:
        if workspace != run_root / WORKSPACE_DIRNAME:
            raise BridgeError("checker workspace differs from owned workspace")
        code, summary = request_check(run_root)
        print(json.dumps(summary, ensure_ascii=False))
        return code
    except (BridgeError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2


# --- grade (最终复评; 重新导出当前最终补丁) ---


def cmd_grade(args) -> int:
    run_root = Path(args.run_root).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else run_root / WORKSPACE_DIRNAME
    try:
        lock = load_lock(run_root)
    except (BridgeError, ValueError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2
    evidence_dir = run_root / EVIDENCE_DIRNAME
    write_supervisor_window(evidence_dir, "grade")
    try:
        code, summary = evaluate_workspace_patch(
            run_root, workspace, lock, evidence_dir, require_external_owner=False)
        write_json(evidence_dir / "grade-last.json", summary)
        return code
    except (BridgeError, OSError, ValueError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="swebench_runner")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="固定题目并建立可验证准备回执")
    prepare.add_argument("--run-root", required=True)
    check = sub.add_parser("check", help="生产 check-1 调用: 导出补丁并在新容器官方评测")
    check.add_argument("--run-root", required=True)
    check.add_argument("--workspace", required=True)
    grade = sub.add_parser("grade", help="重新导出最终补丁并在全新容器复评")
    grade.add_argument("--run-root", required=True)
    grade.add_argument("--workspace", default=None)
    sub.add_parser("dry-run", help="票 03: 全路径离线演练")
    sub.add_parser("run", help="票 02: 生产 CLI 单题运行")
    sub.add_parser("report", help="票 03: 结果汇编")
    args = parser.parse_args()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "grade":
        return cmd_grade(args)
    print("this subcommand is delivered by a later ticket of the plan", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
