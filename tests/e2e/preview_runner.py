"""SWE-bench Pro V2 两题公开实测 harness (PV01 接入)。

统一入口:
  uv run --frozen python tests/e2e/preview_runner.py prepare   --manifest <path> --run-root <dir>
  uv run --frozen python tests/e2e/preview_runner.py dry-run   --manifest <path> --run-root <dir>
  uv run --frozen python tests/e2e/preview_runner.py check     --manifest <path> --run-root <attempt> --workspace <ws>
  uv run --frozen python tests/e2e/preview_runner.py grade     --manifest <path> --run-root <attempt> [--workspace <ws>]
  uv run --frozen python tests/e2e/preview_runner.py report    --manifest <path> --run-root <attempt>
  uv run --frozen python tests/e2e/preview_runner.py run       --manifest <path> --run-root <attempt> \
                                                               --prepared-root <dir> --live

边界 (PV01):
- 只接入两题身份与参数接缝, 不改旧单题 tests/e2e/swebench_runner.py 的命令/协议/工件含义,
  旧固定题检查按原命令保持可用。题目身份只经不可变 TaskSpec 传入, 不 monkeypatch 模块常量。
- run 需要 --live 与另获的明确授权; 本票不实现 live launcher (属 PV02), 因此 run 只拒绝,
  不发真实请求。dry-run 为静态准入核对, 不启动容器、不发请求。
- check 退出码沿用: 0=完整通过, 1=任务未解决, 2=身份/环境/证据异常。
- stdout 只输出有界公共摘要; 不打印用例名、官方日志正文或私有材料内容。
"""

from __future__ import annotations

import argparse
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
from tests.support import swebench_tasks as tasks  # noqa: E402
from tests.support.swebench_tasks import (  # noqa: E402
    ManifestError,
    TaskSpec,
    dataset_row,
    ensure_materials,
    image_identity,
    load_json,
    sha256_file,
    spec_from_manifest,
    verifier_dir_for,
    verifier_identity,
    verify_materials,
    write_json,
)

PUBLIC_DIRNAME = "public"
PRIVATE_DIRNAME = "private"
WORKSPACE_DIRNAME = "workspace"
EVIDENCE_DIRNAME = "evidence"
TEMP_DIRNAME = "temp"
LOCK_FILENAME = "lock.json"
RECEIPT_FILENAME = "prepare-receipt.json"
SUPERVISOR_WINDOW = "window.json"


class PrepareFailed(RuntimeError):
    """准备失败: 保留具体不足即停止, 不换题不削弱基准。"""


def run_root_dirs(run_root: Path) -> dict[str, Path]:
    return {
        "public": run_root / PUBLIC_DIRNAME,
        "private": run_root / PRIVATE_DIRNAME,
        "workspace": run_root / WORKSPACE_DIRNAME,
        "evidence": run_root / EVIDENCE_DIRNAME,
        "temp": run_root / TEMP_DIRNAME,
    }


def write_supervisor_window(evidence_dir: Path, purpose: str) -> dict:
    window = {"pid": os.getpid(), "purpose": purpose, "started": time.time()}
    write_json(evidence_dir / "supervisor" / SUPERVISOR_WINDOW, window)
    return window


# --- 工作区: 镜像清洁基线 → 本地 clone (通用化自旧单题流程) ---


def extract_base_git(docker: bridge.DockerCLI, supervisor: bridge.ContainerSupervisor,
                     temp_dir: Path, image: str, label_prefix: str) -> Path:
    """从镜像 created 容器复制 /app/.git (清洁基线, sanitized history)。"""
    label = supervisor.owner_label(label_prefix)
    supervisor.register_intent(label, "extract-base-git", image, os.getpid())
    create = docker.run("create", "--name", label, "--platform", bridge.IMAGE_PLATFORM,
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
        copy = docker.run("cp", f"{cid}:/app/.git", str(target), timeout_s=900)
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
        ["git", "clone", "--no-checkout", "--quiet", base_git.as_uri(), str(workspace_dir)],
        capture_output=True, timeout=3600)
    command_log.append(bridge.CommandRecord(
        command=["git", "clone", "--no-checkout", base_git.as_uri(), str(workspace_dir)],
        exit_code=clone.returncode,
        stdout=clone.stdout.decode("utf-8", errors="replace")[-2000:],
        stderr=clone.stderr.decode("utf-8", errors="replace")[-2000:],
        seconds=0.0))
    if clone.returncode != 0:
        raise PrepareFailed("workspace clone failed")
    git = bridge.GitCLI(workspace_dir, command_log)
    for key, value in bridge.GIT_FREEZE_CONFIG:
        git.run("config", key, value, check=True)
    git.run("remote", "remove", "origin", check=False)
    git.run("checkout", "--quiet", base_commit, check=True)
    head = git.run("rev-parse", "HEAD", check=True).stdout.strip()
    if head != base_commit:
        raise PrepareFailed(f"workspace HEAD {head} != base {base_commit}")
    if git.run("status", "--porcelain", "--untracked-files=all").stdout.strip():
        raise PrepareFailed("workspace not clean after checkout")


def verify_workspace_identity(workspace_dir: Path, container_ls_files: str,
                              command_log: list) -> dict:
    local = bridge.GitCLI(workspace_dir, command_log).run("ls-files", "-s").stdout
    local_normalized = local.replace("\r\n", "\n")
    match = local_normalized == container_ls_files.replace("\r\n", "\n")
    return {
        "ls_files_match": match,
        "local_entries_sha256": tasks.sha256_bytes(local_normalized.encode("utf-8")),
        "container_entries_sha256": tasks.sha256_bytes(
            container_ls_files.replace("\r\n", "\n").encode("utf-8")),
    }


# --- 校准反证 ---


def evaluate_calibration(spec: TaskSpec, image: dict, verifier_dir: Path,
                         verifier_sha256: dict, patch_bytes: bytes | None,
                         evidence_dir: Path, f2p: list[str], p2p: list[str],
                         timeout_s: float) -> bridge.EvaluationResult:
    def verify() -> None:
        current = verifier_identity(spec)
        if current["sha256"] != verifier_sha256:
            raise PrepareFailed("calibration verifier identity changed")
        for name, digest in verifier_sha256.items():
            if sha256_file(verifier_dir / name) != digest:
                raise PrepareFailed("mounted verifier changed during calibration")

    verify()
    docker = bridge.DockerCLI()
    supervisor = bridge.ContainerSupervisor(evidence_dir, docker)
    result = bridge.ContainerEvaluator(docker, supervisor).evaluate(
        bridge.ContainerSpec(
            image=spec.image_tag, image_digest=image["digest"],
            verifier_dir=verifier_dir,
            owner_label=supervisor.owner_label(
                "calibration-empty" if patch_bytes is None else "calibration-gold")),
        patch_bytes, evidence_dir, f2p, p2p, timeout_s=timeout_s)
    verify()
    return result


def calibration_problems(empty: bridge.EvaluationResult, gold: bridge.EvaluationResult,
                         timeout_s: float, f2p: list[str]) -> list[str]:
    """空补丁必失败、参考补丁必通过的联合判定; 返回全部不足, 不早退掩盖。"""
    problems: list[str] = []
    for name, result in (("empty", empty), ("gold", gold)):
        if result.timed_out:
            problems.append(f"{name}: calibration timed out")
        if result.apply_failed:
            problems.append(f"{name}: patch failed to apply")
        if result.cleanup != "confirmed":
            problems.append(f"{name}: container cleanup not confirmed")
        if result.wall_time_s > timeout_s:
            problems.append(f"{name}: exceeded {timeout_s:.0f}s")
        if not (result.output_json or {}).get("artifacts_complete"):
            problems.append(f"{name}: official verifier artifacts incomplete")

    empty_grade = empty.output_json or {}
    if empty.verifier_exit_code in (None, 0):
        problems.append("empty: verifier exit must be non-zero")
    if empty.reward != 0:
        problems.append("empty: reward must be 0")
    if empty_grade.get("grader_pass"):
        problems.append("empty: empty patch must not be reported as passing")
    if empty_grade.get("parse_failed"):
        problems.append("empty: official output.json unparsable")
    # 空补丁门槛 (主 Agent 2026-10-15 裁决的细化标准):
    #   (a) 官方 exit!=0 且 reward==0;
    #   (b) 基线已存在的 P2P 必须被观测且通过 —— 证明"失败来自缺少实现",
    #       而不是环境/收集被搞坏;
    #   (c) 被观测到的 F2P 必须失败 (部分观测即可, 见下)。
    # 不要求 required_observed == required_total: 当题目 test_patch 新增整个测试文件时
    # (new-multifile 的 test_locale.py 是 new file), 空补丁下该文件导入失败会让整轮
    # pytest 会话中止, 那部分 F2P 根本无法被收集。逐题强度记入 empty_reversal_strength。
    if not empty_grade.get("p2p_passed"):
        problems.append("empty: pre-existing P2P tests must be observed and pass")
    f2p_observed_failed, f2p_total = empty_f2p_failures(empty, f2p)
    if f2p_observed_failed == 0:
        problems.append("empty: no F2P test was observed failing")

    gold_grade = gold.output_json or {}
    if not gold_grade.get("grader_pass"):
        problems.append("gold: reference patch failed official verifier")
    # 参考补丁必须覆盖完整: F2P 与 P2P 全部被观测且通过 (授权包 "覆盖完整")。
    if gold_grade.get("required_observed") != gold.required_total:
        problems.append("gold: not all required tests were observed")
    if not gold_grade.get("f2p_passed"):
        problems.append("gold: F2P tests must all pass after the reference patch")
    if not gold_grade.get("p2p_passed"):
        problems.append("gold: P2P tests must pass after the reference patch")
    return problems


def test_statuses(result: bridge.EvaluationResult) -> dict[str, str]:
    """从官方 output.json 工件解析逐用例状态; 解析失败返回空表 (判定仍以 exit/reward 为准)。"""
    raw = (result.artifacts or {}).get("output.json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    statuses: dict[str, str] = {}
    for entry in payload.get("tests", []):
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            statuses[entry["name"]] = str(entry.get("status", ""))
    return statuses


def empty_f2p_failures(empty: bridge.EvaluationResult,
                       f2p: list[str]) -> tuple[int, int]:
    """空补丁下被观测且失败的 F2P 数量, 与 F2P 总数。"""
    statuses = test_statuses(empty)
    failed = [name for name in f2p if statuses.get(name) == "FAILED"]
    return len(failed), len(f2p)


def empty_reversal_strength(empty: bridge.EvaluationResult,
                            f2p: list[str]) -> dict:
    """记录空补丁反证的实际强度, 避免把弱反证当强结论。"""
    grade = empty.output_json or {}
    observed = grade.get("required_observed")
    total = empty.required_total
    failed_count, f2p_total = empty_f2p_failures(empty, f2p)
    p2p_passed = bool(grade.get("p2p_passed"))
    all_f2p_failed = f2p_total > 0 and failed_count == f2p_total
    if p2p_passed and all_f2p_failed and observed == total:
        strength = "strong"
        detail = "every required test observed and all F2P individually failed"
    elif p2p_passed and failed_count:
        strength = "partial-f2p-observed"
        detail = (f"{failed_count}/{f2p_total} F2P tests observed failing while "
                  f"pre-existing P2P stayed green; the unobserved rest could not be "
                  f"collected under the empty patch ({observed}/{total} observed)")
    elif p2p_passed:
        strength = "p2p-intact-only"
        detail = ("no F2P test was observed failing; reversal rests on the official "
                  "exit code and reward=0 with pre-existing P2P intact")
    else:
        strength = "exit-and-reward-only"
        detail = ("required tests were not all collected under the empty patch "
                  f"({observed}/{total} observed); reversal rests on the official "
                  "exit code and reward=0 only")
    return {"strength": strength, "required_observed": observed,
            "required_total": total, "f2p_observed_failed": failed_count,
            "f2p_total": f2p_total, "f2p_failed": bool(grade.get("f2p_failed")),
            "p2p_passed": p2p_passed, "detail": detail}


# --- prepare ---


def cmd_prepare(args) -> int:
    run_root = Path(args.run_root).resolve()
    started = time.time()
    receipt: dict = {"started": started, "commands": [], "warnings": [], "failures": []}
    command_log: list[bridge.CommandRecord] = []
    dirs: dict[str, Path] = {}
    # 只有确认 run root 为空并由本次 prepare 建立后, 才允许写回执; 否则会覆盖
    # 既有 attempt 的成功回执 (spec: prepare 不覆盖已存在的 prepare/attempt)。
    owns_run_root = False

    def dump() -> None:
        if not owns_run_root:
            return
        receipt["commands"] = [record.to_json() for record in command_log]
        receipt["elapsed_s"] = round(time.time() - started, 3)
        write_json(run_root / RECEIPT_FILENAME, receipt)

    try:
        spec = spec_from_manifest(Path(args.manifest))
        receipt["task"] = spec.to_json()
        dirs = run_root_dirs(run_root)
        if run_root.exists() and any(run_root.iterdir()):
            raise PrepareFailed(f"run root not empty: {run_root}")
        owns_run_root = True
        for name, path in dirs.items():
            if name != "workspace":
                path.mkdir(parents=True, exist_ok=True)
        write_json(dirs["evidence"] / "supervisor" / SUPERVISOR_WINDOW,
                   {"pid": os.getpid(), "purpose": "prepare", "started": started})

        docker = bridge.DockerCLI(command_log)
        available, detail = bridge.docker_available()
        if not available:
            raise PrepareFailed(f"docker engine unavailable: {detail}")
        receipt["environment"] = {"docker": detail}

        ensure_materials(spec)
        materials = verify_materials(spec)
        receipt["materials"] = materials

        row = dataset_row(spec)
        verifier = verifier_identity(spec)
        if (verifier["fail_to_pass"] != row["fail_to_pass"]
                or verifier["pass_to_pass"] != row["pass_to_pass"]):
            raise PrepareFailed("verifier config F2P/P2P differ from dataset row")
        receipt["dataset_row"] = {
            "base_commit": row["base_commit"], "f2p_count": len(row["fail_to_pass"]),
            "p2p_count": len(row["pass_to_pass"]),
            "selected_test_files": row["selected_test_files"]}
        receipt["verifier"] = {"sha256": verifier["sha256"],
                               "config_bytes_sha256": verifier["config_bytes_sha256"]}

        image = image_identity(docker, spec)
        receipt["image"] = image
        if image["architecture"] != "amd64" or image["os"] != "linux":
            raise PrepareFailed("image platform is not linux/amd64")

        # 冻结源码基线核对: 镜像内 HEAD 必须等于 manifest/dataset 声明的 base_commit。
        supervisor = bridge.ContainerSupervisor(dirs["evidence"], docker)
        container_head = tasks.container_git_head(
            docker, supervisor, image["digest"], dirs["evidence"])
        receipt["container_head"] = container_head
        if container_head != row["base_commit"]:
            raise PrepareFailed(
                f"image source HEAD {container_head} != declared base {row['base_commit']}")

        base_git = extract_base_git(docker, supervisor, dirs["temp"], image["digest"], "extract")
        build_workspace(base_git, row["base_commit"], dirs["workspace"], command_log)
        git = bridge.GitCLI(dirs["workspace"], command_log)
        receipt["workspace"] = {
            "path": str(dirs["workspace"]), "base_commit": row["base_commit"],
            "git_config": {key: git.run("config", key).stdout.strip()
                           for key, _ in bridge.GIT_FREEZE_CONFIG}}

        _, container_ls = bridge.capture_container_manifest(
            image["digest"], supervisor, docker, dirs["evidence"])
        identity = verify_workspace_identity(dirs["workspace"], container_ls, command_log)
        if not identity["ls_files_match"]:
            raise PrepareFailed("workspace ls-files -s differs from image container")
        receipt["workspace_identity"] = identity

        # 公开材料 → public/; 参考补丁与 verifier → private/ (不进入模型可见路径)。
        for name in tasks.PUBLIC_MATERIALS:
            source = tasks.downloads_dir(spec) / "upstream" / name
            (dirs["public"] / name).write_bytes(source.read_bytes())
        verifier_mount = verifier_dir_for(spec, dirs["private"])
        gold_patch = tasks.gold_patch_bytes(spec)
        (dirs["private"] / "gold_patch.diff").write_bytes(gold_patch)

        # 干净工作区必须导出空补丁且真实 index 不变。
        empty_export = bridge.export_patch(
            dirs["workspace"], row["base_commit"], spec.allowed_paths, dirs["temp"], git)
        if not empty_export.empty or not empty_export.index_unchanged:
            raise PrepareFailed("clean workspace must export an empty patch with index unchanged")
        receipt["empty_export"] = empty_export.to_json()

        # 校准反证: 空补丁必失败、参考补丁必通过。
        timeout_s = float(json.loads((Path(args.manifest)).read_text(encoding="utf-8"))
                          .get("limits", {}).get("calibration_timeout_s", 120))
        empty_eval = evaluate_calibration(
            spec, image, verifier_mount, verifier["sha256"], None,
            dirs["evidence"] / "calibration-empty", row["fail_to_pass"],
            row["pass_to_pass"], timeout_s)
        receipt["calibration_empty"] = empty_eval.to_json()
        gold_eval = evaluate_calibration(
            spec, image, verifier_mount, verifier["sha256"], gold_patch,
            dirs["evidence"] / "calibration-gold", row["fail_to_pass"],
            row["pass_to_pass"], timeout_s)
        receipt["calibration_gold"] = gold_eval.to_json()

        problems = calibration_problems(empty_eval, gold_eval, timeout_s,
                                         row["fail_to_pass"])
        receipt["calibration_problems"] = problems
        receipt["empty_reversal_strength"] = empty_reversal_strength(
            empty_eval, row["fail_to_pass"])
        if problems:
            raise PrepareFailed("calibration failed: " + "; ".join(problems[:6]))

        write_json(run_root / LOCK_FILENAME,
                   build_lock(spec, row, verifier, image, receipt))
        receipt["outcome"] = "prepared"
        dump()
        print(json.dumps({"outcome": "prepared", "task_key": spec.task_key,
                          "run_root": str(run_root), "elapsed_s": receipt["elapsed_s"],
                          "f2p_count": len(row["fail_to_pass"]),
                          "p2p_count": len(row["pass_to_pass"])}, ensure_ascii=False))
        return 0
    except (ManifestError, PrepareFailed, bridge.BridgeError, bridge.PatchScopeError,
            OSError, ValueError, RuntimeError) as exc:
        receipt["outcome"] = "prepare-failed"
        receipt["failures"].append({"type": type(exc).__name__, "message": str(exc)[:500]})
        if dirs:
            dump()
        print(json.dumps({"outcome": "prepare-failed", "reason": str(exc)[:500]},
                         ensure_ascii=False), file=sys.stderr)
        return 2


def build_lock(spec: TaskSpec, row: dict, verifier: dict, image: dict,
               receipt: dict) -> dict:
    import platform

    from cicada.plugins.coding.process import resolve_pwsh

    git = bridge.GitCLI(receipt["workspace"]["path"])
    manifest = bridge.workspace_manifest(Path(receipt["workspace"]["path"]), git)
    uv_record = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=60)
    pwsh = subprocess.run(
        [resolve_pwsh(), "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
        capture_output=True, text=True, timeout=60)
    manifest_doc = load_json(Path(spec_manifest_path(spec)))
    return {
        "design": manifest_doc["design_id"],
        "schema_version": tasks.SCHEMA_VERSION,
        "task": spec.to_json(),
        "manifest_sha256": sha256_file(Path(spec_manifest_path(spec))),
        "dataset": {
            "id": bridge.DATASET_ID, "config": bridge.DATASET_CONFIG,
            "split": bridge.DATASET_SPLIT, "revision": bridge.DATASET_REVISION,
            "version": bridge.DATASET_VERSION, "parquet_sha256": bridge.DATASET_PARQUET_LFS_SHA256},
        "upstream": {
            "repo": bridge.UPSTREAM_REPO, "revision": spec.upstream_revision,
            "tree": bridge.UPSTREAM_TREE,
            "sha256sums_sha256": receipt["materials"]["v2-SHA256SUMS"]["sha256"]},
        "instance": {
            "instance_id": spec.instance_id, "repo": spec.repo,
            "base_commit": row["base_commit"],
            "fail_to_pass": row["fail_to_pass"], "pass_to_pass": row["pass_to_pass"],
            "fail_to_pass_nonempty": bool(row["fail_to_pass"]),
            "selected_test_files": row["selected_test_files"],
            "allowed_paths": list(spec.allowed_paths)},
        "image": image,
        "instruction_sha256": spec.instruction_sha256,
        "task_toml_sha256": spec.task_toml_sha256,
        "verifier": verifier["sha256"],
        "verifier_config_sha256": verifier["config_bytes_sha256"],
        "workspace": {
            "path_name": WORKSPACE_DIRNAME, "head": row["base_commit"],
            "git_entries_sha256": manifest.entries_sha256,
            "git_path_count": manifest.path_count,
            "manifest": manifest.to_json(),
            "ls_files_match_container": receipt.get("workspace_identity", {}).get("ls_files_match")},
        "cicada": {
            "source_sha256": {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256_file(p)
                              for p in sorted((PROJECT / "src").rglob("*.py"))},
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                       capture_output=True, text=True).stdout.strip(),
            "git_tree": subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=PROJECT,
                                       capture_output=True, text=True).stdout.strip()},
        "environment": {
            "python": platform.python_version(),
            "uv": uv_record.stdout.strip() if uv_record.returncode == 0 else None,
            "pwsh": pwsh.stdout.strip(),
            "docker_server": receipt.get("environment", {}).get("docker"),
            "platform": platform.platform()},
        "model": manifest_doc["model"],
        "limits": manifest_doc["limits"],
        "cost_profile_path": manifest_doc.get("cost_profile_path"),
        "created": time.time(),
        "live_authorized": False,
        "pending_fields": ["cost_profile (live admission; PV02)"],
    }


def spec_manifest_path(spec: TaskSpec) -> str:
    return str(tasks.manifest_path_for(spec.task_key))


def load_lock(run_root: Path) -> dict:
    path = run_root / LOCK_FILENAME
    if not path.is_file():
        raise bridge.BridgeError(f"lock.json missing under {run_root}")
    return load_json(path)


def verify_frozen_identity(spec: TaskSpec, lock: dict, run_root: Path) -> list[str]:
    """每次评测前重验冻结输入; 返回异常列表 (空即健康)。"""
    problems: list[str] = []
    if lock.get("task") != spec.to_json():
        problems.append("lock task identity differs from manifest")
    if lock.get("manifest_sha256") != sha256_file(Path(spec_manifest_path(spec))):
        problems.append("manifest changed since prepare")
    if lock.get("dataset") != {
            "id": bridge.DATASET_ID, "config": bridge.DATASET_CONFIG,
            "split": bridge.DATASET_SPLIT, "revision": bridge.DATASET_REVISION,
            "version": bridge.DATASET_VERSION,
            "parquet_sha256": bridge.DATASET_PARQUET_LFS_SHA256}:
        problems.append("dataset identity differs from frozen selection")
    if lock.get("upstream", {}).get("revision") != spec.upstream_revision:
        problems.append("upstream revision differs from frozen selection")
    if (lock.get("instance", {}).get("base_commit") != spec.base_commit
            or lock.get("image", {}).get("tag") != spec.image_tag):
        problems.append("task/image identity differs from frozen selection")
    try:
        materials = verify_materials(spec)
        row = dataset_row(spec)
    except (ManifestError, OSError, ValueError) as exc:
        return problems + [f"materials: {exc}"]
    for key in ("base_commit", "fail_to_pass", "pass_to_pass", "selected_test_files"):
        if lock["instance"].get(key) != row[key]:
            problems.append("required task identity differs from pinned dataset row")
            break
    for name in bridge.VERIFIER_FILES:
        if materials[f"tests/{name}"]["sha256"] != lock["verifier"][name]:
            problems.append("verifier file hashes differ from lock")
            break
    verifier_dir = run_root / PRIVATE_DIRNAME / "verifier"
    for name, digest in lock["verifier"].items():
        mounted = verifier_dir / name
        if not mounted.is_file() or sha256_file(mounted) != digest:
            problems.append("mounted verifier differs from lock")
            break
    expected_sources = {p.relative_to(PROJECT).as_posix(): sha256_file(p)
                        for p in sorted((PROJECT / "src").rglob("*.py"))}
    if lock.get("cicada", {}).get("source_sha256") != expected_sources:
        problems.append("production source manifest differs from lock")
    docker = bridge.DockerCLI()
    inspect = docker.run("image", "inspect", spec.image_tag)
    if inspect.exit_code != 0:
        problems.append("pinned image not present")
    else:
        payload = json.loads(inspect.stdout)[0]
        digests = [d for d in payload.get("RepoDigests", []) if d.startswith("ghcr.io/scaleapi/")]
        if not digests or digests[0] != lock["image"]["digest"]:
            problems.append("image digest differs from lock")
    return problems


def evaluate_workspace_patch(spec: TaskSpec, run_root: Path, workspace: Path, lock: dict,
                             evidence_dir: Path) -> tuple[int, dict]:
    """导出当前完整补丁并在全新容器评测; 返回 (退出码, 摘要)。"""
    problems = verify_frozen_identity(spec, lock, run_root)
    git = bridge.GitCLI(workspace)
    manifest = bridge.workspace_manifest(workspace, git)
    if manifest.head != spec.base_commit:
        problems.append(f"workspace HEAD {manifest.head} != base")
    if manifest.entries_sha256 != lock["workspace"]["git_entries_sha256"]:
        problems.append("git index fingerprint differs from lock; export refuses")
    if problems:
        summary = {"result": "anomaly", "reasons": problems[:8]}
        print(json.dumps(summary, ensure_ascii=False))
        return 2, summary

    temp_dir = run_root / TEMP_DIRNAME / f"check-{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        exported = bridge.export_patch(
            workspace, spec.base_commit, spec.allowed_paths, temp_dir, git)
    except bridge.PatchScopeError as exc:
        write_json(evidence_dir / "scope-violation.json",
                   {"offending": exc.offending, "exempted": exc.exempted})
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
    verifier_mount = verifier_dir_for(spec, run_root / PRIVATE_DIRNAME)
    docker = bridge.DockerCLI()
    supervisor = bridge.ContainerSupervisor(evidence_dir, docker)
    result = bridge.ContainerEvaluator(docker, supervisor).evaluate(
        bridge.ContainerSpec(
            image=spec.image_tag, image_digest=lock["image"]["digest"],
            verifier_dir=verifier_mount, owner_label=supervisor.owner_label("check")),
        None if exported.empty else exported.patch_bytes, check_dir,
        lock["instance"]["fail_to_pass"], lock["instance"]["pass_to_pass"],
        timeout_s=float(lock["limits"].get("grader_timeout_s", bridge.GRADER_TIMEOUT_S)))

    grade = result.output_json or {}
    identity_after = verify_frozen_identity(spec, lock, run_root)
    if bridge.workspace_manifest(workspace, git).to_json() != exported.source_manifest:
        identity_after.append("workspace changed during evaluation")
    grade["identity_after_problems"] = identity_after
    passed = bool(grade.get("grader_pass")) and result.cleanup == "confirmed" and not identity_after
    unresolved_code = (
        not result.timed_out and not result.apply_failed and result.cleanup == "confirmed"
        and result.verifier_exit_code is not None and grade.get("reward") == 0
        and grade.get("artifacts_complete") and not identity_after
        and grade.get("required_observed") == result.required_total
        and not grade.get("parse_failed") and not grade.get("required_empty"))
    write_json(check_dir / "patch.json", exported.to_json())
    write_json(check_dir / "evaluation.json", result.to_json())
    if passed:
        outcome, code = "passed", 0
    elif unresolved_code:
        outcome, code = "failed", 1
    else:
        outcome, code = "anomaly", 2
    summary = {
        "result": outcome, "task_key": spec.task_key,
        "required_passed": grade.get("required_passed"),
        "required_total": grade.get("required_total"),
        "patch_sha256": exported.sha256, "patch_empty": exported.empty,
        "grade_id": check_id, "verifier_exit_code": result.verifier_exit_code,
        "cleanup": result.cleanup,
        # 记录被评分的正是这一份工作区内容: report 用它判断"工作区是否仍等于被评分者"。
        # 不能用 prepare 时的原始 manifest (解答题工作区必然与之不同)。
        "source_manifest": exported.source_manifest}
    print(json.dumps(summary, ensure_ascii=False))
    return code, summary


# --- check / grade ---


def cmd_check(args) -> int:
    run_root = Path(args.run_root).resolve()
    workspace = Path(args.workspace).resolve()
    try:
        spec = spec_from_manifest(Path(args.manifest))
        lock = load_lock(run_root)
        if workspace != run_root / WORKSPACE_DIRNAME:
            raise bridge.BridgeError("checker workspace differs from owned workspace")
        write_supervisor_window(run_root / EVIDENCE_DIRNAME, "check")
        code, _ = evaluate_workspace_patch(
            spec, run_root, workspace, lock, run_root / EVIDENCE_DIRNAME)
        return code
    except (ManifestError, bridge.BridgeError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2


def cmd_grade(args) -> int:
    run_root = Path(args.run_root).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else run_root / WORKSPACE_DIRNAME
    try:
        spec = spec_from_manifest(Path(args.manifest))
        lock = load_lock(run_root)
        # 与 check 同一所有权约束: 只评自己拥有的工作区 (spec 仅对 check 要求 --workspace)。
        if workspace != run_root / WORKSPACE_DIRNAME:
            raise bridge.BridgeError("grader workspace differs from owned workspace")
        evidence_dir = run_root / EVIDENCE_DIRNAME
        write_supervisor_window(evidence_dir, "grade")
        code, summary = evaluate_workspace_patch(spec, run_root, workspace, lock, evidence_dir)
        # 版本化留档: 失败复评不得覆盖既有通过记录; 每个记录独占创建。
        record_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        write_json(evidence_dir / "grades" / f"{record_id}.json", summary)
        write_json(evidence_dir / "grade-last.json", {**summary, "record_id": record_id})
        return code
    except (ManifestError, bridge.BridgeError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"result": "anomaly", "reason": str(exc)[:300]}, ensure_ascii=False))
        return 2


# --- dry-run / run / report ---


def cmd_dry_run(args) -> int:
    """静态离线准入核对: 校验身份、材料、镜像与锁一致性; 不启动容器、不发请求。"""
    run_root = Path(args.run_root).resolve()
    try:
        spec = spec_from_manifest(Path(args.manifest))
        available, detail = bridge.docker_available()
        admission: dict = {
            "task_key": spec.task_key, "manifest": spec.to_json(),
            "docker": detail, "scope": "static-identity-and-materials-only",
            "containers_started": False, "model_requests": 0,
            "note": "full-path fake/MockTransport rehearsal belongs to PV02"}
        if not available:
            admission["outcome"] = "offline-static-failed"
            admission["reason"] = "docker engine unavailable"
            print(json.dumps(admission, ensure_ascii=False))
            return 2
        ensure_materials(spec)
        admission["materials_verified"] = bool(verify_materials(spec))
        row = dataset_row(spec)
        verifier = verifier_identity(spec)
        admission["dataset_row"] = {
            "base_commit": row["base_commit"], "f2p_count": len(row["fail_to_pass"]),
            "p2p_count": len(row["pass_to_pass"])}
        if (verifier["fail_to_pass"] != row["fail_to_pass"]
                or verifier["pass_to_pass"] != row["pass_to_pass"]):
            raise PrepareFailed("verifier config F2P/P2P differ from dataset row")
        admission["image"] = image_identity(bridge.DockerCLI(), spec)
        # 既有 run root 必须成对具备回执与 lock; 只有其一的残留目录不得报 passed。
        has_receipt = (run_root / RECEIPT_FILENAME).is_file()
        has_lock = (run_root / LOCK_FILENAME).is_file()
        if run_root.exists() and any(run_root.iterdir()):
            if has_receipt != has_lock:
                admission["outcome"] = "offline-static-failed"
                admission["reason"] = (
                    "run root has prepare-receipt.json without lock.json" if has_receipt
                    else "run root has lock.json without prepare-receipt.json")
                print(json.dumps(admission, ensure_ascii=False))
                return 1
            if has_lock:
                lock = load_lock(run_root)
                problems = verify_frozen_identity(spec, lock, run_root)
                admission["lock_problems"] = problems
                admission["prepare_receipt_present"] = has_receipt
                if problems:
                    admission["outcome"] = "offline-static-failed"
                    print(json.dumps(admission, ensure_ascii=False))
                    return 1
        admission["outcome"] = "offline-static-passed"
        print(json.dumps(admission, ensure_ascii=False))
        return 0
    except (ManifestError, PrepareFailed, bridge.BridgeError, OSError, ValueError,
            RuntimeError, KeyError) as exc:
        print(json.dumps({"outcome": "offline-static-failed", "reason": str(exc)[:400]},
                         ensure_ascii=False))
        return 2


def cmd_run(args) -> int:
    """live 入口占位: 不实现 launcher, 不发送真实请求 (PV01 边界)。"""
    if not args.live:
        print(json.dumps({
            "outcome": "refused", "reason": "run requires --live and separate explicit authorization",
            "model_requests": 0}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "outcome": "not-implemented", "model_requests": 0,
        "reason": "live launcher (AttemptBudget, cross-process lifecycle) belongs to PV02 and is not built",
        "prepared_root": args.prepared_root}, ensure_ascii=False), file=sys.stderr)
    return 2


def cmd_report(args) -> int:
    """只重组已有事实; 不发请求、不重新评分、不升级旧回执权威。"""
    run_root = Path(args.run_root).resolve()
    try:
        spec = spec_from_manifest(Path(args.manifest))
        lock = load_lock(run_root)
        evidence_dir = run_root / EVIDENCE_DIRNAME
        prepared = (run_root / RECEIPT_FILENAME).is_file()
        prepare_receipt = load_json(run_root / RECEIPT_FILENAME) if prepared else None
        grade = load_json(evidence_dir / "grade-last.json") if (
            evidence_dir / "grade-last.json").is_file() else None
        checks_dir = evidence_dir / "checks"
        checks = sorted(p.name for p in checks_dir.iterdir()) if checks_dir.is_dir() else []
        prepare_failures = (prepare_receipt or {}).get("failures") or []
        calibration_problems = (prepare_receipt or {}).get("calibration_problems") or []
        coverage = {
            "prepare": prepared and (prepare_receipt or {}).get("outcome") == "prepared",
            "calibration_empty": bool((prepare_receipt or {}).get("calibration_empty")),
            "calibration_gold": bool((prepare_receipt or {}).get("calibration_gold")),
            "checks": len(checks), "final_grade": grade is not None}

        # 身份与工作区漂移必须参与判定: report 是"该 attempt 通过"的权威工件, 不能在
        # 私料/verifier/源码/镜像漂移, 或工作区已不同于被评分内容时仍报 passed
        # (spec: 缺少任何必须证据时不得 overall passed)。
        identity_problems = verify_frozen_identity(spec, lock, run_root)
        workspace = run_root / WORKSPACE_DIRNAME
        # 只在确实存在最终 grade 时才谈"与所评内容是否一致": 没有 grade 时由
        # coverage.final_grade 表达缺失, 不应伪报成 grade 的溯源缺陷。
        if grade is not None:
            if not workspace.is_dir():
                identity_problems.append("graded workspace is missing")
            else:
                current = bridge.workspace_manifest(workspace, bridge.GitCLI(workspace)).to_json()
                # 与"被评分的那份内容"比较, 而非 prepare 时的原始基线: 解答题工作区本就
                # 不同于基线, 用基线比较会让任何合法解答都被误判为漂移。
                graded = grade.get("source_manifest")
                if graded is None:
                    identity_problems.append(
                        "final grade carries no graded source manifest; cannot prove "
                        "the workspace still equals what was scored")
                elif current != graded:
                    identity_problems.append("workspace differs from the graded source manifest")

        overall = bool(
            coverage["prepare"]
            and coverage["calibration_empty"]
            and coverage["calibration_gold"]
            and coverage["checks"] >= 1
            and coverage["final_grade"]
            and (grade or {}).get("result") == "passed"
            and not prepare_failures
            and not calibration_problems
            and not identity_problems)
        summary = {
            "task_key": spec.task_key, "run_root": str(run_root),
            "coverage": coverage, "prepare_failures": prepare_failures,
            "calibration_problems": calibration_problems,
            "final_grade": grade, "check_ids": checks,
            "identity_problems": identity_problems,
            "overall_passed": overall,
            "outcome": "passed" if overall else "not-passed",
            "note": "reassembles stored facts only; no requests, no re-scoring"}
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if overall else 1
    except (ManifestError, bridge.BridgeError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"outcome": "inconclusive-evidence", "reason": str(exc)[:300]},
                         ensure_ascii=False))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="preview_runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
            ("prepare", "冻结一题身份、源码与镜像并完成空/参考补丁反证"),
            ("dry-run", "静态离线准入核对; 不启动容器、不发请求"),
            ("check", "导出当前补丁并在新容器官方评测 (0/1/2)"),
            ("grade", "重新导出最终补丁并在全新容器复评"),
            ("report", "只重组已有事实, 不重新评分"),
            ("run", "live 入口; 需 --live 与另获授权 (PV02 未实现)")):
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("--manifest", required=True)
        sub_parser.add_argument("--run-root", required=True)
        if name == "check":
            sub_parser.add_argument("--workspace", required=True)
        if name == "grade":
            sub_parser.add_argument("--workspace", default=None)
        if name == "run":
            sub_parser.add_argument("--prepared-root", required=True)
            sub_parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    handlers = {"prepare": cmd_prepare, "dry-run": cmd_dry_run, "check": cmd_check,
                "grade": cmd_grade, "run": cmd_run, "report": cmd_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
