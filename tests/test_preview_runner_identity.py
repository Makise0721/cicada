"""PV01 两题身份/允许路径隔离与旧题缺省兼容的最小关键回归。

不调用 Docker、不发模型请求。身份负向用例只依赖仓库内冻结的 selection.json 与
公开输入副本; 当本地公开输入缺失时按 skip 处理 (该目录按交付规则不入库)。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import swebench_tasks as tasks

PUBLIC_INPUTS = tasks.PUBLIC_VALIDATION / "inputs"


def test_runtime_identity_survives_checkout_line_endings_but_detects_edits(monkeypatch):
    """准入身份跨 LF/CRLF 检出保持一致, 但不能漏掉实际源码改动。"""
    from tests.support import swebench_lifecycle as lifecycle

    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda path: read_bytes(path).replace(b"\r\n", b"\n"))
    expected = lifecycle.runtime_identity()
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_bytes(path)
                        .replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert lifecycle.runtime_identity() == expected
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_bytes(path) + b"# edit\n")
    assert all(value != expected[name]
               for name, value in lifecycle.runtime_identity().items())


def _manifest(task_key: str) -> dict:
    path = tasks.manifest_path_for(task_key)
    if not path.is_file():
        pytest.skip(f"manifest not present: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_inputs(task_key: str) -> None:
    """公开输入副本不在库内; 缺失时跳过而不是失败。"""
    doc = _manifest(task_key)
    for entry in doc["public_inputs"].values():
        if not (tasks.PUBLIC_VALIDATION / entry["path"]).is_file():
            pytest.skip("public input copies are not committed")


def test_both_frozen_tasks_load_with_expected_identity():
    """两题都能从 manifest 载入, 且身份与冻结 selection.json 一致。"""
    _require_inputs("repeat-lifecycle")
    _require_inputs("new-multifile")
    selection = tasks.load_selection()
    for task_key, scenario in (("repeat-lifecycle", "restore-compact-continue"),
                               ("new-multifile", "coding")):
        spec = tasks.spec_from_manifest(tasks.manifest_path_for(task_key))
        frozen = selection["tasks"][task_key]["metadata"]
        assert spec.task_key == task_key
        assert spec.scenario == scenario
        assert spec.instance_id == frozen["instance_id"]
        assert spec.repo == frozen["repo"]
        assert spec.base_commit == frozen["base_commit"]
        assert spec.image_tag.endswith(frozen["instance_id"])
        # 两题允许路径互不相同, 且与冻结授权契约一致。
        assert set(spec.allowed_paths) == set(tasks.FROZEN_ALLOWED_PATHS[task_key])


def test_allowlist_cannot_be_widened_by_manifest(tmp_path):
    """manifest 不得自行放宽允许路径 (例如加入 README 或测试文件)。"""
    _require_inputs("repeat-lifecycle")
    doc = _manifest("repeat-lifecycle")
    doc["allowed_paths"] = [*doc["allowed_paths"], "README.md"]
    path = tmp_path / "widened.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(tasks.ManifestError):
        tasks.spec_from_manifest(path)


def test_unknown_task_key_is_rejected(tmp_path):
    """只接受本包冻结的两题; 其他题号一律拒绝。"""
    _require_inputs("repeat-lifecycle")
    doc = _manifest("repeat-lifecycle")
    doc["task_key"] = "not-a-frozen-task"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(tasks.ManifestError):
        tasks.spec_from_manifest(path)


def test_scenario_binding_cannot_be_swapped(tmp_path):
    """生命周期题不能改成 coding 场景运行。"""
    _require_inputs("repeat-lifecycle")
    doc = _manifest("repeat-lifecycle")
    doc["scenario"] = "coding"
    path = tmp_path / "swapped.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(tasks.ManifestError):
        tasks.spec_from_manifest(path)


def test_base_commit_tamper_is_rejected(tmp_path):
    _require_inputs("repeat-lifecycle")
    doc = _manifest("repeat-lifecycle")
    doc["base_commit"] = "0" * 40
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(tasks.ManifestError):
        tasks.spec_from_manifest(path)


def test_legacy_runner_default_identity_is_unchanged():
    """旧单题 runner 缺省身份未被两题接缝改写。

    旧固定题就是"重复题"(repeat-lifecycle)的同一 instance, 因此它与该题 instance_id
    相同、允许路径相同; 新题 new-multifile 则必须是另一个 instance。这同时证明新接缝
    没有反向改写旧 runner 的缺省题。
    """
    from tests.e2e import swebench_runner as legacy

    assert legacy.INSTANCE_ID == tasks.bridge.INSTANCE_ID
    assert legacy.ALLOWED_PATHS == tasks.bridge.ALLOWED_PATHS
    assert legacy.DESIGN_ID == "live-coding-under-5-rmb-20261014-r2"

    repeat = _manifest("repeat-lifecycle")
    new = _manifest("new-multifile")
    assert legacy.INSTANCE_ID == repeat["instance_id"]
    assert set(legacy.ALLOWED_PATHS) == set(repeat["allowed_paths"])
    assert legacy.INSTANCE_ID != new["instance_id"]
    assert set(legacy.ALLOWED_PATHS) != set(new["allowed_paths"])


def test_run_subcommand_refuses_without_authorization(tmp_path):
    """run 的两条零发送闸: 缺价格 profile 拒绝; --live 无授权记录拒绝。

    PV02 后 run 的语义是"缺 --live 跑离线演练, --live 需另获授权"。无论哪条,
    "没有闸就不发送"与"没有授权就不发送"都必须成立且零请求。

    注意: 用**合法 profile** 才能走到 live 分支 —— 传 None 会在缺 profile 处短路,
    那样测的只是第一条闸 (审查 P2-7)。
    """
    from tests.e2e import preview_runner as runner
    from types import SimpleNamespace

    manifest = tasks.manifest_path_for("repeat-lifecycle")
    if not manifest.is_file():
        pytest.skip("manifest not present")
    base = dict(manifest=str(manifest), run_root=str(tmp_path),
                prepared_root=str(tmp_path))
    # 1) 缺 --cost-profile: 没有 attempt 总额度闸, 一律不发送。
    assert runner.cmd_run(SimpleNamespace(live=False, cost_profile=None, **base)) == 2
    # 2) 无授权记录时 --live 必须拒绝 (此例真正走到 live 分支)。
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        "checked_at": "2026-10-15",
        "models": {"deepseek-flash": {"input_cny_per_million": "2",
                                      "output_cny_per_million": "8"}}},
        ensure_ascii=False), encoding="utf-8")
    assert runner.cmd_run(
        SimpleNamespace(live=True, cost_profile=str(profile), **base)) == 2


def test_prepare_does_not_overwrite_existing_attempt_receipt(tmp_path):
    """prepare 拒绝非空 run root 时, 不得覆盖既有 attempt 的回执 (spec: 不覆盖)。"""
    from tests.e2e import preview_runner as runner
    from types import SimpleNamespace

    manifest = tasks.manifest_path_for("repeat-lifecycle")
    if not manifest.is_file():
        pytest.skip("manifest not present")

    run_root = tmp_path / "attempt"
    run_root.mkdir()
    existing = {"task": {"task_key": "repeat-lifecycle"}, "outcome": "prepared",
                "calibration_empty": {"reward": 0}, "calibration_gold": {"reward": 1}}
    receipt_path = run_root / runner.RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")

    args = SimpleNamespace(manifest=str(manifest), run_root=str(run_root))
    assert runner.cmd_prepare(args) == 2
    # 既有成功回执必须原样保留, 不被失败回执覆盖。
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == existing


def _report_args(manifest: Path, run_root: Path):
    from types import SimpleNamespace
    return SimpleNamespace(manifest=str(manifest), run_root=str(run_root),
                           admission=None)


def test_report_refuses_pass_without_calibration_evidence(tmp_path):
    """缺少校准/身份证据时 report 不得报 passed (spec: 缺必须证据不得 overall passed)。"""
    from tests.e2e import preview_runner as runner

    manifest = tasks.manifest_path_for("repeat-lifecycle")
    if not manifest.is_file():
        pytest.skip("manifest not present")
    root = tmp_path / "bare"
    (root / "evidence").mkdir(parents=True)
    (root / "workspace").mkdir()
    (root / runner.RECEIPT_FILENAME).write_text(
        json.dumps({"outcome": "prepared"}), encoding="utf-8")
    (root / runner.LOCK_FILENAME).write_text(json.dumps({}), encoding="utf-8")
    # 只有一个裸 grade 文件, 没有校准与身份证据。
    (root / "evidence" / "grade-last.json").write_text(
        json.dumps({"result": "passed"}), encoding="utf-8")
    assert runner.cmd_report(_report_args(manifest, root)) != 0


def test_report_requires_graded_manifest_to_claim_pass(tmp_path, monkeypatch):
    """存在最终 grade 但未携带被评分工作区指纹时, report 不得据此报 passed。"""
    from tests.e2e import preview_runner as runner
    from tests.support import swebench_bridge as bridge

    manifest = tasks.manifest_path_for("repeat-lifecycle")
    if not manifest.is_file():
        pytest.skip("manifest not present")
    root = tmp_path / "attempt"
    (root / "evidence" / "checks").mkdir(parents=True)
    (root / "workspace").mkdir()
    (root / runner.RECEIPT_FILENAME).write_text(
        json.dumps({"outcome": "prepared", "calibration_empty": {"reward": 0},
                    "calibration_gold": {"reward": 1}}), encoding="utf-8")
    (root / runner.LOCK_FILENAME).write_text(json.dumps({"verifier": {}}), encoding="utf-8")
    (root / "evidence" / "checks" / "c1").mkdir()
    # grade 声称 passed, 但没有 source_manifest。
    (root / "evidence" / "grade-last.json").write_text(
        json.dumps({"result": "passed", "required_passed": 16, "required_total": 16}),
        encoding="utf-8")
    # 隔离外部依赖, 只考察"缺 graded manifest"这一条。
    monkeypatch.setattr(runner, "verify_frozen_identity", lambda *a, **k: [])
    monkeypatch.setattr(bridge, "workspace_manifest",
                        lambda *a, **k: type("M", (), {"to_json": staticmethod(lambda: {})})())
    assert runner.cmd_report(_report_args(manifest, root)) != 0


def test_report_without_any_grade_reports_missing_grade_not_drift(tmp_path, monkeypatch):
    """没有最终 grade 时, 只应表达 grade 缺失, 不得伪报 grade 溯源缺陷。"""
    from tests.e2e import preview_runner as runner

    manifest = tasks.manifest_path_for("repeat-lifecycle")
    if not manifest.is_file():
        pytest.skip("manifest not present")
    root = tmp_path / "nograde"
    (root / "evidence" / "checks").mkdir(parents=True)
    (root / "workspace").mkdir()
    (root / runner.RECEIPT_FILENAME).write_text(
        json.dumps({"outcome": "prepared", "calibration_empty": {"reward": 0},
                    "calibration_gold": {"reward": 1}}), encoding="utf-8")
    (root / runner.LOCK_FILENAME).write_text(json.dumps({"verifier": {}}), encoding="utf-8")
    monkeypatch.setattr(runner, "verify_frozen_identity", lambda *a, **k: [])
    assert runner.cmd_report(_report_args(manifest, root)) != 0


# --- P1 私有材料隔离 ---
#
# 这些用例会真实改写临时目录的 ACL 并启动低完整性子进程 (不碰 Docker、不发模型请求)。
# 保留它们的理由: 隔离是安全边界, 且底层依赖的内核规则被 DSH 文档自标 unconfirmed ——
# 一旦形状或屏障失效, "封印成功"会变成静默放行, 属"已确认的关键风险"。


def test_package_sid_shape_guard_rejects_sids_that_do_not_trip_the_rule():
    """短值/通配/异族 SID 必须被拒: 它们不会触发内核规则, 用了等于没封印。"""
    from tests.support import swebench_isolation as iso

    for bad in ("S-1-15-2-1", "S-1-15-2-2", "S-1-15-2-99", "S-1-15-2-12345",
                "S-1-15-2-1-2", "S-1-5-21-1-2-3", "S-1-15-3-1-2-3", "Everyone"):
        with pytest.raises(iso.IsolationError):
            iso.validate_package_sid(bad)
    # 生产常量必须是多子授权形式, 否则封印静默失效。
    assert iso.package_sid_subauthorities(iso.PROTECTED_PACKAGE_SID) >= 3
    iso.validate_package_sid(iso.PROTECTED_PACKAGE_SID)


def _sealed_private(tmp_path: Path):
    from tests.support import swebench_isolation as iso

    if not sys.platform == "win32":
        pytest.skip("Windows ACL isolation is Windows-only")
    private = tmp_path / "private"
    (private / "verifier").mkdir(parents=True)
    (private / "gold_patch.diff").write_bytes(b"GOLD_SECRET_PAYLOAD\n")
    (private / "verifier" / "config.json").write_bytes(b'{"ok":true}\n')
    return iso, private


def test_sealed_private_passes_the_negative_gate(tmp_path):
    """正常封印后负向验收应为空: 子进程读不到私料, 但评估方读得到。"""
    iso, private = _sealed_private(tmp_path)
    record = iso.seal_private(private)
    assert iso.isolation_problems(private, record) == []


def test_gate_detects_a_removed_package_sid_ace(tmp_path):
    """撤掉包 SID ACE 后, 负向验收必须报出"子进程读到了受保护材料"。"""
    iso, private = _sealed_private(tmp_path)
    record = iso.seal_private(private)
    for path in sorted(private.rglob("*")):
        if path.is_file():
            subprocess.run(["icacls", str(path), "/remove:g",
                            f"*{iso.PROTECTED_PACKAGE_SID}"],
                           capture_output=True, text=True)
    subprocess.run(["icacls", str(private), "/remove:g",
                    f"*{iso.PROTECTED_PACKAGE_SID}"], capture_output=True, text=True)
    problems = iso.isolation_problems(private, record)
    assert any("READ protected material" in item for item in problems)


def test_gate_detects_both_delete_barriers_being_removed(tmp_path):
    """两条删除屏障 (目录包 SID ACE 与 DC deny) 同时撤掉时, 必须报出破坏性访问。

    只撤其中一条不该被误判为"已被破坏": 矩阵实测二者任一即可挡住删除。
    """
    iso, private = _sealed_private(tmp_path)
    record = iso.seal_private(private)
    subprocess.run(["icacls", str(private), "/remove:d", f"*{iso.EVERYONE_SID}"],
                   capture_output=True, text=True)
    problems = iso.isolation_problems(private, record)
    # 目录 ACE 仍在 => 删除仍被挡; 但 DC 屏障已少一层, 必须显式报告。
    assert any("FILE_DELETE_CHILD" in item for item in problems)
    assert not any("DELETED protected material" in item for item in problems)
    subprocess.run(["icacls", str(private), "/remove:g",
                    f"*{iso.PROTECTED_PACKAGE_SID}"], capture_output=True, text=True)
    problems = iso.isolation_problems(private, record)
    assert any("DELETED protected material" in item for item in problems)
