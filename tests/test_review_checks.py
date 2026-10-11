"""02 最小回归: 检查记录详情与累计工件读取的长期关键缺口。

票据 02 验证节点名的两类缺口, 既有检查未覆盖:
- 工件篡改/缺失在审阅详情与工件正文中不被 passed/current 掩盖 (C8);
- 三状态边界: 恢复不从 restored_requests 恢复当前引用, clear 保留最近引用。
真实内核 + 真实插件组装, FakeModel 零工具调用, 不运行任何检查命令;
seed 回执写入服务账本 (与既有测试替换 session._service 同风格的应用 seam),
工件在项目内临时工作区。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from cicada.application import (
    CHECK_TOOL_CAPABILITIES,
    ApplicationConfig,
    ApplicationSession,
    default_definitions,
    open_application,
)
from cicada.boot import bootstrap
from cicada.core.cancel import CancelToken
from cicada.persistence.contracts import RequestOutcomeFacts, RestoredRequest
from cicada.plugins.coding.change_review import CHANGE_REVIEW_CAPABILITY
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    VERIFICATION_CAPABILITY,
    CheckDefinition,
    CheckReceipt,
    VerificationPlan,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.review import ReviewCheckDetails, ReviewPreview

from tests.support.app import prompt_info as _prompt_info, repo as _repo


def _config(root: Path, model: FakeModel, plan: VerificationPlan | None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="review checks system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(model),
        plan=plan,
        max_turns=6,
        model_label="fake",
    )


def _plan(root: Path) -> VerificationPlan:
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )


def _seed_passed_receipt(plan: VerificationPlan, root: Path, artifact: Path,
                         sha256: str, snapshot_ref: str) -> CheckReceipt:
    return CheckReceipt(
        verification_run_id=plan.verification_run_id, receipt_id="chk-review-checks-0001",
        check_id="check-1", command=plan.checks[0].command, cwd=root, timeout_s=30.0,
        elapsed_s=0.01, execution_status="exited", exit_code=0, timed_out=False,
        cancelled=False, output_complete=True, snapshot_before=snapshot_ref,
        snapshot_after=snapshot_ref, verification_status="passed", failure_kind=None,
        output_truncated=False, artifact_truncated=False,
        output_artifact_path=artifact, output_artifact_sha256=sha256,
    )


async def test_check_details_reports_tampered_artifact_without_passing(tmp_path):
    root = _repo(tmp_path)
    model = FakeModel([])
    session = await open_application(_config(root, model, _plan(root)))
    try:
        await session.execute("普通请求")
        session.cancel_review()
        snapshot = await session.open_review()
        assert snapshot.checks_state == "available" and snapshot.checks_target_id
        service = session._service
        artifact_file = session._review_reader.workspace.output_dir / "check-output.txt"
        artifact_file.write_text("output line\n", encoding="utf-8")
        sha256 = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
        view = await service.refresh(CancelToken())
        assert view.snapshot_ref is not None
        service._receipts.append(
            _seed_passed_receipt(session._config.plan, root, artifact_file, sha256,
                                 view.snapshot_ref))

        details = await session.read_review(snapshot.review_id, snapshot.checks_target_id)
        assert isinstance(details, ReviewCheckDetails)
        row = details.rows[0]
        assert (row.verification_status, row.freshness, row.artifact_integrity) == (
            "passed", "current", "verified")
        preview = await session.read_review(snapshot.review_id, details.artifact_target_id)
        assert isinstance(preview, ReviewPreview)
        # artifact_target 是累计证据工件; seed 回执工件只经行内核验 (C8 分列呈现)
        assert preview.integrity == "verified" and "# cicada change evidence" in preview.text
        assert "启动基线" in preview.comparison and preview.observed_at is None

        artifact_file.write_text("tampered\n", encoding="utf-8")
        details = await session.read_review(snapshot.review_id, snapshot.checks_target_id)
        row = details.rows[0]
        # passed/current 是历史事实照实保留, 但工件完整性可见, 不标 verified
        assert row.verification_status == "passed" and row.freshness == "current"
        assert row.artifact_integrity == "mismatch" and row.reason

        artifact_file.unlink()
        details = await session.read_review(snapshot.review_id, snapshot.checks_target_id)
        assert details.rows[0].artifact_integrity == "missing"
        # 证据工件未被触碰: 回执工件完整性与其正文核验互不影响
        preview = await session.read_review(snapshot.review_id, details.artifact_target_id)
        assert preview.integrity == "verified"

        # 查看零新增回执、零模型调用、规范历史不动
        assert len(service.receipts) == 1 and len(model.requests) == 1
        assert len(session.history) == 2
    finally:
        session.cancel_review()
        await session.aclose()


async def test_review_check_states_restore_and_clear(tmp_path):
    (tmp_path / "plain").mkdir()
    plain = await open_application(
        _config(_repo(tmp_path / "plain"), FakeModel([]), None))
    try:
        snapshot = await plain.open_review()
        assert snapshot.checks_state == "not_enabled" and snapshot.checks_target_id is None
    finally:
        plain.cancel_review()
        await plain.aclose()

    (tmp_path / "check").mkdir()
    root = _repo(tmp_path / "check")
    plan = _plan(root)
    session = await open_application(_config(root, FakeModel([]), plan))
    try:
        snapshot = await session.open_review()
        assert snapshot.checks_state == "not_recorded" and snapshot.checks_target_id is None
        session.cancel_review()
        await session.execute("普通请求")
        session.cancel_review()
        snapshot = await session.open_review()
        assert snapshot.checks_state == "available" and snapshot.checks_target_id
        # clear 只清模型历史; 最近 outcome 引用与三状态保留 (C8)
        session.cancel_review()
        session.clear_history()
        snapshot = await session.open_review()
        assert session.history == () and snapshot.checks_state == "available"
    finally:
        session.cancel_review()
        await session.aclose()

    # 恢复会话不从 restored_requests 恢复当前引用: 有 plan 无本进程 outcome -> not_recorded
    app = await bootstrap(
        default_definitions(root, fake_model_plugin(FakeModel([])), plan),
        tool_capabilities=CHECK_TOOL_CAPABILITIES,
        system_prompt="review checks system prompt", max_turns=6,
    )
    restored = ApplicationSession(
        app=app, config=_config(root, FakeModel([]), plan),
        service=app.runtime.capability(VERIFICATION_CAPABILITY), policy=None,
        persistence_mode="resume",
        review_reader=app.runtime.capability(CHANGE_REVIEW_CAPABILITY),
        restored_requests=(RestoredRequest(
            request_id="req-1", epoch_id="epoch-0", seq=1, context_generation=0,
            prompt="旧请求", messages=(),
            outcome=RequestOutcomeFacts(run_id="run-1", stop_reason="stop", error=None,
                                        exit_code=0, finalized=True),
            committed_at=0.0,),),
    )
    try:
        snapshot = await restored.open_review()
        assert snapshot.checks_state == "not_recorded" and snapshot.checks_target_id is None
    finally:
        restored.cancel_review()
        await restored.aclose()
