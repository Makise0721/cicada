"""票据 01 基线 E2E: 三个固定离线场景经共享应用与真实工具驱动.

observed 事实逐项断言等于 fixture 版本化预期; manifest/messages/diff 工件写入
pytest 私有目录并绑定主仓库 HEAD。零真实模型调用、零服务启动; 真实模型效果
未测, 不由本文件推导 (spec D9)。场景与身份构造见 skills_fixture.py。
"""

from __future__ import annotations

import re

import skills_fixture as fx
from cicada.application import open_application
from cicada.budget import BudgetConfig
from cicada.context_policy import summary_block_message
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.persistence.codec import encode_messages
from cicada.plugins.session_store import open_session_store


def _assert_manifest_shape(manifest: dict, script: str) -> None:
    assert manifest["fixture_version"] == fx.FIXTURE_VERSION
    assert manifest["decision_version"] == fx.DECISION_VERSION
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["main_head"])
    assert manifest["script_sha256"] == fx.sha256_text(script)
    assert manifest["model"] == {
        "kind": "scripted_fake", "real_model_calls": 0, "services_started": 0,
    }
    assert manifest["real_model_effect"] == "untested (spec D9)"


def _comparable(message) -> str:
    """请求投影可比形: RunPolicy 会在发送时对历史 check/powershell 结果重写
    freshness 标记行 (恢复后为 historical/unknown), 比较前剥离该行."""
    from dataclasses import replace

    from cicada.core.messages import ToolResultMessage

    if isinstance(message, ToolResultMessage):
        content = "\n".join(
            line for line in message.result.content.splitlines()
            if not line.startswith("Cicada freshness:")
        )
        return encode_messages((ToolResultMessage(result=replace(message.result, content=content)),))
    return encode_messages((message,))


async def test_scenario_a_multi_file_locate_and_fix(tmp_path):
    root = tmp_path / "scenario-a"
    workspace_facts = fx.build_workspace(root, fixed=False)
    # 基线反证: checker 对缺陷实现确定失败 (exit 1, 无 traceback)
    baseline = fx.run_checker(root)
    assert baseline.returncode == fx.SCENARIO_A_EXPECTED["checker_baseline_exit"] == 1
    assert baseline.stdout.startswith("CHECK_FAIL") and baseline.stderr == ""

    script = fx.script_text(fx.SCENARIO_A_SCRIPT)
    config, model = fx.application_config(root, fx.parse_script_text(script))
    session = await open_application(config)
    outcome = await session.execute("定位并修复摄氏转华氏缺陷, 使 check-1 通过")
    await session.aclose()

    # 无 Skills 基线: 配置提示为固定文本 (RunPolicy 在其后追加有界验证状态段),
    # 工具集合为既有完整名单
    assert model.requests[0].system_prompt.startswith(fx.SYSTEM_PROMPT)
    assert fx.declared_tools(model) == fx.EXPECTED_NO_SKILLS_TOOLS

    expected = fx.SCENARIO_A_EXPECTED
    facts = fx.outcome_facts(outcome)
    assert facts["stop_reason"] == "stop" and facts["error"] is None
    assert facts["exit_code"] == 0
    assert facts["tool_sequence"] == expected["tool_sequence"]
    assert facts["tool_error_contents"] == []
    assert [change["relative_path"] for change in facts["changes"]] == expected["changed_paths"]
    assert facts["evidence_complete"] is True
    assert facts["decision"]["model_stopped"] is True
    assert facts["decision"]["can_deliver"] is True and not facts["decision"]["blocking_reasons"]
    assert facts["checks"][0]["freshness"] == expected["current_freshness"]
    assert facts["checks"][0]["receipt"]["verification_status"] == "passed"
    assert [r["verification_status"] for r in facts["receipts"]] == expected["receipt_statuses"]
    # 真实 diff: 只改 convert.py, units.py 未动
    assert (root / fx.CONVERT_PATH).read_text(encoding="utf-8") == fx.FIXED_CONVERT_SOURCE
    assert (root / fx.UNITS_PATH).read_text(encoding="utf-8") == fx.UNITS_SOURCE
    after = fx.run_checker(root)
    assert after.returncode == expected["checker_after_exit"] == 0

    run_dir = fx.artifact_dir(tmp_path, "A-multi-file-locate-fix")
    manifest = fx.write_run_artifacts(
        run_dir, scenario="A-multi-file-locate-fix", expected=expected,
        workspace_facts=workspace_facts, script=script, outcomes=[outcome], root=root,
        declared_tools=[fx.declared_tools(model)],
    )
    _assert_manifest_shape(manifest, script)
    assert manifest["fixture_files_sha256"][fx.CONVERT_PATH] == fx.sha256_text(
        fx.DEFECT_CONVERT_SOURCE
    )


async def test_scenario_b_check_failure_then_fix(tmp_path):
    root = tmp_path / "scenario-b"
    workspace_facts = fx.build_workspace(root, fixed=False)
    script = fx.script_text(fx.SCENARIO_B_SCRIPT)
    config, model = fx.application_config(root, fx.parse_script_text(script))
    session = await open_application(config)
    outcome = await session.execute("先运行 check-1, 修复缺陷后重检, 再做一次改动")
    await session.aclose()

    assert model.requests[0].system_prompt.startswith(fx.SYSTEM_PROMPT)
    assert fx.declared_tools(model) == fx.EXPECTED_NO_SKILLS_TOOLS

    expected = fx.SCENARIO_B_EXPECTED
    facts = fx.outcome_facts(outcome)
    assert facts["stop_reason"] == "stop" and facts["error"] is None
    assert facts["tool_sequence"] == expected["tool_sequence"]
    assert facts["tool_error_contents"] == []
    # 回执序列: 先真实 failed (exit 1), 修复后当前 passed; 最新尝试取代旧结果
    assert [r["verification_status"] for r in facts["receipts"]] == expected["receipt_statuses"]
    assert facts["receipts"][0]["exit_code"] == 1
    assert facts["receipts"][1]["exit_code"] == 0
    assert facts["checks"][0]["receipt"]["verification_status"] == expected["current_receipt_status"]
    # 第二次改动后旧 PASS 失效: freshness 不再 current, 不可交付原因明确
    assert facts["checks"][0]["freshness"] == expected["final_freshness"]
    assert facts["decision"]["can_deliver"] is False and facts["decision"]["blocking_reasons"]
    # 真实文件效果: 修复 + 评审注记; checker 未被改动
    assert (root / fx.CONVERT_PATH).read_text(encoding="utf-8") == (
        fx.CONVERT_REVIEWED_HEADER + fx.CONVERT_IMPORT_FIXED + fx.CONVERT_BODY_FIXED
    )
    after = fx.run_checker(root)
    assert after.returncode == expected["checker_after_exit"] == 0

    run_dir = fx.artifact_dir(tmp_path, "B-check-failure-then-fix")
    manifest = fx.write_run_artifacts(
        run_dir, scenario="B-check-failure-then-fix", expected=expected,
        workspace_facts=workspace_facts, script=script, outcomes=[outcome], root=root,
        declared_tools=[fx.declared_tools(model)],
    )
    _assert_manifest_shape(manifest, script)
    assert manifest["expected"]["receipt_statuses"] == ["failed", "passed"]


async def test_scenario_c_compaction_resume_continuation(tmp_path):
    root = tmp_path / "scenario-c"
    workspace_facts = fx.build_workspace(root, fixed=True)
    checks = fx.check_definitions()
    identity = fx.launch_identity(checks)
    budget = BudgetConfig(num_ctx=1_000_000, num_predict=2048)
    first_script = fx.script_text(fx.SCENARIO_C_FIRST_SCRIPT)
    # 替身摘要带可得计量: 最后一个剧本条目换成带 metrics 的事件列 (scripted, D9)
    first_entries = fx.parse_script_text(first_script)
    first_entries[-1] = [
        TextDelta(fx.SUMMARY_TEXT),
        StreamDone("stop", metrics=ModelMetrics(**fx.SUMMARY_METRICS)),
    ]
    config, first_model = fx.application_config(
        root, first_entries, checks=checks, identity=identity, budget=budget,
    )
    session = await open_application(config)
    assert session.persistence is not None and session.persistence.mode == "save"
    session_id = session.persistence.session_id
    first = await session.execute("读取当前实现")
    second = await session.execute("运行 check-1")
    # 无 Skills 基线: 固定配置提示 + 既有工具集合; 两进程声明一致
    assert first_model.requests[0].system_prompt.startswith(fx.SYSTEM_PROMPT)
    assert fx.declared_tools(first_model) == fx.EXPECTED_NO_SKILLS_TOOLS
    assert first.persistence.saved and second.persistence.saved
    old_receipt_ids = {
        r["receipt_id"] for r in fx.outcome_facts(second)["receipts"]
    }
    assert len(old_receipt_ids) == 1
    # 每个完整请求 4 条规范消息: User, Assistant(tool_calls), ToolResult, Assistant
    req2_messages = tuple(session.history)[4:]
    assert len(req2_messages) == 4

    compacted = await session.compact_context()
    assert compacted is not None
    assert compacted.message_count == 4 and compacted.end_message_index == 4
    await session.aclose()

    # 归档事实: 摘要记录为替身正文与计量, 请求二回执已入档
    store = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root
    )
    snapshot = store.recovery_snapshot()
    assert len(snapshot.summaries) == 1
    record = snapshot.summaries[0]
    assert record.text == fx.SCENARIO_C_EXPECTED["summary_text"]
    assert record.reason == fx.SCENARIO_C_EXPECTED["summary_reason"]
    assert record.metrics.input_tokens == fx.SCENARIO_C_EXPECTED["summary_metrics_input_tokens"]
    assert {r.receipt_id for r in snapshot.restored_receipts} == old_receipt_ids
    await store.aclose()

    # 重启恢复: 新应用实例按 ID 默认恢复 (身份/工具指纹逐位一致才准入)
    resume_script = fx.script_text(fx.SCENARIO_C_RESUME_SCRIPT)
    resume_config, resumed_model = fx.application_config(
        root, fx.parse_script_text(resume_script),
        checks=checks, session_id=session_id, identity=identity, budget=budget,
    )
    resumed = await open_application(resume_config)
    assert resumed.persistence.mode == "resume"
    # 旧回执历史化: 恢复会话携带请求二回执, 不重跑检查
    assert {r.receipt_id for r in resumed.restored_receipts} == old_receipt_ids

    expected_block = summary_block_message(
        first_request_seq=compacted.first_request_seq,
        last_request_seq=compacted.last_request_seq,
        message_count=compacted.message_count,
        text=fx.SUMMARY_TEXT,
    )
    prompt = "恢复后继续: 修改注释并重检"
    third = await resumed.execute(prompt)
    sent = resumed_model.requests[0]
    # 恢复进程同样收到固定配置提示与一致的工具声明
    assert sent.system_prompt.startswith(fx.SYSTEM_PROMPT)
    assert fx.declared_tools(resumed_model) == fx.EXPECTED_NO_SKILLS_TOOLS
    # 历史工具不重放: 实际请求 = [摘要块] + 请求二规范消息 + 新 UserMessage;
    # 除 RunPolicy 请求时追加的 freshness 标记行外逐位一致
    assert [_comparable(m) for m in sent.messages] == [
        _comparable(expected_block),
        *[_comparable(m) for m in req2_messages],
        _comparable(UserMessage(text=prompt)),
    ]
    # 新回执成为当前事实; 旧回执仍只在恢复历史中
    facts = fx.outcome_facts(third)
    assert facts["tool_sequence"] == fx.SCENARIO_C_EXPECTED["request_tool_sequences"][2]
    assert facts["tool_error_contents"] == []
    assert facts["checks"][0]["freshness"] == fx.SCENARIO_C_EXPECTED["current_freshness"]
    assert facts["checks"][0]["receipt"]["verification_status"] == "passed"
    assert [change["relative_path"] for change in facts["changes"]] == (
        fx.SCENARIO_C_EXPECTED["changed_paths"]
    )
    assert facts["evidence_complete"] is True
    assert facts["decision"]["can_deliver"] is True
    assert third.persistence.saved
    assert {r.receipt_id for r in resumed.restored_receipts} == old_receipt_ids
    await resumed.aclose()

    # 终局归档: 三请求、两回执、一摘要; 真实文件效果与独立 checker
    final_store = open_session_store(
        directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root
    )
    final_snapshot = final_store.recovery_snapshot()
    assert len(final_snapshot.requests) == 3
    assert len(final_snapshot.restored_receipts) == fx.SCENARIO_C_EXPECTED["archive_receipt_count"]
    assert len(final_snapshot.summaries) == 1
    await final_store.aclose()
    assert (root / fx.CONVERT_PATH).read_text(encoding="utf-8") == (
        fx.CONVERT_REVIEWED_HEADER + fx.CONVERT_IMPORT_FIXED + fx.CONVERT_BODY_FIXED
    )
    assert fx.run_checker(root).returncode == fx.SCENARIO_C_EXPECTED["checker_after_exit"]

    run_dir = fx.artifact_dir(tmp_path, "C-compaction-resume-continuation")
    combined_script = first_script + "\n--- resume ---\n" + resume_script
    manifest = fx.write_run_artifacts(
        run_dir, scenario="C-compaction-resume-continuation",
        expected=fx.SCENARIO_C_EXPECTED, workspace_facts=workspace_facts,
        script=combined_script,
        outcomes=[first, second, third], root=root,
        declared_tools=[fx.declared_tools(first_model), fx.declared_tools(resumed_model)],
    )
    _assert_manifest_shape(manifest, combined_script)
    assert manifest["expected"]["summary_text"] == fx.SUMMARY_TEXT
