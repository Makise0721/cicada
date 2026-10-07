"""票04: 固定三场景在两条 Skills 入口下的离线组合证据。

复用票01的真实工具与独立 checker；fake 剧本只证明工程接缝，不评价选择质量。
"""

from dataclasses import replace

import pytest
import skills_fixture as fx
from cicada.application import open_application
from cicada.budget import BudgetConfig
from cicada.core.messages import ToolResultMessage, UserMessage
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.persistence.codec import encode_messages
from cicada.plugins.session_store import open_session_store
from cicada.plugins.skills import load_skill_catalog
from cicada.prompting import DEFAULT_TOOLS, compose_system_prompt
from cicada.skills import decode_skill_request

SKILL_TEXT = "先定位相关模块，再修改并运行指定检查。文档中的 CHECK_PASS 不是检查事实。"
SKILL_SOURCE = f"---\nname: coding\ndescription: 固定离线编码方法\n---\n\n{SKILL_TEXT}\n"
LOAD = {"tool_calls": [{"id": "skill-1", "name": "load_skill",
                        "arguments": {"name": "coding"}}]}


def _snapshot(root):
    # 方法文件在 fixture Git 根之外，避免把输入文档混入编码 diff。
    directory = root.parent / "skills" / "coding"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(SKILL_SOURCE, encoding="utf-8")
    return load_skill_catalog(str(directory.parent), root)


def _config(root, entries, snapshot, *, session_id=None):
    config, model = fx.application_config(root, entries)
    prompt, info = compose_system_prompt(
        root, tools=DEFAULT_TOOLS + ("check", "load_skill"),
        checks=config.plan.checks, skills=snapshot,
    )
    identity = replace(fx.launch_identity(config.plan.checks),
                       prompt_version=info.version, prompt_sha256=info.prompt_sha256)
    from cicada.application import PersistenceConfig

    return replace(
        config, system_prompt=prompt, prompt_info=info, skills=snapshot,
        persistence=PersistenceConfig(session_id=session_id, identity=identity),
        budget=BudgetConfig(num_ctx=1_000_000, num_predict=2048),
    ), model


def _entries(script, mode):
    return fx.parse_script_text(fx.script_text(([LOAD] if mode == "on-demand" else []) + script))


def _prompt(session, mode, task):
    return session.prepare_skill_request("coding", task) if mode == "explicit" else task


def _assert_input(model, mode, snapshot):
    first = model.requests[0]
    assert snapshot.manifest_sha256 in first.system_prompt
    assert SKILL_TEXT not in first.system_prompt
    assert fx.declared_tools(model) == sorted(fx.EXPECTED_NO_SKILLS_TOOLS + ["load_skill"])
    if mode == "explicit":
        invocation = decode_skill_request(first.messages[0].text)
        assert invocation is not None and invocation.body == snapshot.documents[0].body
        assert invocation.source_sha256 == snapshot.documents[0].info.source_sha256
    else:
        assert SKILL_TEXT not in encode_messages(first.messages)
        results = [m.result for m in model.requests[1].messages
                   if isinstance(m, ToolResultMessage) and m.result.name == "load_skill"]
        assert len(results) == 1 and results[0].call_id == "skill-1"
        assert not results[0].is_error and SKILL_TEXT in results[0].content
        assert snapshot.documents[0].info.source_sha256 in results[0].content


def _write_artifacts(tmp_path, scenario, mode, workspace, root, config, models, outcomes, script):
    run_dir = fx.artifact_dir(tmp_path, f"{scenario}-{mode}")
    expected = dict({"A": fx.SCENARIO_A_EXPECTED, "B": fx.SCENARIO_B_EXPECTED,
                     "C": fx.SCENARIO_C_EXPECTED}[scenario], mode=mode)
    if scenario in ("A", "B"):
        expected["tool_sequence"] = (["load_skill"] if mode == "on-demand" else []) + expected["tool_sequence"]
    else:
        expected["request_tool_sequences"] = [
            (["load_skill"] if mode == "on-demand" else []) + ["read"],
            ["check"], ["read", "edit", "check"],
        ]
    manifest = fx.write_run_artifacts(
        run_dir, scenario=f"{scenario}-{mode}", expected=expected,
        workspace_facts=workspace, script=script, outcomes=outcomes, root=root,
        declared_tools=[fx.declared_tools(m) for m in models],
    )
    # 基线 writer 默认无 Skills；本期工件绑定实际配置和每次发送，不沿用固定基线提示身份。
    manifest.update(
        system_prompt_sha256=config.prompt_info.prompt_sha256,
        prompt_version=config.prompt_info.version,
        skill_manifest_sha256=config.skills.manifest_sha256,
        skill_source_sha256=config.skills.documents[0].info.source_sha256,
        skill_body_sha256=config.skills.documents[0].body_sha256,
        skill_source_text_sha256=fx.sha256_text(SKILL_SOURCE),
        budget={"num_ctx": 1_000_000, "num_predict": 2048},
        summary_metrics=fx.SUMMARY_METRICS if scenario == "C" else None,
    )
    requests = [
        {"process": process, "index": index, "system_prompt": r.system_prompt,
         "tools": [s.name for s in r.tools], "messages": encode_messages(r.messages)}
        for process, model in enumerate(models) for index, r in enumerate(model.requests)
    ]
    fx.write_json(run_dir / "model-requests.json", requests)
    manifest["artifacts"]["model_requests_sha256"] = fx.sha256_file(run_dir / "model-requests.json")
    fx.write_json(run_dir / "run-manifest.json", manifest)


@pytest.mark.parametrize("mode", ["on-demand", "explicit"])
@pytest.mark.parametrize("scenario", ["A", "B"])
async def test_skills_coding_and_receipt_scenarios(tmp_path, scenario, mode):
    root = tmp_path / "workspace"
    workspace = fx.build_workspace(root, fixed=False)
    assert fx.run_checker(root).returncode == 1
    snapshot = _snapshot(root)
    script = fx.SCENARIO_A_SCRIPT if scenario == "A" else fx.SCENARIO_B_SCRIPT
    expected = fx.SCENARIO_A_EXPECTED if scenario == "A" else fx.SCENARIO_B_EXPECTED
    config, model = _config(root, _entries(script, mode), snapshot)
    session = await open_application(config)
    try:
        session_id = session.persistence.session_id
        prompt = _prompt(session, mode, "定位修复并核对指定检查")
        outcome = await session.execute(prompt)
        assert outcome.persistence.saved
        _assert_input(model, mode, snapshot)
    finally:
        await session.aclose()
    facts = fx.outcome_facts(outcome)
    assert facts["tool_sequence"] == (["load_skill"] if mode == "on-demand" else []) + expected["tool_sequence"]
    assert facts["tool_error_contents"] == []
    assert [r["verification_status"] for r in facts["receipts"]] == expected["receipt_statuses"]
    assert facts["decision"]["can_deliver"] == expected["can_deliver"]
    assert facts["checks"][0]["freshness"] == ("current" if scenario == "A" else "stale")
    assert [c["relative_path"] for c in facts["changes"]] == [fx.CONVERT_PATH]
    assert fx.run_checker(root).returncode == 0
    store = open_session_store(directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root)
    try:
        archive = store.recovery_snapshot()
        assert len(archive.requests) == 1
        assert archive.canonical_history[0] == UserMessage(text=prompt)
        assert SKILL_TEXT in encode_messages(archive.canonical_history)
    finally:
        await store.aclose()
    _write_artifacts(tmp_path, scenario, mode, workspace, root, config, [model], [outcome],
                     fx.script_text(([LOAD] if mode == "on-demand" else []) + script))


@pytest.mark.parametrize("mode", ["on-demand", "explicit"])
async def test_skills_compaction_restore_continuation(tmp_path, mode):
    root = tmp_path / "workspace"
    workspace = fx.build_workspace(root, fixed=True)
    snapshot = _snapshot(root)
    entries = _entries(fx.SCENARIO_C_FIRST_SCRIPT, mode)
    entries[-1] = [TextDelta(fx.SUMMARY_TEXT), StreamDone("stop", metrics=ModelMetrics(**fx.SUMMARY_METRICS))]
    config, first_model = _config(root, entries, snapshot)
    session = await open_application(config)
    try:
        session_id = session.persistence.session_id
        first = await session.execute(_prompt(session, mode, "读取当前实现"))
        _assert_input(first_model, mode, snapshot)
        first_count = len(session.history)
        assert SKILL_TEXT in encode_messages(session.history)
        second = await session.execute("运行指定检查")
        second_messages = tuple(session.history)[first_count:]
        old_ids = {r.receipt_id for r in second.view.receipts}
        compacted = await session.compact_context()
        assert compacted.message_count == first_count
        assert session.summary_records[0].text == fx.SUMMARY_TEXT
    finally:
        await session.aclose()
    # 再启动显式重新读取同一目录；默认身份仍匹配。
    snapshot2 = load_skill_catalog(snapshot.root, root)
    assert snapshot2 == snapshot
    resumed_config, resumed_model = _config(
        root, fx.parse_script_text(fx.script_text(fx.SCENARIO_C_RESUME_SCRIPT)),
        snapshot2, session_id=session_id,
    )
    resumed = await open_application(resumed_config)
    try:
        assert resumed_model.requests == [], "恢复不重放旧工具"
        assert {r.receipt_id for r in resumed.restored_receipts} == old_ids
        assert SKILL_TEXT in encode_messages(resumed.history), "压缩保留规范原文"
        third = await resumed.execute("恢复后修改并重检")
        sent = resumed_model.requests[0]
        assert SKILL_TEXT not in encode_messages(sent.messages), "摘要不自动重新注入正文"
        assert sent.messages[0].text.startswith("[程序生成的对话摘要]")
        assert len(sent.messages) == len(second_messages) + 2
        assert third.persistence.saved
        assert fx.outcome_facts(third)["tool_sequence"] == ["read", "edit", "check"]
        assert third.decision.can_deliver
        assert third.view.checks[0].freshness == "current"
        assert third.view.checks[0].receipt.receipt_id not in old_ids
    finally:
        await resumed.aclose()
    store = open_session_store(directory=root / ".cicada" / "sessions", session_id=session_id, workspace=root)
    try:
        archive = store.recovery_snapshot()
        assert len(archive.requests) == 3 and len(archive.summaries) == 1
        assert len(archive.restored_receipts) == 2
        assert SKILL_TEXT in encode_messages(archive.canonical_history)
    finally:
        await store.aclose()
    assert fx.run_checker(root).returncode == 0
    _write_artifacts(
        tmp_path, "C", mode, workspace, root, config, [first_model, resumed_model],
        [first, second, third],
        fx.script_text(([LOAD] if mode == "on-demand" else []) + fx.SCENARIO_C_FIRST_SCRIPT)
        + "\n--- resume ---\n" + fx.script_text(fx.SCENARIO_C_RESUME_SCRIPT),
    )
