"""Skills 中立契约/目录发现/load_skill 工具/提示段/CLI 入口的聚焦测试 (票据 02).

覆盖 A02 (发现/排序/BOM/CRLF/首输入只含目录)、A03 (路径/格式/编码/限额整体拒绝)、
A11 的工具路径 (Skill 文本不产生回执)、A13 导入边界另见 test_import_boundaries。
零真实模型; CLI 用例经 --script 驱动。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cicada.plugins.skills import load_auto_skill_catalog, load_skill_catalog, skills_plugin
from cicada.plugins.skills import loader as loader_module
from cicada.plugins.skills import tool as tool_module
from cicada.prompting import DEFAULT_TOOLS, compose_system_prompt
from cicada.state_paths import workspace_state_dir
from cicada.skills import (
    DEFAULT_SKILL_TASK,
    MANIFEST_VERSION,
    MAX_SKILL_CATALOG_PROMPT_BYTES,
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_REQUEST_BYTES,
    SKILLS_PROMPT_VERSION,
    AUTO_SKILLS_PROMPT_VERSION,
    SKILL_REQUEST_PREFIX,
    SkillCatalogError,
    SkillCatalogSnapshot,
    SkillDocument,
    SkillInfo,
    SkillPromptError,
    SkillRequestError,
    SkillSelectionError,
    build_skill_invocation,
    build_skill_request,
    catalog_prompt_section,
    decode_skill_request,
    find_skill,
    manifest_json,
    render_skill_request,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def write_skill(root: Path, name: str, *, description: str = "A test skill.",
                body: str = "Body text.", extra_frontmatter: str = "",
                name_override: str | None = None, raw: bytes | None = None,
                bom: bool = False, crlf: bool = False, hidden: bool = False) -> Path:
    directory = root / ("." + name if hidden else name)
    directory.mkdir(parents=True, exist_ok=True)
    if raw is None:
        frontmatter = (
            f"---\nname: {name_override or name}\ndescription: {description}\n"
            f"{extra_frontmatter}---\n\n{body}\n"
        )
        raw = frontmatter.encode("utf-8-sig" if bom else "utf-8")
        if crlf:
            raw = raw.replace(b"\n", b"\r\n")
    (directory / "SKILL.md").write_bytes(raw)
    return directory


def load(root: Path) -> SkillCatalogSnapshot:
    return load_skill_catalog(str(root), root.parent)


def test_auto_catalog_project_wins_and_freezes_sources(tmp_path):
    user_dir = tmp_path / "home" / ".cicada"
    user_root = user_dir / "skills"
    project_root = tmp_path / "workspace" / ".agents" / "skills"
    write_skill(user_root, "shared", body="User body")
    write_skill(user_root, "general")
    write_skill(project_root, "shared", body="Project body")

    snapshot = load_auto_skill_catalog(project_root.parents[1], user_dir=user_dir)
    assert snapshot is not None
    assert [info.name for info in snapshot.infos] == ["general", "shared"]
    assert find_skill(snapshot, "shared", allow_manual=True).body == "Project body\n"
    assert snapshot.infos[-1].scope == "project"
    assert len(snapshot.roots) == 2
    assert any("overrides user source" in item for item in snapshot.diagnostics)
    prompt, info = compose_system_prompt(project_root.parents[1], skills=snapshot)
    assert info.version == AUTO_SKILLS_PROMPT_VERSION
    assert "Project body" not in prompt
    write_skill(project_root, "shared", body="Changed body")
    assert find_skill(snapshot, "shared", allow_manual=True).body == "Project body\n"
    assert load_auto_skill_catalog(project_root.parents[1], user_dir=user_dir).manifest_sha256 != snapshot.manifest_sha256


# --- 发现与元数据 (A02) ---------------------------------------------------------------------


@pytest.mark.parametrize("field", ["source_path", "base_dir", "body", "task"])
def test_decode_surrogate_fields_fall_back(field):
    payload = {"version": 1, "task": "task", "skill": _skill_with_name("alpha")}
    if field == "task":
        payload[field] = "\ud800"
    else:
        payload["skill"][field] = "\ud800"
    text = SKILL_REQUEST_PREFIX + "\n" + json.dumps(payload, ensure_ascii=True)
    assert decode_skill_request(text) is None
    assert decode_skill_request(SKILL_REQUEST_PREFIX + "\n" + "\ud800") is None


def test_candidate_lstat_permission_error_rejects_whole_catalog(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    original = Path.lstat

    def denied(path):
        if path.name == "SKILL.md":
            raise PermissionError("denied candidate")
        return original(path)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(SkillCatalogError) as error:
        load(root)
    assert error.value.kind == "read" and "SKILL.md" in error.value.path


def test_deep_yaml_is_classified_and_cli_exits_two(tmp_path):
    root = tmp_path / "skills"
    nested = "[" * 600 + "0" + "]" * 600
    raw = f"---\nname: alpha\ndescription: desc\nextra: {nested}\n---\nbody".encode()
    write_skill(root, "alpha", raw=raw)
    with pytest.raises(SkillCatalogError) as error:
        load(root)
    assert error.value.kind == "yaml"
    proc = run_cli("--workspace", str(tmp_path), "--skills-dir", str(root),
                   "--provider", "deepseek", "task")
    assert proc.returncode == 2
    assert "invalid skills directory" in proc.stderr and "Traceback" not in proc.stderr


@pytest.mark.parametrize("field", ["description", "unknown-key"])
def test_yaml_surrogate_metadata_rejects_before_manifest_and_provider(tmp_path, field):
    root = tmp_path / "skills"
    metadata = ('description: "\\uD800"' if field == "description"
                else 'description: desc\n"\\uD800": ignored')
    raw = f"---\nname: alpha\n{metadata}\n---\nbody".encode()
    write_skill(root, "alpha", raw=raw)
    with pytest.raises(SkillCatalogError):
        load(root)
    proc = run_cli("--workspace", str(tmp_path), "--skills-dir", str(root),
                   "--provider", "deepseek", "task")
    assert proc.returncode == 2
    assert "invalid skills directory" in proc.stderr and "Traceback" not in proc.stderr


def test_skill_name_rejects_trailing_newline_at_pure_boundary():
    from cicada.skills import validate_skill_name

    with pytest.raises(ValueError):
        validate_skill_name("alpha\n", "alpha\n")


def test_discovers_sorted_documents_with_normalized_body(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "beta", body="Beta body\r\nsecond line.")
    write_skill(root, "alpha", description="Alpha desc")
    snapshot = load(root)
    assert [d.info.name for d in snapshot.documents] == ["alpha", "beta"]
    assert snapshot.documents[0].info.description == "Alpha desc"
    assert snapshot.documents[1].body == "Beta body\nsecond line.\n"
    assert snapshot.documents[0].info.source_path == str(root / "alpha" / "SKILL.md")
    assert snapshot.documents[0].info.base_dir == str(root / "alpha")
    assert snapshot.root == str(root)
    assert snapshot.diagnostics == ()
    # manifest JSON 版本固定; 跨加载稳定性另见 empty_root 用例
    assert json.loads(manifest_json(snapshot.root, snapshot.documents))["version"] == (
        MANIFEST_VERSION
    )


def test_bom_and_crlf_are_handled_and_identity_covers_raw_bytes(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "plain", body="Body.")
    plain = load(root)
    root2 = tmp_path / "skills2"
    write_skill(root2, "plain", body="Body.", bom=True, crlf=True)
    fancy = load(root2)
    # 正文规范化一致, 但原始字节不同 → source_sha256 / manifest 不同
    assert fancy.documents[0].body == plain.documents[0].body == "Body.\n"
    assert fancy.documents[0].info.source_sha256 != plain.documents[0].info.source_sha256
    assert fancy.manifest_sha256 != plain.manifest_sha256
    assert fancy.documents[0].info.source_sha256 == hashlib.sha256(
        (root2 / "plain" / "SKILL.md").read_bytes()
    ).hexdigest()


def test_empty_root_is_valid_and_skips_are_inert(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "notes.txt").write_text("not a skill", encoding="utf-8")
    write_skill(root, "hidden", hidden=True)
    (root / "docless").mkdir()
    snapshot = load(root)
    assert snapshot.documents == ()
    assert load(root).manifest_sha256 == snapshot.manifest_sha256


def test_relative_root_resolves_against_workspace(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "my-skills" / "alpha").mkdir(parents=True)
    write_skill(workspace / "my-skills", "alpha")
    snapshot = load_skill_catalog("my-skills", workspace)
    assert snapshot.root == str(workspace / "my-skills")
    assert [d.info.name for d in snapshot.documents] == ["alpha"]


# --- 错误整体拒绝 (A03) -----------------------------------------------------------------------


def test_missing_or_file_root_rejected(tmp_path):
    with pytest.raises(SkillCatalogError, match="root_not_directory"):
        load_skill_catalog(str(tmp_path / "nope"), tmp_path)
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    with pytest.raises(SkillCatalogError, match="root_not_directory"):
        load_skill_catalog(str(tmp_path / "afile"), tmp_path)


def test_name_must_equal_directory_and_follow_charset(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha", name_override="beta")
    with pytest.raises(SkillCatalogError, match="name must equal"):
        load(root)
    root2 = tmp_path / "skills2"
    write_skill(root2, "alpha", name_override="Alpha")
    with pytest.raises(SkillCatalogError, match="lowercase"):
        load(root2)
    root3 = tmp_path / "skills3"
    write_skill(root3, "a-b--c")
    with pytest.raises(SkillCatalogError, match="single hyphens"):
        load(root3)


@pytest.mark.parametrize(
    "frontmatter", [
        "description: missing name\n",
        "name: alpha\n",
        "name: alpha\ndescription: '   '\n",
        "name: alpha\ndescription: 123\n",
        "name: alpha\ndescription: ok\nflag: yes\n",  # 未知字段列诊断, 不执行
        "name: alpha\ndescription: ok\ndisable-model-invocation: 'true'\n",
    ],
)
def test_frontmatter_value_rules(tmp_path, frontmatter):
    root = tmp_path / "skills"
    extra = "flag: yes\n" if "flag: yes" in frontmatter else frontmatter
    write_skill(root, "alpha", extra_frontmatter=extra)
    if "flag: yes" in frontmatter:
        snapshot = load(root)
        assert snapshot.documents[0].info.disable_model_invocation is False
        assert any("unsupported frontmatter key" in d for d in snapshot.diagnostics)
        return
    with pytest.raises(SkillCatalogError):
        load(root)


@pytest.mark.parametrize(
    "raw", [
        b"name: alpha\ndescription: x\n---\nbody",       # 无首行 ---
        b"---\nname: alpha\ndescription: x\nbody",        # 无结束行
        b"---\nname: alpha\ndescription: x\n---\n   \n",  # 正文纯空白
        b"---\nname: alpha\nname: alpha\ndescription: x\n---\nbody",  # 重复键
        b"---\n&anchor\nname: alpha\ndescription: x\n---\nbody",      # 锚
        b"---\nx: &a hello\nname: *a\ndescription: x\n---\nbody",     # 别名
        b"---\nname: !!weird alpha\ndescription: x\n---\nbody",       # 自定义标签
        b"---\n42: alpha\nname: a\ndescription: x\n---\nbody",        # 非字符串键
    ],
)
def test_yaml_and_format_wholesale_rejection(tmp_path, raw):
    root = tmp_path / "skills"
    write_skill(root, "alpha", raw=raw)
    with pytest.raises(SkillCatalogError):
        load(root)
    # 整体拒绝: 同目录里的合法候选也不得到达快照
    write_skill(root, "zeta")
    with pytest.raises(SkillCatalogError):
        load(root)


def test_reparse_point_inside_root_rejected(tmp_path):
    root = tmp_path / "skills"
    target = tmp_path / "elsewhere"
    write_skill(target, "alpha")
    root.mkdir()
    junction = root / "linked"
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)], capture_output=True
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot create junction: {proc.stderr.decode(errors='replace').strip()}")
    with pytest.raises(SkillCatalogError, match="reparse_point"):
        load(root)


def test_limits_reject_wholesale(tmp_path, monkeypatch):
    root = tmp_path / "entries"
    root.mkdir()
    for index in range(257):
        (root / f"f{index:03d}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SkillCatalogError, match="entry_limit"):
        load(root)

    root2 = tmp_path / "skills-count"
    for index in range(33):
        write_skill(root2, f"s-{index:02d}")
    with pytest.raises(SkillCatalogError, match="skill_limit"):
        load(root2)

    root3 = tmp_path / "skills-file"
    write_skill(root3, "big", body="x" * (16 * 1024 + 1))
    with pytest.raises(SkillCatalogError, match="file_limit"):
        load(root3)

    root4 = tmp_path / "skills-frontmatter"
    write_skill(root4, "alpha", description="d" * (8 * 1024))
    with pytest.raises(SkillCatalogError, match="frontmatter_limit"):
        load(root4)

    root5 = tmp_path / "skills-total"
    monkeypatch.setattr(loader_module, "MAX_SKILL_TOTAL_BYTES", 1024)
    write_skill(root5, "a1", body="x" * 700)
    write_skill(root5, "a2", body="x" * 700)
    with pytest.raises(SkillCatalogError, match="total_limit"):
        load(root5)

    root6 = tmp_path / "skills-path"
    monkeypatch.setattr(loader_module, "MAX_SKILL_PATH_BYTES", 10)
    write_skill(root6, "alpha")
    with pytest.raises(SkillCatalogError, match="path_limit"):
        load(root6)


def test_manifest_changes_with_any_raw_change(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha", body="v1")
    first = load(root)
    (root / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: A test skill.\n---\n\nv2\n", encoding="utf-8"
    )
    second = load(root)
    assert first.manifest_sha256 != second.manifest_sha256
    assert second.documents[0].body == "v2\n"


# --- 查找 / 提示段 / 工具 -------------------------------------------------------------------


def _snapshot_with_manual(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    write_skill(root, "manual", extra_frontmatter="disable-model-invocation: true\n")
    return load(root)


def test_find_skill_rules(tmp_path):
    snapshot = _snapshot_with_manual(tmp_path)
    assert find_skill(snapshot, "alpha", allow_manual=False).info.name == "alpha"
    assert find_skill(snapshot, "manual", allow_manual=True).info.name == "manual"
    with pytest.raises(SkillSelectionError, match="manual-only"):
        find_skill(snapshot, "manual", allow_manual=False)
    with pytest.raises(SkillSelectionError, match="unknown skill"):
        find_skill(snapshot, "nope", allow_manual=False)


def test_catalog_prompt_section_disclosure_and_limit(tmp_path):
    snapshot = _snapshot_with_manual(tmp_path)
    section = catalog_prompt_section(snapshot)
    assert snapshot.root in section and snapshot.manifest_sha256 in section
    assert '"name": "alpha"' in section
    assert '"description": "A test skill."' in section
    assert "CICADA_SKILL_REQUEST_V1" in section
    assert "manual" not in section, "manual-only 条目不得披露"
    # 多行描述保持单行 (JSON 转义); YAML 双引号转义解析为真实换行
    root = tmp_path / "ml"
    write_skill(
        root, "alpha",
        raw=b'---\nname: alpha\ndescription: "line1\\nline2"\n---\n\nBody.\n',
    )
    section2 = catalog_prompt_section(load(root))
    assert "line1\\nline2" in section2

    root3 = tmp_path / "huge"
    for index in range(16):
        write_skill(root3, f"s-{index:02d}", description="d" * 1024)
    with pytest.raises(SkillPromptError, match="byte limit"):
        catalog_prompt_section(load(root3))


def test_compose_system_prompt_skills_enabled_and_unchanged_disabled(tmp_path):
    snapshot = _snapshot_with_manual(tmp_path)
    prompt, info = compose_system_prompt(
        tmp_path, tools=DEFAULT_TOOLS + ("load_skill",), skills=snapshot
    )
    assert info.version == SKILLS_PROMPT_VERSION
    assert catalog_prompt_section(snapshot) in prompt
    assert "- load_skill" in prompt
    # 未启用: 输出与不传 kwarg 逐位一致, 版本不变
    base_prompt, base_info = compose_system_prompt(tmp_path)
    none_prompt, none_info = compose_system_prompt(tmp_path, skills=None)
    assert none_prompt == base_prompt and none_info.version == base_info.version == "builtin:p4-v2"


async def test_load_skill_tool_success_errors_and_details(tmp_path, monkeypatch):
    from cicada.core.cancel import CancelToken
    from cicada.core.ports import ToolContext

    snapshot = _snapshot_with_manual(tmp_path)
    instance = skills_plugin(snapshot)
    tool = tool_module.LoadSkillTool(snapshot)
    assert instance is not None and tool.spec.name == "load_skill"
    assert tool.spec.parameters["additionalProperties"] is False

    result = await tool.execute({"name": "alpha"}, ToolContext("c1", CancelToken()))
    assert result.is_error is False and result.call_id == "c1"
    assert "[skill name=alpha" in result.content
    assert "source_sha256=" in result.content and "Body text." in result.content
    assert result.details["name"] == "alpha"
    assert result.details["manifest_sha256"] == snapshot.manifest_sha256
    assert "body" not in result.details, "details 不得复制第二份正文"

    unknown = await tool.execute({"name": "nope"}, ToolContext("c2", CancelToken()))
    assert unknown.is_error and "unknown skill" in unknown.content
    manual = await tool.execute({"name": "manual"}, ToolContext("c3", CancelToken()))
    assert manual.is_error and "manual-only" in manual.content

    monkeypatch.setattr(tool_module, "MAX_SKILL_TOOL_CONTENT_BYTES", 32)
    limited = await tool.execute({"name": "alpha"}, ToolContext("c4", CancelToken()))
    assert limited.is_error and "content limit" in limited.content

    cancelled = CancelToken()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tool.execute({"name": "alpha"}, ToolContext("c5", cancelled))


# --- CLI 入口 (D8; --script 驱动, 零真实模型) -------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    # 子进程主目录固定在工作区内, 与本文件 workspace_state_dir 断言同一口径。
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    workspace = Path(args[args.index("--workspace") + 1])
    env["USERPROFILE"] = str(workspace / "test-user-home")
    env["HOME"] = str(workspace / "test-user-home")
    return subprocess.run(
        [sys.executable, "-m", "cicada", *args],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
    )


def _script_file(tmp_path: Path, name: str = "script.json") -> Path:
    script = tmp_path / name
    script.write_text(
        json.dumps([{"text": "完成", "stop": True}]),
        encoding="utf-8",
    )
    return script


def test_cli_skills_dir_rejects_bad_directory_before_model(tmp_path):
    dead = "http://127.0.0.1:1"
    proc = run_cli(
        "--workspace", str(tmp_path), "--provider", "deepseek",
        "--skills-dir", str(tmp_path / "nope"), "任务",
    )
    assert proc.returncode == 2
    assert "invalid skills directory" in proc.stderr
    assert "DEEPSEEK_API_KEY" not in proc.stderr


def test_cli_skills_dir_with_script_runs_and_announces(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    script = _script_file(tmp_path)
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "任务",
    )
    assert proc.returncode == 0, proc.stderr
    assert "[skills root=" in proc.stdout
    assert "skills=1" in proc.stdout


def test_cli_list_sessions_does_not_load_skills(tmp_path):
    proc = run_cli(
        "--workspace", str(tmp_path), "--list-sessions",
        "--skills-dir", str(tmp_path / "definitely-missing"),
    )
    assert proc.returncode == 0, proc.stderr
    assert "invalid skills directory" not in proc.stderr


def test_cli_resume_requires_skills_dir_retransmission(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    script = _script_file(tmp_path)
    saved = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--save-session", "任务",
    )
    assert saved.returncode == 0, saved.stderr
    session_id = next(
        line.split("id=")[1].split()[0]
        for line in saved.stdout.splitlines() if line.startswith("[session id=")
    )
    resumed = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--resume", session_id, "任务",
    )
    assert resumed.returncode == 2
    assert "--skills-dir 不从存档继承" in resumed.stderr
    again = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--resume", session_id, "任务",
    )
    assert again.returncode == 0, again.stderr


def test_cli_auto_skills_explicit_override_and_resume(tmp_path):
    project_root = tmp_path / ".agents" / "skills"
    write_skill(project_root, "project", body="Project body")
    explicit_root = tmp_path / "explicit"
    write_skill(explicit_root, "explicit", body="Explicit body")
    script = _script_file(tmp_path)

    saved = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--save-session", "--skill", "project",
    )
    assert saved.returncode == 0, saved.stderr
    assert "skills=1" in saved.stdout
    session_id = next(
        line.split("id=")[1].split()[0]
        for line in saved.stdout.splitlines() if line.startswith("[session id=")
    )
    resumed = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--resume", session_id, "任务",
    )
    assert resumed.returncode == 0, resumed.stderr

    explicit = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(explicit_root), "--skill", "explicit",
    )
    assert explicit.returncode == 0, explicit.stderr
    assert "skills=1" in explicit.stdout
    disabled = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--no-auto-skills", "--skill", "project",
    )
    assert disabled.returncode == 2
    assert "--skill 需要 --skills-dir" in disabled.stderr

    write_skill(project_root, "project", body="Changed body")
    conflict = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--resume", session_id, "任务",
    )
    assert conflict.returncode == 2
    assert "config conflicts" in conflict.stderr
    adopted = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--resume", session_id, "--resume-with-current-config", "任务",
    )
    assert adopted.returncode == 0, adopted.stderr


# --- 显式请求封装 (票据 03, 契约 §4): 纯构造与有界解码 ------------------------------------------


def _document(body: str, *, manual: bool = False) -> SkillDocument:
    info = SkillInfo(
        name="alpha", description="The alpha skill.",
        source_path="C:/skills/alpha/SKILL.md", base_dir="C:/skills/alpha",
        source_sha256="a" * 64, disable_model_invocation=manual,
    )
    return SkillDocument(
        info=info, body=body, body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest()
    )


def _envelope(payload: dict) -> str:
    return SKILL_REQUEST_PREFIX + "\n" + json.dumps(payload, ensure_ascii=False)


def _valid_payload(**overrides) -> dict:
    body = overrides.pop("body", "Body text.")
    skill = overrides.pop("skill", None)
    if skill is None:
        skill = {
            "name": "alpha",
            "source_path": "C:/skills/alpha/SKILL.md",
            "base_dir": "C:/skills/alpha",
            "source_sha256": "a" * 64,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "body": body,
        }
    payload = {"version": 1, "skill": skill, "task": "任务"}
    payload.update(overrides)
    return payload


def _skill_with_name(name: str) -> dict:
    return {
        "name": name,
        "source_path": "p",
        "base_dir": "b",
        "source_sha256": "a" * 64,
        "body_sha256": hashlib.sha256(b"x").hexdigest(),
        "body": "x",
    }


def test_explicit_request_roundtrip_preserves_task_and_body():
    body = (
        'Body with "quotes", ' + chr(92) + "backslash, --- frontmatter 樣分隔符 and {json:like}\n"
        "第二行 CICADA_SKILL_REQUEST_V1 出现在正文中间\n\t含控制字面与 unicode ✓"
    )
    task = '修复登录 bug\n第二行任务 "带引号" 与 ' + chr(92) + "反斜杠"
    text = build_skill_request(_document(body), task)
    assert text.startswith(SKILL_REQUEST_PREFIX + "\n")
    # JSON 对象占前缀后的完整余文; 封装字符不出现在首行之外的结构位
    decoded = decode_skill_request(text)
    assert decoded is not None
    assert decoded.version == 1
    assert decoded.name == "alpha"
    assert decoded.body == body and decoded.task == task
    assert decoded.body_sha256 == _document(body).body_sha256


def test_explicit_request_default_task_and_manual_only_build():
    default_text = build_skill_request(_document("Body."), "")
    decoded = decode_skill_request(default_text)
    assert decoded is not None and decoded.task == DEFAULT_SKILL_TASK
    # 手动专用 Skill 允许显式选择构造 (模型路由拒绝是 find_skill 的 allow_manual 职责)
    manual = build_skill_request(_document("Manual body.", manual=True), "任务")
    assert decode_skill_request(manual) is not None


def test_decode_rejects_malformed_variants():
    baseline = _envelope(_valid_payload())
    assert decode_skill_request(baseline) is not None
    body = "Body text."
    variants = [
        pytest.param("plain text without prefix", id="no-prefix"),
        pytest.param(SKILL_REQUEST_PREFIX, id="prefix-without-newline"),
        pytest.param(
            SKILL_REQUEST_PREFIX + "\n" + json.dumps(_valid_payload()) + "\n尾部垃圾",
            id="trailing-garbage",
        ),
        pytest.param(
            SKILL_REQUEST_PREFIX + "\n" + json.dumps(_valid_payload())
            + "\n" + json.dumps(_valid_payload()),
            id="second-object",
        ),
        pytest.param(
            SKILL_REQUEST_PREFIX + "\n"
            + '{"version":1,"task":"t","skill":{"name":"a","body_sha256":"x"},'
            + '"skill":{"name":"a","body_sha256":"x"}}',
            id="duplicate-top-key",
        ),
        pytest.param(_envelope(_valid_payload(extra="x")), id="extra-top-key"),
        pytest.param(
            _envelope({k: v for k, v in _valid_payload().items() if k != "task"}),
            id="missing-task",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "description": "d"}}),
            id="extra-skill-key",
        ),
        pytest.param(_envelope(_valid_payload(version="1")), id="version-string"),
        pytest.param(_envelope(_valid_payload(version=2)), id="version-2"),
        pytest.param(_envelope(_valid_payload(version=True)), id="version-bool"),
        pytest.param(_envelope(_valid_payload(task=7)), id="task-int"),
        pytest.param(
            _envelope({**_valid_payload(), "skill": ["not", "a", "dict"]}),
            id="skill-list",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "body": 5}}),
            id="body-int",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "body_sha256": "0" * 64}}),
            id="body-hash-mismatch",
        ),
        pytest.param(_envelope(_valid_payload(body="   \n\t ")), id="whitespace-body"),
        pytest.param(_envelope(_valid_payload(body="")), id="empty-body"),
        pytest.param(
            _envelope(_valid_payload(skill=_skill_with_name("ABC"))),
            id="uppercase-name",
        ),
        pytest.param(
            _envelope(_valid_payload(skill=_skill_with_name("alpha\n"))),
            id="name-trailing-newline",
        ),
        pytest.param(
            _envelope(_valid_payload(skill=_skill_with_name("a" * 65))),
            id="name-too-long",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "source_sha256": "A" * 64}}),
            id="sha-uppercase",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "source_sha256": "short"}}),
            id="sha-short",
        ),
        pytest.param(
            _envelope({**_valid_payload(),
                       "skill": {**_valid_payload()["skill"], "base_dir": ""}}),
            id="empty-base-dir",
        ),
    ]
    for param in variants:
        assert decode_skill_request(param.values[0]) is None, param.id


def test_decode_rejects_oversized_fields_and_total():
    # 任务超限 → 总体超 64 KiB; 字段超限独立生效 (正文 > 16 KiB, hash 自洽也拒绝)
    huge_task = _envelope(_valid_payload(task="x" * 70_000))
    assert decode_skill_request(huge_task) is None
    assert len(huge_task.encode("utf-8")) > MAX_SKILL_REQUEST_BYTES
    big_body = "y" * (MAX_SKILL_FILE_BYTES + 1)
    big = _envelope(_valid_payload(skill={
        "name": "alpha", "source_path": "p", "base_dir": "b",
        "source_sha256": "a" * 64,
        "body_sha256": hashlib.sha256(big_body.encode("utf-8")).hexdigest(),
        "body": big_body,
    }))
    assert decode_skill_request(big) is None
    assert decode_skill_request(
        SKILL_REQUEST_PREFIX + "\n" + "z" * (MAX_SKILL_REQUEST_BYTES + 1)
    ) is None


def test_render_skill_request_over_limit_raises():
    invocation = build_skill_invocation(_document("Body."), "x" * 70_000)
    with pytest.raises(SkillRequestError):
        render_skill_request(invocation)


def test_explicit_request_from_loaded_snapshot_roundtrips(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha", body="Loader body with CRLF check.",
                description="从真实快照构造", crlf=True)
    snapshot = load(root)
    document = find_skill(snapshot, "alpha", allow_manual=True)
    text = build_skill_request(document, "任务 X")
    decoded = decode_skill_request(text)
    assert decoded is not None
    assert decoded.body == document.body
    assert decoded.source_path == document.info.source_path
    assert decoded.base_dir == document.info.base_dir
    assert decoded.source_sha256 == document.info.source_sha256
    assert decoded.body_sha256 == document.body_sha256
    assert decoded.task == "任务 X"


# --- CLI --skill (票据 03, D8) ----------------------------------------------------------------


def test_cli_skill_runs_without_prompt_and_selects_manual(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha", body="Explicit body.")
    write_skill(root, "manual", body="Manual body.",
                extra_frontmatter="disable-model-invocation: true\n")
    script = _script_file(tmp_path)
    # 无位置 prompt: 任务用固定指令
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--skill", "manual",
    )
    assert proc.returncode == 0, proc.stderr
    # 位置 prompt 作为任务
    proc_task = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--skill", "alpha", "修复登录",
    )
    assert proc_task.returncode == 0, proc_task.stderr
    assert "[skills root=" in proc_task.stdout


def test_cli_skill_requires_skills_dir(tmp_path):
    script = _script_file(tmp_path)
    proc = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skill", "alpha",
    )
    assert proc.returncode == 2
    assert "--skill 需要 --skills-dir" in proc.stderr


def test_cli_skill_unknown_rejected_before_provider_probe(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    dead = "http://127.0.0.1:1"
    proc = run_cli(
        "--workspace", str(tmp_path), "--provider", "deepseek",
        "--skills-dir", str(root), "--skill", "nope",
    )
    assert proc.returncode == 2
    assert "invalid --skill" in proc.stderr and "unknown skill" in proc.stderr
    assert "DEEPSEEK_API_KEY" not in proc.stderr


def test_cli_skill_rejected_with_list_sessions(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "alpha")
    proc = run_cli(
        "--workspace", str(tmp_path), "--list-sessions",
        "--skills-dir", str(root), "--skill", "alpha",
    )
    assert proc.returncode == 2
    assert "只读列表入口" in proc.stderr


def test_decode_deep_nesting_returns_none_instead_of_recursion_error():
    """审查处置 F1: 64 KiB 内深嵌套 JSON 的 RecursionError 不得逃出解码."""
    assert decode_skill_request(SKILL_REQUEST_PREFIX + "\n" + "[" * 60_000) is None
    assert decode_skill_request(
        SKILL_REQUEST_PREFIX + "\n" + '{"a":' * 20_000 + "1" + "}" * 20_000
    ) is None


def test_cli_skill_envelope_reaches_committed_archive(tmp_path):
    """A06 (审查处置 F3): CLI --skill 接线直接证据——封装文本进入完整提交存档."""
    import asyncio

    from cicada.plugins.session_store import open_session_store

    root = tmp_path / "skills"
    write_skill(root, "alpha", body="Wired body.")
    script = _script_file(tmp_path)
    saved = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--skill", "alpha", "CLI 任务",
        "--save-session",
    )
    assert saved.returncode == 0, saved.stderr
    session_id = next(
        line.split("id=")[1].split()[0]
        for line in saved.stdout.splitlines() if line.startswith("[session id=")
    )
    store = open_session_store(
        directory=workspace_state_dir(tmp_path, user_dir=tmp_path / "test-user-home" / ".cicada") / "sessions", session_id=session_id, workspace=tmp_path
    )
    try:
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 1
        first = snapshot.requests[0].messages[0]
        assert first.text.startswith(SKILL_REQUEST_PREFIX + "\n")
        decoded = decode_skill_request(first.text)
        assert decoded is not None
        assert decoded.name == "alpha" and decoded.task == "CLI 任务"
        assert "Wired body." in decoded.body
    finally:
        asyncio.run(store.aclose())
    # 无位置 prompt: 固定指令作为任务进入同一封装路径
    saved_default = run_cli(
        "--workspace", str(tmp_path), "--script", str(script),
        "--skills-dir", str(root), "--skill", "alpha", "--save-session",
    )
    assert saved_default.returncode == 0, saved_default.stderr
    session_id_default = next(
        line.split("id=")[1].split()[0]
        for line in saved_default.stdout.splitlines() if line.startswith("[session id=")
    )
    store_default = open_session_store(
        directory=workspace_state_dir(tmp_path, user_dir=tmp_path / "test-user-home" / ".cicada") / "sessions", session_id=session_id_default,
        workspace=tmp_path,
    )
    try:
        first_default = store_default.recovery_snapshot().requests[0].messages[0]
        decoded_default = decode_skill_request(first_default.text)
        assert decoded_default is not None
        assert decoded_default.task == DEFAULT_SKILL_TASK
    finally:
        asyncio.run(store_default.aclose())
