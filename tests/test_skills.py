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

from cicada.plugins.skills import load_skill_catalog, skills_plugin
from cicada.plugins.skills import loader as loader_module
from cicada.plugins.skills import tool as tool_module
from cicada.prompting import DEFAULT_TOOLS, compose_system_prompt
from cicada.skills import (
    MANIFEST_VERSION,
    MAX_SKILL_CATALOG_PROMPT_BYTES,
    SKILLS_PROMPT_VERSION,
    SkillCatalogError,
    SkillCatalogSnapshot,
    SkillPromptError,
    SkillSelectionError,
    catalog_prompt_section,
    find_skill,
    manifest_json,
    manifest_sha256,
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


# --- 发现与元数据 (A02) ---------------------------------------------------------------------


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
    # manifest 确定性: 两次加载一致; JSON 版本固定
    assert snapshot.manifest_sha256 == manifest_sha256(snapshot.root, snapshot.documents)
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
    assert none_prompt == base_prompt and none_info.version == base_info.version == "builtin:p4-v1"


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
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
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
        "--workspace", str(tmp_path), "--ollama-url", dead,
        "--skills-dir", str(tmp_path / "nope"), "任务",
    )
    assert proc.returncode == 2
    assert "invalid skills directory" in proc.stderr
    assert "Ollama 不可达" not in proc.stderr


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
