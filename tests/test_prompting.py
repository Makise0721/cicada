"""prompting 纯函数聚焦测试: 内置提示形态、指令文件有界加载、哈希溯源、来源选择."""

from __future__ import annotations

import hashlib
import os

import pytest

from cicada.prompting import (
    DEFAULT_TOOLS,
    MAX_INSTRUCTIONS_BYTES,
    PROMPT_VERSION,
    CheckPrompt,
    InstructionsError,
    build_builtin_prompt,
    compose_system_prompt,
    load_instructions,
    select_instructions_file,
)


def test_builtin_prompt_shape(tmp_path):
    prompt = build_builtin_prompt(tmp_path)
    assert "coding agent" in prompt
    assert str(tmp_path.resolve()) in prompt
    for tool in ("read", "edit", "write", "powershell"):
        assert f"- {tool}" in prompt
    assert "PowerShell" in prompt
    # 无时间/动态状态: 两次构造完全一致
    assert prompt == build_builtin_prompt(tmp_path)


def test_read_only_prompt_scopes_tools_and_rules(tmp_path):
    prompt = build_builtin_prompt(tmp_path, read_only=True)
    for tool in ("read", "glob", "grep"):
        assert f"- {tool}" in prompt
    for tool in ("write", "edit", "powershell"):
        assert f"- {tool}" not in prompt
    assert "Read-only mode is active" in prompt
    # 只读声明只影响只读档: 默认提示仍列出写入与命令工具
    assert "- write" in build_builtin_prompt(tmp_path)


def test_compose_without_instructions(tmp_path):
    prompt, info = compose_system_prompt(tmp_path)
    assert prompt == build_builtin_prompt(tmp_path)
    assert info.version == PROMPT_VERSION
    assert info.instructions_path is None
    assert info.instructions_sha256 is None
    assert info.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def test_compose_appends_instructions_after_builtin(tmp_path):
    (tmp_path / "TEAM.md").write_text("永远先跑测试。", encoding="utf-8")
    prompt, info = compose_system_prompt(tmp_path, "TEAM.md")
    builtin_end = prompt.index("--- Project instructions")
    assert prompt[:builtin_end] == build_builtin_prompt(tmp_path) + "\n"  # 空行分隔
    assert "永远先跑测试。" in prompt
    assert f"source: {(tmp_path / 'TEAM.md')}" in prompt
    raw = (tmp_path / "TEAM.md").read_bytes()
    assert info.instructions_sha256 == hashlib.sha256(raw).hexdigest()
    # raw bytes hash 与组装文本 hash 分别标注, 不混用
    assert info.instructions_sha256 != info.prompt_sha256


def test_compose_resolves_relative_against_workspace(tmp_path):
    nested = tmp_path / "docs"
    nested.mkdir()
    (nested / "rules.txt").write_text("rule one", encoding="utf-8")
    prompt, info = compose_system_prompt(tmp_path, "docs/rules.txt")
    assert "rule one" in prompt
    assert info.instructions_path == nested / "rules.txt"


def test_compose_accepts_absolute_path(tmp_path):
    target = tmp_path / "abs.txt"
    target.write_text("absolute rule", encoding="utf-8")
    prompt, _ = compose_system_prompt(tmp_path / "sub", str(target))
    assert "absolute rule" in prompt


def test_bom_and_crlf_preserved_in_text_but_not_hash_confusion(tmp_path):
    raw = b"\xef\xbb\xbfline1\r\nline2\r\n"
    (tmp_path / "bom.txt").write_bytes(raw)
    text, _, raw_hash = load_instructions("bom.txt", tmp_path)
    assert text == "line1\r\nline2\r\n"  # utf-8-sig 去 BOM, 换行原样
    assert raw_hash == hashlib.sha256(raw).hexdigest()


def test_empty_and_whitespace_instructions_add_no_section(tmp_path):
    for name, content in (("empty.txt", ""), ("blank.txt", "  \n\t\n")):
        (tmp_path / name).write_text(content, encoding="utf-8")
        prompt, info = compose_system_prompt(tmp_path, name)
        assert "Project instructions" not in prompt
        assert prompt == build_builtin_prompt(tmp_path)
        # 来源信息仍记录文件与 raw hash (显式选择的输入被如实报告)
        assert info.instructions_path is not None


def test_chinese_path_and_content(tmp_path):
    (tmp_path / "规范.md").write_text("中文指令内容", encoding="utf-8")
    prompt, _ = compose_system_prompt(tmp_path, "规范.md")
    assert "中文指令内容" in prompt


def test_missing_file_rejected(tmp_path):
    with pytest.raises(InstructionsError, match="cannot read"):
        load_instructions("nope.md", tmp_path)


def test_directory_rejected(tmp_path):
    with pytest.raises(InstructionsError, match="cannot read"):
        load_instructions(".", tmp_path)


def test_invalid_utf8_rejected(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe invalid \x9f")
    with pytest.raises(InstructionsError, match="not valid UTF-8"):
        load_instructions("bad.txt", tmp_path)


def test_oversize_rejected_at_boundary(tmp_path):
    (tmp_path / "big.txt").write_bytes(b"x" * (MAX_INSTRUCTIONS_BYTES + 1))
    with pytest.raises(InstructionsError, match="exceeds"):
        load_instructions("big.txt", tmp_path)
    # 恰好等于上限可通过
    (tmp_path / "exact.txt").write_bytes(b"x" * MAX_INSTRUCTIONS_BYTES)
    text, _, _ = load_instructions("exact.txt", tmp_path)
    assert len(text) == MAX_INSTRUCTIONS_BYTES


def test_hash_stability_across_calls(tmp_path):
    (tmp_path / "a.md").write_text("same input", encoding="utf-8")
    _, info1 = compose_system_prompt(tmp_path, "a.md")
    _, info2 = compose_system_prompt(tmp_path, "a.md")
    assert info1.prompt_sha256 == info2.prompt_sha256
    assert info1.instructions_sha256 == info2.instructions_sha256


# --- project-context-v1 C1: 启动来源选择与仅信息 mode -------------------------------------


def test_select_instructions_modes(tmp_path):
    # 缺失 = absent, 不报错; 存在 = automatic; 显式与关闭优先, 不查磁盘候选
    user_dir = tmp_path / "test-user-home"
    assert select_instructions_file(tmp_path, user_dir=user_dir).mode == "absent"
    assert select_instructions_file(tmp_path, user_dir=user_dir).path is None
    (tmp_path / "AGENTS.md").write_text("规则", encoding="utf-8")
    auto = select_instructions_file(tmp_path, user_dir=user_dir)
    assert auto.mode == "automatic" and auto.path == str(tmp_path / "AGENTS.md")
    other = select_instructions_file(tmp_path, explicit_path="TEAM.md", user_dir=user_dir)
    assert other.mode == "explicit" and other.path == "TEAM.md"
    off = select_instructions_file(tmp_path, auto_enabled=False, user_dir=user_dir)
    assert off.mode == "disabled" and off.path is None
    # 关闭自动时不检查根候选: 即便 AGENTS.md 是目录也不报错
    (tmp_path / "AGENTS.md").unlink()
    (tmp_path / "AGENTS.md").mkdir()
    assert select_instructions_file(tmp_path, auto_enabled=False, user_dir=user_dir).mode == "disabled"


def test_select_instructions_rejects_directory_and_reparse(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()
    with pytest.raises(InstructionsError, match="directory"):
        select_instructions_file(tmp_path)
    (tmp_path / "AGENTS.md").rmdir()
    link = tmp_path / "AGENTS.md"
    try:
        if os.name == "nt":
            import _winapi

            (tmp_path / "junction-target").mkdir()
            _winapi.CreateJunction(str(tmp_path / "junction-target"), str(link))
        else:
            os.symlink("junction-target", link)
    except OSError as exc:
        pytest.skip(f"无法创建 reparse point: {exc}")
    with pytest.raises(InstructionsError, match="reparse"):
        select_instructions_file(tmp_path)


def test_compose_rejects_contradictory_instructions_mode(tmp_path):
    with pytest.raises(ValueError, match="contradictory"):
        compose_system_prompt(tmp_path, "TEAM.md", instructions_mode="disabled")
    with pytest.raises(ValueError, match="contradictory"):
        compose_system_prompt(tmp_path, None, instructions_mode="automatic")
    with pytest.raises(ValueError, match="unknown instructions mode"):
        compose_system_prompt(tmp_path, None, instructions_mode="auto")


def test_automatic_and_explicit_same_file_same_prompt_and_identity(tmp_path):
    (tmp_path / "AGENTS.md").write_text("先跑测试。", encoding="utf-8")
    auto_prompt, auto_info = compose_system_prompt(
        tmp_path, str(tmp_path / "AGENTS.md"), instructions_mode="automatic"
    )
    explicit_prompt, explicit_info = compose_system_prompt(
        tmp_path, "AGENTS.md", instructions_mode="explicit"
    )
    # 同路径同字节: 提示逐位一致, raw/prompt hash 一致 (mode 不参与身份)
    assert auto_prompt == explicit_prompt
    assert auto_info.instructions_sha256 == explicit_info.instructions_sha256
    assert auto_info.prompt_sha256 == explicit_info.prompt_sha256
    assert auto_info.instructions_mode == "automatic"
    assert explicit_info.instructions_mode == "explicit"
    # mode 不进提示文本: 与不传 mode 的旧组装逐位一致 (旧推导 explicit)
    legacy_prompt, legacy_info = compose_system_prompt(tmp_path, str(tmp_path / "AGENTS.md"))
    assert auto_prompt == legacy_prompt
    assert legacy_info.instructions_mode == "explicit"
    # 空白文件: 不追加正文, 来源与 raw hash 保留
    (tmp_path / "AGENTS.md").write_bytes(b"  \n")
    blank_prompt, blank_info = compose_system_prompt(
        tmp_path, str(tmp_path / "AGENTS.md"), instructions_mode="automatic"
    )
    assert "Project instructions" not in blank_prompt
    assert blank_info.instructions_path == tmp_path / "AGENTS.md"
    assert blank_info.instructions_sha256 == hashlib.sha256(b"  \n").hexdigest()

def test_compose_default_derivation_unchanged(tmp_path):
    # 纯默认调用 (不传 mode) 保持旧行为: 无路径=absent, 有路径=explicit
    _, no_path = compose_system_prompt(tmp_path)
    assert no_path.instructions_mode == "absent"
    (tmp_path / "TEAM.md").write_text("规则", encoding="utf-8")
    _, with_path = compose_system_prompt(tmp_path, "TEAM.md")
    assert with_path.instructions_mode == "explicit"


def test_user_and_project_instructions_are_ordered_and_identified(tmp_path):
    user_dir = tmp_path / "home" / ".cicada"
    user_dir.mkdir(parents=True)
    workspace = tmp_path / "project"
    workspace.mkdir()
    (user_dir / "AGENTS.md").write_text("通用规则", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("项目规则", encoding="utf-8")

    selected = select_instructions_file(workspace, user_dir=user_dir)
    prompt, info = compose_system_prompt(
        workspace, selected.path, instructions_mode=selected.mode,
        instruction_sources=selected.sources,
    )
    assert prompt.index("通用规则") < prompt.index("项目规则")
    assert tuple(scope for scope, _, _ in info.instruction_sources) == ("user", "project")
    assert info.instructions_sha256 != info.instruction_sources[-1][2]
    assert compose_system_prompt(
        workspace, selected.path, instructions_mode=selected.mode,
        instruction_sources=selected.sources,
    )[1].prompt_sha256 == info.prompt_sha256

    explicit = select_instructions_file(workspace, explicit_path="AGENTS.md", user_dir=user_dir)
    explicit_prompt, _ = compose_system_prompt(
        workspace, explicit.path, instructions_mode=explicit.mode,
    )
    assert "通用规则" not in explicit_prompt
    assert "项目规则" in explicit_prompt
    assert select_instructions_file(workspace, auto_enabled=False, user_dir=user_dir).sources == ()


# --- P4 §1/§6: 实际工具名单与指定检查声明 -----------------------------------------------


def test_default_prompt_lists_glob_and_grep(tmp_path):
    prompt = build_builtin_prompt(tmp_path)
    for tool in ("read", "edit", "write", "powershell", "glob", "grep"):
        assert f"- {tool}\n" in prompt
    assert "- check" not in prompt  # 普通模式没有 check
    assert prompt == build_builtin_prompt(tmp_path)  # 无时间/动态状态


def test_default_tools_tuple_matches_prompt():
    assert DEFAULT_TOOLS == ("read", "edit", "write", "powershell", "glob", "grep")


def test_check_mode_prompt_declares_launcher_checks(tmp_path):
    checks = (
        CheckPrompt("check-1", "pytest -q", 120.0),
        CheckPrompt("check-2", "Write-Output 'line one\nline two'", 30.0),
    )
    prompt = build_builtin_prompt(tmp_path, DEFAULT_TOOLS + ("check",), checks)
    assert "- check\n" in prompt
    assert "Required checks (declared by the launcher; you cannot change them):" in prompt
    assert "- check-1: command=pytest -q timeout=120s" in prompt
    # 多行命令压平为一行, 不破坏提示结构
    assert "command=Write-Output 'line one line two' timeout=30s" in prompt
    assert "Delivery is judged by the program" in prompt
    assert "stale" in prompt


def test_compose_passes_tools_and_checks(tmp_path):
    checks = (CheckPrompt("check-1", "exit 0", 120.0),)
    prompt, info = compose_system_prompt(
        tmp_path, None, tools=DEFAULT_TOOLS + ("check",), checks=checks
    )
    assert "- check\n" in prompt
    assert "- check-1: command=exit 0 timeout=120s" in prompt
    assert info.version == PROMPT_VERSION
    # 指令追加语义不变: 检查声明在指令之前 (内置提示部分)
    (tmp_path / "TEAM.md").write_text("先跑测试。", encoding="utf-8")
    prompt2, _ = compose_system_prompt(
        tmp_path, "TEAM.md", tools=DEFAULT_TOOLS + ("check",), checks=checks
    )
    assert prompt2.index("- check-1: command=exit 0") < prompt2.index("--- Project instructions")
