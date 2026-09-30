"""prompting 纯函数聚焦测试: 内置提示形态、指令文件有界加载、哈希溯源."""

from __future__ import annotations

import hashlib

import pytest

from cicada.prompting import (
    MAX_INSTRUCTIONS_BYTES,
    PROMPT_VERSION,
    InstructionsError,
    build_builtin_prompt,
    compose_system_prompt,
    load_instructions,
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
