import json
import os
import subprocess

import pytest

import cicada.plugins.coding.tool_read as tool_read_module
from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.process import resolve_pwsh
from cicada.plugins.coding.tool_read import MAX_CONTENT_BYTES, MAX_LINES, ReadTool
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, ReadTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


def expected_content(path, start, end, total, body_lines, footer):
    source = json.dumps(str(path), ensure_ascii=False)
    return "\n".join(
        [f"[file={source} lines={start}-{end} total_lines={total}]", *body_lines, footer]
    )


async def test_basic_slice_has_source_header_and_footer(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert not r.is_error
    assert r.content == expected_content(
        f, 1, 3, 3, ["1\tone", "2\ttwo", "3\tthree"], "[truncated=false next_offset=none]"
    )
    assert r.details == {
        "total_lines": 3,
        "truncated": False,
        "next_offset": None,
        "truncation_reason": None,
        "path": str(f),
        "start_line": 1,
        "end_line": 3,
        "partial_last_line": False,
    }
    r2 = await run(tool, path="a.txt", offset=2, limit=1)
    # limit=1 只取一行, 其后仍有内容 → lines 截断并给出续读位置
    assert r2.content == expected_content(
        f, 2, 2, 3, ["2\ttwo"], "[truncated=true reason=lines next_offset=3]"
    )
    assert r2.details["start_line"] == 2
    assert r2.details["end_line"] == 2
    assert r2.details["truncated"] is True
    assert r2.details["next_offset"] == 3
    r3 = await run(tool, path="a.txt", offset=2)
    assert r3.content == expected_content(
        f, 2, 3, 3, ["2\ttwo", "3\tthree"], "[truncated=false next_offset=none]"
    )


async def test_header_path_is_json_escaped_canonical_absolute(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("x\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert not r.is_error
    first_line = r.content.split("\n")[0]
    # Windows 反斜杠路径必须 JSON 转义 (\"...\" 且 \\), 模型可直接按 JSON 读取
    assert first_line.startswith("[file=")
    encoded = first_line[len("[file=") : first_line.index(" lines=")]
    assert json.loads(encoded) == str(f)
    assert "\\" in encoded  # 转义后的反斜杠


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive path alias")
async def test_source_uses_actual_file_casing(tmp_path):
    ws, tool = make(tmp_path)
    target = ws.root / "MixedCase.TXT"
    target.write_text("actual\n", encoding="utf-8")
    result = await run(tool, path="mixedcase.txt")
    assert not result.is_error
    assert result.details["path"] == str(target.resolve())
    assert result.content == expected_content(
        target.resolve(), 1, 1, 1, ["1\tactual"], "[truncated=false next_offset=none]"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junction")
async def test_junction_source_is_actual_target_and_remains_readable(tmp_path):
    ws, tool = make(tmp_path)
    target_dir = tmp_path / "actual"
    target_dir.mkdir()
    target = target_dir / "a.txt"
    target.write_text("junction target\n", encoding="utf-8")
    alias = ws.root / "alias"
    subprocess.run(
        [
            str(resolve_pwsh()), "-NoProfile", "-NonInteractive", "-Command",
            "New-Item -ItemType Junction -Path $env:CICADA_TEST_ALIAS "
            "-Target $env:CICADA_TEST_TARGET -ErrorAction Stop | Out-Null",
        ],
        env={**os.environ, "CICADA_TEST_ALIAS": str(alias), "CICADA_TEST_TARGET": str(target_dir)},
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=30,
    )
    try:
        result = await run(tool, path="alias/a.txt")
        assert not result.is_error
        assert result.details["path"] == str(target.resolve())
        assert result.content == expected_content(
            target.resolve(), 1, 1, 1, ["1\tjunction target"],
            "[truncated=false next_offset=none]",
        )
    finally:
        alias.rmdir()  # Remove only the junction, never its target directory.


async def test_absolute_path_outside_root_is_readable(tmp_path):
    ws, tool = make(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("ext\n", encoding="utf-8")
    r = await run(tool, path=str(outside))
    assert not r.is_error
    assert r.content == expected_content(
        outside, 1, 1, 1, ["1\text"], "[truncated=false next_offset=none]"
    )


async def test_offset_beyond_eof(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "a.txt").write_text("one\n", encoding="utf-8")
    r = await run(tool, path="a.txt", offset=5)
    assert r.is_error
    assert "offset" in r.content


async def test_line_truncation_and_next_offset(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("\n".join(f"L{i}" for i in range(1, 2501)) + "\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert not r.is_error
    assert r.details["truncated"] is True
    assert r.details["truncation_reason"] == "lines"
    assert r.details["next_offset"] == MAX_LINES + 1
    assert r.details["start_line"] == 1
    assert r.details["end_line"] == MAX_LINES
    content_lines = r.content.split("\n")
    assert len(content_lines) == MAX_LINES + 2  # 头 + 正文 + footer
    assert content_lines[0].startswith("[file=")
    assert content_lines[1] == "1\tL1"
    assert content_lines[-2] == f"{MAX_LINES}\tL{MAX_LINES}"
    assert content_lines[-1] == f"[truncated=true reason=lines next_offset={MAX_LINES + 1}]"
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    r2 = await run(tool, path="a.txt", offset=MAX_LINES + 1)
    assert r2.details["truncated"] is False
    assert r2.details["start_line"] == MAX_LINES + 1
    assert r2.details["end_line"] == 2500
    assert r2.content.split("\n")[-1] == "[truncated=false next_offset=none]"
    tail_lines = r2.content.split("\n")
    assert tail_lines[1] == f"{MAX_LINES + 1}\tL{MAX_LINES + 1}"
    assert len(tail_lines) == 500 + 2


async def test_byte_truncation(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    line = "x" * 100
    f.write_text("\n".join([line] * 2000) + "\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert not r.is_error
    assert r.details["truncated"] is True
    assert r.details["truncation_reason"] == "bytes"
    n = r.details["next_offset"]
    assert isinstance(n, int) and 1 < n < 2000
    content_lines = r.content.split("\n")
    assert len(content_lines) == n + 1  # 头 + (n-1) 正文 + footer
    assert content_lines[-2] == f"{n - 1}\t{line}"
    assert content_lines[-1] == f"[truncated=true reason=bytes next_offset={n}]"
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    r2 = await run(tool, path="a.txt", offset=n)
    assert r2.content.split("\n")[1] == f"{n}\t{line}"


async def test_utf8_content_intact(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "u.txt").write_text("你好，世界\n🎉 蜡烛 emoji\n", encoding="utf-8")
    r = await run(tool, path="u.txt")
    assert not r.is_error
    assert r.content == expected_content(
        ws.root / "u.txt",
        1,
        2,
        2,
        ["1\t你好，世界", "2\t🎉 蜡烛 emoji"],
        "[truncated=false next_offset=none]",
    )


async def test_bom_is_stripped_and_content_intact(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "bom.txt").write_bytes(b"\xef\xbb\xbf" + "hello\n".encode("utf-8"))
    r = await run(tool, path="bom.txt")
    assert not r.is_error
    assert "\n1\thello" in r.content
    assert "\xef\xbb\xbf" not in r.content


async def test_crlf_lines_have_no_cr(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "c.txt").write_bytes(b"a\r\nb\r\n")
    r = await run(tool, path="c.txt")
    assert not r.is_error
    assert "\r" not in r.content


async def test_missing_file(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, path="nope.txt")
    assert r.is_error
    assert "not found" in r.content


async def test_binary_rejected(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "nul.bin").write_bytes(b"ab\x00cd")
    r = await run(tool, path="nul.bin")
    assert r.is_error
    assert "binary" in r.content
    (ws.root / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    r2 = await run(tool, path="img.png")
    assert r2.is_error
    assert "binary" in r2.content


async def test_text_starting_with_bm_is_not_binary(tmp_path):
    """2 字节 BM 前缀不足以判定 BMP, 不得误伤以 BM 开头的文本 (审查 F2)."""
    ws, tool = make(tmp_path)
    (ws.root / "bm.txt").write_text("BMW is a car brand\n", encoding="utf-8")
    r = await run(tool, path="bm.txt")
    assert not r.is_error
    assert "\n1\tBMW is a car brand\n" in r.content


async def test_single_oversize_line_truncated_with_hint(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "long.txt").write_text("y" * 60000, encoding="utf-8")
    r = await run(tool, path="long.txt")
    assert not r.is_error
    assert r.details["truncated"] is True
    assert r.details["truncation_reason"] == "bytes"
    assert r.details["next_offset"] is None
    assert r.details["partial_last_line"] is True
    assert r.details["start_line"] == 1
    assert r.details["end_line"] == 1
    content_lines = r.content.split("\n")
    assert content_lines[-1] == (
        "[truncated=true reason=bytes partial_last_line=true next_offset=none]"
    )
    assert content_lines[1].startswith("1\t")
    assert content_lines[2].startswith("[line 1 truncated: ")
    assert "powershell" in content_lines[2]
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert len(r.content.encode("utf-8")) < 60000


async def test_total_content_within_50k_for_wide_file(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "wide.txt"
    f.write_text("\n".join("宽" * 200 for _ in range(2000)) + "\n", encoding="utf-8")
    r = await run(tool, path="wide.txt")
    assert not r.is_error
    assert r.details["truncation_reason"] == "bytes"
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES


async def test_metadata_alone_over_cap_is_explicit_error(tmp_path, monkeypatch):
    """元信息自身放不进 50 KiB 上限时返回明确工具错误, 而不是悄悄截断."""
    monkeypatch.setattr(tool_read_module, "MAX_CONTENT_BYTES", 128)
    ws, tool = make(tmp_path)
    (ws.root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert r.is_error
    assert "128 byte output cap" in r.content
