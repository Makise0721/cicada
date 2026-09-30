import asyncio

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.tool_edit import EditTool
from cicada.plugins.coding.workspace import Workspace

BOM = b"\xef\xbb\xbf"


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, EditTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


def apply_hunks(original: str, patch: str) -> str:
    lines = original.split("\n")
    result: list[str] = []
    i = 0  # 已消费的原始行数
    for line in patch.split("\n"):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            old_start = int(line.split()[1].lstrip("-").split(",")[0])
            result.extend(lines[i : old_start - 1])
            i = old_start - 1
            continue
        if line.startswith("-"):
            assert lines[i] == line[1:], f"patch mismatch at original line {i + 1}"
            i += 1
        elif line.startswith("+"):
            result.append(line[1:])
        else:
            assert lines[i] == line[1:], f"patch context mismatch at original line {i + 1}"
            result.append(line[1:] if len(line) > 1 else "")
            i += 1
    result.extend(lines[i:])
    return "\n".join(result)


async def test_single_replacement(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    r = await run(tool, path="a.txt", edits=[{"old_text": "two", "new_text": "TWO"}])
    assert not r.is_error
    assert f.read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert r.details["first_changed_line"] == 2
    assert "-two" in r.details["diff"]
    assert "+TWO" in r.details["diff"]
    assert apply_hunks("one\ntwo\nthree\n", r.details["diff"]) == "one\nTWO\nthree\n"


async def test_batch_non_overlapping(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("aa\nbb\ncc\ndd\n", encoding="utf-8")
    r = await run(
        tool,
        path="a.txt",
        edits=[
            {"old_text": "aa", "new_text": "AA"},
            {"old_text": "cc", "new_text": "CC"},
        ],
    )
    assert not r.is_error
    assert f.read_text(encoding="utf-8") == "AA\nbb\nCC\ndd\n"
    assert r.details["first_changed_line"] == 1
    assert apply_hunks("aa\nbb\ncc\ndd\n", r.details["diff"]) == "AA\nbb\nCC\ndd\n"


async def test_invalid_edits_rejected_and_file_unchanged(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    original = b"dup\ndup\nabcd\nend\n"
    f.write_bytes(original)

    cases = [
        ({"old_text": "nope", "new_text": "x"}, "not found"),
        ({"old_text": "dup", "new_text": "x"}, "2"),
        ({"old_text": "", "new_text": "x"}, "empty"),
        ({"old_text": "end", "new_text": "end"}, "identical"),
        (
            [
                {"old_text": "abc", "new_text": "X"},
                {"old_text": "bcd", "new_text": "Y"},
            ],
            "overlap",
        ),
    ]
    for edits, expected_fragment in cases:
        if isinstance(edits, dict):
            edits = [edits]
        r = await run(tool, path="a.txt", edits=edits)
        assert r.is_error, f"expected error for {edits}"
        assert expected_fragment in r.content, r.content
        assert f.read_bytes() == original  # 任一非法 → 整体不写盘


async def test_lf_roundtrip_preserves_trailing_no_newline(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"keep1\nold\nkeep2")
    r = await run(tool, path="a.txt", edits=[{"old_text": "old", "new_text": "new"}])
    assert not r.is_error
    assert f.read_bytes() == b"keep1\nnew\nkeep2"


async def test_crlf_preserved_and_lf_old_text_matches(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    r = await run(
        tool,
        path="a.txt",
        edits=[{"old_text": "one\ntwo", "new_text": "1\n2"}],
    )
    assert not r.is_error
    assert f.read_bytes() == b"1\r\n2\r\nthree\r\n"


async def test_bom_preserved(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(BOM + "x\ny\n".encode("utf-8"))
    r = await run(tool, path="a.txt", edits=[{"old_text": "y", "new_text": "z"}])
    assert not r.is_error
    assert f.read_bytes() == BOM + "x\nz\n".encode("utf-8")


async def test_mixed_newline_normalized_to_first_style(tmp_path):
    """混合换行文件按首个换行风格整体归一 (审查 F1 既定语义)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    # 首个换行是 LF: 全文件归一为 LF, 不因后文出现 CRLF 而整体改写为 CRLF
    f.write_bytes(b"a\nb\r\nc\n")
    r = await run(tool, path="a.txt", edits=[{"old_text": "c", "new_text": "C"}])
    assert not r.is_error
    assert f.read_bytes() == b"a\nb\nC\n"
    # 首个换行是裸 CR: 全文件归一为 CR
    f.write_bytes(b"a\rb\nc\r")
    r2 = await run(tool, path="a.txt", edits=[{"old_text": "c", "new_text": "C"}])
    assert not r2.is_error
    assert f.read_bytes() == b"a\rb\rC\r"


async def test_outside_root_rejected(tmp_path):
    ws, tool = make(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"one\n")
    r = await run(
        tool, path=str(outside), edits=[{"old_text": "one", "new_text": "x"}]
    )
    assert r.is_error
    assert "outside" in r.content
    assert outside.read_bytes() == b"one\n"
    r2 = await run(tool, path="..\\evil.txt", edits=[{"old_text": "one", "new_text": "x"}])
    assert r2.is_error
    assert "outside" in r2.content


async def test_missing_file(tmp_path):
    ws, tool = make(tmp_path)
    r = await run(tool, path="nope.txt", edits=[{"old_text": "a", "new_text": "b"}])
    assert r.is_error
    assert "not found" in r.content


async def test_concurrent_edits_no_lost_update(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    results = await asyncio.gather(
        run(tool, path="a.txt", edits=[{"old_text": "l2", "new_text": "L2"}]),
        run(tool, path="A.TXT", edits=[{"old_text": "l4", "new_text": "L4"}]),
    )
    assert all(not r.is_error for r in results)
    assert f.read_text(encoding="utf-8") == "l1\nL2\nl3\nL4\nl5\n"
