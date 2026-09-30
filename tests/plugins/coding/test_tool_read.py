from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.tool_read import MAX_LINES, ReadTool
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, ReadTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


async def test_basic_slice_and_prefix(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    r = await run(tool, path="a.txt")
    assert not r.is_error
    assert r.content == "1\tone\n2\ttwo\n3\tthree"
    assert r.details == {
        "total_lines": 3,
        "truncated": False,
        "next_offset": None,
        "truncation_reason": None,
    }
    r2 = await run(tool, path="a.txt", offset=2, limit=1)
    assert r2.content == "2\ttwo"


async def test_absolute_path_outside_root_is_readable(tmp_path):
    ws, tool = make(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("ext\n", encoding="utf-8")
    r = await run(tool, path=str(outside))
    assert not r.is_error
    assert r.content == "1\text"


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
    content_lines = r.content.split("\n")
    assert len(content_lines) == MAX_LINES
    assert content_lines[0] == "1\tL1"
    assert content_lines[-1] == f"{MAX_LINES}\tL{MAX_LINES}"
    r2 = await run(tool, path="a.txt", offset=MAX_LINES + 1)
    assert r2.details["truncated"] is False
    tail_lines = r2.content.split("\n")
    assert tail_lines[0] == f"{MAX_LINES + 1}\tL{MAX_LINES + 1}"
    assert len(tail_lines) == 500


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
    assert len(content_lines) == n - 1  # 停在该行之前, 无半行
    assert content_lines[-1] == f"{n - 1}\t{line}"
    assert len(r.content.encode("utf-8")) <= 50 * 1024 + 200
    r2 = await run(tool, path="a.txt", offset=n)
    assert r2.content.split("\n")[0] == f"{n}\t{line}"


async def test_utf8_content_intact(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "u.txt").write_text("你好，世界\n🎉 蜡烛 emoji\n", encoding="utf-8")
    r = await run(tool, path="u.txt")
    assert not r.is_error
    assert r.content == "1\t你好，世界\n2\t🎉 蜡烛 emoji"


async def test_crlf_lines_have_no_cr(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "c.txt").write_bytes(b"a\r\nb\r\n")
    r = await run(tool, path="c.txt")
    assert not r.is_error
    assert r.content == "1\ta\n2\tb"
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
    assert r.content == "1\tBMW is a car brand"


async def test_single_oversize_line_truncated_with_hint(tmp_path):
    ws, tool = make(tmp_path)
    (ws.root / "long.txt").write_text("y" * 60000, encoding="utf-8")
    r = await run(tool, path="long.txt")
    assert not r.is_error
    assert r.details["truncated"] is True
    assert r.details["truncation_reason"] == "bytes"
    assert r.details["next_offset"] is None
    assert "truncated" in r.content
    assert "powershell" in r.content
    assert len(r.content.encode("utf-8")) < 60000
