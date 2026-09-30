import asyncio

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.tool_edit import EditTool
from cicada.plugins.coding.tool_write import WriteTool
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, WriteTool(ws), EditTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


async def test_creates_with_nested_parents(tmp_path):
    ws, tool, _ = make(tmp_path)
    r = await run(tool, path="deep\\nest\\dir\\f.txt", content="hello")
    assert not r.is_error
    f = ws.root / "deep" / "nest" / "dir" / "f.txt"
    assert f.read_text(encoding="utf-8") == "hello"


async def test_overwrites_existing(tmp_path):
    ws, tool, _ = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"\xef\xbb\xbfold\r\n")  # 原 BOM + CRLF, 覆盖后不继承
    r = await run(tool, path="a.txt", content="new\n")
    assert not r.is_error
    assert f.read_bytes() == b"new\n"


async def test_utf8_roundtrip(tmp_path):
    ws, tool, _ = make(tmp_path)
    content = "第一行 🎉\nsecond line\n"
    r = await run(tool, path="u.txt", content=content)
    assert not r.is_error
    assert (ws.root / "u.txt").read_text(encoding="utf-8") == content


async def test_outside_root_rejected(tmp_path):
    ws, tool, _ = make(tmp_path)
    outside = tmp_path / "outside.txt"
    r = await run(tool, path=str(outside), content="x")
    assert r.is_error
    assert "outside" in r.content
    assert not outside.exists()
    r2 = await run(tool, path="..\\evil.txt", content="x")
    assert r2.is_error
    assert not (tmp_path / "evil.txt").exists()


async def test_edit_after_write_reads_new_content(tmp_path):
    ws, write_tool, edit_tool = make(tmp_path)
    r1 = await run(write_tool, path="a.txt", content="v1\nv2\nv3\n")
    r2 = await run(edit_tool, path="A.TXT", edits=[{"old_text": "v2", "new_text": "V2"}])
    assert not r1.is_error and not r2.is_error
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "v1\nV2\nv3\n"


async def test_concurrent_write_and_edit_serialized_no_corruption(tmp_path):
    ws, write_tool, edit_tool = make(tmp_path)
    (ws.root / "a.txt").write_text("v1\nv2\nv3\n", encoding="utf-8")
    results = await asyncio.gather(
        run(write_tool, path="a.txt", content="v1\nv2\nv3\nv4\n"),
        run(edit_tool, path="A.TXT", edits=[{"old_text": "v2", "new_text": "V2"}]),
    )
    assert all(not r.is_error for r in results), results
    final = (ws.root / "a.txt").read_text(encoding="utf-8")
    # 队列串行: 终态必为两种合法串行结果之一, 无交错损坏
    assert final in ("v1\nv2\nv3\nv4\n", "v1\nV2\nv3\nv4\n")
