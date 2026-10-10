import asyncio
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding import file_commit_windows as native
from cicada.plugins.coding.tool_edit import EditTool
from cicada.plugins.coding.tool_read import ReadTool
from cicada.plugins.coding.tool_write import WriteTool
from cicada.plugins.coding.workspace import Workspace


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, WriteTool(ws), EditTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


async def read_basis(ws, path):
    r = await ReadTool(ws).execute({"path": path}, ToolContext(call_id="rd", cancel=CancelToken()))
    assert not r.is_error, r.content


async def test_creates_with_nested_parents(tmp_path):
    ws, tool, _ = make(tmp_path)
    r = await run(tool, path="deep\\nest\\dir\\f.txt", content="hello")
    assert not r.is_error
    f = ws.root / "deep" / "nest" / "dir" / "f.txt"
    assert f.read_text(encoding="utf-8") == "hello"
    mutation = r.details["mutation"]
    assert mutation["state"] == "committed"
    assert mutation["current_matches"] is True
    assert "[mutation=committed]" in r.content and "created" in r.content


async def test_overwrites_existing_requires_read_first(tmp_path):
    ws, tool, _ = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"\xef\xbb\xbfold\r\n")  # 原 BOM + CRLF, 覆盖后不继承
    r = await run(tool, path="a.txt", content="new\n")
    assert r.is_error  # 已有文件未读: 拒绝且目标不变
    assert r.details["mutation"]["reason"] == "read_required"
    assert f.read_bytes() == b"\xef\xbb\xbfold\r\n"
    await read_basis(ws, "a.txt")
    r2 = await run(tool, path="a.txt", content="new\n")
    assert not r2.is_error
    assert f.read_bytes() == b"new\n"
    assert "wrote" in r2.content


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


async def test_write_establishes_basis_for_edit(tmp_path):
    """write 成功后依据更新: 经大小写别名直接 edit, 无需再 read (A05)."""
    ws, write_tool, edit_tool = make(tmp_path)
    r1 = await run(write_tool, path="a.txt", content="v1\nv2\nv3\n")
    assert not r1.is_error
    r2 = await run(edit_tool, path="A.TXT", edits=[{"old_text": "v2", "new_text": "V2"}])
    assert not r2.is_error
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "v1\nV2\nv3\n"


async def test_concurrent_write_and_edit_serialized_no_corruption(tmp_path):
    ws, write_tool, edit_tool = make(tmp_path)
    (ws.root / "a.txt").write_text("v1\nv2\nv3\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    results = await asyncio.gather(
        run(write_tool, path="a.txt", content="v1\nv2\nv3\nv4\n"),
        run(edit_tool, path="A.TXT", edits=[{"old_text": "v2", "new_text": "V2"}]),
    )
    # 队列串行, 无交错损坏; 先入队者成功并更新依据, 后入队者在新依据上继续
    write_result, edit_result = results
    assert not write_result.is_error
    assert not edit_result.is_error
    final = (ws.root / "a.txt").read_text(encoding="utf-8")
    assert final in ("v1\nv2\nv3\nv4\n", "v1\nV2\nv3\nv4\n")


async def test_create_conflict_does_not_overwrite(tmp_path, monkeypatch):
    """发布前同名竞争: 真实 MoveFileExW flags=0 拒绝覆盖, 判 not_written/create_conflict (A05)."""
    ws, tool, _ = make(tmp_path)
    target = ws.root / "race.txt"
    real_create = native.create_file

    def competitor_then_create(t, staged):
        target.write_bytes(b"raced")  # 竞争者在发布前抢建同名目标
        real_create(t, staged)        # 真实系统调用: 同名失败, 不覆盖

    monkeypatch.setattr(native, "create_file", competitor_then_create)
    r = await run(tool, path="race.txt", content="new")
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "create_conflict"
    assert mutation["winerror"] in (native.ERROR_FILE_EXISTS, native.ERROR_ALREADY_EXISTS)
    assert target.read_bytes() == b"raced"  # 不切换为覆盖
    assert mutation["retained_paths"] == []  # 新件已核对, staging 已清理
    staging_root = ws.root / ".cicada" / "mutations"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []
    # 冲突后不自动转覆盖: 仍需先 read
    r2 = await run(tool, path="race.txt", content="again")
    assert r2.is_error
    assert r2.details["mutation"]["reason"] == "read_required"


async def test_reserved_staging_area_rejects_write(tmp_path):
    """.cicada/mutations 保留区不允许 write (C3)."""
    ws, tool, _ = make(tmp_path)
    staged = ws.root / ".cicada" / "mutations" / "deadbeef" / "evidence.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("evidence\n", encoding="utf-8")
    r = await run(tool, path=str(staged.relative_to(ws.root)), content="x")
    assert r.is_error
    assert r.details["mutation"]["reason"] == "unsupported_target"
    assert "reserved mutation staging" in r.content
    assert staged.read_text(encoding="utf-8") == "evidence\n"


async def test_cancelled_write_creates_nothing(tmp_path):
    """取消的 write: 未提交事实与 caller_cancelled 真实可见 (A04)."""
    ws, tool, _ = make(tmp_path)
    token = CancelToken()
    token.cancel()
    r = await tool.execute(
        {"path": "c.txt", "content": "x"}, ToolContext(call_id="c9", cancel=token)
    )
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "cancelled_before_commit"
    assert mutation["caller_cancelled"] is True
    assert not (ws.root / "c.txt").exists()


async def test_write_after_deletion_refuses_then_can_recreate(tmp_path):
    """读到的原件被删除: 本次拒绝且不转创建; 失效后独立 write 可新建 (A05/C1)."""
    ws, tool, _ = make(tmp_path)
    f = ws.root / "d.txt"
    f.write_text("v1\n", encoding="utf-8")
    await read_basis(ws, "d.txt")
    f.unlink()
    r = await run(tool, path="d.txt", content="v2\n")
    assert r.is_error
    assert r.details["mutation"]["reason"] == "version_conflict"
    assert not f.exists()
    r2 = await run(tool, path="d.txt", content="v2\n")
    assert not r2.is_error
    assert f.read_text(encoding="utf-8") == "v2\n"


async def test_creation_path_change_rejected(tmp_path, monkeypatch):
    """发布前路径绑定变化: 创建分支同样拒绝并清理暂存 (A06)."""
    ws, tool, _ = make(tmp_path)
    real = native.reparse_ancestor
    calls = {"n": 0}

    def flip_after_target_checks(path):
        # 放行创建支持性核对 (original+canonical 两次), 只在发布前重核对时触发
        calls["n"] += 1
        return None if calls["n"] <= 2 else f"{path}\\junction"

    monkeypatch.setattr(native, "reparse_ancestor", flip_after_target_checks)
    assert real(ws.root / "p.txt") is None  # 实际路径无 reparse, 注入只控分支
    r = await run(tool, path="p.txt", content="x")
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "version_conflict"
    assert not (ws.root / "p.txt").exists()
    staging_root = ws.root / ".cicada" / "mutations"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []


async def test_creation_unknown_failure_retains_materials(tmp_path, monkeypatch):
    """无法证明的创建失败: unknown 保留材料, 不直接写或重试 (A06)."""
    ws, tool, _ = make(tmp_path)

    def fail(t, staged):
        raise native.NativePublishError("MoveFileExW", native.ERROR_ACCESS_DENIED)

    monkeypatch.setattr(native, "create_file", fail)
    r = await run(tool, path="u.txt", content="x")
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "unknown"
    assert mutation["reason"] == "publish_failed"
    assert mutation["winerror"] == native.ERROR_ACCESS_DENIED
    assert mutation["retained_paths"]
    for retained in mutation["retained_paths"]:
        assert Path(retained).exists()
    assert not (ws.root / "u.txt").exists()
    assert "do not assume" in r.content
