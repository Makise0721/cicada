import asyncio
import ctypes
import json
import os
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding import file_commit_windows
from cicada.plugins.coding.tool_edit import MAX_CONTENT_BYTES, EditTool
from cicada.plugins.coding.tool_read import ReadTool
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace

BOM = b"\xef\xbb\xbf"


def make(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    return ws, EditTool(ws)


async def run(tool, **arguments):
    return await tool.execute(arguments, ToolContext(call_id="c1", cancel=CancelToken()))


async def read_basis(ws: Workspace, path: str):
    """read 前置 fixture: 成功读取以登记版本依据."""
    r = await ReadTool(ws).execute(
        {"path": path}, ToolContext(call_id="rd", cancel=CancelToken())
    )
    assert not r.is_error, r.content


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
    await read_basis(ws, "a.txt")
    r = await run(tool, path="a.txt", edits=[{"old_text": "two", "new_text": "TWO"}])
    assert not r.is_error
    assert f.read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert r.details["first_changed_line"] == 2
    assert r.details["diff_truncated"] is False
    assert r.details["mutation"]["state"] == "committed"
    assert r.details["mutation"]["reason"] == "applied"
    assert "full_diff_path" not in r.details
    assert "-two" in r.details["diff"]
    assert "+TWO" in r.details["diff"]
    assert apply_hunks("one\ntwo\nthree\n", r.details["diff"]) == "one\nTWO\nthree\n"
    # 模型可见 content: [mutation=committed] + applied/路径/替换数/first_changed_line 在前, diff 在后
    lines = r.content.split("\n")
    assert lines[0] == (
        "[mutation=committed] edit applied: a.txt, 1 replacement(s), first changed line 2"
    )
    assert lines[1].startswith("--- a/a.txt")
    assert lines[2].startswith("+++ b/a.txt")
    assert "@@" in lines[3]
    assert "-two" in r.content and "+TWO" in r.content
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES


async def test_batch_non_overlapping(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("aa\nbb\ncc\ndd\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
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
    await read_basis(ws, "a.txt")

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
        assert r.details["mutation"]["state"] == "not_written"
        assert r.details["mutation"]["reason"] == "invalid_edit"
        assert expected_fragment in r.content, r.content
        assert f.read_bytes() == original  # 任一非法 → 整体不写盘


async def test_lf_roundtrip_preserves_trailing_no_newline(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"keep1\nold\nkeep2")
    await read_basis(ws, "a.txt")
    r = await run(tool, path="a.txt", edits=[{"old_text": "old", "new_text": "new"}])
    assert not r.is_error
    assert f.read_bytes() == b"keep1\nnew\nkeep2"


async def test_crlf_preserved_and_lf_old_text_matches(tmp_path):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    await read_basis(ws, "a.txt")
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
    await read_basis(ws, "a.txt")
    r = await run(tool, path="a.txt", edits=[{"old_text": "y", "new_text": "z"}])
    assert not r.is_error
    assert f.read_bytes() == BOM + "x\nz\n".encode("utf-8")


async def test_mixed_newline_normalized_to_first_style(tmp_path):
    """混合换行文件按首个换行风格整体归一 (审查 F1 既定语义)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    # 首个换行是 LF: 全文件归一为 LF, 不因后文出现 CRLF 而整体改写为 CRLF
    f.write_bytes(b"a\nb\r\nc\n")
    await read_basis(ws, "a.txt")
    r = await run(tool, path="a.txt", edits=[{"old_text": "c", "new_text": "C"}])
    assert not r.is_error
    assert f.read_bytes() == b"a\nb\nC\n"
    # 首个换行是裸 CR: 全文件归一为 CR
    f.write_bytes(b"a\rb\nc\r")
    await read_basis(ws, "a.txt")
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
    """同文件并发 edit 经队列串行: 本进程提交会更新依据, 两笔按序生效, 不丢更新."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    results = await asyncio.gather(
        run(tool, path="a.txt", edits=[{"old_text": "l2", "new_text": "L2"}]),
        run(tool, path="A.TXT", edits=[{"old_text": "l4", "new_text": "L4"}]),
    )
    assert all(not r.is_error for r in results), [r.content for r in results]
    assert f.read_text(encoding="utf-8") == "l1\nL2\nl3\nL4\nl5\n"


def big_edit_args(ws: Workspace, count: int = 4000):
    f = ws.root / "big.txt"
    original_lines = [f"line{i:04d}" for i in range(count)]
    new_lines = [f"LINE{i:04d}" for i in range(count)]
    f.write_text("\n".join(original_lines) + "\n", encoding="utf-8")
    return f, [
        {"old_text": "\n".join(original_lines), "new_text": "\n".join(new_lines)}
    ], "\n".join(new_lines) + "\n"


async def test_large_diff_spills_to_artifact_and_is_readable(tmp_path):
    ws, tool = make(tmp_path)
    f, edits, expected_text = big_edit_args(ws)
    await read_basis(ws, "big.txt")
    r = await run(tool, path="big.txt", edits=edits)
    assert not r.is_error
    # 文件已改
    assert f.read_text(encoding="utf-8") == expected_text
    # diff 超预算 → artifact 落盘
    assert r.details["diff_truncated"] is True
    diff_bytes = len(r.details["diff"].encode("utf-8"))
    assert diff_bytes > MAX_CONTENT_BYTES
    artifact = Path(r.details["full_diff_path"])
    assert artifact.parent == ws.output_dir
    assert artifact.exists()
    full = artifact.read_text(encoding="utf-8")
    assert full == r.details["diff"]  # 完整 diff 落盘
    assert "-line0000" in full and "+LINE0000" in full
    # content 有界, 带 diff_truncated=true 与可 read 路径
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert "diff_truncated=true" in r.content
    display = str(artifact.relative_to(ws.root))
    assert display in r.content
    assert "read it with the read tool" in r.content
    assert "--- a/big.txt" in r.content  # 预览含 diff 开头
    # 给模型的路径确实能被 read 工具读回 (read 每页 2000 行, 首页覆盖 diff 的 - 半)
    back = await ReadTool(ws).execute(
        {"path": display}, ToolContext(call_id="c2", cancel=CancelToken())
    )
    assert not back.is_error
    assert "@@ -1,4001 +1,4001 @@" in back.content
    assert "-line0000" in back.content


async def test_artifact_write_failure_still_reports_applied(tmp_path, monkeypatch):
    """文件已改但 artifact 写失败 (junction 越界): 不回滚、不谎称未改."""
    ws, tool = make(tmp_path)
    f, edits, expected_text = big_edit_args(ws)
    await read_basis(ws, "big.txt")
    original_resolve = ws.resolve_for_write

    def guarded(raw: str):
        if ".cicada" in raw:
            raise PathNotAllowedError(
                f"path {raw!r} resolves outside workspace root (junction probe)"
            )
        return original_resolve(raw)

    monkeypatch.setattr(ws, "resolve_for_write", guarded)
    r = await run(tool, path="big.txt", edits=edits)
    assert not r.is_error
    # edit 已生效且未被回滚
    assert f.read_text(encoding="utf-8") == expected_text
    assert r.content.startswith("[mutation=committed] edit applied: big.txt, 1 replacement(s)")
    assert "edit applied; full diff unavailable" in r.content
    assert "re-read the edited file" in r.content
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert r.details["diff_truncated"] is True
    assert "artifact_error" in r.details
    assert "full_diff_path" not in r.details  # 不提供假路径
    # 确实没有 artifact 残留
    assert list(ws.output_dir.iterdir()) == []


async def test_empty_diff_is_stated_explicitly(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    monkeypatch.setattr(tool, "_diff", lambda path, original, applied: ("", 1))
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "two"}])
    assert not r.is_error
    assert r.content.startswith("[mutation=committed] edit applied: a.txt, 1 replacement(s)")
    assert "no text differences" in r.content
    assert r.details["diff_truncated"] is False


async def test_spilled_artifact_paths_are_unique(tmp_path):
    ws, tool = make(tmp_path)
    f, edits, _ = big_edit_args(ws)
    await read_basis(ws, "big.txt")
    first = await run(tool, path="big.txt", edits=edits)
    # 反向再改一次, 同样产生超大 diff
    revert = [{"old_text": e["new_text"], "new_text": e["old_text"]} for e in edits]
    second = await run(tool, path="big.txt", edits=revert)
    assert first.details["full_diff_path"] != second.details["full_diff_path"]
    assert len(list(ws.output_dir.iterdir())) == 2


async def test_long_header_keeps_success_content_bounded_and_artifact_readable(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    f, edits, expected_text = big_edit_args(ws)
    await read_basis(ws, "big.txt")
    display = tool._display
    long_name = "宽🎉" * 10000
    monkeypatch.setattr(tool, "_display", lambda path: long_name if path == f else display(path))
    result = await run(tool, path="big.txt", edits=edits)
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == expected_text
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert result.details["metadata_truncated"] is True
    assert long_name in result.details["full_result_content"]
    assert "edit applied" in result.content and "metadata_truncated=true" in result.content
    artifact = Path(result.details["full_diff_path"])
    assert artifact.read_text(encoding="utf-8") == result.details["diff"]
    assert json.dumps(str(artifact), ensure_ascii=False) in result.content
    back = await ReadTool(ws).execute(
        {"path": str(artifact)}, ToolContext(call_id="c2", cancel=CancelToken())
    )
    assert not back.is_error  # Complete usable path, not a clipped prefix.


async def test_long_artifact_error_is_bounded_after_edit_applied(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    f, edits, expected_text = big_edit_args(ws)
    await read_basis(ws, "big.txt")
    original_resolve = ws.resolve_for_write
    error_text = "失败🎉" * 10000

    def guarded(raw):
        if ".cicada" in raw:
            raise OSError(error_text)
        return original_resolve(raw)

    monkeypatch.setattr(ws, "resolve_for_write", guarded)
    result = await run(tool, path="big.txt", edits=edits)
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == expected_text
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert "edit applied" in result.content and "full diff unavailable" in result.content
    assert result.details["metadata_truncated"] is True
    assert result.details["artifact_error"] == error_text
    assert "full_diff_path" not in result.details
    assert list(ws.output_dir.iterdir()) == []


async def test_empty_diff_with_long_header_is_bounded(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    monkeypatch.setattr(tool, "_display", lambda path: "汉" * 18000)
    monkeypatch.setattr(tool, "_diff", lambda path, original, applied: ("", 1))
    result = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "two"}])
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "two\n"
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert "edit applied" in result.content and "no text differences" in result.content
    assert result.details["metadata_truncated"] is True
    assert result.details["diff_truncated"] is False


def test_wide_path_metadata_fallback_stays_within_byte_cap(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    path = ws.root.joinpath(*(["汉" * 200] * 88), "a.txt")
    monkeypatch.setattr(
        ws, "resolve_for_write",
        lambda raw: (_ for _ in ()).throw(PathNotAllowedError("artifact destination rejected")),
    )
    header = f"edit applied: {tool._display(path)}, 1 replacement(s), first changed line 1"
    result = tool._spill_diff(
        ToolContext(call_id="c1", cancel=CancelToken()), path, header, "-a\n+b",
        {"diff": "-a\n+b", "first_changed_line": 1},
    )
    assert not result.is_error
    assert len(result.content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert result.details["metadata_truncated"] is True
    assert "metadata_truncated=true" in result.content
    assert "edit applied" in result.content and "full diff unavailable" in result.content


# ---- 版本前置与取消 (文件修改可靠性 01 的关键回归) ----


async def test_edit_requires_prior_read(tmp_path):
    """未读过 → read_required, 文件不变 (A02)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "x"}])
    assert r.is_error
    assert r.details["mutation"]["state"] == "not_written"
    assert r.details["mutation"]["reason"] == "read_required"
    assert "read" in r.content
    assert f.read_text(encoding="utf-8") == "one\ntwo\n"


async def test_edit_rejects_external_change_since_read(tmp_path):
    """读后其他区域被外部改动 → version_conflict (局部匹配不足以放行), 重读后可编辑."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("keep\nold\nkeep\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    # 外部改动未读区域: old_text 仍唯一匹配, 但依据已失效
    f.write_text("CHANGED\nold\nkeep\n", encoding="utf-8")
    r = await run(tool, path="a.txt", edits=[{"old_text": "old", "new_text": "new"}])
    assert r.is_error
    assert r.details["mutation"]["reason"] == "version_conflict"
    assert f.read_text(encoding="utf-8") == "CHANGED\nold\nkeep\n"
    await read_basis(ws, "a.txt")
    r2 = await run(tool, path="a.txt", edits=[{"old_text": "old", "new_text": "new"}])
    assert not r2.is_error
    assert f.read_text(encoding="utf-8") == "CHANGED\nnew\nkeep\n"


async def test_edit_rejects_original_deleted_after_read(tmp_path):
    """读到的原件被删除 → 拒绝且不转创建 (A02)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    f.unlink()
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "x"}])
    assert r.is_error
    assert r.details["mutation"]["reason"] == "version_conflict"
    assert not f.exists()


async def test_continuous_edits_update_basis(tmp_path):
    """成功提交并核对相符后依据更新, 连续 edit 无需重新 read (A02)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    r1 = await run(tool, path="a.txt", edits=[{"old_text": "a", "new_text": "A"}])
    assert not r1.is_error
    r2 = await run(tool, path="a.txt", edits=[{"old_text": "b", "new_text": "B"}])
    assert not r2.is_error
    assert f.read_text(encoding="utf-8") == "A\nB\nc\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive path alias")
async def test_case_alias_shares_basis(tmp_path):
    """大小写别名共享同一依据与队列 (A02)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    r = await run(tool, path="A.TXT", edits=[{"old_text": "one", "new_text": "1"}])
    assert not r.is_error
    assert f.read_text(encoding="utf-8") == "1\n"


async def test_sliced_read_registers_full_version_basis(tmp_path):
    """切片读取登记完整字节版本, 不代表模型读完, 但足以作为编辑依据 (A01)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    r = await ReadTool(ws).execute(
        {"path": "a.txt", "offset": 2, "limit": 1},
        ToolContext(call_id="rd", cancel=CancelToken()),
    )
    assert not r.is_error
    # 编辑切片之外的行 1 仍被接受: 依据是整份字节而非所读行
    r2 = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "ONE"}])
    assert not r2.is_error
    assert f.read_text(encoding="utf-8") == "ONE\ntwo\nthree\n"


async def test_cancelled_before_publish_leaves_file_unchanged(tmp_path):
    """令牌已取消 (含排队/核对阶段) → 未提交, 文件不变 (A04)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    token = CancelToken()
    token.cancel()
    r = await tool.execute(
        {"path": "a.txt", "edits": [{"old_text": "one", "new_text": "x"}]},
        ToolContext(call_id="c1", cancel=token),
    )
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "cancelled_before_commit"
    assert mutation["caller_cancelled"] is True
    assert f.read_text(encoding="utf-8") == "one\n"


async def test_cancel_during_publish_returns_committed_result(tmp_path, monkeypatch):
    """原生调用进入后取消不中断发布: 真实 committed 结果被保留, 令牌仍置位 (A04)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    entered = threading.Event()  # 发布运行在 worker 线程
    real_replace = file_commit_windows._ReplaceFileW

    def slow_replace(target, replacement, backup, flags, exclude, reserved):
        entered.set()
        time.sleep(0.25)  # 已进入的不可中断窗口
        return real_replace(target, replacement, backup, flags, exclude, reserved)

    monkeypatch.setattr(file_commit_windows, "_ReplaceFileW", slow_replace)
    token = CancelToken()
    task = asyncio.create_task(
        tool.execute(
            {"path": "a.txt", "edits": [{"old_text": "two", "new_text": "TWO"}]},
            ToolContext(call_id="c1", cancel=token),
        )
    )
    for _ in range(500):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    task.cancel()
    result = await task  # 不抛 CancelledError: 已取得文件结果
    assert not result.is_error
    mutation = result.details["mutation"]
    assert mutation["state"] == "committed"
    assert mutation["reason"] == "applied"
    assert mutation["caller_cancelled"] is True
    assert f.read_text(encoding="utf-8") == "one\nTWO\n"
    assert token.cancelled  # 工具接住取消后仍使既有令牌置位


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junction")
async def test_edit_rejects_junction_alias_path(tmp_path):
    """原始路径穿越 reparse 点 → unsupported_target, 即便 canonical 目标合法 (A06)."""
    ws, tool = make(tmp_path)
    target_dir = ws.root / "real"
    target_dir.mkdir()
    f = target_dir / "a.txt"
    f.write_text("x\n", encoding="utf-8")
    alias = ws.root / "alias"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target_dir)], capture_output=True
    )
    if result.returncode != 0:
        pytest.skip("junction creation unavailable on this host")
    try:
        # read 允许穿透 junction (只读范围保持), 登记的是真实路径依据
        await read_basis(ws, "alias\\a.txt")
        r = await run(tool, path="alias\\a.txt", edits=[{"old_text": "x", "new_text": "y"}])
        assert r.is_error
        assert r.details["mutation"]["reason"] == "unsupported_target"
        assert f.read_text(encoding="utf-8") == "x\n"
        # 用真实路径编辑同一依据 → 成功
        r2 = await run(tool, path="real\\a.txt", edits=[{"old_text": "x", "new_text": "y"}])
        assert not r2.is_error
        assert f.read_text(encoding="utf-8") == "y\n"
    finally:
        alias.rmdir()  # 只删 junction 本身


async def test_queued_cancel_returns_not_written_without_worker(tmp_path, monkeypatch):
    """排队中取消: op 不启动, 返回 not_written 结果而非 CancelledError (A04)."""
    import cicada.plugins.coding.file_access as file_access_module

    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    entered = threading.Event()
    release = threading.Event()  # 发布运行在 worker 线程, 用线程事件
    real_replace = file_commit_windows._ReplaceFileW

    def held_replace(target, replacement, backup, flags, exclude, reserved):
        entered.set()
        release.wait(timeout=5)  # 第一笔持有队列
        return real_replace(target, replacement, backup, flags, exclude, reserved)

    monkeypatch.setattr(file_commit_windows, "_ReplaceFileW", held_replace)
    first = asyncio.create_task(
        tool.execute(
            {"path": "a.txt", "edits": [{"old_text": "one", "new_text": "1"}]},
            ToolContext(call_id="c1", cancel=CancelToken()),
        )
    )
    for _ in range(500):  # 等待第一笔进入发布 (线程事件无 async wait)
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    token2 = CancelToken()
    second = asyncio.create_task(
        tool.execute(
            {"path": "a.txt", "edits": [{"old_text": "one", "new_text": "2"}]},
            ToolContext(call_id="c2", cancel=token2),
        )
    )
    await asyncio.sleep(0.05)  # 第二笔进入队列等待
    second.cancel()  # 排队中取消: 不抢占在途第一笔, 自身未启动
    result = await second
    assert result.is_error
    mutation = result.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "cancelled_before_commit"
    assert mutation["caller_cancelled"] is True
    assert token2.cancelled
    release.set()
    first_result = await asyncio.wait_for(first, timeout=5)
    assert not first_result.is_error
    assert f.read_text(encoding="utf-8") == "1\n"


async def test_publish_error_1176_stays_unknown_with_retained_materials(tmp_path, monkeypatch):
    """无法证明未提交的原生失败 (1176) 保持 unknown, 材料保留, 依据失效 (A06)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")

    def failing_replace(target, replacement, backup):
        raise file_commit_windows.NativePublishError("ReplaceFileW", 1176)

    monkeypatch.setattr(file_commit_windows, "replace_file", failing_replace)
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "x"}])
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "unknown"
    assert mutation["reason"] == "publish_failed"
    assert mutation["winerror"] == 1176
    assert mutation["retained_paths"], "unknown 必须保留材料路径"
    for path in mutation["retained_paths"]:
        assert Path(path).exists()
    assert f.read_text(encoding="utf-8") == "one\n"
    # 依据已失效: 不重读直接再 edit 仍被 read_required 拒绝
    r2 = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "x"}])
    assert r2.is_error
    assert r2.details["mutation"]["reason"] == "read_required"


async def test_committed_current_mismatch_is_error_not_falsely_unwritten(tmp_path, monkeypatch):
    """提交成功但回读不符: committed+current_matches=false, is_error, 不假报未写 (A06)."""
    import cicada.plugins.coding.file_access as fa

    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    real_read_all = fa._read_all
    readback_calls = {"count": 0}

    def tampered_read(path):
        data = real_read_all(path)
        if path.name == "a.txt":  # 目标回读: 模拟提交后被外部改写
            readback_calls["count"] += 1
            if readback_calls["count"] == 1:
                return data + b"external\n"
        return data

    monkeypatch.setattr(fa, "_read_all", tampered_read)
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "TWO"}])
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "committed"
    assert mutation["current_matches"] is False
    assert "[mutation=committed]" in r.content
    assert "no longer matches" in r.content


async def test_reserved_staging_area_rejects_edit(tmp_path):
    """.cicada/mutations 保留区不允许 edit: unknown 保留材料是证据 (C3)."""
    ws, tool = make(tmp_path)
    staged_dir = ws.root / ".cicada" / "mutations" / "deadbeef"
    staged_dir.mkdir(parents=True)
    f = staged_dir / "old.bak"
    f.write_text("evidence\n", encoding="utf-8")
    await read_basis(ws, str(f.relative_to(ws.root)))
    r = await run(tool, path=str(f.relative_to(ws.root)), edits=[{"old_text": "evidence", "new_text": "x"}])
    assert r.is_error
    assert r.details["mutation"]["reason"] == "unsupported_target"
    assert "reserved mutation staging" in r.content
    assert f.read_text(encoding="utf-8") == "evidence\n"


_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value


async def test_locked_target_proves_not_written_with_stat_evidence(tmp_path, monkeypatch):
    """读后加锁注入: 目标不可读时以 size+mtime 旁证判未提交 (01 审查 note-2 交接)."""
    ws, tool = make(tmp_path)
    f = ws.root / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    await read_basis(ws, "a.txt")
    real_replace = file_commit_windows.replace_file
    locked: dict = {}

    def open_exclusive(path):
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        handle = kernel32.CreateFileW(
            str(path), _GENERIC_READ, 0, None, _OPEN_EXISTING, 0, None
        )
        assert handle not in (None, 0, _INVALID_HANDLE), "cannot lock target in test"
        return handle

    def lock_then_replace(target, replacement, backup):
        # 锁在发布前注入并保持到分类完成: 分类期目标不可读, 只能走 stat 级旁证
        locked["handle"] = open_exclusive(target)
        return real_replace(target, replacement, backup)

    monkeypatch.setattr(file_commit_windows, "replace_file", lock_then_replace)
    try:
        r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "TWO"}])
    finally:
        if "handle" in locked:
            ctypes.windll.kernel32.CloseHandle(locked["handle"])
    assert r.is_error
    mutation = r.details["mutation"]
    assert mutation["state"] == "not_written"
    assert mutation["reason"] == "publish_failed"
    assert mutation["winerror"] == file_commit_windows.ERROR_SHARING_VIOLATION
    assert f.read_text(encoding="utf-8") == "one\n"
    staging_root = ws.root / ".cicada" / "mutations"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []

async def test_staging_junction_cannot_publish_or_cleanup_external_material(tmp_path):
    from cicada.plugins.coding.tool_write import WriteTool
    ws, tool = make(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = ws.root / ".cicada" / "mutations"
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(outside)], capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assigned = alias / ("a" * 32)
    evidence = outside / ("a" * 32)
    evidence.mkdir()
    (evidence / "new.tmp").write_bytes(b"keep")
    try:
        result = await run(WriteTool(ws), path="new.txt", content="x")
        assert result.details["mutation"]["reason"] == "unsupported_target"
        assert not (ws.root / "new.txt").exists()
        assert sorted(p.name for p in outside.iterdir()) == ["a" * 32]
        assert ws.files._cleanup_staging(assigned, [assigned / "new.tmp"])
        assert (evidence / "new.tmp").read_bytes() == b"keep"
    finally:
        alias.rmdir()


async def test_diff_failure_and_cleanup_warning_preserve_committed_fact(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    (ws.root / "a.txt").write_bytes(b"one\n")
    await read_basis(ws, "a.txt")
    monkeypatch.setattr(ws.files, "_cleanup_staging", lambda *args: "cleanup denied")
    result = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "two"}])
    assert not result.is_error and "cleanup warning: cleanup denied" in result.content

    def fail_diff(*args):
        raise RuntimeError("presentation failed")

    monkeypatch.setattr(tool, "_diff", fail_diff)
    result = await run(tool, path="a.txt", edits=[{"old_text": "two", "new_text": "three"}])
    assert not result.is_error and result.details["mutation"]["state"] == "committed"
    assert result.content.startswith("[mutation=committed]")
    assert "presentation unavailable" in result.content and "cleanup denied" in result.content
    assert (ws.root / "a.txt").read_bytes() == b"three\n"


async def test_cancel_during_watcher_cleanup_sets_token_and_preserves_result(tmp_path, monkeypatch):
    from cicada.plugins.coding.tool_write import WriteTool
    ws, _ = make(tmp_path)
    token = CancelToken()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_watcher(cancel, on_cancelled):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            entered.set()
            await release.wait()

    monkeypatch.setattr(ws.files, "_watch_cancel_token", slow_watcher)
    task = asyncio.create_task(WriteTool(ws).execute(
        {"path": "new.txt", "content": "x"}, ToolContext(call_id="write", cancel=token),
    ))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert token.cancelled and result.details["mutation"]["caller_cancelled"]
    assert result.details["mutation"]["state"] == "committed"
    assert (ws.root / "new.txt").read_bytes() == b"x"
