import asyncio
import json
from pathlib import Path

import pytest

from cicada.core.cancel import CancelToken
from cicada.core.ports import ToolContext
from cicada.plugins.coding.tool_edit import MAX_CONTENT_BYTES, EditTool
from cicada.plugins.coding.tool_read import ReadTool
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace

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
    assert r.details["diff_truncated"] is False
    assert "full_diff_path" not in r.details
    assert "-two" in r.details["diff"]
    assert "+TWO" in r.details["diff"]
    assert apply_hunks("one\ntwo\nthree\n", r.details["diff"]) == "one\nTWO\nthree\n"
    # 模型可见 content: applied/路径/替换数/first_changed_line 在前, diff 在后
    lines = r.content.split("\n")
    assert lines[0] == "edit applied: a.txt, 1 replacement(s), first changed line 2"
    assert lines[1].startswith("--- a/a.txt")
    assert lines[2].startswith("+++ b/a.txt")
    assert "@@" in lines[3]
    assert "-two" in r.content and "+TWO" in r.content
    assert len(r.content.encode("utf-8")) <= MAX_CONTENT_BYTES


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
    assert r.content.startswith("edit applied: big.txt, 1 replacement(s)")
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
    monkeypatch.setattr(tool, "_diff", lambda path, original, applied: ("", 1))
    r = await run(tool, path="a.txt", edits=[{"old_text": "one", "new_text": "two"}])
    assert not r.is_error
    assert r.content.startswith("edit applied: a.txt, 1 replacement(s)")
    assert "no text differences" in r.content
    assert r.details["diff_truncated"] is False


async def test_spilled_artifact_paths_are_unique(tmp_path):
    ws, tool = make(tmp_path)
    f, edits, _ = big_edit_args(ws)
    first = await run(tool, path="big.txt", edits=edits)
    # 反向再改一次, 同样产生超大 diff
    revert = [{"old_text": e["new_text"], "new_text": e["old_text"]} for e in edits]
    second = await run(tool, path="big.txt", edits=revert)
    assert first.details["full_diff_path"] != second.details["full_diff_path"]
    assert len(list(ws.output_dir.iterdir())) == 2


async def test_long_header_keeps_success_content_bounded_and_artifact_readable(tmp_path, monkeypatch):
    ws, tool = make(tmp_path)
    f, edits, expected_text = big_edit_args(ws)
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
