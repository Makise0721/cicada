"""read 工具: UTF-8 文本文件行切片读取, head 有界, 二进制拒绝."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

MAX_LINES = 2000
MAX_BYTES = 50 * 1024

_BINARY_MAGIC = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"%PDF",
    b"PK\x03\x04",
    b"RIFF",
)


class ReadTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read",
            description=(
                "读取 UTF-8 文本文件的行切片. path 支持相对 workspace 与绝对路径; "
                "offset 为 1 起始行号; 输出行带行号前缀, 超限时截断并在 details 给出 next_offset 续读."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = arguments["path"]
        path = self._workspace.resolve(raw)
        if not path.exists():
            return self._error(ctx, f"file not found: {path}")
        if path.is_dir():
            return self._error(ctx, f"path is a directory: {path}")
        data = path.read_bytes()
        if b"\x00" in data or any(data.startswith(magic) for magic in _BINARY_MAGIC):
            return self._error(ctx, f"binary content is not readable: {path}")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return self._error(ctx, f"file is not valid UTF-8 text: {path}")

        lines = text.splitlines()
        total_lines = len(lines)
        offset = arguments.get("offset", 1)
        limit = min(arguments.get("limit", MAX_LINES), MAX_LINES)
        if offset > total_lines:
            return self._error(ctx, f"offset {offset} beyond end of file ({total_lines} lines)")

        out_lines: list[str] = []
        used = 0
        truncated = False
        reason: str | None = None
        next_offset: int | None = None
        index = offset - 1
        end = min(offset - 1 + limit, total_lines)
        while index < end:
            formatted = f"{index + 1}\t{lines[index]}"
            need = len(formatted.encode("utf-8")) + 1
            if used + need <= MAX_BYTES:
                out_lines.append(formatted)
                used += need
                index += 1
                continue
            truncated = True
            reason = "bytes"
            if used == 0:
                # 单行自身超预算: 截断该行内容并标注 (标注不计入预算)
                prefix = f"{index + 1}\t"
                available = MAX_BYTES - 1 - len(prefix.encode("utf-8"))
                carved_bytes = lines[index].encode("utf-8")[:available]
                carved = carved_bytes.decode("utf-8", errors="ignore")
                omitted = len(lines[index].encode("utf-8")) - len(carved.encode("utf-8"))
                out_lines.append(prefix + carved)
                out_lines.append(
                    f"[line {index + 1} truncated: {omitted} bytes omitted; "
                    f"use the powershell tool to read byte ranges of this file]"
                )
                next_offset = None
                return self._view(ctx, out_lines, total_lines, True, None, "bytes")
            next_offset = index + 1
            break
        if not truncated and index < total_lines:
            truncated = True
            reason = "lines"
            next_offset = index + 1
        return self._view(ctx, out_lines, total_lines, truncated, next_offset, reason)

    @staticmethod
    def _view(
        ctx: ToolContext,
        out_lines: list[str],
        total_lines: int,
        truncated: bool,
        next_offset: int | None,
        reason: str | None,
    ) -> ToolResult:
        return ToolResult(
            call_id=ctx.call_id,
            name="read",
            content="\n".join(out_lines),
            details={
                "total_lines": total_lines,
                "truncated": truncated,
                "next_offset": next_offset,
                "truncation_reason": reason,
            },
        )

    @staticmethod
    def _error(ctx: ToolContext, message: str) -> ToolResult:
        return ToolResult(call_id=ctx.call_id, name="read", content=message, is_error=True)


def read_plugin() -> PluginDefinition:
    """tool-read 插件: 依赖 coding.workspace, 提供 tool.read."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        ctx.provide("tool.read", ReadTool(workspace))

    return PluginDefinition(
        name="tool-read",
        setup=setup,
        provides=frozenset({"tool.read"}),
        requires=frozenset({"coding.workspace"}),
    )
