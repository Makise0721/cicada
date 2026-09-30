"""read 工具: UTF-8 文本文件行切片读取, 来源头/续读 footer 有界, 二进制拒绝."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

MAX_LINES = 2000
MAX_CONTENT_BYTES = 50 * 1024  # 头/正文/footer 全部内容的 UTF-8 总预算

_BINARY_MAGIC = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"%PDF",
    b"PK\x03\x04",
    b"RIFF",
)  # 不含 2 字节的 BMP "BM" 前缀: 不足以与以 BM 开头的文本区分

_NO_TRUNC_FOOTER = "[truncated=false next_offset=none]"
_PARTIAL_FOOTER = "[truncated=true reason=bytes partial_last_line=true next_offset=none]"


class ReadTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read",
            description=(
                "读取 UTF-8 文本文件的行切片. path 支持相对 workspace 与绝对路径; "
                "offset 为 1 起始行号. 输出以来源头开始 "
                "(canonical 绝对路径/实际行范围/总行数), 正文行带行号前缀, "
                "结尾 footer 给出截断状态、原因与 next_offset 续读; "
                "最终输出 (头/正文/footer) 不超过 50 KiB UTF-8, 超长单行只返回部分字符 "
                "并以 partial 标记 + powershell 字节读取提示说明."
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

        source = json.dumps(str(path), ensure_ascii=False)
        planned_end = min(offset - 1 + limit, total_lines)
        # 头按最大可能行尾预算, 实际行号位数只会更短; footer 按各分支最长形态预算
        header = f"[file={source} lines={offset}-{planned_end} total_lines={total_lines}]"
        footer_cap = max(
            len(_NO_TRUNC_FOOTER.encode("utf-8")),
            len(f"[truncated=true reason=lines next_offset={total_lines + 1}]".encode("utf-8")),
            len(f"[truncated=true reason=bytes next_offset={total_lines + 1}]".encode("utf-8")),
            len(_PARTIAL_FOOTER.encode("utf-8")),
        )
        body_budget = MAX_CONTENT_BYTES - len(header.encode("utf-8")) - 2 - footer_cap
        if body_budget <= 0:
            return self._metadata_overflow(ctx, path)

        out_lines: list[str] = []
        used = 0  # Σ(行 UTF-8 字节 + 1 换行)
        truncated = False
        reason: str | None = None
        next_offset: int | None = None
        partial_last_line = False
        index = offset - 1
        while index < planned_end:
            formatted = f"{index + 1}\t{lines[index]}"
            need = len(formatted.encode("utf-8")) + 1
            if used + need <= body_budget:
                out_lines.append(formatted)
                used += need
                index += 1
                continue
            truncated = True
            reason = "bytes"
            if used == 0:
                # 单行自身超预算: 截断该行内容并标注 (UTF-8 安全截断, 不拆字符)
                line_bytes = lines[index].encode("utf-8")
                prefix = f"{index + 1}\t"
                digits = len(str(len(line_bytes)))
                marker = (
                    f"[line {index + 1} truncated: {'9' * digits} bytes omitted; "
                    f"use the powershell tool to read byte ranges of this file]"
                )
                fixed = len(prefix.encode("utf-8")) + 1 + len(marker.encode("utf-8")) + 1
                available = body_budget - fixed
                if available < 0:
                    return self._metadata_overflow(ctx, path)
                carved = line_bytes[:available].decode("utf-8", errors="ignore")
                omitted = len(line_bytes) - len(carved.encode("utf-8"))
                out_lines.append(prefix + carved)
                out_lines.append(
                    f"[line {index + 1} truncated: {omitted} bytes omitted; "
                    f"use the powershell tool to read byte ranges of this file]"
                )
                partial_last_line = True
                next_offset = None
                index += 1
            else:
                next_offset = index + 1
            break
        if not truncated and index < total_lines:
            truncated = True
            reason = "lines"
            next_offset = index + 1

        end_line = index  # 下一条未读行的 0 基序号即末条已读行的 1 基行号
        header = f"[file={source} lines={offset}-{end_line} total_lines={total_lines}]"
        if not truncated:
            footer = _NO_TRUNC_FOOTER
        elif partial_last_line:
            footer = _PARTIAL_FOOTER
        else:
            footer = f"[truncated=true reason={reason} next_offset={next_offset}]"
        content = "\n".join([header, *out_lines, footer])
        return ToolResult(
            call_id=ctx.call_id,
            name="read",
            content=content,
            details={
                "total_lines": total_lines,
                "truncated": truncated,
                "next_offset": next_offset,
                "truncation_reason": reason,
                "path": str(path),
                "start_line": offset,
                "end_line": end_line,
                "partial_last_line": partial_last_line,
            },
        )

    @staticmethod
    def _metadata_overflow(ctx: ToolContext, path) -> ToolResult:
        return ReadTool._error(
            ctx,
            f"read result for {path} exceeds the {MAX_CONTENT_BYTES} byte output cap "
            f"before any content: source metadata alone does not fit",
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
