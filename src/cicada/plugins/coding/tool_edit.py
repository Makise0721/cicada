"""edit 工具: 批量精确替换, 任一编辑非法则整体不写盘; 经同文件队列串行."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

BOM = b"\xef\xbb\xbf"


class _EditInvalid(Exception):
    """可预期编辑失败 (由 execute 映射为 is_error 结果, 不向内核抛出)."""


class EditTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit",
            description=(
                "对文件做批量精确替换. 全部 old_text 在原始内容上匹配且必须唯一; "
                "任一编辑非法则整体不写入."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"},
                            },
                            "required": ["old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        edits = arguments["edits"]
        try:
            path = self._workspace.resolve_for_write(arguments["path"])
        except PathNotAllowedError as exc:
            return self._error(ctx, str(exc))
        if not path.exists():
            return self._error(ctx, f"file not found: {path}")

        try:
            diff, first_changed_line = await self._workspace.mutate(
                path, lambda: self._apply(path, edits)
            )
        except _EditInvalid as exc:
            return self._error(ctx, str(exc))
        except UnicodeDecodeError:
            return self._error(ctx, f"file is not valid UTF-8 text: {path}")
        return ToolResult(
            call_id=ctx.call_id,
            name="edit",
            content=f"edited {self._display(path)}: {len(edits)} replacement(s), first changed line {first_changed_line}",
            details={"diff": diff, "first_changed_line": first_changed_line},
        )

    async def _apply(self, path, edits: list[dict[str, str]]) -> tuple[str, int]:
        """读-验证-改-写 原子地跑在同文件队列内, 避免并发丢更新."""
        data = path.read_bytes()
        has_bom = data.startswith(BOM)
        try:
            text = (data[3:] if has_bom else data).decode("utf-8")
        except UnicodeDecodeError:
            raise

        # 恢复为文件首个换行风格: 取最早出现的换行字符, \r 后紧跟 \n 才算 CRLF;
        # 混合换行文件整体归一为首个风格 (既定语义)
        p_cr = text.find("\r")
        p_lf = text.find("\n")
        if p_cr == -1 and p_lf == -1:
            newline = "\n"
        elif p_lf == -1 or (p_cr != -1 and p_cr < p_lf):
            newline = "\r\n" if text.startswith("\r\n", p_cr) else "\r"
        else:
            newline = "\n"
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        spans: list[tuple[int, int, str]] = []
        for index, edit in enumerate(edits):
            old = edit["old_text"].replace("\r\n", "\n").replace("\r", "\n")
            new = edit["new_text"].replace("\r\n", "\n").replace("\r", "\n")
            if old == "":
                raise _EditInvalid(f"edit {index}: old_text must not be empty")
            if new == old:
                raise _EditInvalid(f"edit {index}: new_text is identical to old_text")
            count = normalized.count(old)
            if count == 0:
                raise _EditInvalid(f"edit {index}: old_text not found in {path}")
            if count > 1:
                raise _EditInvalid(
                    f"edit {index}: old_text matches {count} times in {path}; make it unique"
                )
            start = normalized.find(old)
            spans.append((start, start + len(old), new))

        spans.sort(key=lambda span: span[0])
        for (s1, e1, _), (s2, _, _) in zip(spans, spans[1:]):
            if s2 < e1:
                raise _EditInvalid(f"edits overlap between offsets {s1}..{e1} and {s2}; rejected")

        applied = normalized
        for start, end, new in reversed(spans):
            applied = applied[:start] + new + applied[end:]

        out_text = applied.replace("\n", newline) if newline != "\n" else applied
        path.write_bytes((BOM if has_bom else b"") + out_text.encode("utf-8"))

        return self._diff(path, normalized, applied)

    def _diff(self, path, original: str, applied: str) -> tuple[str, int]:
        old_lines = original.split("\n")
        new_lines = applied.split("\n")
        first_changed = min(len(old_lines), len(new_lines))
        for i, (a, b) in enumerate(zip(old_lines, new_lines)):
            if a != b:
                first_changed = i
                break
        patch = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{self._display(path)}",
            tofile=f"b/{self._display(path)}",
            lineterm="",
        )
        return "\n".join(patch), first_changed + 1

    def _display(self, path) -> str:
        try:
            return str(path.relative_to(self._workspace.root))
        except ValueError:
            return str(path)

    @staticmethod
    def _error(ctx: ToolContext, message: str) -> ToolResult:
        return ToolResult(call_id=ctx.call_id, name="edit", content=message, is_error=True)


def edit_plugin() -> PluginDefinition:
    """tool-edit 插件: 依赖 coding.workspace, 提供 tool.edit."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        ctx.provide("tool.edit", EditTool(workspace))

    return PluginDefinition(
        name="tool-edit",
        setup=setup,
        provides=frozenset({"tool.edit"}),
        requires=frozenset({"coding.workspace"}),
    )
