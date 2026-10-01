"""edit 工具: 批量精确替换, 任一编辑非法则整体不写盘; 成功结果附有界 diff."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

BOM = b"\xef\xbb\xbf"
MAX_CONTENT_BYTES = 50 * 1024  # applied 头 + diff 的 UTF-8 总预算


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
                "任一编辑非法则整体不写入. 成功结果先给出 applied/路径/替换数/"
                "first_changed_line, 再附 unified diff; 总输出不超过 50 KiB UTF-8, "
                "超限时完整 diff 写入 workspace 的 .cicada/outputs/ 唯一文件, "
                "结果标 diff_truncated=true 并给出可 read 的路径."
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
        header = (
            f"edit applied: {self._display(path)}, {len(edits)} replacement(s), "
            f"first changed line {first_changed_line}"
        )
        details = {"diff": diff, "first_changed_line": first_changed_line}
        if diff == "":
            # 语义上罕见 (new==old 已在校验拒绝), 仍明示而不是静默
            return self._success_result(
                ctx, f"{header}\n(no text differences)",
                {**details, "diff_truncated": False},
            )
        if len(header.encode("utf-8")) + 1 + len(diff.encode("utf-8")) <= MAX_CONTENT_BYTES:
            return self._success_result(
                ctx, f"{header}\n{diff}", {**details, "diff_truncated": False},
            )
        return self._spill_diff(ctx, path, header, diff, details)

    def _spill_diff(
        self, ctx: ToolContext, path: Path, header: str, diff: str, details: dict
    ) -> ToolResult:
        """diff 超预算: 完整 diff 落盘 .cicada/outputs, content 给有界预览."""
        artifact: Path | None = None
        artifact_error: str | None = None
        try:
            candidate = self._workspace.new_output_file(f"edit-diff-{self._display(path)}")
            # 写入前再经 resolve_for_write 校验 canonical 目标, 防 .cicada/outputs
            # 被 junction 指到工作区外 (P3 C2)
            target = self._workspace.resolve_for_write(str(candidate))
            target.write_bytes(diff.encode("utf-8"))
            artifact = target
        except Exception as exc:  # artifact 失败不回滚 edit; 取消 (BaseException) 仍传播
            artifact_error = str(exc)
        header_bytes = len(header.encode("utf-8"))
        if artifact is not None:
            display_path = self._display(artifact)
            marker = (
                f"[diff_truncated=true full diff of {len(diff.encode('utf-8'))} bytes "
                f"written to {display_path}; read it with the read tool]"
            )
            preview = self._bounded_preview(
                diff, MAX_CONTENT_BYTES - header_bytes - 2 - len(marker.encode("utf-8"))
            )
            parts = [header, preview, marker] if preview else [header, marker]
            return self._success_result(
                ctx, "\n".join(parts),
                {**details, "diff_truncated": True, "full_diff_path": str(artifact)},
            )
        # 文件已改但 artifact 写失败: 不回滚、不谎称未改, 明确告知 full diff 不可得
        marker = (
            f"[edit applied; full diff unavailable: {artifact_error}; "
            f"re-read the edited file with the read tool to inspect the changes]"
        )
        preview = self._bounded_preview(
            diff, MAX_CONTENT_BYTES - header_bytes - 2 - len(marker.encode("utf-8"))
        )
        parts = [header, preview, marker] if preview else [header, marker]
        return self._success_result(
            ctx, "\n".join(parts),
            {**details, "diff_truncated": True, "artifact_error": artifact_error},
        )

    @staticmethod
    def _success_result(ctx: ToolContext, content: str, details: dict) -> ToolResult:
        """元信息也计入最终预算; 已写入的编辑不能因反馈超限变成错误."""
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            details = {**details, "metadata_truncated": True, "full_result_content": content}
            parts = ["edit applied; metadata_truncated=true; full metadata retained in details."]
            if details.get("full_diff_path") is not None:
                artifact_path = json.dumps(details["full_diff_path"], ensure_ascii=False)
                marker = f"[diff_truncated=true full_diff_path={artifact_path}; read it with the read tool]"
                if len((parts[0] + "\n" + marker).encode("utf-8")) <= MAX_CONTENT_BYTES:
                    parts.append(marker)
                else:
                    parts.append(
                        "[diff_truncated=true; full diff saved; artifact path omitted to fit the cap; "
                        "re-read the edited file using the original edit path]"
                    )
            elif "artifact_error" in details:
                parts.append(
                    "[diff_truncated=true; full diff unavailable; "
                    "re-read the edited file using the original edit path]"
                )
            elif details["diff"] == "":
                parts.append("(no text differences)")
            else:
                details["diff_truncated"] = True
                parts.append(
                    "[diff_truncated=true; re-read the edited file using the original edit path]"
                )
            # 不截断成半个可回读路径; marker 已整体核对预算。短说明亦遵守总界。
            content = "\n".join(parts).encode("utf-8")[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
        return ToolResult(call_id=ctx.call_id, name="edit", content=content, details=details)

    @staticmethod
    def _bounded_preview(diff: str, budget: int) -> str:
        if budget <= 0:
            return ""
        out: list[str] = []
        used = 0
        for line in diff.split("\n"):
            need = len(line.encode("utf-8")) + 1
            if used + need > budget:
                break
            out.append(line)
            used += need
        return "\n".join(out)

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
