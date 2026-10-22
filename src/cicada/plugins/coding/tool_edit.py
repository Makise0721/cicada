"""edit 工具: 批量精确替换 + 版本前置 + 可靠提交; 任一编辑非法则整体不写盘.

已有文件必须先经 read 建立进程内版本依据; 读后变化会被拒绝并要求重读.
提交事实由 FileAccess 单次原生发布给出, committed 不得假报未写, unknown 不掩盖.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.file_access import (
    STATE_COMMITTED,
    MutationOutcome,
    MutationRejected,
)
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition
from cicada.tool_outcomes import mutation_details

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

BOM = b"\xef\xbb\xbf"
MAX_CONTENT_BYTES = 50 * 1024  # applied 头 + diff 的 UTF-8 总预算


class EditTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit",
            description=(
                "对文件做批量精确替换. 已有文件必须先用 read 读取: 未读过会被拒绝, "
                "读取后文件被改动也会因版本不一致被拒绝, 需重新 read. "
                "全部 old_text 在原始内容上匹配且必须唯一; 任一编辑非法则整体不写入. "
                "成功结果先给出 applied/路径/替换数/first_changed_line, 再附 unified diff; "
                "总输出不超过 50 KiB UTF-8, 超限时完整 diff 写入用户级运行状态目录的唯一文件, "
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
        captured: dict[str, str] = {}

        def transform(original: bytes) -> bytes:
            return self._transform(original, edits, captured)

        outcome = await self._workspace.files.mutate(
            arguments["path"], transform, ctx.cancel, mode="edit"
        )
        try:
            return self._render(ctx, outcome, len(edits), captured)
        except Exception as exc:
            # 结果排版是附属操作: 不能让核心把已经提交的事实泛化成 tool raised。
            mutation = {
                "version": 1, "state": outcome.state, "reason": outcome.reason,
                "canonical_path": outcome.canonical_path,
                "current_matches": outcome.current_matches,
                "caller_cancelled": outcome.caller_cancelled, "winerror": outcome.winerror,
                "retained_paths": list(outcome.retained_paths),
                "cleanup_warning": outcome.cleanup_warning,
            }
            parts = [f"[mutation={outcome.state}] {outcome.canonical_path}; {outcome.reason}",
                     f"result presentation unavailable: {type(exc).__name__}; re-read the file"]
            if outcome.cleanup_warning:
                parts.append(f"cleanup warning: {outcome.cleanup_warning}")
            if outcome.retained_paths:
                parts.append("retained materials: " + "; ".join(outcome.retained_paths))
            content = "\n".join(parts).encode("utf-8")[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
            return ToolResult(
                call_id=ctx.call_id, name="edit", content=content,
                is_error=outcome.state != STATE_COMMITTED or outcome.current_matches is not True
                         or outcome.reason == "conflict_at_commit",
                details=mutation_details(mutation, extras={
                    "presentation_warning": type(exc).__name__,
                }),
            )

    # ---- 纯字节转换 (队列内核对依据后调用; 可预期失败抛 MutationRejected) ----

    def _transform(self, original: bytes, edits: list[dict[str, str]], captured: dict[str, str]) -> bytes:
        has_bom = original.startswith(BOM)
        try:
            text = (original[3:] if has_bom else original).decode("utf-8")
        except UnicodeDecodeError:
            raise MutationRejected("invalid_utf8", "file is not valid UTF-8 text") from None

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
                raise MutationRejected("invalid_edit", f"edit {index}: old_text must not be empty")
            if new == old:
                raise MutationRejected(
                    "invalid_edit", f"edit {index}: new_text is identical to old_text"
                )
            count = normalized.count(old)
            if count == 0:
                raise MutationRejected(
                    "invalid_edit", f"edit {index}: old_text not found in file"
                )
            if count > 1:
                raise MutationRejected(
                    "invalid_edit",
                    f"edit {index}: old_text matches {count} times in file; make it unique",
                )
            start = normalized.find(old)
            spans.append((start, start + len(old), new))

        spans.sort(key=lambda span: span[0])
        for (s1, e1, _), (s2, _, _) in zip(spans, spans[1:]):
            if s2 < e1:
                raise MutationRejected(
                    "invalid_edit",
                    f"edits overlap between offsets {s1}..{e1} and {s2}; rejected",
                )

        applied = normalized
        for start, end, new in reversed(spans):
            applied = applied[:start] + new + applied[end:]

        captured["original"] = normalized
        captured["applied"] = applied
        out_text = applied.replace("\n", newline) if newline != "\n" else applied
        return (BOM if has_bom else b"") + out_text.encode("utf-8")

    # ---- 结果渲染 (契约 C4: 真实状态优先, 提交事实先于附属成败) ----

    def _render(
        self, ctx: ToolContext, outcome: MutationOutcome, count: int, captured: dict[str, str]
    ) -> ToolResult:
        path = Path(outcome.canonical_path)
        display = self._display(path)
        mutation = {
            "version": 1,
            "state": outcome.state,
            "reason": outcome.reason,
            "canonical_path": outcome.canonical_path,
            "current_matches": outcome.current_matches,
            "caller_cancelled": outcome.caller_cancelled,
            "winerror": outcome.winerror,
            "retained_paths": list(outcome.retained_paths),
            "cleanup_warning": outcome.cleanup_warning,
        }
        details: dict[str, Any] = mutation_details(mutation)

        if outcome.state != STATE_COMMITTED:
            parts = [f"[mutation={outcome.state}] {outcome.message or outcome.reason}"]
            if outcome.cleanup_warning:
                parts.append(f"cleanup warning: {outcome.cleanup_warning}")
            if outcome.retained_paths:
                parts.append("retained materials: " + "; ".join(outcome.retained_paths))
            if outcome.state == "unknown":
                parts.append(
                    f"do not assume the file was or was not modified; "
                    f"read {display} again before any further change"
                )
            elif outcome.reason == "read_required":
                parts.append(f"read {display} with the read tool first, then edit")
            elif outcome.reason == "version_conflict":
                parts.append(f"read {display} again, then edit with fresh content")
            elif outcome.reason == "cancelled_before_commit":
                parts.append("the file was not modified; retry after the cancellation")
            elif outcome.reason == "restore_capture_failed":
                parts.append(
                    "the file was not modified; the restore point could not be captured "
                    "(check the workspace disk, or disable restore capture to edit anyway)"
                )
            content = "\n".join(parts)
            return self._error_result(ctx, content, details)

        diff, first_changed_line = self._diff(
            path, captured.get("original", ""), captured.get("applied", "")
        )
        header = (
            f"[mutation=committed] edit applied: {display}, {count} replacement(s), "
            f"first changed line {first_changed_line}"
        )
        if outcome.cleanup_warning:
            header += f"; cleanup warning: {outcome.cleanup_warning}"
        details.update({"diff": diff, "first_changed_line": first_changed_line})
        warning = self._commit_warning(outcome)
        if warning is None:
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

        # committed 但需按错误报告: 短头 + warning 优先, diff 只给有界预览
        parts = [header, warning]
        if outcome.retained_paths:
            parts.append("retained materials: " + "; ".join(outcome.retained_paths))
        used = sum(len(part.encode("utf-8")) + 1 for part in parts)
        parts.append(f"re-read {display} to observe the current state")
        used += len(parts[-1].encode("utf-8")) + 1
        preview = self._bounded_preview(diff, MAX_CONTENT_BYTES - used)
        if preview:
            parts.append(preview)
        return self._error_result(ctx, "\n".join(parts), details)

    @staticmethod
    def _commit_warning(outcome: MutationOutcome) -> str | None:
        if outcome.reason == "conflict_at_commit":
            return (
                "commit-window conflict: the replaced original did not match the verified "
                "version; materials retained"
            )
        if outcome.current_matches is False:
            return "target no longer matches the committed content (changed externally after commit)"
        if outcome.current_matches is None:
            return "target readback unavailable; the commit itself stands"
        return None

    @staticmethod
    def _error_result(ctx: ToolContext, content: str, details: dict) -> ToolResult:
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            content = (
                content.encode("utf-8")[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
            )
        return ToolResult(call_id=ctx.call_id, name="edit", content=content, is_error=True,
                          details=details)

    def _spill_diff(
        self, ctx: ToolContext, path: Path, header: str, diff: str, details: dict
    ) -> ToolResult:
        """diff 超预算: 完整 diff 落盘输出目录, content 给有界预览."""
        artifact: Path | None = None
        artifact_error: str | None = None
        try:
            candidate = self._workspace.new_output_file(f"edit-diff-{self._display(path)}")
            target = self._workspace.resolve_output_for_write(candidate)
            with target.open("xb") as handle:
                handle.write(diff.encode("utf-8"))
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
                {**details, "diff_truncated": True, "artifact_path": str(artifact)},
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
            parts = [
                "[mutation=committed] edit applied; "
                "metadata_truncated=true; full metadata retained in details."
            ]
            if details.get("artifact_path") is not None:
                artifact_path = json.dumps(details["artifact_path"], ensure_ascii=False)
                marker = f"[diff_truncated=true artifact_path={artifact_path}; read it with the read tool]"
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
