"""write 工具: UTF-8 全量创建/覆盖 + 版本前置 + 可靠提交 (FileAccess 单次原生发布).

新路径直接创建并自动建父目录; 已有文件必须先经 read 建立进程内版本依据,
读后变化会被拒绝并要求重读; 发布时同名竞争目标不会被覆盖 (create_conflict).
不继承原文件 BOM/换行约定; committed 不得假报未写, unknown 不掩盖.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.plugins.coding.file_access import STATE_COMMITTED, MutationOutcome
from cicada.plugins.coding.workspace import Workspace
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

MAX_CONTENT_BYTES = 50 * 1024  # 与 edit 相同的 content 总预算


class WriteTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write",
            description=(
                "以 UTF-8 全量写入文件, 自动创建父目录; 不继承已有文件的 BOM/换行约定. "
                "新路径直接创建; 已有文件必须先用 read 读取, 未读或读取后被改动都会被拒绝, "
                "需重新 read. 发布时若同名目标抢先出现, 不覆盖现有文件并报 create_conflict. "
                "成功后该文件的版本依据更新, 可直接 edit; 结果以 [mutation=state] "
                "给出真实提交事实 (not_written/committed/unknown)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        new_bytes = arguments["content"].encode("utf-8")  # 严格 UTF-8, 不继承 BOM/换行
        created: dict[str, bool] = {}

        def transform(original: bytes | None) -> bytes:
            # None 只出现在本次最初判断为创建的分支 (契约 C1); 已有文件收到原件字节
            created["this"] = original is None
            return new_bytes

        outcome = await self._workspace.files.mutate(
            arguments["path"], transform, ctx.cancel, mode="write"
        )
        return self._render(ctx, outcome, len(new_bytes), created.get("this", False))

    # ---- 结果渲染 (契约 C4: 真实状态优先, 提交事实先于附属成败) ----

    def _render(
        self, ctx: ToolContext, outcome: MutationOutcome, n_bytes: int, created: bool
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
        details: dict[str, Any] = {"mutation": mutation}

        if outcome.state != STATE_COMMITTED:
            parts = [f"[mutation={outcome.state}] {outcome.message or outcome.reason}"]
            if outcome.retained_paths:
                parts.append("retained materials: " + "; ".join(outcome.retained_paths))
            if outcome.state == "unknown":
                parts.append(
                    f"do not assume the file was or was not created; "
                    f"read {display} again before any further change"
                )
            elif outcome.reason == "read_required":
                parts.append(f"read {display} with the read tool first, then write")
            elif outcome.reason == "version_conflict":
                parts.append(f"read {display} again, then write with fresh content")
            elif outcome.reason == "create_conflict":
                parts.append(
                    f"a file already exists at {display}; it was not overwritten; "
                    f"read it first if you intend to replace it"
                )
            elif outcome.reason == "cancelled_before_commit":
                parts.append("the file was not modified; retry after the cancellation")
            return self._result(ctx, "\n".join(parts), details, is_error=True)

        if created:
            header = f"[mutation=committed] created {display} ({n_bytes} bytes)"
        else:
            header = f"[mutation=committed] wrote {n_bytes} bytes to {display}"
        warning = self._commit_warning(outcome)
        if warning is not None:
            # committed 但需按错误报告: 短头 + warning, 不假报未写
            parts = [header, warning]
            if outcome.retained_paths:
                parts.append("retained materials: " + "; ".join(outcome.retained_paths))
            parts.append(f"re-read {display} to observe the current state")
            return self._result(ctx, "\n".join(parts), details, is_error=True)
        if outcome.cleanup_warning:
            header = f"{header}; cleanup warning: {outcome.cleanup_warning}"
        return self._result(ctx, header, details, is_error=False)

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
    def _result(
        ctx: ToolContext, content: str, details: dict, *, is_error: bool
    ) -> ToolResult:
        """write 的 content 无正文回显, 构造上即有界; 保留截断兜底与 edit 对齐."""
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            content = content.encode("utf-8")[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
        return ToolResult(call_id=ctx.call_id, name="write", content=content,
                          is_error=is_error, details=details)

    def _display(self, path) -> str:
        try:
            return str(path.relative_to(self._workspace.root))
        except ValueError:
            return str(path)


def write_plugin() -> PluginDefinition:
    """tool-write 插件: 依赖 coding.workspace, 提供 tool.write."""

    def setup(ctx: PluginContext) -> None:
        workspace: Workspace = ctx.require("coding.workspace")
        ctx.provide("tool.write", WriteTool(workspace))

    return PluginDefinition(
        name="tool-write",
        setup=setup,
        provides=frozenset({"tool.write"}),
        requires=frozenset({"coding.workspace"}),
    )
