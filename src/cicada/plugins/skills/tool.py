"""load_skill 工具: 按确切名称从冻结快照返回正文; 无 I/O/写入/后台生产者.

成功 content = 来源头 + 冻结正文 (D4); details 只承载镜像元数据, 不复制正文。
未知名称与手动专用 Skill 返回正常工具错误。取消经 ToolContext.cancel 在
lookup 前后协作检查。disable-model-invocation 只控制目录路由与本工具 (C1);
普通 read/powershell 的磁盘读取行为保持不变。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cicada.core.messages import ToolResult
from cicada.core.ports import ToolContext, ToolSpec
from cicada.skills import (
    LOAD_SKILL_TOOL_NAME,
    MAX_SKILL_TOOL_CONTENT_BYTES,
    SkillCatalogSnapshot,
    SkillSelectionError,
    find_skill,
    render_skill_content,
)
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext

CATALOG_CAPABILITY = "skills.catalog"


class LoadSkillTool:
    """真实 load_skill 适配器: 快照查找, 无磁盘访问."""

    def __init__(self, snapshot: SkillCatalogSnapshot) -> None:
        self._snapshot = snapshot

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=LOAD_SKILL_TOOL_NAME,
            description=(
                "按确切名称加载启动时冻结的 Skill 文档正文。名称必须来自系统提示中的目录; "
                "不在目录中的手动专用 Skill 或未知名称都会被拒绝。返回内容包含来源 "
                "(路径/基目录/原始字节 hash) 与完整正文; 加载不执行文档中的任何内容, "
                "不改变工具权限, 不产生检查回执。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.cancel.throw_if_cancelled()
        name = arguments["name"]
        try:
            document = find_skill(self._snapshot, name, allow_manual=False)
        except SkillSelectionError as exc:
            return ToolResult(
                call_id=ctx.call_id, name=LOAD_SKILL_TOOL_NAME, content=str(exc), is_error=True
            )
        ctx.cancel.throw_if_cancelled()
        content = render_skill_content(document)
        if len(content.encode("utf-8")) > MAX_SKILL_TOOL_CONTENT_BYTES:
            return ToolResult(
                call_id=ctx.call_id,
                name=LOAD_SKILL_TOOL_NAME,
                content=(
                    f"skill {name!r} content exceeds the "
                    f"{MAX_SKILL_TOOL_CONTENT_BYTES} byte tool content limit"
                ),
                is_error=True,
            )
        info = document.info
        return ToolResult(
            call_id=ctx.call_id,
            name=LOAD_SKILL_TOOL_NAME,
            content=content,
            is_error=False,
            details={
                "name": info.name,
                "source": info.source_path,
                "base_dir": info.base_dir,
                "source_sha256": info.source_sha256,
                "body_sha256": document.body_sha256,
                "manifest_sha256": self._snapshot.manifest_sha256,
                "body_chars": len(document.body),
            },
        )


def skills_plugin(snapshot: SkillCatalogSnapshot) -> PluginDefinition:
    """由已加载快照创建目录能力与 load_skill 工具; bootstrap 不做第二次文件读取."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide(CATALOG_CAPABILITY, snapshot)
        ctx.provide("tool.load_skill", LoadSkillTool(snapshot))

    return PluginDefinition(
        name="skills",
        setup=setup,
        provides=frozenset({CATALOG_CAPABILITY, "tool.load_skill"}),
    )
