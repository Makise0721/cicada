"""Skills 能力插件: 目录发现/冻结快照 (loader) 与真实 load_skill 工具 (tool).

快照由启动层经 load_skill_catalog 一次读取; 插件只持有不可变快照,
bootstrap 不做第二次文件读取。共享接口: skills_plugin(snapshot)。
"""

from __future__ import annotations

from cicada.plugins.skills.loader import load_auto_skill_catalog, load_skill_catalog
from cicada.plugins.skills.tool import CATALOG_CAPABILITY, LoadSkillTool, skills_plugin

__all__ = [
    "CATALOG_CAPABILITY",
    "LoadSkillTool",
    "load_skill_catalog",
    "load_auto_skill_catalog",
    "skills_plugin",
]
