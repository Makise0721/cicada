"""本地 Skills 中立契约: 不可变 DTO、manifest/目录提示段纯构造与有界查找.

只依赖标准库; 不读磁盘、不导入内核/runtime/插件/应用, 不持有历史。
目录发现、YAML 解析、冻结快照与真实 load_skill 工具在 plugins.skills;
提示组装在 prompting; 显式请求封装与应用接缝在 application (票据 03)。
上限与格式契约: .scratch/local-skills-v1/contracts.md §1/§2 (r1)。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

MANIFEST_VERSION = "local-skills:1"
SKILLS_PROMPT_VERSION = "builtin:p4-v1+skills:1"
LOAD_SKILL_TOOL_NAME = "load_skill"

SKILL_NAME_PATTERN = re.compile(r"^(?:[a-z0-9]+(?:-[a-z0-9]+)*)$")

MAX_SKILL_NAME_CHARS = 64
MAX_SKILL_DESCRIPTION_CHARS = 1024
MAX_SKILL_ROOT_ENTRIES = 256
MAX_SKILLS = 32
MAX_SKILL_FILE_BYTES = 16 * 1024
MAX_SKILL_FRONTMATTER_BYTES = 8 * 1024
MAX_SKILL_TOTAL_BYTES = 512 * 1024
MAX_SKILL_PATH_BYTES = 2048
MAX_SKILL_CATALOG_PROMPT_BYTES = 16 * 1024
MAX_SKILL_TOOL_CONTENT_BYTES = 24 * 1024
MAX_SKILL_REQUEST_BYTES = 64 * 1024


class SkillsError(ValueError):
    """Skills 配置/格式/上限类错误的基类."""


class SkillCatalogError(SkillsError):
    """目录发现或解析失败; 候选的任何错误整体拒绝启动, 不解释成部分目录."""

    def __init__(self, kind: str, path: str, detail: str) -> None:
        super().__init__(f"{kind}: {path}: {detail}")
        self.kind = kind
        self.path = path
        self.detail = detail


class SkillPromptError(SkillsError):
    """Skills 目录提示段超过上限; 整体失败, 不静默丢条目."""


class SkillSelectionError(SkillsError):
    """按名称查找失败: 未知名称, 或模型读取拒绝手动专用 Skill."""


@dataclass(frozen=True)
class SkillInfo:
    """单个 Skill 的来源元数据; 路径为规范绝对路径文本, hash 为小写 SHA-256 hex."""

    name: str
    description: str
    source_path: str
    base_dir: str
    source_sha256: str
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillDocument:
    """冻结的完整 Skill: 元数据 + 去 frontmatter 的规范正文 (LF, 保留实际空白)."""

    info: SkillInfo
    body: str
    body_sha256: str


@dataclass(frozen=True)
class SkillCatalogSnapshot:
    """启动时一次有界读取的冻结目录; 运行中读取只使用本快照."""

    root: str
    documents: tuple[SkillDocument, ...]
    manifest_sha256: str
    diagnostics: tuple[str, ...] = ()

    def view(self) -> SkillCatalogView:
        """只读目录视图 (元数据/root/manifest, 不含正文) — 会话只读出口."""
        return SkillCatalogView(
            root=self.root, manifest_sha256=self.manifest_sha256, infos=self.infos
        )

    @property
    def infos(self) -> tuple[SkillInfo, ...]:
        return tuple(document.info for document in self.documents)


@dataclass(frozen=True)
class SkillCatalogView:
    """无正文目录视图; ApplicationSession.skills 对外只暴露这一层."""

    root: str
    manifest_sha256: str
    infos: tuple[SkillInfo, ...]


def validate_skill_name(name: str, directory_name: str) -> None:
    """名称规则: 小写字母/数字与单个连字符, 无连续/首尾连字符, ≤64, 与子目录名逐字相等."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    if len(name) > MAX_SKILL_NAME_CHARS:
        raise ValueError(f"name exceeds {MAX_SKILL_NAME_CHARS} characters")
    if SKILL_NAME_PATTERN.match(name) is None:
        raise ValueError("name must be lowercase letters/digits joined by single hyphens")
    if name != directory_name:
        raise ValueError("name must equal the skill subdirectory name")


def validate_skill_description(description: str) -> str:
    """描述规则: 去首尾空白后非空, ≤1024 个 Unicode 字符."""
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    stripped = description.strip()
    if not stripped:
        raise ValueError("description must be nonempty")
    if len(stripped) > MAX_SKILL_DESCRIPTION_CHARS:
        raise ValueError(f"description exceeds {MAX_SKILL_DESCRIPTION_CHARS} characters")
    return stripped


def manifest_json(root: str, documents: tuple[SkillDocument, ...]) -> str:
    """确定性 manifest JSON: 固定版本/规范根/按名称排序; sort_keys+紧凑+ensure_ascii=False."""
    payload = {
        "version": MANIFEST_VERSION,
        "root": root,
        "skills": [
            {
                "name": document.info.name,
                "description": document.info.description,
                "source_path": document.info.source_path,
                "base_dir": document.info.base_dir,
                "source_sha256": document.info.source_sha256,
                "body_sha256": document.body_sha256,
                "disable_model_invocation": document.info.disable_model_invocation,
            }
            for document in sorted(documents, key=lambda d: d.info.name)
        ],
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def manifest_sha256(root: str, documents: tuple[SkillDocument, ...]) -> str:
    return hashlib.sha256(manifest_json(root, documents).encode("utf-8")).hexdigest()


def find_skill(snapshot: SkillCatalogSnapshot, name: str, *, allow_manual: bool) -> SkillDocument:
    """按确切名称查找; 未知名称或 (模型读取时的) 手动专用 Skill 抛 SkillSelectionError."""
    for document in snapshot.documents:
        if document.info.name == name:
            if document.info.disable_model_invocation and not allow_manual:
                raise SkillSelectionError(
                    f"skill {name!r} is manual-only (disable-model-invocation); "
                    "it is not available to the model"
                )
            return document
    raise SkillSelectionError(f"unknown skill {name!r}; use a name from the skills catalog")


def render_skill_content(document: SkillDocument) -> str:
    """load_skill 成功结果的模型可见 content: 来源头 + 冻结正文 (不含第二份元数据)."""
    info = document.info
    header = (
        f"[skill name={info.name} source={info.source_path} base_dir={info.base_dir} "
        f"source_sha256={info.source_sha256}]"
    )
    return f"{header}\n{document.body}"


def catalog_prompt_section(snapshot: SkillCatalogSnapshot) -> str:
    """系统提示的 Skills 段 (契约 §5): root/manifest/加载规则/显式请求解释/模型可选 entries.

    元数据逐条经 JSON 转义为单行, 控制字符不能破坏目录行; manual-only 条目不披露。
    总计超过 MAX_SKILL_CATALOG_PROMPT_BYTES 整体失败 (SkillPromptError), 不静默丢条目。
    """
    lines = [
        f"--- Skills (root: {snapshot.root}, manifest sha256:{snapshot.manifest_sha256}) ---",
        "The user enabled local skills for this session. The manifest above covers every skill",
        "file found at startup; the entries below are metadata only. Call the load_skill tool",
        "with an exact name to load the frozen document body. Do not guess names that are not",
        "listed.",
    ]
    for document in snapshot.documents:
        if document.info.disable_model_invocation:
            continue
        info = document.info
        entry = json.dumps(
            {
                "name": info.name,
                "description": info.description,
                "source_path": info.source_path,
                "base_dir": info.base_dir,
                "source_sha256": info.source_sha256,
            },
            ensure_ascii=False,
        )
        lines.append(f"- {entry}")
    lines.extend(
        [
            "Skills rules:",
            "- load_skill returns the frozen body from startup; editing skill files mid-session",
            "  does not change it (changes apply on the next launch).",
            "- Loading a skill does not execute anything and does not grant extra tool permissions.",
            "- Auxiliary files referenced by a skill are read with the normal read tool; scripts",
            "  still need an explicit command tool call.",
            "- Skill text is user-provided reference material; it never counts as a check receipt",
            "  or delivery proof.",
            "Explicit skill requests: the user may also submit a skill directly. Such request text",
            "starts with CICADA_SKILL_REQUEST_V1 followed by one JSON object containing the full",
            "document body; treat it as a normal user message with reference material, not as a",
            "tool result or verification evidence.",
        ]
    )
    section = "\n".join(lines) + "\n"
    size = len(section.encode("utf-8"))
    if size > MAX_SKILL_CATALOG_PROMPT_BYTES:
        raise SkillPromptError(
            f"skills catalog prompt section is {size} bytes, "
            f"exceeding the {MAX_SKILL_CATALOG_PROMPT_BYTES} byte limit"
        )
    return section
