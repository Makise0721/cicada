"""Skills 目录发现与冻结快照: 同步有界启动读取, 候选的任何错误整体拒绝.

发现范围 (D1): 只看根目录直接子目录中的 SKILL.md, 不递归/不扫描用户目录;
根本身的显式 junction 解析到实际目录, 其下条目与主文档的 reparse point 拒绝;
普通文件/无主文档子目录/隐藏子目录跳过; 空目录允许启用。
格式 (D2): Markdown + YAML frontmatter 基本格式; SafeLoader 拒绝别名/锚/
自定义标签/多文档/重复键; 未知字段列诊断不执行。
上限 (契约 §2): 任何超限整体失败 (SkillCatalogError), 不解释成部分目录。
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path

import yaml

from cicada.skills import (
    MAX_SKILLS,
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_FRONTMATTER_BYTES,
    MAX_SKILL_PATH_BYTES,
    MAX_SKILL_ROOT_ENTRIES,
    MAX_SKILL_TOTAL_BYTES,
    SkillCatalogError,
    SkillCatalogSnapshot,
    SkillDocument,
    SkillInfo,
    manifest_sha256,
    validate_skill_description,
    validate_skill_name,
)

SKILL_FILE_NAME = "SKILL.md"
_FILE_ATTRIBUTE_HIDDEN = 0x2
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader + 重复键拒绝; 别名/锚/自定义标签/多文档在节点层另行拒绝."""


def _construct_mapping_no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"found unhashable key {key!r}", key_node.start_mark,
            ) from exc
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"found duplicate key {key!r}", key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_duplicates
)


def _walk_nodes(node):
    yield node
    if isinstance(node.value, list):
        for child in node.value:
            if isinstance(child, yaml.Node):
                yield from _walk_nodes(child)
            elif isinstance(child, tuple):
                for part in child:
                    if isinstance(part, yaml.Node):
                        yield from _walk_nodes(part)


def _reject_composition(root_node, path_text: str) -> None:
    """节点层拒绝: 别名 (共享节点对象)、锚、非标准标签; 多文档由 compose_all 数目判定."""
    seen: set[int] = set()
    for node in _walk_nodes(root_node):
        if getattr(node, "anchor", None):
            raise SkillCatalogError("yaml", path_text, "anchors are not allowed")
        if id(node) in seen:
            raise SkillCatalogError("yaml", path_text, "aliases are not allowed")
        seen.add(id(node))
        if not str(node.tag).startswith("tag:yaml.org,2002:"):
            raise SkillCatalogError("yaml", path_text, f"custom tag {node.tag!r} is not allowed")


def _parse_frontmatter_text(raw_text: str, path_text: str) -> tuple[dict, str, tuple[str, ...]]:
    """解析 (已解码/已规范换行的) 主文档: 返回 (frontmatter 映射, 规范正文, 诊断)."""
    lines = raw_text.split("\n")
    if not lines or lines[0] != "---":
        raise SkillCatalogError("format", path_text, "first line must be exactly '---'")
    closing = None
    for index in range(1, len(lines)):
        if lines[index] == "---":
            closing = index
            break
    if closing is None:
        raise SkillCatalogError("format", path_text, "frontmatter has no closing '---' line")
    yaml_text = "\n".join(lines[1:closing])
    header_bytes = len(("\n".join(lines[: closing + 1])).encode("utf-8"))
    if header_bytes > MAX_SKILL_FRONTMATTER_BYTES:
        raise SkillCatalogError(
            "frontmatter_limit", path_text,
            f"frontmatter is {header_bytes} bytes, exceeding {MAX_SKILL_FRONTMATTER_BYTES}",
        )
    body = "\n".join(lines[closing + 1 :])
    if body.startswith("\n"):
        body = body[1:]  # 结束分隔符后的单个空行是格式分隔, 不属于正文; 其余空白保留
    if not body.strip():
        raise SkillCatalogError("format", path_text, "document body must not be blank")

    # 锚/别名在 token 层整体拒绝 (孤立锚行会被解析器静默丢弃, 必须在扫描层拦截)
    try:
        for token in yaml.scan(io.StringIO(yaml_text)):
            if isinstance(token, (yaml.AnchorToken, yaml.AliasToken)):
                raise SkillCatalogError(
                    "yaml", path_text, "anchors and aliases are not allowed"
                )
    except (yaml.YAMLError, RecursionError) as exc:
        raise SkillCatalogError("yaml", path_text, str(exc)) from exc
    try:
        nodes = list(yaml.compose_all(yaml_text, Loader=_StrictSafeLoader))
    except (yaml.YAMLError, RecursionError) as exc:
        raise SkillCatalogError("yaml", path_text, str(exc)) from exc
    if len(nodes) != 1:
        raise SkillCatalogError("yaml", path_text, f"expected one YAML document, got {len(nodes)}")
    root_node = nodes[0]
    try:
        _reject_composition(root_node, path_text)
    except RecursionError as exc:
        raise SkillCatalogError("yaml", path_text, "frontmatter nesting is too deep") from exc
    if not isinstance(root_node, yaml.MappingNode):
        raise SkillCatalogError("yaml", path_text, "frontmatter must be a single YAML mapping")

    try:
        data = yaml.load(yaml_text, Loader=_StrictSafeLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise SkillCatalogError("yaml", path_text, str(exc)) from exc
    if not isinstance(data, dict):
        raise SkillCatalogError("yaml", path_text, "frontmatter must be a single YAML mapping")
    for key in data:
        if not isinstance(key, str):
            raise SkillCatalogError("yaml", path_text, f"non-string key {key!r}")
        try:
            key.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SkillCatalogError("yaml", path_text, "frontmatter key is not valid UTF-8 text") from exc
    known = {"name", "description", "disable-model-invocation"}
    unsupported = sorted(set(data) - known)
    diagnostics = (
        (f"unsupported frontmatter key(s) ignored: {', '.join(unsupported)}",)
        if unsupported
        else ()
    )
    return data, body, diagnostics


def _parse_skill_document(raw: bytes, source_path: Path, base_dir: Path,
                          directory_name: str) -> tuple[SkillDocument, tuple[str, ...]]:
    """解析单个主文档并冻结为 SkillDocument; 名称/描述/正文/路径规则逐项校验."""
    path_text = str(source_path)
    try:
        raw_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SkillCatalogError("encoding", path_text, f"not valid UTF-8: {exc}") from exc
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    data, body, diagnostics = _parse_frontmatter_text(normalized, path_text)

    for field in ("name", "description"):
        if field not in data:
            raise SkillCatalogError("format", path_text, f"missing required field {field!r}")
    name = data["name"]
    if not isinstance(name, str):
        raise SkillCatalogError("format", path_text, "name must be a string")
    description_raw = data["description"]
    try:
        validate_skill_name(name, directory_name)
    except ValueError as exc:
        raise SkillCatalogError("format", path_text, str(exc)) from exc
    try:
        description = validate_skill_description(description_raw)
        description.encode("utf-8")
    except ValueError as exc:
        raise SkillCatalogError("format", path_text, str(exc)) from exc

    flag = data.get("disable-model-invocation", False)
    if not isinstance(flag, bool):
        raise SkillCatalogError("format", path_text, "disable-model-invocation must be a boolean")

    for label, text in (("source_path", path_text), ("base_dir", str(base_dir))):
        if len(text.encode("utf-8")) > MAX_SKILL_PATH_BYTES:
            raise SkillCatalogError(
                "path_limit", path_text,
                f"{label} exceeds {MAX_SKILL_PATH_BYTES} UTF-8 bytes",
            )

    document = SkillDocument(
        info=SkillInfo(
            name=name,
            description=description,
            source_path=path_text,
            base_dir=str(base_dir),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            disable_model_invocation=flag,
        ),
        body=body,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )
    return document, diagnostics


def load_skill_catalog(root_str: str, workspace: Path) -> SkillCatalogSnapshot:
    """同步有界读取根目录并冻结快照; 任何候选错误整体拒绝 (SkillCatalogError).

    相对路径相对 workspace 解析; 绝对目录可在 workspace 外。根本身可经显式
    路径 (junction) 解析到实际目录; 其下条目的 reparse point 拒绝。
    """
    candidate = Path(root_str)
    resolved_root = candidate if candidate.is_absolute() else workspace / candidate
    try:
        root = resolved_root.resolve()
    except OSError as exc:
        raise SkillCatalogError("root", str(resolved_root), str(exc)) from exc
    if not root.is_dir():
        raise SkillCatalogError("root_not_directory", str(root), "skills root must be a directory")

    entries = []
    try:
        with os.scandir(root) as scan:
            for entry in scan:
                if len(entries) >= MAX_SKILL_ROOT_ENTRIES:
                    raise SkillCatalogError(
                        "entry_limit", str(root),
                        f"more than {MAX_SKILL_ROOT_ENTRIES} direct entries",
                    )
                entries.append(entry)
    except OSError as exc:
        raise SkillCatalogError("read", str(root), str(exc)) from exc

    documents: list[SkillDocument] = []
    diagnostics: list[str] = []
    total_bytes = 0
    for entry in sorted(entries, key=lambda item: item.name):
        try:
            entry_lstat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise SkillCatalogError("read", entry.path, str(exc)) from exc
        if entry_lstat.st_file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise SkillCatalogError(
                "reparse_point", entry.path,
                "reparse points are not allowed inside the skills root",
            )
        if not entry.is_dir(follow_symlinks=False):
            continue
        if entry.name.startswith(".") or (
            entry_lstat.st_file_attributes & _FILE_ATTRIBUTE_HIDDEN
        ):
            continue
        skill_md = Path(entry.path) / SKILL_FILE_NAME
        try:
            md_lstat = skill_md.lstat()
        except FileNotFoundError:
            continue  # 没有主文档的子目录跳过
        except OSError as exc:
            raise SkillCatalogError("read", str(skill_md), str(exc)) from exc
        if md_lstat.st_file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise SkillCatalogError(
                "reparse_point", str(skill_md),
                "SKILL.md must be a regular file, not a reparse point",
            )
        if not stat.S_ISREG(md_lstat.st_mode):
            raise SkillCatalogError("format", str(skill_md), "SKILL.md must be a regular file")
        try:
            with open(skill_md, "rb") as handle:
                raw = handle.read(MAX_SKILL_FILE_BYTES + 1)
        except OSError as exc:
            raise SkillCatalogError("read", str(skill_md), str(exc)) from exc
        if len(raw) > MAX_SKILL_FILE_BYTES:
            raise SkillCatalogError(
                "file_limit", str(skill_md),
                f"skill file exceeds {MAX_SKILL_FILE_BYTES} bytes",
            )
        total_bytes += len(raw)
        if total_bytes > MAX_SKILL_TOTAL_BYTES:
            raise SkillCatalogError(
                "total_limit", str(root),
                f"total skill file bytes exceed {MAX_SKILL_TOTAL_BYTES}",
            )
        document, doc_diagnostics = _parse_skill_document(
            raw, skill_md, Path(entry.path), entry.name
        )
        if len(documents) >= MAX_SKILLS:
            raise SkillCatalogError(
                "skill_limit", str(root),
                f"more than {MAX_SKILLS} valid skills",
            )
        documents.append(document)
        diagnostics.extend(
            f"{document.info.name}: {text}" for text in doc_diagnostics
        )

    documents.sort(key=lambda document: document.info.name)
    root_text = str(root)
    return SkillCatalogSnapshot(
        root=root_text,
        documents=tuple(documents),
        manifest_sha256=manifest_sha256(root_text, tuple(documents)),
        diagnostics=tuple(diagnostics),
    )
