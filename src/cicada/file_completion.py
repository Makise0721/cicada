"""中立文件补全契约与纯匹配规则 (project-context-v1 C3/C5)。

本模块只依赖标准库与 core 的 CancelToken；不导入插件运行时、Coding 插件或
UI。候选是查询时观察，不带文件正文或读取依据；路径文字进请求仍按原文发送，
手写请求不按候选规则改写。排序与限额借鉴固定版本 Pi (pi-mono@1a584a7a
packages/tui/src/autocomplete.ts, MIT)；scope/reason 词表为本项目契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Protocol

from cicada.core.cancel import CancelToken

MAX_CANDIDATES = 20
MAX_QUERY_BYTES = 4096

SCOPE_GIT = "git"
SCOPE_DIRECTORY = "directory"
SCOPE_UNAVAILABLE = "unavailable"
_SCOPES = (SCOPE_GIT, SCOPE_DIRECTORY, SCOPE_UNAVAILABLE)

# C6 分隔字符: Pi 的引号/等号加常用 CJK 标点; 中文文字仍是路径的一部分。
SEPARATORS = "，．：；！？（）［］｛｝“”‘’…—。、《》「」『』【】"


class UnsupportedPathError(ValueError):
    """query 含绝对盘符/UNC、~ 前缀、.. 段、控制字符或超出字节限额。"""


@dataclass(frozen=True)
class FileCompletionCandidate:
    """workspace 相对路径候选: 分隔符 `/`，目录尾部含 `/`，不带 @/引号/空格。"""

    relative_path: str
    is_directory: bool

    def __post_init__(self) -> None:
        if not self.relative_path or "\\" in self.relative_path:
            raise ValueError("relative_path must be a non-empty /-separated path")
        if self.is_directory != self.relative_path.endswith("/"):
            raise ValueError("directory candidates end with '/', files must not")
        if self.relative_path.startswith("/"):
            raise ValueError("candidate must stay inside the workspace")


@dataclass(frozen=True)
class FileCompletionResult:
    """一次候选查询的结果: scope 为 git/directory/unavailable。

    candidates 已排序且最多 MAX_CANDIDATES 项；无匹配是可用 scope 加空候选，
    不是失败。reason 是稳定短原因标识 (由实现词表定义)，UI 负责映射中文；
    预期查找失败才用 unavailable，不显示任意 stderr。
    """

    candidates: tuple[FileCompletionCandidate, ...]
    scope: str
    truncated: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}, got {self.scope!r}")
        if len(self.candidates) > MAX_CANDIDATES:
            raise ValueError(f"candidates exceed {MAX_CANDIDATES}")
        if self.scope == SCOPE_UNAVAILABLE and self.candidates:
            raise ValueError("unavailable results carry no candidates")
        if self.reason is not None and self.scope != SCOPE_UNAVAILABLE:
            raise ValueError("reason is only set on unavailable results")


class FileCompletionPort(Protocol):
    """Coding 能力实现的只读候选端口; 取消必须继续传播，不能转为空匹配。"""

    async def complete(self, query: str, cancel: CancelToken) -> FileCompletionResult:
        """query 是去掉 `@` 与包围引号的路径文字；实现自有界与失败词表。"""
        ...


def normalize_query(query: str) -> str:
    """校验并规范查询文字: `\\` 规范为 `/`；危险形态抛 UnsupportedPathError。

    空查询合法 (搜索工作区候选)。只拒绝展开形态的 `~` 前缀；文件名中部的
    `~` 是合法字符。盘符/UNC 用 PureWindowsPath.drive 判定，覆盖 `C:`、
    `C:x`、`C:/x` 与 `//server/share`。
    """
    if not isinstance(query, str):
        raise UnsupportedPathError("query must be str")
    try:
        encoded = query.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedPathError("query is not representable as UTF-8") from exc
    if len(encoded) > MAX_QUERY_BYTES:
        raise UnsupportedPathError("query exceeds 4096 UTF-8 bytes")
    normalized = query.replace("\\", "/")
    if any(ord(ch) < 0x20 or 0x7f <= ord(ch) <= 0x9f for ch in normalized):
        raise UnsupportedPathError("query contains control characters")
    if normalized.startswith(("~", "/")):
        raise UnsupportedPathError("query must stay inside the workspace")
    if PureWindowsPath(normalized).drive:
        raise UnsupportedPathError("query must be workspace-relative")
    if any(ch in '<>:"|?*' for ch in normalized):
        raise UnsupportedPathError("query is not a safe Windows path")
    if any(segment == ".." for segment in normalized.split("/")):
        raise UnsupportedPathError("query must not contain '..'")
    return normalized


def score_candidate(relative_path: str, is_directory: bool, fragment: str) -> int:
    """C5 评分: casefold 后文件名精确 100/前缀 80/包含 50/完整路径包含 30。

    目录加 10，但无匹配不因加分入选 (返回 0)；空 fragment 基础分 0。
    """
    if not fragment:
        return 0
    bare = relative_path[:-1] if is_directory else relative_path
    folded_name = bare.rsplit("/", 1)[-1].casefold()
    folded_fragment = fragment.casefold()
    if folded_name == folded_fragment:
        score = 100
    elif folded_name.startswith(folded_fragment):
        score = 80
    elif folded_fragment in folded_name:
        score = 50
    elif folded_fragment in relative_path.casefold():
        score = 30
    else:
        return 0
    return score + 10 if is_directory else score


def rank_candidates(
    paths: list[tuple[str, bool]], fragment: str
) -> tuple[tuple[FileCompletionCandidate, ...], bool]:
    """按得分降序、深度/长度/casefold/原路径升序排序并截取 top N。

    返回 (候选, 是否截断)；fragment 为空时全部按基础分 0 参加排序。
    """
    scored = []
    for path, is_directory in paths:
        score = score_candidate(path, is_directory, fragment)
        if score > 0 or not fragment:
            depth = path.rstrip("/").count("/") + 1
            scored.append((score, depth, len(path), path.casefold(), path, is_directory))
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    truncated = len(scored) > MAX_CANDIDATES
    candidates = tuple(
        FileCompletionCandidate(path, is_directory)
        for _, _, _, _, path, is_directory in scored[:MAX_CANDIDATES]
    )
    return candidates, truncated


def is_separator(character: str) -> bool:
    """token 分隔判定: 空白、引号/等号与 C6 标点集合；中文文字不是分隔。"""
    return character.isspace() or character in "\"'=" or character in SEPARATORS


__all__ = [
    "MAX_CANDIDATES",
    "MAX_QUERY_BYTES",
    "SCOPE_DIRECTORY",
    "SCOPE_GIT",
    "SCOPE_UNAVAILABLE",
    "FileCompletionCandidate",
    "FileCompletionPort",
    "FileCompletionResult",
    "SEPARATORS",
    "UnsupportedPathError",
    "is_separator",
    "normalize_query",
    "rank_candidates",
    "score_candidate",
]
