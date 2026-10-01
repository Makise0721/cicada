"""文件发现与字面量内容搜索: 限定 glob、Git 清单、reparse 事实、有界结果.

范围口径: 只用 Git 清单给候选 (tracked + nonignored untracked), 匹配与读取都在 Python
内完成; 解引用前先看清单原始路径上的 lstat/reparse 事实, 不先 realpath 再猜。
"""

from __future__ import annotations

import asyncio
import codecs
import json
import re
import stat
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import FileInventory, GitInventory, InventoryError
from cicada.plugins.coding.workspace import PathNotAllowedError, Workspace

MAX_CONTENT_BYTES = 8192  # 头/JSON 转义/正文/footer 与下一步提示全部计入
QUERY_TIMEOUT_S = 10.0  # 整个查询 (清单 + 匹配 + 内容扫描)
MAX_FILE_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 65536
MAX_GLOB_BYTES = 512
MAX_PATH_BYTES = 4096
GREP_MAX_PATTERN_BYTES = 1024
MAX_MATCH_TEXT_BYTES = 1024  # 单条文本元信息占比上限
MAX_HEADER_FIELD_BYTES = 512  # 头字段预算: 路径/模式本身也可能很长
GLOB_DEFAULT_LIMIT = 200
GLOB_MAX_LIMIT = 1000
GREP_DEFAULT_LIMIT = 100
GREP_MAX_LIMIT = 500
GREP_MAX_CONTEXT = 2
NO_MATCHES_TEXT = "No matches found in the searched files."
_TRUNCATED_NOTE = "[results truncated; narrow path, include or pattern and query again]"


class SearchError(Exception):
    """整个查询失败; 部分扫描不得解释成完整零命中."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True)
class SearchScope:
    """一次查询的已解析范围.

    base: 匹配基准目录 (单文件 path 时是其父目录)
    root_prefix: 把基准相对路径还原为 root 相对路径的前缀 ('' 或 '<base>/')
    files: 候选的基准相对 POSIX 路径; pattern/include 都以此匹配
    """

    base: Path
    root_prefix: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Rendered:
    """渲染结果; shown/truncated 描述 content 实际展示的内容."""

    content: str
    shown: int
    truncated: bool
    reason: str | None = None


# --- 限定 glob 语法 ---


def parse_glob(pattern: str, *, field: str = "pattern") -> tuple[re.Pattern[str], ...]:
    """逐段 `*`/`?` 与独立段 `**`; 比较用 Unicode casefold.

    返回匹配器变体。`**` 至少要吃掉一个路径段, 因此含 `**` 的模式给出两个变体:
    常规形式, 以及首个 `**` 恰好吃掉一段 (使相邻 `/` 之一消失) 的形式。
    """
    if not isinstance(pattern, str) or not pattern:
        raise SearchError("invalid_pattern", f"{field} must be a nonempty string")
    if len(pattern.encode("utf-8")) > MAX_GLOB_BYTES:
        raise SearchError("invalid_pattern", f"{field} exceeds {MAX_GLOB_BYTES} UTF-8 bytes")
    if any(character in pattern for character in "[]{}"):
        raise SearchError(
            "invalid_pattern", f"{field} does not support bracket or brace expansion"
        )
    if "\x00" in pattern or "\n" in pattern or "\r" in pattern:
        raise SearchError("invalid_pattern", f"{field} must not contain control characters")
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("/"):
        raise SearchError("invalid_pattern", f"{field} must not be absolute")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise SearchError(
            "invalid_pattern",
            f"{field} must not contain empty, '.' or '..' path segments",
        )
    segments: list[str] = []
    for part in parts:
        if part == "**":
            # `**` 匹配 1 个或多个完整段
            segments.append(r"[^/]+(?:/[^/]+)*")
            continue
        if "**" in part:
            raise SearchError(
                "invalid_pattern",
                f"{field} allows '**' only as a standalone path segment",
            )
        segments.append(_segment_regex(part.casefold()))
    matchers = [re.compile("^" + "/".join(segments) + "$", re.IGNORECASE)]
    if "**" in parts:
        index = parts.index("**")
        collapsed = segments[:index] + segments[index + 1 :]
        if collapsed:
            matchers.append(re.compile("^" + "/".join(collapsed) + "$", re.IGNORECASE))
    return tuple(matchers)


def _segment_regex(part: str) -> str:
    # 段内 token 放进非捕获组, 段首的 * / ? 才有可重复目标
    body = []
    for character in part:
        if character == "*":
            body.append("[^/]*")
        elif character == "?":
            body.append("[^/]")
        else:
            body.append(re.escape(character))
    return "(?:" + "".join(body) + ")"


def glob_matches(matchers: Sequence[re.Pattern[str]], relative: str) -> bool:
    """整串匹配 (含路径段), 用 casefold 比较; 不改写真实路径."""
    folded = relative.casefold()
    return any(matcher.match(folded) is not None for matcher in matchers)


def matches_any(globs: Sequence[Sequence[re.Pattern[str]]], relative: str) -> bool:
    return any(glob_matches(matchers, relative) for matchers in globs)


# --- 文件系统事实 ---


def is_reparse(info) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def clipped(text: str, max_bytes: int) -> str:
    """按 UTF-8 边界从右侧裁剪, 保留前缀."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", "ignore")


def _lstat(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SearchError("path_unreadable", f"cannot inspect {path}: {exc}") from exc


def first_reparse(root: Path, relative: str, *, prefix: str = "") -> str | None:
    """返回 relative (或 prefix 子路径) 上第一个 reparse 路径, 不继续解引用."""
    current = root
    parts = relative.split("/")
    limit = len(prefix.split("/")) if prefix else len(parts)
    for index, part in enumerate(parts[:limit], 1):
        current = current / part
        info = _lstat(current)
        if info is None:
            return None
        if is_reparse(info):
            return "/".join(parts[:index])
    return None


def classify_candidate(root: Path, relative: str) -> str:
    """可读性判定: 'ok' / 'reparse' / 'missing' / 'not_regular'."""
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts, 1):
        current = current / part
        info = _lstat(current)
        if info is None:
            return "missing"
        if is_reparse(info):
            return "reparse"
        if index < len(parts) and not stat.S_ISDIR(info.st_mode):
            return "not_regular"
    try:
        return "ok" if stat.S_ISREG(current.lstat().st_mode) else "not_regular"
    except OSError as exc:
        raise SearchError("path_unreadable", f"cannot inspect {relative!r}: {exc}") from exc


def read_text_file_chunked(path: Path) -> str | None:
    """分块读取, 仍整文件校验编码; 候选已由调用方限制在 MAX_FILE_BYTES 内."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    chunks: list[str] = []
    try:
        with open(path, "rb") as handle:
            first = handle.read(READ_CHUNK_BYTES + 1)
            if b"\x00" in first:
                return None
            if first.startswith(codecs.BOM_UTF8):
                first = first[len(codecs.BOM_UTF8) :]
            chunks.append(decoder.decode(first))
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                if b"\x00" in chunk:
                    return None
                chunks.append(decoder.decode(chunk))
            chunks.append(decoder.decode(b"", True))
    except UnicodeDecodeError:
        return None
    return "".join(chunks)


# --- 范围解析 ---


def resolve_scope(
    workspace: Workspace,
    record: FileInventory,
    raw_path: str,
) -> SearchScope:
    """把 path 解析为基准目录 + 候选文件; 基准祖先的 reparse 先看事实再解引用."""
    if not isinstance(raw_path, str) or not raw_path:
        raise SearchError("invalid_path", "path must be a nonempty string")
    if len(raw_path.encode("utf-8")) > MAX_PATH_BYTES:
        raise SearchError("invalid_path", f"path exceeds {MAX_PATH_BYTES} UTF-8 bytes")
    try:
        resolved = workspace.resolve_within_root(raw_path)
    except PathNotAllowedError as exc:
        raise SearchError("invalid_path", str(exc)) from exc
    root = workspace.root
    if resolved == root:
        base_relative = ""
    else:
        try:
            base_relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:  # pragma: no cover - resolve_within_root 已保证
            raise SearchError("invalid_path", f"path is outside the workspace: {resolved}") from exc
        ancestor = first_reparse(root, base_relative)
        if ancestor is not None:
            raise SearchError(
                "reparse_path",
                f"path traverses a reparse point ({ancestor!r}); search does not follow it",
            )
    if not resolved.exists():
        raise SearchError("not_found", f"path does not exist: {resolved}")
    if resolved.is_dir():
        scope_prefix = f"{base_relative}/" if base_relative else ""
        root_prefix = scope_prefix
        selected = [path for path in record.paths if path.startswith(scope_prefix)]
        base = resolved
    else:
        if base_relative not in record.paths:
            raise SearchError(
                "not_in_inventory",
                f"file is not part of the Git-visible search scope: {base_relative}",
            )
        # 单文件 path: 以父目录为基准, 匹配路径就是文件名
        scope_prefix = f"{Path(base_relative).parent.as_posix()}/"
        if scope_prefix == "./":
            scope_prefix = ""
        root_prefix = scope_prefix
        selected = [base_relative]
        base = resolved.parent
    files = tuple(path[len(scope_prefix) :] for path in selected)
    return SearchScope(base=base, root_prefix=root_prefix, files=files)


def scope_files(scope: SearchScope) -> Iterator[tuple[str, str]]:
    """候选 (基准相对匹配路径, root 相对路径)."""
    for relative in scope.files:
        yield relative, f"{scope.root_prefix}{relative}"


# --- 查询入口 ---


async def run_glob(
    workspace: Workspace,
    inventory: GitInventory,
    *,
    pattern: str,
    raw_path: str,
    limit: int,
    cancel: CancelToken,
) -> tuple[Rendered, dict[str, int]]:
    started = time.monotonic()
    matchers = parse_glob(pattern)
    record = await _list_inventory(inventory, cancel, started)
    scope = resolve_scope(workspace, record, raw_path)
    rows: list[str] = []
    total = 0
    skipped = {"missing": 0, "reparse": 0, "not_regular": 0}
    for index, (match_path, relative) in enumerate(scope_files(scope)):
        _checkpoint(cancel, started, index)
        verdict = classify_candidate(workspace.root, relative)
        if verdict != "ok":
            skipped[verdict] = skipped.get(verdict, 0) + 1
            continue
        if not glob_matches(matchers, match_path):
            continue
        total += 1
        if total > limit:
            break
        rows.append(json.dumps(str(workspace.root / relative), ensure_ascii=False))
    truncated = total > limit
    header = (
        f"[glob pattern={_json(pattern)} path={_json(raw_path)} limit={limit} "
        f"files={len(scope.files)} policy={record.policy_id} "
        f"excluded={record.excluded_count}]"
    )
    rendered = _render(
        header=header,
        rows=rows,
        shown=len(rows),
        truncated=truncated,
        reason="limit" if truncated else None,
        skipped=skipped,
        no_match_text=NO_MATCHES_TEXT,
    )
    return rendered, skipped


async def run_grep(
    workspace: Workspace,
    inventory: GitInventory,
    *,
    pattern: str,
    raw_path: str,
    include: str,
    ignore_case: bool,
    context: int,
    limit: int,
    cancel: CancelToken,
) -> tuple[Rendered, dict[str, int]]:
    started = time.monotonic()
    needle = _prepare_needle(pattern, ignore_case)
    include_glob = (parse_glob(include, field="include"),)
    record = await _list_inventory(inventory, cancel, started)
    scope = resolve_scope(workspace, record, raw_path)
    skipped = {
        "missing": 0,
        "reparse": 0,
        "not_regular": 0,
        "too_large": 0,
        "non_text": 0,
    }
    rows: list[str] = []
    shown = 0
    truncated = False
    reason: str | None = None
    for index, (match_path, relative) in enumerate(scope_files(scope)):
        _checkpoint(cancel, started, index)
        if not matches_any(include_glob, match_path):
            continue
        verdict = classify_candidate(workspace.root, relative)
        if verdict != "ok":
            skipped[verdict] = skipped.get(verdict, 0) + 1
            continue
        absolute = workspace.root / relative
        try:
            if absolute.stat().st_size > MAX_FILE_BYTES:
                skipped["too_large"] += 1
                continue
        except OSError as exc:
            raise SearchError("path_unreadable", f"cannot stat {relative!r}: {exc}") from exc
        try:
            text = read_text_file_chunked(absolute)
        except OSError as exc:
            raise SearchError("path_unreadable", f"cannot read {relative!r}: {exc}") from exc
        if text is None:
            skipped["non_text"] += 1
            continue
        hits = _match_lines(text, needle, ignore_case)
        if not hits:
            continue
        remaining = limit - shown
        for block in _group_hits(relative, text.splitlines(), hits, context):
            if remaining <= 0:
                truncated = True
                reason = reason or "limit"
                break
            block, taken = _cap_block(block, remaining)
            rows.append(_render_block(workspace, block))
            shown += taken
            remaining -= taken
        if truncated:
            break
    header = (
        f"[grep pattern={_json(pattern)} path={_json(raw_path)} include={_json(include)} "
        f"ignore_case={str(ignore_case).lower()} context={context} limit={limit} "
        f"files={len(scope.files)} policy={record.policy_id} "
        f"excluded={record.excluded_count}]"
    )
    rendered = _render(
        header=header,
        rows=rows,
        shown=shown,
        truncated=truncated,
        reason=reason,
        skipped=skipped,
        no_match_text=NO_MATCHES_TEXT,
    )
    return rendered, skipped


async def _list_inventory(
    inventory: GitInventory, cancel: CancelToken, started: float
) -> FileInventory:
    if time.monotonic() - started >= QUERY_TIMEOUT_S:
        raise SearchError("timeout", "search query deadline exceeded")
    try:
        return await inventory.list_files(cancel)
    except InventoryError as exc:
        kind = "timeout" if exc.kind == "timeout" else exc.kind
        raise SearchError(kind, str(exc)) from exc


def _checkpoint(cancel: CancelToken, started: float, index: int) -> None:
    cancel.throw_if_cancelled()
    if time.monotonic() - started >= QUERY_TIMEOUT_S:
        raise SearchError("timeout", "search query deadline exceeded")


def _json(value: str) -> str:
    """有界 JSON 转义值; 超长时按 UTF-8 边界裁剪并标记, 不产生半个转义序列."""
    if len(value.encode("utf-8")) <= MAX_HEADER_FIELD_BYTES:
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(clipped(value, MAX_HEADER_FIELD_BYTES) + "…[clipped]", ensure_ascii=False)


# --- 命中分组与渲染 ---


def _prepare_needle(pattern: str, ignore_case: bool) -> str:
    if not isinstance(pattern, str) or not pattern:
        raise SearchError("invalid_pattern", "pattern must be a nonempty string")
    if len(pattern.encode("utf-8")) > GREP_MAX_PATTERN_BYTES:
        raise SearchError(
            "invalid_pattern", f"pattern exceeds {GREP_MAX_PATTERN_BYTES} UTF-8 bytes"
        )
    if pattern.splitlines() != [pattern] or "\x00" in pattern:
        raise SearchError(
            "invalid_pattern", "pattern must not contain line separators or NUL"
        )
    return pattern.casefold() if ignore_case else pattern


def _match_lines(text: str, needle: str, ignore_case: bool) -> list[int]:
    """行号与 read 同口径: BOM 剥离后的 splitlines, 从 1 开始."""
    haystack = text.casefold() if ignore_case else text
    return [number for number, line in enumerate(haystack.splitlines(), 1) if needle in line]


@dataclass(frozen=True)
class Block:
    """一段合并后的上下文窗口: 匹配行号 + 窗口内全部 (行号, 文本)."""

    relative_path: str
    matched: tuple[int, ...]
    lines: tuple[tuple[int, str], ...]


def _group_hits(relative: str, lines: list[str], hits: list[int], context: int) -> list[Block]:
    if not hits:
        return []
    if context == 0:
        return [
            Block(relative, (number,), ((number, lines[number - 1]),)) for number in hits
        ]
    blocks: list[Block] = []
    start = max(hits[0] - context, 1)
    matched: list[int] = [hits[0]]
    previous = hits[0]
    for number in hits[1:]:
        if number - previous <= 2 * context:
            matched.append(number)
            previous = number
            continue
        blocks.append(_make_block(relative, lines, start, previous + context, matched))
        start = max(number - context, 1)
        matched = [number]
        previous = number
    blocks.append(_make_block(relative, lines, start, previous + context, matched))
    return blocks


def _make_block(
    relative: str, lines: list[str], start: int, end: int, matched: list[int]
) -> Block:
    end = min(end, len(lines))
    return Block(
        relative,
        tuple(matched),
        tuple((number, lines[number - 1]) for number in range(start, end + 1)),
    )


def _cap_block(block: Block, room: int) -> tuple[Block, int]:
    if len(block.matched) <= room:
        return block, len(block.matched)
    allowed = set(block.matched[:room])
    keep = [item for item in block.lines if item[0] in allowed]
    return Block(block.relative_path, tuple(sorted(allowed)), tuple(keep)), room


def _display_path(workspace: Workspace, relative: str) -> str:
    """canonical absolute 路径; 契约要求模型拿到的路径可直接继续 read."""
    return str(workspace.root / relative)


def _render_block(workspace: Workspace, block: Block) -> str:
    path_text = _display_path(workspace, block.relative_path)
    if len(block.lines) == 1 and len(block.matched) == 1:
        number, line = block.lines[0]
        text, clipped_flag = _clip_match_text(line)
        return _json_row(path_text, text, number, clipped_flag)
    text, clipped_lines = _clip_block_lines(block)
    return _json_block(
        path_text, block.lines[0][0], block.lines[-1][0], text, clipped_lines
    )


def _clip_match_text(text: str) -> tuple[str, bool]:
    if len(text.encode("utf-8")) <= MAX_MATCH_TEXT_BYTES:
        return text, False
    return clipped(text, MAX_MATCH_TEXT_BYTES), True


def _clip_block_lines(block: Block) -> tuple[list[str], list[int]]:
    text: list[str] = []
    clipped_lines: list[int] = []
    matched = set(block.matched)
    for number, line in block.lines:
        prefix = ">" if number in matched else " "
        body, was_clipped = _clip_match_text(line)
        if was_clipped:
            clipped_lines.append(number)
        text.append(f"{prefix}{number}:{body}")
    return text, clipped_lines


def _json_row(path_text: str, text: str, line: int, partial: bool) -> str:
    payload: dict[str, object] = {"path": path_text, "line": line, "text": text}
    if partial:
        payload["partial_text"] = True
    return json.dumps(payload, ensure_ascii=False)


def _json_block(
    path_text: str, start: int, end: int, lines: list[str], partial_lines: list[int]
) -> str:
    payload: dict[str, object] = {
        "path": path_text,
        "lines": f"{start}-{end}",
        "text": lines,
    }
    if partial_lines:
        payload["partial_text_lines"] = partial_lines
    return json.dumps(payload, ensure_ascii=False)


def _render(
    *,
    header: str,
    rows: list[str],
    shown: int,
    truncated: bool,
    reason: str | None,
    skipped: dict[str, int],
    no_match_text: str,
) -> Rendered:
    """把行装配进 8192 bytes 预算; 放不下就丢尾部行并标 truncated/bytes."""
    budget = MAX_CONTENT_BYTES
    kept: list[str] = []
    used = len(header.encode("utf-8")) + 1
    dropped = False
    for row in rows:
        size = len(row.encode("utf-8")) + 1
        if used + size > budget:
            dropped = True
            break
        kept.append(row)
        used += size
    if dropped:
        truncated = True
        reason = "bytes"
    kept_shown = _shown_of(kept, shown, len(rows))
    lines = [header]
    if kept:
        lines.extend(kept)
    else:
        kept_shown = 0
        lines.append(no_match_text)
    skipped_note = ", ".join(f"{name}={value}" for name, value in sorted(skipped.items()) if value)
    if skipped_note:
        lines.append(f"[skipped: {clipped(skipped_note, 256)}]")
    lines.append(
        f"[shown={kept_shown} truncated={str(truncated).lower()} "
        f"complete={str(not truncated).lower()}"
        + (f" reason={reason}" if truncated and reason else "")
        + "]"
    )
    if truncated:
        lines.append(_TRUNCATED_NOTE)
    content = "\n".join(lines)
    if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        content = clipped(content, MAX_CONTENT_BYTES)
    return Rendered(content, kept_shown, truncated, reason)


def _shown_of(kept: list[str], shown: int, total_rows: int) -> int:
    """content 里实际展示的条目数 (glob 一行 = 一个路径, grep 一行 = 匹配行数)."""
    if len(kept) == total_rows:
        return shown
    if not kept:
        return 0
    matched = 0
    for row in kept:
        try:
            payload = json.loads(row)
        except json.JSONDecodeError:  # pragma: no cover - 行由本模块生成
            continue
        if isinstance(payload, str):
            matched += 1
        elif "line" in payload:
            matched += 1
        else:
            matched += sum(1 for line in payload.get("text", []) if line.startswith(">"))
    return matched
