"""Git 原生比较的只读审阅来源；列表与正文是独立的有界观察。"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import stat
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.git_io import GitIOError, read_git
from cicada.plugins.coding.inventory import DEFAULT_EXCLUSIONS, GitInventory, InventoryError
from cicada.plugins.coding.workspace import Workspace
from cicada.review import ReviewGroupState, ReviewItem, ReviewPreview, ReviewSnapshot
from cicada.runtime.plugin import PluginContext, PluginDefinition

CHANGE_REVIEW_CAPABILITY = "coding.change_review"
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ITEMS = 10_000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_PREVIEW_BYTES = 256 * 1024
REVIEW_TIMEOUT_S = 10.0
GROUPS = ("unstaged", "staged", "untracked")
_DIFF_OPTIONS = ("--no-ext-diff", "--no-textconv", "--no-renames", "--no-color",
                 "--ignore-submodules=dirty")
_KINDS = {"A": "added", "M": "modified", "D": "deleted", "T": "type_changed",
          "U": "conflict"}
_COMPARISONS = {"unstaged": "index → 工作区", "staged": "HEAD → index",
                "untracked": "未跟踪新增（原始文件）"}
_ATTRIBUTES = ("filter", "diff", "text", "eol", "working-tree-encoding")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(record: bytes) -> str:
    try:
        path = record.decode("utf-8")
    except UnicodeError as exc:
        raise GitIOError("invalid_path", "Git path is not UTF-8") from exc
    parts = path.split("/")
    if (not path or PurePosixPath(path).is_absolute() or "\\" in path or ":" in path
            or any(part in ("", ".", "..") for part in parts)):
        raise GitIOError("invalid_path", "invalid Git relative path")
    return path


def _records(data: bytes) -> list[bytes]:
    if data and not data.endswith(b"\0"):
        raise GitIOError("invalid_metadata", "Git metadata lacks a terminating NUL")
    return data.split(b"\0")[:-1]


@dataclass(frozen=True)
class _FileObservation:
    data: bytes
    sha256: str
    stamp: tuple[int, ...]


class _Observation:
    """一个操作共用 deadline 与累计元数据预算。"""

    def __init__(self, root: Path, cancel: CancelToken) -> None:
        self.root = root
        self.cancel = cancel
        self.deadline = time.monotonic() + REVIEW_TIMEOUT_S
        self.remaining = MAX_METADATA_BYTES

    def checkpoint(self) -> None:
        self.cancel.throw_if_cancelled()
        if time.monotonic() >= self.deadline:
            raise GitIOError("timeout", "review observation deadline exceeded")

    async def git(self, *args: str, input_data: bytes | None = None,
                  allowed_codes: tuple[int, ...] = (0,), content: bool = False,
                  preview: bool = False) -> tuple[bytes, int, bool]:
        self.checkpoint()
        result = await read_git(
            self.root, tuple(args), self.cancel, MAX_FILE_BYTES if content else self.remaining,
            input_data=input_data, allowed_codes=allowed_codes,
            preview_bytes=MAX_PREVIEW_BYTES if preview else None,
        )
        if not content:
            self.remaining -= len(result[0])
        self.checkpoint()
        return result

    async def file(self, path: Path) -> _FileObservation | None:
        """worker 受当前 reader task 拥有，外层取消也先等实际读取终结。"""
        task = asyncio.create_task(asyncio.to_thread(self._read_file, path))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise

    def _read_file(self, path: Path) -> _FileObservation | None:
        self.checkpoint()
        try:
            before = path.lstat()
        except FileNotFoundError:
            return None
        if (stat.S_ISLNK(before.st_mode) or getattr(before, "st_file_attributes", 0) & 0x400
                or not stat.S_ISREG(before.st_mode) or Path(os.path.realpath(path)) != path):
            raise GitIOError("special_path", "links and non-regular paths have no content preview")
        if before.st_size > MAX_FILE_BYTES:
            raise GitIOError("file_limit", "complete file exceeds 16 MiB")
        result = bytearray()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise GitIOError("stale", "file changed while opening")
            while True:
                self.checkpoint()
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                result.extend(chunk)
                digest.update(chunk)
                if len(result) > MAX_FILE_BYTES:
                    raise GitIOError("file_limit", "complete file exceeds 16 MiB")
        after = path.lstat()
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_mode,
                                getattr(value, "st_file_attributes", 0))
        if fields(before) != fields(after) or len(result) != after.st_size:
            raise GitIOError("stale", "file changed while reading")
        self.checkpoint()
        return _FileObservation(bytes(result), digest.hexdigest(), fields(after))


class ChangeReview:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._inventory = GitInventory(workspace)

    async def _basis(self, observation: _Observation) -> tuple[str, ...]:
        head, code, _ = await observation.git("rev-parse", "--verify", "--quiet", "HEAD",
                                              allowed_codes=(0, 1))
        if code:
            branch, branch_code, _ = await observation.git("symbolic-ref", "--quiet", "HEAD",
                                                          allowed_codes=(0, 1))
            if branch_code or not branch.startswith(b"refs/heads/"):
                raise GitIOError("head_unknown", "HEAD cannot be established")
            _, exists, _ = await observation.git("show-ref", "--verify", "--quiet",
                                                 branch.decode("utf-8").strip(), allowed_codes=(0, 1))
            if exists != 1:
                raise GitIOError("head_unknown", "HEAD resolution failed")
            head = b"unborn:" + branch.strip()
        index_path, _, _ = await observation.git("rev-parse", "--path-format=absolute", "--git-path", "index")
        index = Path(index_path.decode("utf-8").rstrip("\r\n"))
        content = await observation.file(index)
        return (head.decode("utf-8").strip(), str(index),
                "missing" if content is None else content.sha256)

    async def _filters(self, observation: _Observation, paths: tuple[str, ...]) -> tuple[str, bool]:
        # 先核对全部有效 filter 命令及候选属性，再允许任何工作区 diff。
        config, _, _ = await observation.git("config", "--null", "--get-regexp",
                                             r"^(filter\..*\.(clean|process)|core\.(autocrlf|eol|safecrlf|attributesfile)|diff\..*)$",
                                             allowed_codes=(0, 1))
        attrs = b""
        if paths:
            attrs, _, _ = await observation.git("check-attr", "-z", "--stdin", *_ATTRIBUTES,
                                               input_data=b"".join(p.encode("utf-8") + b"\0" for p in paths))
        commands = {}
        for record in _records(config):
            key, sep, value = record.partition(b"\n")
            if not sep:
                raise GitIOError("filter_unknown", "filter configuration cannot be checked")
            commands[key] = value
        fields = _records(attrs)
        if len(fields) != len(paths) * len(_ATTRIBUTES) * 3:
            raise GitIOError("filter_unknown", "filter attributes cannot be checked")
        active = False
        for offset in range(0, len(fields), 3):
            if fields[offset + 1] != b"filter":
                continue
            value = fields[offset + 2]
            if value not in (b"unspecified", b"unset", b"set"):
                active |= any(commands.get(b"filter." + value + b"." + suffix)
                              for suffix in (b"clean", b"process"))
        return hashlib.sha256(config + b"\0" + attrs).hexdigest(), active

    def _included(self, path: str) -> bool:
        parts = path.split("/")
        if any(part.casefold() in DEFAULT_EXCLUSIONS for part in parts[:-1]):
            return False
        _, _, directory = self._inventory._inspect_path(parts)
        # 删除目录中的默认排除路径同样不纳入正文。
        return not (parts[-1].casefold() in DEFAULT_EXCLUSIONS and directory)

    async def capture(self, review_id: str, cancel: CancelToken) -> ReviewSnapshot:
        observation = _Observation(self.workspace.root, cancel)
        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                return await self._capture(review_id, observation)
        except (GitIOError, InventoryError, OSError, UnicodeError, TimeoutError) as exc:
            reason = getattr(exc, "kind", "timeout" if isinstance(exc, TimeoutError) else "unreadable")
            return ReviewSnapshot(review_id, str(self.workspace.root), _now(),
                                  tuple(ReviewGroupState(group, "unavailable", reason) for group in GROUPS), ())

    async def _capture(self, review_id: str, observation: _Observation) -> ReviewSnapshot:
        root, _, _ = await observation.git("rev-parse", "--show-toplevel")
        if Path(os.path.realpath(root.decode("utf-8").rstrip("\r\n"))) != self.workspace.root:
            raise GitIOError("root_mismatch", "workspace must be the Git root")
        before = await self._basis(observation)
        tracked, _, _ = await observation.git("ls-files", "--cached", "--stage", "-z")
        tracked_paths = tuple(dict.fromkeys(_path(record.partition(b"\t")[2])
                                           for record in _records(tracked)))
        # Git 的工作区比较会读取文件。已观察到 reparse 祖先时整组拒绝，
        # 保留 staged 对象比较与 untracked 原始读取，避免跟随链接目标。
        special_paths = False
        filter_paths = []
        for index, path in enumerate(tracked_paths):
            if index % 128 == 0:
                await asyncio.sleep(0)
                observation.checkpoint()
            if self._inventory._inspect_path(path.split("/"))[1]:
                special_paths = True
            else:
                filter_paths.append(path)
        filter_paths = tuple(filter_paths)
        filter_basis = "unknown"
        filter_reason = None
        try:
            filter_basis, active = await self._filters(observation, filter_paths)
            if active:
                filter_reason = "external_filter"
        except GitIOError as exc:
            filter_reason = exc.kind
        if special_paths:
            filter_reason = "special_path"
        groups = []
        items = []
        aliases: dict[str, str] = {}
        for group in GROUPS:
            observation.checkpoint()
            if group == "unstaged" and filter_reason:
                groups.append(ReviewGroupState(group, "unavailable", filter_reason))
                continue
            try:
                if group == "untracked":
                    data, _, _ = await observation.git("ls-files", "--others", "--exclude-standard", "-z")
                    candidates = [("A", record.rstrip(b"/")) for record in _records(data)]
                else:
                    data, _, _ = await observation.git("diff", *(('--cached',) if group == "staged" else ()),
                                                       *_DIFF_OPTIONS, "--name-status", "-z", "--")
                    fields = _records(data)
                    if len(fields) % 2:
                        raise GitIOError("invalid_metadata", "invalid Git name-status")
                    candidates = [(fields[i].decode("ascii"), fields[i+1]) for i in range(0, len(fields), 2)]
                selected: dict[str, ReviewItem] = {}
                for index, (status, raw) in enumerate(candidates):
                    if index % 128 == 0:
                        await asyncio.sleep(0)
                        observation.checkpoint()
                    path = _path(raw)
                    if aliases.setdefault(path.casefold(), path) != path:
                        raise GitIOError("alias_conflict", "case-insensitive path aliases")
                    if not self._included(path):
                        continue
                    if status not in _KINDS:
                        raise GitIOError("invalid_metadata", "unsupported Git status")
                    if path not in selected and len(items) + len(selected) >= MAX_ITEMS:
                        raise GitIOError("path_limit", "review exceeds 10000 entries")
                    if path not in selected or status == "U":
                        selected[path] = ReviewItem(uuid.uuid4().hex, group, path, _KINDS[status])
                items.extend(selected.values())
                groups.append(ReviewGroupState(group, "complete"))
            except (GitIOError, InventoryError, OSError, UnicodeError) as exc:
                groups.append(ReviewGroupState(group, "unavailable", getattr(exc, "kind", "unreadable")))
        after = await self._basis(observation)
        if before != after:
            raise GitIOError("stale", "HEAD/index changed during observation")
        if filter_basis != "unknown":
            final_filter, _ = await self._filters(observation, filter_paths)
            if final_filter != filter_basis:
                groups = [ReviewGroupState(g.group, "unknown", "stale") if g.group == "unstaged" else g
                          for g in groups]
                items = [item for item in items if item.group != "unstaged"]
        return ReviewSnapshot(review_id, str(self.workspace.root), _now(), tuple(groups), tuple(items),
                              _basis=before + (filter_basis,), _filter_paths=filter_paths)

    async def read_file(self, snapshot: ReviewSnapshot, item_id: str, cancel: CancelToken) -> ReviewPreview:
        item = next(item for item in snapshot.items if item.item_id == item_id)
        observation = _Observation(self.workspace.root, cancel)
        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                return await self._read(snapshot, item, observation)
        except (GitIOError, InventoryError, OSError, UnicodeError, TimeoutError) as exc:
            reason = getattr(exc, "kind", "timeout" if isinstance(exc, TimeoutError) else "unreadable")
            return ReviewPreview(snapshot.review_id, item_id, item.relative_path, _COMPARISONS[item.group],
                                 "stale" if reason == "stale" else "unavailable", notice=reason,
                                 observed_at=_now())

    async def _read(self, snapshot: ReviewSnapshot, item: ReviewItem,
                    observation: _Observation) -> ReviewPreview:
        if await self._basis(observation) != snapshot._basis[:3]:
            raise GitIOError("stale", "HEAD/index changed; refresh the list")
        if snapshot._basis[3] != "unknown":
            filters, _ = await self._filters(observation, snapshot._filter_paths)
            if filters != snapshot._basis[3]:
                raise GitIOError("stale", "filter configuration changed; refresh the list")
        path = item.relative_path
        stage, _, _ = await observation.git("ls-files", "--stage", "-z", "--", path)
        index_records = _records(stage)
        header = index_records[0].partition(b"\t")[0].split() if len(index_records) == 1 else []
        head_info = b""
        if not snapshot._basis[0].startswith("unborn:"):
            head_info, _, _ = await observation.git("ls-tree", "-z", "HEAD", "--", path)
        modes = {record.partition(b" ")[0] for record in (*index_records, *_records(head_info))}
        exists, reparses, directory = self._inventory._inspect_path(path.split("/"))
        if item.kind == "conflict" or len(index_records) > 1 or (header and header[-1] != b"0"):
            return ReviewPreview(snapshot.review_id, item.item_id, path, _COMPARISONS[item.group],
                                 "unavailable", notice="conflict：不合成多个 stage 的正文", observed_at=_now())
        if reparses or directory or modes & {b"120000", b"160000"}:
            return ReviewPreview(snapshot.review_id, item.item_id, path, _COMPARISONS[item.group],
                                 "unavailable", notice="链接/子模块/目录：仅元数据，不读取目标正文", observed_at=_now())
        filter_before, active = await self._filters(observation, (path,))
        if item.group == "unstaged" and active:
            raise GitIOError("external_filter", "external filter is configured")
        work_before = await observation.file(self.workspace.root / path)
        blobs = []
        if item.group != "untracked":
            if header:
                blob, _, _ = await observation.git("cat-file", "blob", header[1].decode("ascii"), content=True)
                blobs.append(blob)
            if item.group == "staged" and head_info:
                object_id = _records(head_info)[0].partition(b"\t")[0].split()[2]
                blob, _, _ = await observation.git("cat-file", "blob", object_id.decode("ascii"), content=True)
                blobs.append(blob)
        compared = blobs + ([work_before.data] if item.group != "staged" and work_before is not None else [])
        textual = True
        notice = None
        for content in compared:
            if b"\0" in content:
                notice, textual = "binary：只显示元数据", False
                break
            try:
                content.decode("utf-8")
            except UnicodeError:
                notice, textual = "非 UTF-8：只显示元数据", False
                break
        truncated = False
        if item.group == "untracked":
            current, _, _ = await observation.git("ls-files", "--others", "--exclude-standard", "-z", "--", path)
            if path.encode("utf-8") not in _records(current) or work_before is None:
                raise GitIOError("stale", "entry is no longer untracked; refresh")
            patch = b""
            if textual:
                # 全量文件已在 worker 核对；只为有界前缀建立逻辑行，
                # 避免大量短行把 16 MiB 文件膨胀成数百万 Python 字符串。
                source = work_before.data[:MAX_PREVIEW_BYTES]
                truncated = len(source) < len(work_before.data)
                pieces = difflib.unified_diff([], source.decode("utf-8", "ignore").splitlines(keepends=True),
                                              fromfile="/dev/null", tofile=path)
                output = bytearray()
                for line in pieces:
                    observation.checkpoint()
                    encoded = line.encode("utf-8")
                    remaining = MAX_PREVIEW_BYTES - len(output)
                    output.extend(encoded[:remaining])
                    if len(encoded) > remaining:
                        truncated = True
                        break
                patch = bytes(output)
        else:
            status, _, _ = await observation.git("diff", *(('--cached',) if item.group == "staged" else ()),
                                                 *_DIFF_OPTIONS, "--name-status", "-z", "--", path)
            fields = _records(status)
            if not fields or fields[0].decode("ascii") not in _KINDS or _KINDS[fields[0].decode("ascii")] != item.kind:
                raise GitIOError("stale", "entry status changed; refresh")
            patch = b""
            if textual:
                patch, _, truncated = await observation.git(
                    "diff", *(('--cached',) if item.group == "staged" else ()), *_DIFF_OPTIONS,
                    "--patch", "--", path, content=True, preview=True)
        work_after = await observation.file(self.workspace.root / path)
        filter_after, _ = await self._filters(observation, (path,))
        exists_after, reparse_after, directory_after = self._inventory._inspect_path(path.split("/"))
        if (work_before != work_after or filter_before != filter_after
                or (exists, reparses, directory) != (exists_after, reparse_after, directory_after)
                or await self._basis(observation) != snapshot._basis[:3]):
            raise GitIOError("stale", "comparison changed during reading; refresh")
        if textual:
            facts = []
            if work_before is not None:
                if work_before.data.startswith(b"\xef\xbb\xbf"):
                    facts.append("工作区 UTF-8 BOM")
                facts.append("工作区 CRLF" if b"\r\n" in work_before.data else "工作区 LF/无换行")
            facts.append("Git 原生 EOL 比较" if item.group != "untracked" else "原始读取")
            if not patch:
                facts.append("空文本差异；仍有新增/模式等元数据变化")
            notice = " · ".join(facts)
        # 截断时仅丢掉最后一个不完整 UTF-8 码点；完整非 UTF-8 在上面已明确拒绝。
        text = patch.decode("utf-8", "ignore" if truncated else "strict")
        return ReviewPreview(snapshot.review_id, item.item_id, path, _COMPARISONS[item.group],
                             "truncated" if truncated else "complete", text, notice, _now())


def change_review_plugin() -> PluginDefinition:
    def setup(ctx: PluginContext) -> None:
        ctx.provide(CHANGE_REVIEW_CAPABILITY, ChangeReview(ctx.require("coding.workspace")))

    return PluginDefinition("coding-change-review", setup,
                            provides=frozenset({CHANGE_REVIEW_CAPABILITY}),
                            requires=frozenset({"coding.workspace"}))
