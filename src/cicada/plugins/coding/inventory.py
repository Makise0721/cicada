"""有界 Git 文件清单；保留原始路径上的 reparse/gitlink 事实，不读取内容."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.workspace import Workspace, WorkspaceError
from cicada.runtime.plugin import PluginContext, PluginDefinition

INVENTORY_CAPABILITY = "coding.inventory"
POLICY_ID = "cicada-git-files-v1"
DEFAULT_EXCLUSIONS = (
    ".git", ".cicada", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".pytest-tmp",
)
MAX_GIT_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_PATHS = 10_000
QUERY_TIMEOUT_S = 10.0


class InventoryError(WorkspaceError):
    """整个清单失败；不得把部分候选解释为完整搜索范围."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True)
class InventoryEntry:
    relative_path: str
    git_modes: tuple[str, ...]  # 空表示 untracked；合并冲突可能有多个 mode。
    exists: bool | None  # reparse 祖先后的候选存在性未知，不继续 lstat。
    reparse_paths: tuple[str, ...]  # 第一个遇到的 reparse 祖先/候选，不继续解引用。
    is_submodule: bool = False

    @property
    def tracked(self) -> bool:
        return bool(self.git_modes)

    @property
    def is_reparse(self) -> bool:
        return bool(self.reparse_paths) or "120000" in self.git_modes


@dataclass(frozen=True)
class FileInventory:
    root: Path
    files: tuple[InventoryEntry, ...]
    excluded_count: int
    policy_id: str = POLICY_ID
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.relative_path for entry in self.files)


class GitInventory:
    """同一 deadline 覆盖 root 检查、两类 Git 清单和路径元信息检查.

    可降低限额以适配更小工作区；不能超过本阶段公布的硬上限。
    只在 list_files 时检查 Git/root，普通非 Git 工作区仍能启动原工具。
    """

    def __init__(self, workspace: Workspace, *, max_output_bytes: int = MAX_GIT_OUTPUT_BYTES,
                 max_paths: int = MAX_PATHS, timeout_s: float = QUERY_TIMEOUT_S) -> None:
        for name, value, ceiling in (("max_output_bytes", max_output_bytes, MAX_GIT_OUTPUT_BYTES),
                                     ("max_paths", max_paths, MAX_PATHS)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if isinstance(timeout_s, bool) or not 0 < timeout_s <= QUERY_TIMEOUT_S:
            raise ValueError(f"timeout_s must be in (0, {QUERY_TIMEOUT_S}]")
        self.workspace = workspace
        self.max_output_bytes = max_output_bytes
        self.max_paths = max_paths
        self.timeout_s = timeout_s

    async def list_files(self, cancel: CancelToken) -> FileInventory:
        cancel.throw_if_cancelled()
        deadline = time.monotonic() + self.timeout_s
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self._list_files(cancel, deadline)
        except TimeoutError as exc:
            raise InventoryError("timeout", "Git inventory query deadline exceeded") from exc
        except OSError as exc:
            raise InventoryError("path_unreadable", f"inventory path inspection failed: {exc}") from exc

    async def _list_files(self, cancel: CancelToken, deadline: float) -> FileInventory:
        root = self.workspace.root
        remaining = self.max_output_bytes
        root_bytes = await self._git(("rev-parse", "--show-toplevel"), cancel, remaining)
        remaining -= len(root_bytes)
        try:
            git_root = Path(os.path.realpath(root_bytes.decode("utf-8").rstrip("\r\n")))
        except UnicodeError as exc:
            raise InventoryError("invalid_path", "Git root is not UTF-8") from exc
        if git_root != root:
            raise InventoryError("root_mismatch", f"workspace must be the Git worktree root: {git_root}")
        tracked = await self._git(("ls-files", "--cached", "--stage", "-z"), cancel, remaining)
        remaining -= len(tracked)
        untracked = await self._git(("ls-files", "--others", "--exclude-standard", "-z"), cancel, remaining)
        modes: dict[str, set[str]] = {}
        nested_repos: set[str] = set()
        aliases: dict[str, str] = {}
        for data, staged in ((tracked, True), (untracked, False)):
            if data and not data.endswith(b"\0"):
                raise InventoryError("invalid_inventory", "Git inventory lacks a terminating NUL")
            for index, record in enumerate(data.split(b"\0")[:-1]):
                if index % 128 == 0:
                    await asyncio.sleep(0)
                    self._checkpoint(cancel, deadline)
                mode = None
                if staged:
                    header, sep, record = record.partition(b"\t")
                    fields = header.split(b" ")
                    if not sep or len(fields) != 3 or fields[0] not in (b"100644", b"100755", b"120000", b"160000"):
                        raise InventoryError("invalid_inventory", "invalid Git stage record")
                    mode = fields[0].decode("ascii")
                try:
                    path = record.decode("utf-8")
                except UnicodeError as exc:
                    raise InventoryError("invalid_path", "Git path is not UTF-8") from exc
                nested = not staged and path.endswith("/")
                if nested:
                    path = path[:-1]
                    nested_repos.add(path)
                parts = path.split("/")
                if not path or PurePosixPath(path).is_absolute() or any(p in ("", ".", "..") for p in parts) or "\\" in path or ":" in path:
                    raise InventoryError("invalid_path", f"invalid Git relative path: {path!r}")
                folded = path.casefold()
                if folded in aliases and aliases[folded] != path:
                    raise InventoryError("alias_conflict", f"case-insensitive path aliases: {aliases[folded]!r}, {path!r}")
                aliases[folded] = path
                modes.setdefault(path, set())
                if mode is not None:
                    modes[path].add(mode)
                if len(modes) > self.max_paths:
                    raise InventoryError("path_limit", f"Git inventory exceeds {self.max_paths} paths")
        entries: list[InventoryEntry] = []
        excluded = 0
        for index, path in enumerate(sorted(modes, key=lambda p: (p.casefold(), p))):
            if index % 128 == 0:
                await asyncio.sleep(0)
            self._checkpoint(cancel, deadline)
            parts = path.split("/")
            if any(p.casefold() in DEFAULT_EXCLUSIONS for p in parts[:-1]):
                excluded += 1
                continue
            exists, reparse, directory = self._inspect_path(parts)
            if directory and parts[-1].casefold() in DEFAULT_EXCLUSIONS:
                excluded += 1
                continue
            entries.append(InventoryEntry(path, tuple(sorted(modes[path])), exists, reparse,
                                          "160000" in modes[path] or path in nested_repos))
        self._checkpoint(cancel, deadline)
        return FileInventory(root, tuple(entries), excluded)

    @staticmethod
    def _checkpoint(cancel: CancelToken, deadline: float) -> None:
        cancel.throw_if_cancelled()
        if time.monotonic() >= deadline:
            raise InventoryError("timeout", "Git inventory query deadline exceeded")

    def _inspect_path(self, parts: list[str]) -> tuple[bool | None, tuple[str, ...], bool]:
        current = self.workspace.root
        for index, part in enumerate(parts):
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                return False, (), False
            directory = stat.S_ISDIR(info.st_mode)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return (True if index == len(parts) - 1 else None), ("/".join(parts[:index + 1]),), directory
            if index < len(parts) - 1 and not directory:
                raise InventoryError("path_unreadable", f"inventory ancestor is not a directory: {current}")
        return True, (), directory

    async def _git(self, arguments: tuple[str, ...], cancel: CancelToken, budget: int) -> bytes:
        environment = {k: v for k, v in os.environ.items() if k.upper() not in {
            "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"}}
        environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "-c", "core.fsmonitor=false", "-C", str(self.workspace.root), *arguments,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=environment)
        except FileNotFoundError as exc:
            raise InventoryError("git_unavailable", "Git executable is unavailable") from exc
        except OSError as exc:
            raise InventoryError("git_launch_failed", f"cannot launch Git: {exc}") from exc

        async def read_stdout() -> bytes:
            output = bytearray()
            while chunk := await process.stdout.read(4096):
                if len(output) + len(chunk) > budget:
                    raise InventoryError("output_limit", f"Git stdout exceeds {self.max_output_bytes} bytes")
                output.extend(chunk)
            return bytes(output)

        async def read_stderr() -> bytes:
            tail = b""
            while chunk := await process.stderr.read(4096):
                tail = (tail + chunk)[-4096:]
            return tail

        readers = [asyncio.create_task(read_stdout()), asyncio.create_task(read_stderr()),
                   asyncio.create_task(process.wait())]
        operation = asyncio.gather(*readers)
        cancellation = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait((operation, cancellation), return_when=asyncio.FIRST_COMPLETED)
            if cancellation in done:
                cancel.throw_if_cancelled()
            output, stderr, returncode = await operation
            if returncode:
                raise InventoryError("git_failed", f"Git failed ({returncode}): {stderr.decode('utf-8', 'replace')}")
            return output
        finally:
            cancellation.cancel()
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            operation.cancel()
            for reader in readers:
                reader.cancel()
            await asyncio.gather(operation, cancellation, *readers, return_exceptions=True)
            try:
                await asyncio.wait_for(process.wait(), timeout=1)
            except TimeoutError:
                pass  # 已请求 kill；清理不无限延长查询。


def inventory_plugin() -> PluginDefinition:
    def setup(ctx: PluginContext) -> None:
        ctx.provide(INVENTORY_CAPABILITY, GitInventory(ctx.require("coding.workspace")))

    return PluginDefinition("coding-inventory", setup, provides=frozenset({INVENTORY_CAPABILITY}),
                            requires=frozenset({"coding.workspace"}))
