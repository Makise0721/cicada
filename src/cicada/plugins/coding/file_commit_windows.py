"""Windows 发布 Adapter (file_access 私有): 支持性核对、独占暂存与单次原生发布.

ctypes 显式声明 Unicode/Win32 类型与 use_last_error, 无第三方依赖.
政策与结果分类归 file_access.py; 本模块只做原生机制 (契约 C3):
- 首版范围: 本地盘符 NTFS、普通单链接、无 reparse/ADS;
- 覆盖发布 ReplaceFileW flags=0 + 本次 backup; 创建分支 (MoveFileExW) 由票据 02 交付.
"""

from __future__ import annotations

import ctypes
import os
import stat
import sys
from ctypes import wintypes
from pathlib import Path

ERROR_FILE_NOT_FOUND = 2
ERROR_PATH_NOT_FOUND = 3
ERROR_ACCESS_DENIED = 5
ERROR_SHARING_VIOLATION = 32
ERROR_HANDLE_EOF = 38

# 只有同时具备实物证据 (目标仍等于原件/新件仍在/无 backup) 时才允许判未提交的
# 文档化前置失败错误; 1176/1177 等一律保持 unknown.
NOT_COMMITTED_WINERRORS = frozenset(
    {ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION}
)


class UnsupportedTargetError(Exception):
    """目标不属于首版支持范围 (非本地 NTFS/UNC/目录/reparse/ADS/多硬链接/非 Windows)."""


class NativePublishError(Exception):
    """原生发布调用失败; winerror 为 GetLastError 原始值, 不做文本猜测."""

    def __init__(self, action: str, winerror: int) -> None:
        self.winerror = winerror
        super().__init__(f"{action} failed with winerror {winerror}")


class _FIND_STREAM_DATA(ctypes.Structure):
    _fields_ = [
        ("StreamSize", wintypes.LARGE_INTEGER),
        ("cStreamName", ctypes.c_wchar * 296),  # MAX_PATH + 流名最大后缀
    ]


_INVALID_HANDLE = ctypes.c_void_p(-1).value

_kernel32 = None
_ReplaceFileW = None
_GetVolumeInformationW = None
_FindFirstStreamW = None
_FindNextStreamW = None
_FindClose = None

if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _ReplaceFileW = _kernel32.ReplaceFileW
    _ReplaceFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID,
    ]
    _ReplaceFileW.restype = wintypes.BOOL

    _GetVolumeInformationW = _kernel32.GetVolumeInformationW
    _GetVolumeInformationW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD,
        wintypes.LPWSTR, wintypes.DWORD,
    ]
    _GetVolumeInformationW.restype = wintypes.BOOL

    _FindFirstStreamW = _kernel32.FindFirstStreamW
    _FindFirstStreamW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                  ctypes.POINTER(_FIND_STREAM_DATA), wintypes.DWORD]
    _FindFirstStreamW.restype = ctypes.c_void_p

    _FindNextStreamW = _kernel32.FindNextStreamW
    _FindNextStreamW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FIND_STREAM_DATA)]
    _FindNextStreamW.restype = wintypes.BOOL

    _FindClose = _kernel32.FindClose
    _FindClose.argtypes = [wintypes.HANDLE]
    _FindClose.restype = wintypes.BOOL


def _require_windows() -> None:
    if sys.platform != "win32":
        raise UnsupportedTargetError(
            f"native publication supports Windows only (current platform: {sys.platform})"
        )


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def reparse_ancestor(path: Path) -> str | None:
    """返回 path 自身或祖先中第一个 reparse 点; 无则 None. 不存在组件视为无 (由存在性检查处理)."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError:
            return None
        if _is_reparse(info):
            return str(current)
    return None


def volume_fs_name(path: Path) -> str:
    _require_windows()
    root = os.path.splitdrive(str(path))[0] + "\\"
    fs_buf = ctypes.create_unicode_buffer(64)
    if not _GetVolumeInformationW(root, None, 0, None, None, None, fs_buf, 64):
        raise UnsupportedTargetError(
            f"cannot query volume information for {root} (winerror {ctypes.get_last_error()})"
        )
    return fs_buf.value


def has_alternate_streams(path: Path) -> bool:
    """默认流 '::$DATA' 之外仍有命名流 → ADS. 枚举失败按不支持目标拒绝."""
    _require_windows()
    data = _FIND_STREAM_DATA()
    handle = _FindFirstStreamW(str(path), 0, ctypes.byref(data), 0)
    if handle in (None, 0) or handle == _INVALID_HANDLE:
        raise UnsupportedTargetError(
            f"cannot enumerate streams of {path} (winerror {ctypes.get_last_error()})"
        )
    try:
        while True:
            if data.cStreamName != "::$DATA":
                return True
            if not _FindNextStreamW(handle, ctypes.byref(data)):
                if ctypes.get_last_error() != ERROR_HANDLE_EOF:
                    raise UnsupportedTargetError(
                        f"stream enumeration of {path} failed (winerror {ctypes.get_last_error()})"
                    )
                return False
    finally:
        _FindClose(handle)


def check_mutation_target(original: Path, canonical: Path) -> None:
    """修改支持范围: 本地盘符、原始路径与 canonical 均无 reparse、NTFS、普通单链接、无 ADS."""
    _require_windows()
    raw = str(canonical)
    if raw.startswith("\\\\") or len(canonical.drive) < 2 or not canonical.is_absolute():
        raise UnsupportedTargetError(
            f"only local absolute drive-letter paths are supported: {canonical}"
        )
    for label, candidate in (("original", original), ("canonical", canonical)):
        reparse = reparse_ancestor(candidate)
        if reparse is not None:
            raise UnsupportedTargetError(
                f"{label} path crosses a reparse point (junction/symlink): {reparse}; "
                "edit the real path directly"
            )
    fs = volume_fs_name(canonical)
    if fs.upper() != "NTFS":
        raise UnsupportedTargetError(f"only local NTFS volumes are supported: {canonical} is {fs}")
    try:
        info = canonical.lstat()
    except OSError as exc:
        raise UnsupportedTargetError(f"cannot inspect target {canonical}: {exc}") from exc
    if stat.S_ISDIR(info.st_mode):
        raise UnsupportedTargetError(f"target is a directory: {canonical}")
    if not stat.S_ISREG(info.st_mode):
        raise UnsupportedTargetError(f"target is not a regular file: {canonical}")
    if info.st_nlink > 1:
        raise UnsupportedTargetError(
            f"target has {info.st_nlink} hard links; only single-link files are supported"
        )
    if has_alternate_streams(canonical):
        raise UnsupportedTargetError(f"target has alternate data streams: {canonical}")


def write_staged_file(staged: Path, data: bytes) -> None:
    """独占创建 staging 文件, 完整写入后 flush + fsync 再关闭."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(staged, flags)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def replace_file(target: Path, replacement: Path, backup: Path) -> None:
    """覆盖发布: ReplaceFileW flags=0, 不忽略 ACL/合并错误. 失败抛 NativePublishError."""
    _require_windows()
    if not _ReplaceFileW(str(target), str(replacement), str(backup), 0, None, None):
        raise NativePublishError(
            f"ReplaceFileW(target={target}, replacement={replacement}, backup={backup})",
            ctypes.get_last_error(),
        )
