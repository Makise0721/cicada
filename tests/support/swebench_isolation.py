"""P1 私有材料隔离: private/ 封印 + 受限低权子进程 + 每次准入负向验收。

机制 (2026-10-15 本机实跑核验, 见 .scratch/preview-readiness/p1-isolation-design.md):
- 挡读: DACL 上的 AppContainer 包 SID (``S-1-15-2-<多段>``) ACE。Windows 内核对
  Low 完整性令牌执行强制检查, 带该 ACE 的对象对其不可达; Medium 完整性评估方不受影响。
- 挡删/改名: **两条独立屏障, 任一即可** —— 实测 2x2 矩阵 (probe_p1_matrix.py):
    父目录有可继承包 SID ACE  → 删/改名被拒
    父目录 deny Everyone:(DC)  → 删/改名被拒
    仅**文件**盖包 SID ACE     → 读被拒, 但**删/改名仍成功**
  因为删除可由父目录的 FILE_DELETE_CHILD 授权, 与文件自身 ACE 无关 (DSH 文档原话:
  "Windows can also authorize a write or a delete from the parent directory's
  FILE_DELETE_CHILD right, which no restricting SID has to co-sign")。故封印**同时**
  盖目录 + 拒绝 DC, 二者互为纵深。
- 断继承: private/ 若继承到 ``CodexSandboxUsers:(M)`` / ``S-1-4-*:(W,D,DC)`` 等 ACE,
  封印形同虚设, 故先 ``/inheritance:r``。
- 执行身份: 未提权进程可自行 ``CreateRestrictedToken`` + 降 Low 完整性 +
  ``CreateProcessAsUserW``, 无需新账户、凭据或服务。

破坏性判定只认**效果** (文件是否还在、内容是否未变): 实测 ``del`` 被拒时
``errorlevel`` 仍为 0, 按退出码判定会把"删除失败"误报成"删除成功"。

边界 (必须如实记录, 不得声称已覆盖):
- 只挡"模型侧进程经本地文件系统读取"。用户自己开的 pwsh 是 Owner + Medium, 仍可读。
- 挡不住联网: gold 与 tests/config.json 可从公开 raw.githubusercontent.com 匿名取回。
- 依赖未公开内核规则 (DSH 文档自标 unconfirmed), 故负向验收每次准入重跑, 不信一次结论。
- 只保护 private/; workspace/temp 等可写目录的同类处理未在本模块内验证。

定向: 只服务 tests/e2e/preview_runner.py 与后继 launcher; 不改生产插件行为。
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 受保护材料的包 SID。挡读要求**多子授权** (>=3 段) 形式: 实测
# `S-1-15-2-1` / `-2` / `-99` / `-12345` 等短值**不触发**内核规则, 而 `-1`/`-2`
# 是 "ALL APPLICATION PACKAGES" 通配组 (DSH 诊断脚本正则 '^S-1-15-2-(?![12]$)')。
# 本值固定为常量并写入 lock 供复核; asar 内不存在可读取的"真实包 SID 接缝"
# (packageSid/package_sid 出现 0 次), 故不假装从沙箱 seam 取值。
PROTECTED_PACKAGE_SID = "S-1-15-2-2694941230-1403535211-3243528075-2169540821"

EVERYONE_SID = "S-1-1-0"
LOW_INTEGRITY_SID = "S-1-16-4096"
CANARY_NAME = "isolation-canary.txt"
CANARY_PAYLOAD = b"ISOLATION_CANARY_MUST_NOT_BE_READABLE_BY_THE_MODEL_SIDE\n"


class IsolationError(RuntimeError):
    """封印或负向验收失败; 保留具体不足即停止, 不降级放行。"""


def package_sid_subauthorities(sid: str) -> int:
    """返回 ``S-1-15-2-...`` 的子授权段数; 非该族返回 0。"""
    parts = sid.split("-")
    if len(parts) < 4 or parts[:3] != ["S", "1", "15"] or parts[3] != "2":
        return 0
    return len(parts) - 4


def validate_package_sid(sid: str) -> None:
    """封印前守卫: 形状错误会让"封印成功"变成静默失效。"""
    count = package_sid_subauthorities(sid)
    if count == 0:
        raise IsolationError(f"package SID is not in the S-1-15-2- family: {sid}")
    tail = sid.split("-")[4:]
    if len(tail) < 3:
        raise IsolationError(
            f"package SID needs >=3 subauthorities to trip the kernel rule: {sid}")
    if len(tail) == 1 and tail[0] in ("1", "2"):
        raise IsolationError(
            f"package SID {sid} names ALL application packages; refusing to stamp with it")


# --- Win32: 受限 + Low 完整性令牌 ---

if sys.platform == "win32":
    import ctypes.wintypes as _wt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    _kernel32.GetCurrentProcess.restype = _wt.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.CloseHandle.argtypes = [_wt.HANDLE]
    _kernel32.WaitForSingleObject.argtypes = [_wt.HANDLE, _wt.DWORD]
    _kernel32.GetExitCodeProcess.argtypes = [_wt.HANDLE, ctypes.POINTER(_wt.DWORD)]
    _advapi32.OpenProcessToken.argtypes = [
        _wt.HANDLE, _wt.DWORD, ctypes.POINTER(_wt.HANDLE)]
    _advapi32.OpenProcessToken.restype = _wt.BOOL
    _advapi32.CreateRestrictedToken.argtypes = [
        _wt.HANDLE, _wt.DWORD, _wt.DWORD, ctypes.c_void_p, _wt.DWORD, ctypes.c_void_p,
        _wt.DWORD, ctypes.c_void_p, ctypes.POINTER(_wt.HANDLE)]
    _advapi32.CreateRestrictedToken.restype = _wt.BOOL
    _advapi32.SetTokenInformation.argtypes = [
        _wt.HANDLE, ctypes.c_int, ctypes.c_void_p, _wt.DWORD]
    _advapi32.SetTokenInformation.restype = _wt.BOOL
    _advapi32.ConvertStringSidToSidW.argtypes = [
        _wt.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    _advapi32.ConvertStringSidToSidW.restype = _wt.BOOL
    _advapi32.CreateProcessAsUserW.argtypes = [
        _wt.HANDLE, _wt.LPCWSTR, _wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
        _wt.BOOL, _wt.DWORD, ctypes.c_void_p, _wt.LPCWSTR, ctypes.c_void_p,
        ctypes.c_void_p]
    _advapi32.CreateProcessAsUserW.restype = _wt.BOOL
    _advapi32.LookupAccountSidW.argtypes = [
        _wt.LPCWSTR, ctypes.c_void_p, _wt.LPWSTR, ctypes.POINTER(_wt.DWORD),
        _wt.LPWSTR, ctypes.POINTER(_wt.DWORD), ctypes.POINTER(ctypes.c_int)]
    _advapi32.LookupAccountSidW.restype = _wt.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    class _SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", _wt.DWORD)]

    class _TOKEN_MANDATORY_LABEL(ctypes.Structure):
        _fields_ = [("Label", _SID_AND_ATTRIBUTES)]

    class _STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", _wt.DWORD), ("lpReserved", _wt.LPWSTR), ("lpDesktop", _wt.LPWSTR),
            ("lpTitle", _wt.LPWSTR), ("dwX", _wt.DWORD), ("dwY", _wt.DWORD),
            ("dwXSize", _wt.DWORD), ("dwYSize", _wt.DWORD),
            ("dwXCountChars", _wt.DWORD), ("dwYCountChars", _wt.DWORD),
            ("dwFillAttribute", _wt.DWORD), ("dwFlags", _wt.DWORD),
            ("wShowWindow", _wt.WORD), ("cbReserved2", _wt.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
            ("hStdInput", _wt.HANDLE), ("hStdOutput", _wt.HANDLE),
            ("hStdError", _wt.HANDLE)]

    class _PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", _wt.HANDLE), ("hThread", _wt.HANDLE),
                    ("dwProcessId", _wt.DWORD), ("dwThreadId", _wt.DWORD)]

    _TOKEN_DUPLICATE = 0x0002
    _TOKEN_QUERY = 0x0008
    _TOKEN_ASSIGN_PRIMARY = 0x0001
    _TOKEN_ADJUST_DEFAULT = 0x0080
    _TOKEN_ADJUST_SESSIONID = 0x0100
    _DISABLE_MAX_PRIVILEGE = 0x1
    _SE_GROUP_INTEGRITY = 0x20
    _TOKEN_INTEGRITY_LEVEL = 25
    _INFINITE = 0xFFFFFFFF


def _low_token():
    """当前用户的受限令牌, 完整性降为 Low。供负向验收与后继 launcher 共用。"""
    if sys.platform != "win32":
        raise IsolationError("the isolation gate requires Windows")
    handle = _wt.HANDLE()
    if not _advapi32.OpenProcessToken(
            _kernel32.GetCurrentProcess(),
            _TOKEN_DUPLICATE | _TOKEN_QUERY | _TOKEN_ASSIGN_PRIMARY
            | _TOKEN_ADJUST_DEFAULT | _TOKEN_ADJUST_SESSIONID, ctypes.byref(handle)):
        raise IsolationError(f"OpenProcessToken failed: {ctypes.get_last_error()}")
    restricted = _wt.HANDLE()
    if not _advapi32.CreateRestrictedToken(
            handle, _DISABLE_MAX_PRIVILEGE, 0, None, 0, None, 0, None,
            ctypes.byref(restricted)):
        _kernel32.CloseHandle(handle)
        raise IsolationError(
            f"CreateRestrictedToken failed: {ctypes.get_last_error()}")
    psid = ctypes.c_void_p()
    if not _advapi32.ConvertStringSidToSidW(LOW_INTEGRITY_SID, ctypes.byref(psid)):
        raise IsolationError("ConvertStringSidToSidW(Low) failed")
    label = _TOKEN_MANDATORY_LABEL()
    label.Label.Sid = psid
    label.Label.Attributes = _SE_GROUP_INTEGRITY
    if not _advapi32.SetTokenInformation(
            restricted, _TOKEN_INTEGRITY_LEVEL, ctypes.byref(label),
            ctypes.sizeof(label) + 8):
        raise IsolationError(
            f"SetTokenInformation(Low) failed: {ctypes.get_last_error()}")
    _kernel32.CloseHandle(handle)
    return restricted


def low_child_exit_code(command: str, cwd: Path) -> int:
    """在受限 + Low 完整性子进程内跑一条 cmd 命令, 返回其退出码。

    刻意只回退出码而不抓管道: 受限环境下管道/命名管道不可用, 且"文件是否还在、
    内容是否未变"这类事实必须由父侧 (评估方) 直接核验, 不采信子进程自报。
    """
    token = _low_token()
    try:
        startup = _STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        info = _PROCESS_INFORMATION()
        cmd_exe = str(Path(os.environ.get("SystemRoot", r"C:\Windows"))
                      / "System32" / "cmd.exe")
        argv = ctypes.create_unicode_buffer(f'"{cmd_exe}" /c {command}')
        ok = _advapi32.CreateProcessAsUserW(
            token, None, argv, None, None, False, 0, None, str(cwd),
            ctypes.byref(startup), ctypes.byref(info))
        if not ok:
            raise IsolationError(
                f"CreateProcessAsUserW failed: {ctypes.get_last_error()}")
        try:
            _kernel32.WaitForSingleObject(info.hProcess, _INFINITE)
            code = _wt.DWORD()
            _kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
            return int(code.value)
        finally:
            _kernel32.CloseHandle(info.hThread)
            _kernel32.CloseHandle(info.hProcess)
    finally:
        _kernel32.CloseHandle(token)


def _icacls(*args: str) -> subprocess.CompletedProcess:
    record = subprocess.run(["icacls", *args], capture_output=True, text=True,
                            timeout=300)
    return record


class LowProcess:
    """评估方保有真实进程句柄, 可轮询退出并在协作取消失败后终止整棵树。"""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str]):
        token = _low_token()
        startup = _STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        info = _PROCESS_INFORMATION()
        environment = ctypes.create_unicode_buffer(
            "\0".join(f"{key}={value}" for key, value in sorted(env.items())) + "\0\0")
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
        try:
            if not _advapi32.CreateProcessAsUserW(
                    token, str(argv[0]), command, None, None, False,
                    0x00000400 | 0x08000000, environment, str(cwd),
                    ctypes.byref(startup), ctypes.byref(info)):
                raise IsolationError(f"low process launch failed: {ctypes.get_last_error()}")
            self.handle, self.pid = info.hProcess, int(info.dwProcessId)
            _kernel32.CloseHandle(info.hThread)
        finally:
            _kernel32.CloseHandle(token)

    def poll(self) -> int | None:
        if _kernel32.WaitForSingleObject(self.handle, 0) == 0x102:
            return None
        code = _wt.DWORD()
        if not _kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise IsolationError("cannot confirm child exit")
        return int(code.value)

    def terminate_tree(self) -> None:
        subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"],
                       capture_output=True, timeout=20)

    def close(self) -> None:
        _kernel32.CloseHandle(self.handle)


def allow_low_write(directory: Path) -> None:
    """只在显式列出的模型工作目录给 Low 子进程读写权限。"""
    directory.mkdir(parents=True, exist_ok=True)
    for args in ((str(directory), "/grant", f"*{EVERYONE_SID}:(OI)(CI)M"),
                 (str(directory), "/setintegritylevel", "(OI)(CI)L", "/T")):
        if _icacls(*args).returncode:
            raise IsolationError("cannot authorize low writable directory")


def _acl_text(path: Path) -> str:
    record = _icacls(str(path))
    return (record.stdout or "") + (record.stderr or "")


def _sid_account_name(sid: str) -> str | None:
    """把 SID 解析成 icacls 会打印的账户名 (如 S-1-1-0 → Everyone)。

    icacls 输出用的是**账户名**而非 SID 字符串, 按 SID 字面量核对 ACL 会永远落空;
    解析失败 (如合成的包 SID) 返回 None, 此时 icacls 原样打印 SID 字符串。
    """
    if sys.platform != "win32":
        return None
    psid = ctypes.c_void_p()
    if not _advapi32.ConvertStringSidToSidW(sid, ctypes.byref(psid)):
        return None
    try:
        name = ctypes.create_unicode_buffer(256)
        domain = ctypes.create_unicode_buffer(256)
        name_len, domain_len = _wt.DWORD(256), _wt.DWORD(256)
        use = ctypes.c_int()
        if not _advapi32.LookupAccountSidW(
                None, psid, name, ctypes.byref(name_len), domain,
                ctypes.byref(domain_len), ctypes.byref(use)):
            return None
        return name.value
    finally:
        if psid:
            _kernel32.LocalFree(psid)


def _acl_has_sid(path: Path, sid: str) -> bool:
    """ACL 中是否存在该 SID 对应的主体 (按账户名或 SID 字面量匹配)。"""
    text = _acl_text(path)
    if sid in text:
        return True
    account = _sid_account_name(sid)
    return bool(account) and account in text


def _inherited_access_aces(path: Path) -> list[str]:
    """仍从父目录继承下来的**访问** ACE。

    完整性标签 (``Mandatory Label\\...``) 的继承规则与访问 ACE 独立, 且标签行自身
    常带 ``(I)``; 故必须排除标签行, 只看访问 ACE。断继承不彻底 = 封印形同虚设
    (继承的 ``CodexSandboxUsers:(M)`` / ``S-1-4-*:(W,D,DC)`` 会直接放行)。
    """
    leaked: list[str] = []
    for line in _acl_text(path).splitlines():
        stripped = line.strip()
        if not stripped or "Mandatory Label" in stripped:
            continue
        if "(I)" in stripped and ":" in stripped:
            leaked.append(stripped)
    return leaked


def _has_delete_child_deny(path: Path) -> bool:
    """DACL 中是否有一条拒绝 FILE_DELETE_CHILD (``DC``) 的条目。

    icacls 输出形如 ``Everyone:(DENY)(DC)``; 这里要求同一行同时出现 DENY 与 DC,
    避免把别处的 ``DC`` 字样误当成本条屏障。
    """
    for line in _acl_text(path).splitlines():
        upper = line.upper()
        if "DENY" in upper and "(DC)" in upper:
            return True
    return False


def current_user() -> str:
    domain = os.environ.get("USERDOMAIN", "")
    user = os.environ.get("USERNAME", "")
    return f"{domain}\\{user}" if domain else user


# --- 封印 ---


def seal_private(private_dir: Path) -> dict:
    """封印 private/: 断继承 + 评估方全权 + 包 SID ACE + 拒绝删除子项。

    每步都核对**效果** (DACL 是否真的含该 SID), 不只信 icacls 的退出码 ——
    中文 Windows 下 icacls 报文是本地化的, 按文本匹配不可靠, 按 effect 核对才可靠。
    """
    private_dir = Path(private_dir)
    if not private_dir.is_dir():
        raise IsolationError(f"private dir missing: {private_dir}")
    validate_package_sid(PROTECTED_PACKAGE_SID)
    owner = current_user()

    steps: list[dict] = []

    def step(label: str, args: tuple[str, ...], verify) -> None:
        record = _icacls(*args)
        ok = record.returncode == 0 and verify()
        steps.append({"step": label, "exit_code": record.returncode, "ok": bool(ok)})
        if not ok:
            raise IsolationError(f"sealing step failed: {label}")

    def has(sid: str):
        return lambda: _acl_has_sid(private_dir, sid)

    def dropped_inheritance() -> bool:
        return not _inherited_access_aces(private_dir)

    step("break-inheritance", (str(private_dir), "/inheritance:r"), dropped_inheritance)
    step("evaluator-full", (str(private_dir), "/grant:r", f"{owner}:(OI)(CI)F"),
         has(owner))
    step("everyone-read", (str(private_dir), "/grant", f"*{EVERYONE_SID}:(OI)(CI)R"),
         has(EVERYONE_SID))
    # 删除屏障之二: 拒绝 Everyone 的 FILE_DELETE_CHILD。与目录的包 SID ACE 互为纵深
    # (矩阵实测: 只盖文件时"读被拒但删成功"; 目录 ACE 与 DC deny 任一即可挡住删除)。
    step("deny-delete-child", (str(private_dir), "/deny", f"*{EVERYONE_SID}:(DC)"),
         lambda: "DC" in _acl_text(private_dir))
    # 删除屏障之一 + 防列目录: 目录自身盖章 (可继承给内容)。
    step("stamp-directory",
         (str(private_dir), "/grant", f"*{PROTECTED_PACKAGE_SID}:(OI)(CI)R"),
         has(PROTECTED_PACKAGE_SID))

    stamped: list[str] = []
    for path in sorted(private_dir.rglob("*")):
        if not path.is_file():
            continue
        step(f"stamp-file:{path.relative_to(private_dir).as_posix()}",
             (str(path), "/grant", f"*{PROTECTED_PACKAGE_SID}:(R)"),
             lambda p=path: _acl_has_sid(p, PROTECTED_PACKAGE_SID))
        stamped.append(path.relative_to(private_dir).as_posix())
    if not stamped:
        raise IsolationError(f"private dir has no material to protect: {private_dir}")

    # 金丝雀: 破坏性验收的靶子, 与真实材料同处一个被封印目录。
    canary = private_dir / CANARY_NAME
    canary.write_bytes(CANARY_PAYLOAD)
    step("canary-stamp",
         (str(canary), "/grant", f"*{PROTECTED_PACKAGE_SID}:(R)"),
         lambda: _acl_has_sid(canary, PROTECTED_PACKAGE_SID))

    return {
        "package_sid": PROTECTED_PACKAGE_SID,
        "owner": owner,
        "private_dir": private_dir.name,
        "stamped_files": stamped,
        "canary": CANARY_NAME,
        "steps": steps,
        "mechanism": "appcontainer-package-sid-ace + file-delete-child-deny",
    }


# --- 负向验收 (每次准入重跑) ---


def _escape_quotes(value: str) -> str:
    return value.replace('"', "")


def isolation_problems(private_dir: Path, record: dict | None = None) -> list[str]:
    """每次准入的负向验收; 返回不足清单 (空即健康)。

    断言三类事实, 任一不符即隔离不成立:
      1. 低权子进程**读不到**被封印材料, 但**能读到** `private/` 之外未盖章的对照文件
         (后者证明"被拒"来自 ACE, 而不是环境整体损坏);
      2. 低权子进程删/改名/覆写金丝雀**必须失败**, 且父侧复核文件仍在、内容未变;
      3. 评估方读同一材料**必须成功** (证明没有把所有人一起挡住)。
    """
    problems: list[str] = []
    private_dir = Path(private_dir)
    if not private_dir.is_dir():
        return [f"private dir missing: {private_dir}"]

    validate_package_sid(PROTECTED_PACKAGE_SID)

    try:
        probe = low_child_exit_code("exit /b 0", private_dir)
    except IsolationError as exc:
        return [f"cannot launch low-integrity child: {exc}"]
    if probe != 0:
        problems.append("low-integrity child cannot run at all (environment broken)")

    # 对照: 子进程必须能读到**未被封印**的普通文件, 否则"读不到私有材料"可能只是
    # 环境整体损坏, 结论不可归因到包 SID ACE。对照放在 private/ 之外 —— 放进去会继承
    # 目录上的包 SID ACE, 就不是对照了。
    #
    # 对照目录**自带**显式授权而不依赖继承: 实测对照的可读性一度受 run root 所在父目录的
    # 继承 ACL 影响而偶发失败 (fail-closed 但把环境问题误报成隔离缺陷)。自带授权后,
    # 对照只随"受限低权子进程能否读普通文件"这一件事变化。
    control_dir = private_dir.parent / "isolation-control"
    shutil.rmtree(control_dir, ignore_errors=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    control = control_dir / "readable.txt"
    control.write_bytes(b"CONTROL_READABLE\n")
    # 自带授权 (评估方全权 + Everyone 只读), 不依赖继承; 必须保留评估方写权限,
    # 否则下一轮 rmtree 清理对照目录会因缺 DELETE 而失败。
    _icacls(str(control_dir), "/inheritance:r")
    _icacls(str(control_dir), "/grant:r", f"{current_user()}:(OI)(CI)F")
    _icacls(str(control_dir), "/grant", f"*{EVERYONE_SID}:(OI)(CI)R")
    try:
        control_code = low_child_exit_code(
            f'type "{_escape_quotes(str(control))}" >nul 2>&1', private_dir)
        if control_code != 0:
            problems.append(
                "low-integrity child cannot read an unstamped control file; "
                "denial cannot be attributed to the package-SID ACE")
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)

    materials = sorted(p for p in private_dir.rglob("*")
                       if p.is_file() and p.name != CANARY_NAME)
    if not materials:
        problems.append("no protected material to verify")
    for path in materials:
        rel = path.relative_to(private_dir).as_posix()
        read_code = low_child_exit_code(
            f'type "{_escape_quotes(str(path))}" >nul 2>&1', private_dir)
        if read_code == 0:
            problems.append(f"low-integrity child READ protected material: {rel}")
        try:
            path.read_bytes()
        except OSError as exc:
            problems.append(f"evaluator cannot read protected material {rel}: {exc}")

    canary = private_dir / CANARY_NAME
    if not canary.is_file():
        problems.append("isolation canary is missing; destructive checks cannot run")
        return problems
    escaped = _escape_quotes(str(canary))

    # 破坏性尝试一律**只按效果判定**: 实测 `del` 被拒时 errorlevel 仍为 0
    # (cmd 的已知行为), 按退出码判定会把"删除失败"误报成"删除成功"。
    low_child_exit_code(f'del /f /q "{escaped}"', private_dir)
    if not canary.is_file():
        problems.append("low-integrity child DELETED protected material (canary gone)")
        return problems
    if canary.read_bytes() != CANARY_PAYLOAD:
        problems.append("canary content changed after a delete attempt")

    renamed = canary.with_name("isolation-canary-renamed.txt")
    low_child_exit_code(f'ren "{escaped}" "{renamed.name}"', private_dir)
    if not canary.is_file() or renamed.exists():
        problems.append("low-integrity child RENAMED protected material")
        renamed.unlink(missing_ok=True)
        canary.write_bytes(CANARY_PAYLOAD)
        return problems

    low_child_exit_code(f'echo HACKED> "{escaped}"', private_dir)
    if canary.read_bytes() != CANARY_PAYLOAD:
        problems.append("low-integrity child OVERWROTE protected material")
        canary.write_bytes(CANARY_PAYLOAD)

    if low_child_exit_code(
            f'icacls "{escaped}" /remove:g *{PROTECTED_PACKAGE_SID}', private_dir) == 0:
        problems.append("low-integrity child removed the package-SID ACE")
    if not _acl_has_sid(canary, PROTECTED_PACKAGE_SID):
        problems.append("package-SID ACE is no longer present on the canary")
    # 目录自身的 ACE 是删除屏障之一 (矩阵实测): 只盖文件时"读被拒但删成功"。
    if not _acl_has_sid(private_dir, PROTECTED_PACKAGE_SID):
        problems.append(
            "package-SID ACE is missing from the private directory; delete is then "
            "authorized via the parent's FILE_DELETE_CHILD right")
    # 另一条删除屏障必须**静态**核对: 它的缺失不一定立刻表现为"删除成功"
    # (目录 ACE 仍在时删除照样被挡), 但纵深已少一层, 必须显式报告。
    if not _has_delete_child_deny(private_dir):
        problems.append(
            "private directory no longer denies FILE_DELETE_CHILD (DC); the delete "
            "barrier relies on the package-SID ACE alone")
    for leaked in _inherited_access_aces(private_dir):
        problems.append(f"private directory still inherits an access ACE: {leaked[:80]}")

    if record is not None and record.get("package_sid") != PROTECTED_PACKAGE_SID:
        problems.append("protection record was made with a different package SID")
    return problems
