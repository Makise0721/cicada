"""A05/A06/A07 崩溃恢复独立进程证据: 握手 kill -> 恢复恰好得到已提交前缀.

kill 点由父进程轮询子进程存档 (只读连接) 判定; fixture kill 只针对本测试创建
并核对命令行的子进程树 (taskkill /T), 不触碰任何用户进程。不把 kill 结果宣传
为物理掉电实测; SQLite journal 回滚由恢复结果核对。

kill 点 (script 四幕: 答复 -> powershell A(gateA) -> powershell B(gateB) -> 答复):
- after-begin:       请求开始后立即 kill (首活动登记与否按存档实况分支断言)
- activity-open:     活动已登记、进程被 gate 挂住 (状态可持有, 确定性)
- activity-terminal: A 已证终结、B 仍开放 (状态可持有, 确定性)
- before-commit:     全部终结已登记、完整提交未落 (窄窗口, 失败换会话重试)
- mid-commit:        提交事务 journal 在盘时 kill (窄窗口, 失败换会话重试)
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
HARNESS = Path(__file__).resolve().parent / "crash_harness.py"

POLL_TIMEOUT_S = 30.0
ATTEMPTS = 6


def _spawn(mode: str, workspace: Path, session_id: str, script: Path) -> subprocess.Popen:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.Popen(
        [sys.executable, str(HARNESS), mode, str(workspace), session_id, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def _wait_line(proc: subprocess.Popen, marker: str, timeout: float = 60.0) -> str:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line == "":
            break
        if marker in line:
            return line
    raise AssertionError(f"child never reached {marker!r}; stderr={proc.stderr.read() if proc.stderr else ''}")


def _kill_tree(proc: subprocess.Popen) -> None:
    """杀掉本测试创建并核对身份的子进程树; 不触碰任何其他进程."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )
    proc.wait(timeout=30)


def _connect_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)


def _poll(db: Path, query: str, predicate, timeout: float = POLL_TIMEOUT_S):
    """轮询存档直到 predicate 成立; 只读连接遇写锁短暂重试."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            connection = _connect_ro(db)
            try:
                rows = connection.execute(query).fetchall()
            finally:
                connection.close()
            last = rows
            if predicate(rows):
                return rows
        except sqlite3.OperationalError:
            pass  # 子进程正写: 下一轮重试
        time.sleep(0.01)
    raise AssertionError(f"poll timed out for {query!r}; last={last}")


def _counts(db: Path) -> tuple[int, int, int, int]:
    """(pending intents, open activities, ended activities, committed requests)."""
    rows = _poll(db, (
        "SELECT (SELECT COUNT(*) FROM request_intents WHERE status='pending'),"
        " (SELECT COUNT(*) FROM process_activity WHERE end_json IS NULL),"
        " (SELECT COUNT(*) FROM process_activity WHERE end_json IS NOT NULL),"
        " (SELECT COUNT(*) FROM records WHERE kind='request')"
    ), lambda _rows: True)
    return tuple(int(value) for value in rows[0])  # type: ignore[return-value]


def _build_workspace(tmp_path: Path, gates: tuple[Path, Path]) -> tuple[Path, Path, Path]:
    work = tmp_path / "ws"
    work.mkdir()
    marker = work / "marker.txt"
    marker.write_text("", encoding="utf-8")
    gate_a, gate_b = gates

    def blocked(gate: Path) -> str:
        return (
            f"while (-not (Test-Path '{gate.as_posix()}')) "
            f"{{ Start-Sleep -Milliseconds 20 }}; "
            f"Add-Content -Path '{marker.as_posix()}' -Value 'run'; exit 0"
        )

    script = tmp_path / "script.json"
    # 6 MiB 尾轮文本拉长完整提交事务的窗口 (编码在调用方线程、事务写盘在 worker),
    # 让 before-commit/mid-commit 的窄窗口轮询可命中; 载荷仍低于 8 MiB v1 限额
    script.write_text(json.dumps([
        {"text": "第一问回答", "stop": True},
        {"tool_calls": [{"id": "a1", "name": "powershell",
                         "arguments": {"command": blocked(gate_a)}}], "tool_use": True},
        {"tool_calls": [{"id": "b1", "name": "powershell",
                         "arguments": {"command": blocked(gate_b)}}], "tool_use": True},
        {"text": "x" * 6_000_000, "stop": True},
    ]), encoding="utf-8")
    resume_script = tmp_path / "resume_script.json"
    resume_script.write_text(json.dumps([{"text": "第三问回答", "stop": True}]), encoding="utf-8")
    return work, script, resume_script


def _resume_facts(work: Path, session_id: str, resume_script: Path) -> dict:
    """恢复进程的事实; 返回 {restored, view_only, interrupted, refused, saved, exit}."""
    proc = _spawn("resume", work, session_id, resume_script)
    try:
        session_line = _wait_line(proc, "[session id=")
        restored = int(session_line.split("restored=", 1)[1].rsplit("]", 1)[0])
        view_line = _wait_line(proc, "[view_only=")
        interrupted_line = _wait_line(proc, "[interrupted=")
        view_only = view_line.split("[view_only=", 1)[1].rsplit("]", 1)[0]
        interrupted = int(interrupted_line.split("[interrupted=", 1)[1].rsplit("]", 1)[0])
        refused = False
        saved = None
        outcome_line = ""
        if view_only != "-":
            _wait_line(proc, "[refused]")
            refused = True
        exit_code = proc.wait(timeout=60)
        output = proc.stdout.read() if proc.stdout is not None else ""
        if view_only == "-":
            for line in output.splitlines():
                if line.startswith("[outcome "):
                    outcome_line = line
            saved = outcome_line == "[outcome exit=0 saved=True]"
    finally:
        if proc.poll() is None:
            _kill_tree(proc)
    return {
        "restored": restored,
        "view_only": None if view_only == "-" else view_only,
        "interrupted": interrupted,
        "refused": refused,
        "saved": saved,
        "exit": exit_code,
        "outcome_line": outcome_line,
    }


def _marker_runs(marker: Path) -> int:
    return len([line for line in marker.read_text(encoding="utf-8").splitlines() if line == "run"])


def _session_dir(work: Path) -> Path:
    return work / ".cicada" / "sessions"


_CRASH_CASES = {
    # 名称: (到达 kill 点的动作, 断言路径)
    "after-begin": None,
    "activity-open": None,
    "activity-terminal": None,
    "before-commit": None,
    "mid-commit": None,
}


@pytest.mark.parametrize("kill_point", sorted(_CRASH_CASES))
def test_crash_then_resume_yields_committed_prefix(tmp_path, kill_point):
    gate_a = tmp_path / "gateA.txt"
    gate_b = tmp_path / "gateB.txt"
    attempts = 1 if kill_point in ("after-begin", "activity-open", "activity-terminal") else ATTEMPTS
    last_error: Exception | None = None
    for attempt in range(attempts):
        scope = tmp_path / f"attempt-{attempt}"
        scope.mkdir()
        work, script, resume_script = _build_workspace(scope, (gate_a, gate_b))
        for gate in (gate_a, gate_b):
            if gate.exists():
                gate.unlink()
        child = _spawn("run", work, "new", script)
        try:
            _wait_line(child, "[request1 committed]")
            db = _session_dir(work)
            db_files = list(db.glob("*.sqlite3"))
            session_file = db_files[0]

            if kill_point == "after-begin":
                _poll(session_file,
                      "SELECT COUNT(*) FROM request_intents WHERE status='pending'",
                      lambda rows: int(rows[0][0]) == 1)
            elif kill_point == "activity-open":
                _poll(session_file,
                      "SELECT COUNT(*) FROM process_activity WHERE end_json IS NULL",
                      lambda rows: int(rows[0][0]) >= 1)
            elif kill_point == "activity-terminal":
                _poll(session_file,
                      "SELECT COUNT(*) FROM process_activity WHERE end_json IS NULL",
                      lambda rows: int(rows[0][0]) >= 1)
                gate_a.write_text("go", encoding="utf-8")
                _poll(session_file, (
                    "SELECT (SELECT COUNT(*) FROM process_activity WHERE end_json IS NOT NULL),"
                    " (SELECT COUNT(*) FROM process_activity WHERE end_json IS NULL)"
                ), lambda rows: int(rows[0][0]) >= 1 and int(rows[0][1]) >= 1)
            else:  # before-commit / mid-commit: 窄窗口
                _poll(session_file,
                      "SELECT COUNT(*) FROM process_activity WHERE end_json IS NULL",
                      lambda rows: int(rows[0][0]) >= 1)
                gate_a.write_text("go", encoding="utf-8")
                _poll(session_file,
                      "SELECT COUNT(*) FROM process_activity WHERE end_json IS NOT NULL",
                      lambda rows: int(rows[0][0]) >= 1)
                gate_b.write_text("go", encoding="utf-8")
                hit = False
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    try:
                        connection = _connect_ro(session_file)
                        try:
                            pending, committed = connection.execute(
                                "SELECT (SELECT COUNT(*) FROM request_intents WHERE status='pending'),"
                                " (SELECT COUNT(*) FROM records WHERE kind='request')"
                            ).fetchone()
                        finally:
                            connection.close()
                    except sqlite3.OperationalError:
                        time.sleep(0.002)
                        continue
                    pending, committed = int(pending), int(committed)
                    if committed >= 2:
                        break  # 已完整提交: 本 attempt 未命中窗口
                    if kill_point == "before-commit":
                        ended_open = _counts(session_file)
                        if pending == 1 and ended_open[1] == 0 and ended_open[2] >= 2:
                            hit = True
                            break
                    else:  # mid-commit: ended>=2 后任何 journal 出现就是提交事务在盘
                        journal = session_file.with_name(session_file.name + "-journal")
                        if journal.exists():
                            hit = True
                            break
                    time.sleep(0.002)
                if not hit:
                    _kill_tree(child)
                    last_error = AssertionError(f"{kill_point} window missed")
                    continue

            marker = work / "marker.txt"
            runs_before_kill = _marker_runs(marker)
            _kill_tree(child)
            time.sleep(0.2)

            pending, open_activities, ended, committed = _counts(session_file)
            if kill_point in ("before-commit", "mid-commit"):
                if committed != 1 or pending != 1:
                    last_error = AssertionError(
                        f"window missed (committed={committed}, pending={pending})"
                    )
                    continue
            assert committed == 1, f"恰好第一条请求完整提交 (实际 {committed})"
            assert pending == 1, "第二条请求保持未提交 intent"

            facts = _resume_facts(work, session_file.stem, resume_script)
            assert facts["restored"] == 1, "恢复恰好得到之前完整请求"
            assert facts["interrupted"] == 1, "未提交请求标中断"
            expected_view_only = open_activities > 0
            assert (
                facts["view_only"] is not None if expected_view_only else facts["view_only"] is None
            ), f"view-only 判定须与存档活动证据一致 (open={open_activities})"
            if facts["view_only"] is not None:
                assert facts["exit"] == 3 and facts["refused"], "view-only 拒绝新请求"
            else:
                assert facts["exit"] == 0 and facts["saved"], (
                    f"可续写恢复追加保存 (outcome_line={facts['outcome_line']!r})"
                )
            assert _marker_runs(marker) == runs_before_kill, "恢复不得重放旧工具"

            connection = _connect_ro(session_file)
            try:
                statuses = connection.execute(
                    "SELECT status, (SELECT COUNT(*) FROM records r WHERE r.request_id ="
                    " i.request_id AND r.kind='request') FROM request_intents i"
                ).fetchall()
            finally:
                connection.close()
            by_status = {status: has_record for status, has_record in statuses}
            assert by_status.get("committed") == 1, "request1 已提交且带记录"
            assert by_status.get("interrupted") == 0, "request2 标中断且没有 request 记录"
            return  # 本 attempt 命中并全部通过
        finally:
            for gate in (gate_a, gate_b):
                if not gate.exists():
                    gate.write_text("teardown", encoding="utf-8")
            if child.poll() is None:
                _kill_tree(child)
    raise AssertionError(f"{kill_point}: {ATTEMPTS} 次尝试均未命中窗口: {last_error}")
