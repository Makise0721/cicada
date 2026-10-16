"""PV02 跨进程 attempt 总额度闸 (AttemptBudget)。

职责: 为同一授权 attempt 的**两个进程**提供持久总额度与次数; 普通与摘要两条通道
在真实 transport 发送前都必须先经此闸。金额用生产费用模块的精确整数向上取分算法
(``cicada.costs.ModelPrice.upper_cents``) 与冻结输入上界 ``SERVER_INPUT_BOUND``;
**不维护旧固定 2/8 公式**。

与生产进程账本的关系: 生产 ``cicada.costs.CostLedger`` 仍按产品规则"重启新建额度";
本模块是**额外的** attempt 级总闸。两者互不覆盖, 也不把历史消费当当前进程消费。

边界 (规格 §3, 每条都有对应拒绝):
- 发送前顺序: 校验总闸与 phase → 计算 reserve → 检查 ``settled+pending+reserve <= 上限``
  且 ``calls+1 <= 上限`` → 原子写盘并 fsync 预留 → 才交给生产发送口。任一步不过都不发送。
- 生产闸随后拒绝且有明确零发送事实 → 记 ``refused-before-send`` 释放金额;
  **调用尝试号保守保留**, 绝不重用。
- 只有"完整合法 usage + 成功终局 + 响应关闭 + 生产确认"齐备才结算一次并释放差额;
  任何部分异常/usage 缺失或超界/取消/长度终止/关闭失败都**保留 pending 并封口**。
- 第二进程续跑前须持独占锁、身份 hash 全等、并从 append-only 记录**重算**后与
  checkpoint 比对; 任一矛盾即拒绝并封口, 不自动修账/清零/释放。

记录格式: ``calls.jsonl`` 是**真正的 append-only 事件流**。同一 ``call_id`` 的状态
转换也追加新行 (reserve → settle/seal/refuse), 由 ``fold()`` 折叠。这样"重复预留"
与"重复结算"都能被结构性地检出, 而不是靠覆盖写把矛盾抹掉。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cicada.costs import SERVER_INPUT_BOUND, CostError, CostProfile

SCHEMA_VERSION = 1

# 授权包冻结值: 执行期 450 分 (4.50 元) 与 24 次发送, 两进程共享。
DEFAULT_LIMIT_CENT = 450
DEFAULT_MAX_CALLS = 24

PROCESS_1 = "process-1"
PROCESS_2 = "process-2"
PHASES = (PROCESS_1, PROCESS_2)


class AttemptBudgetError(RuntimeError):
    """总闸拒绝或状态矛盾; 保留具体原因即停止, 不降级放行。"""


def _fsync_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise AttemptBudgetError(f"attempt ledger write failed: {exc}") from None


def _append_line(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AttemptBudgetError(f"call record write failed: {exc}") from None


class AttemptBudget:
    """一个 attempt 的持久总闸; 两个进程串行使用, 一次仅一个模型流。

    ``root`` 下布局:
      attempt.json    当前 checkpoint (身份 + 汇总 + 阶段回执)
      calls.jsonl     append-only 事件流
      attempt.lock    独占锁 (记录持有者 pid)
    """

    def __init__(
        self,
        root: Path,
        *,
        attempt_id: str,
        manifest_hash: str,
        prepared_lock_hash: str,
        profile: CostProfile,
        model_identity: dict,
        limit_cent: int = DEFAULT_LIMIT_CENT,
        max_calls: int = DEFAULT_MAX_CALLS,
    ) -> None:
        if isinstance(limit_cent, bool) or not isinstance(limit_cent, int) or limit_cent <= 0:
            raise AttemptBudgetError("limit_cent must be a positive integer")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
            raise AttemptBudgetError("max_calls must be a positive integer")
        if not isinstance(profile, CostProfile):
            raise AttemptBudgetError(
                "attempt budget requires an explicit price profile; no send without it")
        if not isinstance(model_identity, dict) or "model" not in model_identity:
            raise AttemptBudgetError("model_identity must carry the selected model name")
        self.root = Path(root)
        self.limit_cent = limit_cent
        self.max_calls = max_calls
        self.profile = profile
        # 身份 hash 逐位冻结: 第二进程据此判定"同一 attempt"。
        self.identity = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "manifest_hash": manifest_hash,
            "prepared_lock_hash": prepared_lock_hash,
            "profile_hash": profile.sha256,
            "model_identity": model_identity,
            "limit_cent": limit_cent,
            "max_calls": max_calls,
        }
        self.phase: str | None = None
        self.sealed_reason: str | None = None
        self._locked = False

    # ---- 路径 ----

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "attempt.json"

    @property
    def calls_path(self) -> Path:
        return self.root / "calls.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.root / "attempt.lock"

    # ---- 独占锁 ----

    def acquire(self, phase: str) -> dict:
        """取得 attempt 独占锁; 陈旧锁仅在确认持有者不存活后才顶替。"""
        if phase not in PHASES:
            raise AttemptBudgetError(f"unknown phase {phase!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            holder = self._read_lock()
            if holder.get("corrupt") or _process_alive(holder.get("pid")):
                raise AttemptBudgetError(
                    "attempt is locked by a live or unreadable holder "
                    f"(pid={holder.get('pid')}, phase={holder.get('phase')}); "
                    "refusing to run concurrently")
            self.lock_path.unlink(missing_ok=True)
        record = {"pid": os.getpid(), "phase": phase, "acquired": time.time(),
                  "attempt_id": self.identity["attempt_id"]}
        try:
            handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise AttemptBudgetError("attempt lock is contended") from exc
        except OSError as exc:
            raise AttemptBudgetError(f"attempt lock failed: {exc}") from None
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        self._locked = True
        self.phase = phase
        return record

    def release(self) -> None:
        """释放独占锁; 幂等。"""
        if self._locked:
            self.lock_path.unlink(missing_ok=True)
            self._locked = False

    def _read_lock(self) -> dict:
        try:
            return json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"pid": None, "phase": None, "corrupt": True}

    # ---- append-only 事件流 ----

    def events(self) -> list[dict]:
        """读取原始事件流; 坏行/截断即拒绝 (不猜测修复)。"""
        if not self.calls_path.is_file():
            return []
        try:
            raw = self.calls_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AttemptBudgetError(f"call records unreadable: {exc}") from None
        records: list[dict] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                raise AttemptBudgetError(
                    f"call records truncated or corrupt at line {number}") from None
            if not isinstance(record, dict) or not isinstance(record.get("call_id"), str):
                raise AttemptBudgetError(f"call record {number} has no call_id")
            records.append(record)
        return records

    def fold(self) -> dict:
        """把事件流折叠为逐 call 状态; 结构性矛盾即拒绝。

        检出: 同一 call_id 重复预留、重复结算、结算/封口/拒发先于预留、未知事件、
        非法金额。这些是"账被写坏或被伪造"的证据, 一律不自动修。
        """
        calls: dict[str, dict] = {}
        settled_cent = 0
        pending_cent = 0
        order: list[str] = []
        for record in self.events():
            call_id = record["call_id"]
            event = record.get("event")
            if event == "reserve":
                if call_id in calls:
                    raise AttemptBudgetError(f"duplicate reservation for call {call_id}")
                reserve = record.get("reserve_cent")
                if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
                    raise AttemptBudgetError("reservation has an invalid amount")
                calls[call_id] = {**record, "state": "pending"}
                order.append(call_id)
                pending_cent += reserve
            elif event in ("settle", "seal", "refuse"):
                if call_id not in calls:
                    raise AttemptBudgetError(
                        f"{event} for call {call_id} without a reservation")
                current = calls[call_id]
                if current["state"] != "pending":
                    raise AttemptBudgetError(
                        f"call {call_id} was already resolved as {current['state']}")
                if event == "settle":
                    actual = record.get("settled_cent")
                    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                        raise AttemptBudgetError("settlement has an invalid amount")
                    if actual > current["reserve_cent"]:
                        raise AttemptBudgetError(
                            f"settlement for {call_id} exceeds its reservation")
                    # 只有完备结算才释放预留差额。
                    pending_cent -= current["reserve_cent"]
                    settled_cent += actual
                    current.update({"state": "settled", "settled_cent": actual})
                elif event == "refuse":
                    # 明确零发送: 释放金额, 但尝试号已保守消耗。
                    pending_cent -= current["reserve_cent"]
                    current.update({"state": "refused-before-send",
                                    "reason": record.get("reason")})
                else:
                    # 封口 (seal): **保留 pending**。预留如何了结未知, 不能当成已释放。
                    current.update({"state": "sealed", "reason": record.get("reason")})
            else:
                raise AttemptBudgetError(f"unknown call event {event!r}")
        return {"calls": len(order), "settled_cent": settled_cent,
                "pending_cent": pending_cent, "order": order,
                "states": {cid: calls[cid]["state"] for cid in order}}

    def totals(self) -> dict:
        folded = self.fold()
        return {key: folded[key] for key in ("settled_cent", "pending_cent", "calls")}

    def recorded_calls(self) -> list[dict]:
        """逐 call 的最终事实 (供汇报与 scope 核对)。"""
        folded = self.fold()
        latest: dict[str, dict] = {}
        for record in self.events():
            latest.setdefault(record["call_id"], {}).update(record)
        return [{**latest[call_id], "state": folded["states"][call_id]}
                for call_id in folded["order"]]

    # ---- 准入 ----

    def reserve(self, *, call_id: str, phase: str, kind: str, output_cap: int,
                session: str | None = None, request: str | None = None) -> dict:
        """预留一次发送; 返回记录。任一检查不过即抛错 (调用方不得发送)。"""
        if not self._locked:
            raise AttemptBudgetError("attempt budget is not acquired")
        if phase != self.phase:
            raise AttemptBudgetError(
                f"phase mismatch: budget held for {self.phase}, send asked for {phase}")
        if self.sealed_reason:
            raise AttemptBudgetError(f"attempt sealed: {self.sealed_reason}")
        if isinstance(output_cap, bool) or not isinstance(output_cap, int) or output_cap <= 0:
            raise AttemptBudgetError("output_cap must be a positive integer")
        if any(r["call_id"] == call_id for r in self.events()):
            raise AttemptBudgetError(f"call_id already used in this attempt: {call_id}")
        totals = self.totals()
        try:
            price = self.profile.price(self.identity["model_identity"]["model"])
        except (CostError, KeyError):
            raise AttemptBudgetError(
                "cost profile has no price for the selected model; send refused") from None
        reserve = price.upper_cents(SERVER_INPUT_BOUND, output_cap)
        if totals["calls"] + 1 > self.max_calls:
            self.seal("call limit reached")
            raise AttemptBudgetError(
                f"attempt call limit reached ({self.max_calls}); not sent")
        if totals["settled_cent"] + totals["pending_cent"] + reserve > self.limit_cent:
            self.seal("amount limit reached")
            raise AttemptBudgetError(
                "attempt cost upper bound exceeds the remaining allowance; not sent")
        _append_line(self.calls_path, {
            "call_id": call_id, "event": "reserve", "phase": phase, "kind": kind,
            "process": os.getpid(), "session": session, "request": request,
            "output_cap": output_cap, "reserve_cent": reserve, "at": time.time()})
        self.write_checkpoint()
        return {"call_id": call_id, "reserve_cent": reserve, "output_cap": output_cap,
                "kind": kind}

    def refuse_before_send(self, call_id: str, reason: str) -> None:
        """生产闸拒绝且有零发送事实: 释放金额, **保留尝试号**。"""
        _append_line(self.calls_path, {
            "call_id": call_id, "event": "refuse", "reason": reason, "at": time.time()})
        self.write_checkpoint()

    def settle(self, call_id: str, metrics, *, complete: bool, closed: bool = True) -> bool:
        """结算一次并释放差额; 条件不齐即保留 pending 并封口。返回是否已结算。"""
        folded = self.fold()
        if call_id not in folded["states"]:
            raise AttemptBudgetError(f"no such call: {call_id}")
        if folded["states"][call_id] != "pending":
            raise AttemptBudgetError(
                f"call {call_id} is already {folded['states'][call_id]}")
        reserve = next(c["reserve_cent"] for c in self.events()
                       if c["call_id"] == call_id and c.get("event") == "reserve")
        price = self.profile.price(self.identity["model_identity"]["model"])
        valid = bool(complete and closed and metrics is not None)
        if valid:
            valid = (
                isinstance(metrics.input_tokens, int)
                and not isinstance(metrics.input_tokens, bool)
                and 0 <= metrics.input_tokens <= SERVER_INPUT_BOUND
                and isinstance(metrics.output_tokens, int)
                and not isinstance(metrics.output_tokens, bool)
                and 0 <= metrics.output_tokens <= reserve_output_cap(self, call_id)
            )
        if not valid:
            # 计量缺失不得当 0 结算: 那是"部分异常", 保留 pending 并封口。
            self.seal(f"call {call_id} lacks complete, in-range usage")
            _append_line(self.calls_path, {
                "call_id": call_id, "event": "seal", "at": time.time(),
                "reason": "usage missing, out of range, or response not confirmed"})
            self.write_checkpoint()
            return False
        actual = price.upper_cents(metrics.input_tokens, metrics.output_tokens)
        if actual > reserve:
            self.seal(f"call {call_id} settled above its reservation")
            _append_line(self.calls_path, {
                "call_id": call_id, "event": "seal", "at": time.time(),
                "reason": "actual usage exceeded the reservation"})
            self.write_checkpoint()
            return False
        _append_line(self.calls_path, {
            "call_id": call_id, "event": "settle", "settled_cent": actual,
            "input_tokens": metrics.input_tokens, "output_tokens": metrics.output_tokens,
            "at": time.time()})
        self.write_checkpoint()
        return True

    def seal(self, reason: str) -> None:
        """封口: 拒绝后续发送并保留全部预留, 不自动释放。"""
        if not self.sealed_reason:
            self.sealed_reason = reason
            self.write_checkpoint()

    # ---- checkpoint ----

    def write_checkpoint(self, extra: dict | None = None) -> dict:
        payload = {
            **self.identity,
            "phase": self.phase,
            "sealed_reason": self.sealed_reason,
            "updated": time.time(),
            **self.totals(),
        }
        if extra:
            payload.update(extra)
        _fsync_json(self.checkpoint_path, payload)
        return payload

    def read_checkpoint(self) -> dict:
        if not self.checkpoint_path.is_file():
            raise AttemptBudgetError("no attempt checkpoint; cannot continue")
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise AttemptBudgetError("attempt checkpoint is unreadable") from None

    def verify_continuation(self, *, expect_phase: str) -> dict:
        """第二进程续跑前的完整核对; 任一矛盾即拒绝并封口。"""
        checkpoint = self.read_checkpoint()
        for key, value in self.identity.items():
            if checkpoint.get(key) != value:
                raise AttemptBudgetError(
                    f"attempt identity drift on {key!r}; refusing to continue")
        if checkpoint.get("phase") != expect_phase:
            raise AttemptBudgetError(
                f"checkpoint phase is {checkpoint.get('phase')!r}, expected {expect_phase!r}")
        totals = self.totals()
        for key in ("settled_cent", "pending_cent", "calls"):
            if checkpoint.get(key) != totals[key]:
                raise AttemptBudgetError(
                    f"checkpoint {key} contradicts call records "
                    f"({checkpoint.get(key)} != {totals[key]}); refusing to continue")
        if totals["pending_cent"] > 0:
            self.seal("a reservation from the previous process is still pending")
            raise AttemptBudgetError(
                "previous process left a pending reservation; sealed, not auto-released")
        if checkpoint.get("sealed_reason"):
            raise AttemptBudgetError(f"attempt is sealed: {checkpoint['sealed_reason']}")
        return {"checkpoint": checkpoint, "totals": totals}

    def close_checkpoint(self, *, process_identity: dict) -> dict:
        """第一进程正常完结后写 phase checkpoint (无 pending / 无封口才允许)。"""
        totals = self.totals()
        if totals["pending_cent"] > 0 or self.sealed_reason:
            raise AttemptBudgetError(
                "cannot checkpoint with a pending reservation or a sealed attempt")
        return self.write_checkpoint({
            "process_identity": process_identity,
            "phase_receipt": {"phase": self.phase, "closed": True, "at": time.time()}})


def reserve_output_cap(budget: AttemptBudget, call_id: str) -> int:
    for record in budget.events():
        if record["call_id"] == call_id and record.get("event") == "reserve":
            return record["output_cap"]
    raise AttemptBudgetError(f"no reservation for call {call_id}")


def _process_alive(pid) -> bool:
    """pid 是否仍存活; 探测失败或 pid 非法一律保守视为存活 (不顶替)。"""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    if pid == os.getpid():
        return True
    if os.name != "nt":
        return True
    try:
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            # 打不开句柄**不等于进程已死**: 权限不足也会失败。顶替一个可能仍存活
            # 的持有者会让两个进程同时消费同一 attempt, 故一律保守当作存活。
            return True
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001 - 探测失败即保守
        return True
