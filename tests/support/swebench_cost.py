"""SWE-bench 单题费用账本与透明 transport (票 02, 契约 C5/C6)。

状态机: attempt 只允许 prepared → running → finalized 或 running → sealed;
单线程单在途; reserve 在真实发送前可靠落盘, settle 仅在完整成功终结
(生产收尾确认 + transport DONE/EOF/close 联合确认) 后执行一次。
金额用 Decimal; 计价身份冻结: 输入 2 元/M、输出 8 元/M, 服务端输入上界
1_100_000 tokens (公开 1M 上下文的向上界)。缓存输入仍按未命中计。

transport 是只读观察: 请求/响应字节不改; SSE 流经 wrapper 透传时记录
usage/DONE/EOF/close 事实。内层禁止重试、重定向与并发。
"""

from __future__ import annotations

import json
import copy
import hashlib

import httpx
import os
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

EXECUTION_BUDGET_CNY = Decimal("4.50")
INPUT_CNY_PER_M = Decimal("2")
OUTPUT_CNY_PER_M = Decimal("8")
SERVER_INPUT_TOKEN_BOUND = 1_100_000
# 冻结的请求 cap (spec D3): 普通 8192; 摘要 2048（harness 独立基准，不是产品默认）。
NORMAL_OUTPUT_CAP = 8192
SUMMARY_OUTPUT_CAP = 2048
# 修订 2026-10-15 (amendment-proposal-20261015 提案A): 12 -> 24;
# 费用硬闸 S+U+R<=4.50 与次数无关，次数仅作 usage 谎报的纵深防御。
MAX_HTTP_CALLS = 24

PRICING_IDENTITY = {
    "currency": "CNY",
    "input_cny_per_million_tokens": "2",
    "output_cny_per_million_tokens": "8",
    "server_input_token_bound": SERVER_INPUT_TOKEN_BOUND,
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "checked": "2026-10-14",
}


def ceil_cent(cny: Decimal) -> Decimal:
    return cny.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def reserve_amount(output_cap: int) -> Decimal:
    """R = ceil_cent((bound×input + cap×output)/1e6); 授权余量不成为旁路。"""
    return ceil_cent((Decimal(SERVER_INPUT_TOKEN_BOUND) * INPUT_CNY_PER_M
                      + Decimal(output_cap) * OUTPUT_CNY_PER_M) / Decimal(1_000_000))


def settle_amount(prompt_tokens: int, completion_tokens: int) -> Decimal:
    return ceil_cent((Decimal(prompt_tokens) * INPUT_CNY_PER_M
                      + Decimal(completion_tokens) * OUTPUT_CNY_PER_M) / Decimal(1_000_000))


class LedgerError(RuntimeError):
    """拒绝发送或封存; message 固定分类, 不含凭据。"""


class LedgerSealed(LedgerError):
    pass


class BudgetExceeded(LedgerError):
    pass


@dataclass
class UsageFact:
    prompt_tokens: int
    completion_tokens: int
    cache_hit_input_tokens: int | None
    cache_miss_input_tokens: int | None


def _fsync_write(path: Path, text: str, mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    # Windows: 目录项落盘由 replace 语义保证; 再补一次文件 fsync 已在上方完成。


class CostLedger:
    """一次 attempt 的唯一账本; 文件锁只管本次 attempt (契约 C6)。"""

    def __init__(self, attempt_dir: Path):
        self.dir = attempt_dir
        self.calls_path = self.dir / "calls.jsonl"
        self.budget_path = self.dir / "budget.json"
        self.lock_path = self.dir / "ledger.lock"
        self.calls: list[dict] = []
        self.budget: dict = {}
        self._lock_fd: int | None = None

    # --- 生命周期 ---

    def acquire(self, attempt_identity: dict) -> None:
        """新建或接管校验; 已存在 running/sealed 状态的 attempt 拒绝续消费。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._lock_fd, str(os.getpid()).encode())
            os.fsync(self._lock_fd)
        except FileExistsError as exc:
            raise LedgerError("attempt ledger lock exists; no resume/consume across processes") from exc
        if self.budget_path.exists():
            raise LedgerError("attempt already has a budget state; reuse refused")
        self.budget = {
            "state": "running",
            "attempt_identity": attempt_identity,
            "pricing": PRICING_IDENTITY,
            "settled_upper": "0.00",
            "unresolved_reserve": "0.00",
            "total_upper": "0.00",
            "calls": 0,
            "seal_reason": None,
            "sealed_at": None,
        }
        self._persist()

    def acquire_existing_for_observation(self) -> dict:
        """只读打开 (check/report 用); 不取得消费资格。"""
        return json.loads(self.budget_path.read_text(encoding="utf-8"))

    def release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            self.lock_path.unlink(missing_ok=True)

    @property
    def sealed(self) -> bool:
        return self.budget.get("state") == "sealed"

    @property
    def reserved_upper(self) -> Decimal:
        return Decimal(self.budget["settled_upper"]) + Decimal(self.budget["unresolved_reserve"])

    def _persist(self) -> None:
        self.budget["total_upper"] = str(self.reserved_upper)
        _atomic_json(self.budget_path, self.budget)

    # --- reserve / settle / seal ---

    def reserve(self, *, kind: str, output_cap: int, model: str,
                request_sha256: str) -> dict:
        """发送前调用; 可靠落盘后才允许真实发出。失败即零发送。"""
        if self.sealed:
            raise LedgerSealed(f"ledger sealed: {self.budget.get('seal_reason')}")
        if self.budget["state"] != "running":
            raise LedgerError("ledger not running; send refused")
        if self.budget["calls"] + 1 > MAX_HTTP_CALLS:
            raise BudgetExceeded(f"call limit {MAX_HTTP_CALLS} reached; send refused")
        amount = reserve_amount(output_cap)
        if self.reserved_upper + amount > EXECUTION_BUDGET_CNY:
            raise BudgetExceeded(
                f"reserve {amount} exceeds budget: S+U+R="
                f"{self.reserved_upper + amount} > {EXECUTION_BUDGET_CNY}")
        call_id = uuid.uuid4().hex
        record = {
            "call_id": call_id, "kind": kind, "model": model,
            "output_cap": output_cap, "request_sha256": request_sha256,
            "reserve_amount": str(amount), "reserved_at": time.time(),
            "state": "reserved", "usage": None, "settled_amount": None,
            "seal_reason": None, "transport_events": {},
        }
        self.calls.append(record)
        self.budget["calls"] = len(self.calls)
        self.budget["unresolved_reserve"] = str(
            Decimal(self.budget["unresolved_reserve"]) + amount)
        try:
            _fsync_write(self.calls_path, json.dumps(record, ensure_ascii=False) + "\n")
            self._persist()
        except OSError as exc:
            raise LedgerError("ledger persist failed before send; send refused") from exc
        return record

    def record_transport_event(self, call_id: str, event: dict) -> None:
        for record in self.calls:
            if record["call_id"] == call_id:
                record["transport_events"] = {**record.get("transport_events", {}), **event}
                _fsync_write(self.calls_path, json.dumps({"call_id": call_id,
                    "event": "transport", "facts": record["transport_events"]}, ensure_ascii=False) + "\n")
                break

    def settle(self, call_id: str, usage: UsageFact) -> Decimal:
        """完整成功终结后的唯一一次结算; 校验失败即封存, 不用前帧解锁。"""
        if self.sealed:
            raise LedgerSealed("sealed ledger cannot settle")
        record = next((c for c in self.calls if c["call_id"] == call_id), None)
        if record is None or record["state"] != "reserved":
            raise LedgerError("settle target is not a pending reservation")
        if any(type(v) is not int or v < 0 for v in (usage.prompt_tokens, usage.completion_tokens)):
            self.seal("invalid-usage-type", call_id)
            raise LedgerError("usage tokens must be nonnegative integers; sealed")
        if usage.completion_tokens > record["output_cap"]:
            self.seal("completion-over-cap", call_id)
            raise LedgerError("completion exceeds frozen cap; sealed")
        if usage.prompt_tokens > SERVER_INPUT_TOKEN_BOUND:
            self.seal("prompt-over-server-bound", call_id)
            raise LedgerError("prompt exceeds frozen server bound; sealed")
        hit, miss = usage.cache_hit_input_tokens, usage.cache_miss_input_tokens
        if any(v is not None and (type(v) is not int or v < 0) for v in (hit, miss)):
            self.seal("invalid-cache-usage", call_id)
            raise LedgerError("invalid cache usage; sealed")
        if hit is not None and miss is not None and hit + miss != usage.prompt_tokens:
            self.seal("cache-usage-contradiction", call_id)
            raise LedgerError("cache usage contradicts prompt tokens; sealed")
        amount = settle_amount(usage.prompt_tokens, usage.completion_tokens)
        reserve_amt = Decimal(record["reserve_amount"])
        previous_budget = copy.deepcopy(self.budget)
        previous_record = copy.deepcopy(record)
        record.update({
            "state": "settled",
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cache_hit_input_tokens": hit, "cache_miss_input_tokens": miss,
            },
            "settled_amount": str(amount),
        })
        self.budget["settled_upper"] = str(Decimal(self.budget["settled_upper"]) + amount)
        self.budget["unresolved_reserve"] = str(
            Decimal(self.budget["unresolved_reserve"]) - reserve_amt)
        try:
            _fsync_write(self.calls_path, json.dumps(
                {"call_id": call_id, "event": "settled", "amount": str(amount), "usage": record["usage"]},
                ensure_ascii=False) + "\n")
            self._persist()
        except OSError as exc:
            self.budget = previous_budget
            record.clear()
            record.update(previous_record)
            self.seal("ledger-persist-failure", call_id)
            raise LedgerError("settle persist failed; sealed") from exc
        return amount

    def seal(self, reason: str, call_id: str | None = None) -> None:
        """封存: 保留全部未决预留, 停止后继发送; 不清除任何费用事实。"""
        if self.sealed:
            return
        self.budget["state"] = "sealed"
        self.budget["seal_reason"] = reason
        self.budget["sealed_at"] = time.time()
        if call_id is not None:
            for record in self.calls:
                if record["call_id"] == call_id and record["state"] == "reserved":
                    record["state"] = "unresolved"
                    record["seal_reason"] = reason
        try:
            if call_id is not None:
                _fsync_write(self.calls_path, json.dumps(
                    {"call_id": call_id, "event": "sealed", "reason": reason},
                    ensure_ascii=False) + "\n")
            self._persist()
        except OSError:
            # 封存写失败: 保留内存事实; budget.json 上一次持久状态仍是权威磁盘事实。
            pass

    def finalize(self) -> None:
        """全部调用已结算且无未决时置 finalized; 否则保持 sealed。"""
        if self.sealed:
            return
        pending = [c for c in self.calls if c["state"] == "reserved"]
        if pending:
            self.seal("finalize-with-pending-reserve")
            return
        self.budget["state"] = "finalized"
        self._persist()


# --- 透明 transport (契约 C5) ---


def classify_request(body_bytes: bytes) -> tuple[str, int, str]:
    body = json.loads(body_bytes)
    model = body.get("model")
    cap = body.get("max_tokens")
    thinking = (body.get("thinking") or {}).get("type")
    if model != "deepseek-flash" or body.get("stream") is not True or type(cap) is not int:
        raise LedgerError("request identity outside frozen contract")
    if cap == NORMAL_OUTPUT_CAP and thinking == "enabled" and body.get("reasoning_effort") == "low":
        return "normal", cap, model
    if cap == SUMMARY_OUTPUT_CAP and thinking == "disabled":
        return "summary", cap, model
    raise LedgerError("request shape outside frozen contract")


class CostTransport(httpx.AsyncBaseTransport):
    """发送前持久预留；普通/摘要共用，生产终结确认才解除单在途。"""

    def __init__(self, inner, ledger: CostLedger, *, deadline=None, on_stop=None):
        self.inner = inner
        self.ledger = ledger
        self.deadline = deadline
        self.on_stop = on_stop
        self.network_sealed = False
        self.stop_reason = None
        self.active_call_id = None
        self.admitting = False
        self.observations = {}
        self.handoffs = 0
        self.client_closed = False

    def seal_network(self, reason):
        self.network_sealed = True
        self.stop_reason = self.stop_reason or reason

    def stop(self, reason):
        already_sealed = self.network_sealed
        self.seal_network(reason)
        if self.ledger.budget:
            self.ledger.seal(reason, self.active_call_id)
        if self.on_stop is not None and not already_sealed:
            self.on_stop(reason)

    async def aclose(self):
        try:
            await self.inner.aclose()
            self.client_closed = True
        except BaseException:
            self.stop("client-close-failure")
            raise

    async def handle_async_request(self, request):
        if self.network_sealed or self.ledger.sealed:
            raise LedgerSealed("network sealed")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop("run-timeout")
            raise LedgerSealed("run timeout")
        if self.active_call_id is not None or self.admitting:
            self.stop("concurrent-send-refused")
            raise LedgerSealed("concurrent send refused")
        self.admitting = True
        try:
            if request.method != "POST" or str(request.url) != "https://api.deepseek.com/chat/completions":
                raise LedgerError("request endpoint outside frozen contract")
            body = await request.aread()
            if self.network_sealed or self.ledger.sealed:
                raise LedgerSealed("network sealed during request read")
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.stop("run-timeout")
                raise LedgerSealed("run deadline during request read")
            kind, cap, model = classify_request(body)
            record = self.ledger.reserve(kind=kind, output_cap=cap, model=model,
                request_sha256=hashlib.sha256(body).hexdigest())
        except BudgetExceeded:
            self.stop("call-limit" if self.ledger.budget["calls"] >= MAX_HTTP_CALLS else "budget-limit")
            raise
        except (LedgerError, ValueError, TypeError, AttributeError):
            self.stop("request-or-ledger-failure")
            raise
        finally:
            self.admitting = False
        call_id = record["call_id"]
        self.active_call_id = call_id
        observation = {"usage_frame": None, "done_seen": False, "eof_seen": False,
                       "closed": False, "transport_error": None, "http_status": None,
                       "production_terminal": None}
        self.observations[call_id] = observation
        try:
            self.handoffs += 1
            record["handed_off"] = True
            response = await self.inner.handle_async_request(request)
            observation["http_status"] = response.status_code
            response.stream = _ObservedStream(response.stream, observation)
            return response
        except BaseException as exc:
            observation["transport_error"] = type(exc).__name__
            self.stop("http-handoff-failure")
            raise

    def confirm(self, call_id, terminal):
        """只读生产 stream 已耗尽、其 finally 完成后的结算点。"""
        obs = self.observations[call_id]
        obs["production_terminal"] = terminal.stop_reason if terminal is not None else None
        try:
            self.ledger.record_transport_event(call_id, obs.copy())
            if (terminal is None or terminal.stop_reason not in ("stop", "tool_use")
                    or terminal.error is not None or not obs["done_seen"] or not obs["eof_seen"]
                    or not obs["closed"] or obs["transport_error"] is not None
                    or not isinstance(obs["http_status"], int) or not 200 <= obs["http_status"] < 300):
                raise LedgerError("response termination incomplete")
            usage = obs["usage_frame"]
            if not isinstance(usage, dict):
                raise LedgerError("final usage missing")
            fact = UsageFact(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens"))
            metrics = terminal.metrics
            if (metrics is None or metrics.input_tokens != fact.prompt_tokens
                    or metrics.output_tokens != fact.completion_tokens
                    or metrics.cache_hit_input_tokens != fact.cache_hit_input_tokens
                    or metrics.cache_miss_input_tokens != fact.cache_miss_input_tokens):
                raise LedgerError("transport and production usage disagree")
            self.ledger.settle(call_id, fact)
        except (LedgerError, OSError):
            self.stop("response-unresolved")
            raise
        finally:
            self.active_call_id = None


class _ObservedStream(httpx.AsyncByteStream):
    def __init__(self, inner, observation):
        self.inner = inner
        self.observation = observation
        self.buffer = b""

    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                self.buffer += chunk
                # 与生产 wire 上界一致；超大单行不无限缓存。
                if len(self.buffer) > 2 * 1024 * 1024:
                    raise LedgerError("SSE observation exceeds wire bound")
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    self.consume(line)
                yield chunk
            if self.buffer:
                self.consume(self.buffer)
                self.buffer = b""
            self.observation["eof_seen"] = True
        except BaseException as exc:
            self.observation["transport_error"] = type(exc).__name__
            raise

    def consume(self, line):
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload == b"[DONE]":
            self.observation["done_seen"] = True
            return
        try:
            frame = json.loads(payload)
        except ValueError:
            return  # 生产 decoder 保持协议裁决权。
        if isinstance(frame, dict) and frame.get("usage") is not None:
            self.observation["usage_frame"] = frame["usage"]

    async def aclose(self):
        try:
            await self.inner.aclose()
        except BaseException as exc:
            self.observation["transport_error"] = type(exc).__name__
            raise
        self.observation["closed"] = True


def observe_model_stream(original, transport):
    """同时覆盖 DeepSeekModel 普通和摘要实例；事件与请求内容不变。"""
    from cicada.core.ports import StreamDone

    async def observed(model, request, cancel):
        before = len(transport.ledger.calls)
        terminal = None
        completed = False
        try:
            async for event in original(model, request, cancel):
                if isinstance(event, StreamDone):
                    terminal = event
                else:
                    yield event
            completed = True
        finally:
            calls = transport.ledger.calls[before:]
            if calls:
                if len(calls) != 1:
                    transport.stop("ambiguous-call-correlation")
                elif calls[0]["call_id"] not in transport.observations:
                    transport.stop("reservation-not-sent")
                else:
                    transport.confirm(calls[0]["call_id"], terminal if completed else None)
        if terminal is not None:
            yield terminal
    return observed
