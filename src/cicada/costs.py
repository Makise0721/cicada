"""进程级模型费用上界；价格是显式本地配置，账本不恢复消费额度。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from cicada.core.ports import ModelMetrics

SERVER_INPUT_BOUND = 1_100_000


def _valid_amount(amount: Decimal) -> bool:
    return (isinstance(amount, Decimal) and amount.is_finite() and amount > 0
            and amount <= Decimal('1000000000') and amount.as_tuple().exponent >= -8)


class CostError(ValueError):
    """固定错误分类，不回显请求、凭据或配置正文。"""


def _money(cents: int) -> Decimal:
    return Decimal((0, tuple(int(char) for char in str(cents)), -2))


def positive_amount(value: str) -> Decimal:
    try:
        if not isinstance(value, str) or len(value) > 32:
            raise ValueError
        amount = Decimal(value)
        if not _valid_amount(amount):
            raise ValueError
        return amount
    except (ValueError, InvalidOperation):
        raise CostError('cost amount must be positive, <= 1000000000, with at most 8 decimal places') from None


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CostError('duplicate cost profile key')
        result[key] = value
    return result


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_rate: Decimal
    output_rate: Decimal

    def __post_init__(self) -> None:
        if not _valid_amount(self.input_rate) or not _valid_amount(self.output_rate):
            raise CostError('model prices require positive amounts with at most 8 decimal places')

    def upper(self, inputs: int, outputs: int) -> Decimal:
        return _money(self.upper_cents(inputs, outputs))

    def upper_cents(self, inputs: int, outputs: int) -> int:
        # 整数有理数计算后取分的上整，避免 Decimal context 在取整前丢正尾数。
        input_num, input_den = self.input_rate.as_integer_ratio()
        output_num, output_den = self.output_rate.as_integer_ratio()
        numerator = inputs * input_num * output_den + outputs * output_num * input_den
        denominator = input_den * output_den * 1_000_000
        return (100 * numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class CostProfile:
    source: str
    checked_at: str
    sha256: str
    prices: tuple[ModelPrice, ...]

    @classmethod
    def load(cls, path: Path) -> CostProfile:
        try:
            with path.open('rb') as handle:
                raw = handle.read(65537)
            if len(raw) > 65536:
                raise ValueError
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_pairs,
                              parse_constant=lambda _: None)
            if set(data) != {'schema_version', 'source', 'checked_at', 'models'} or type(data['schema_version']) is not int or data['schema_version'] != 1:
                raise ValueError
            url = urlsplit(data['source'])
            if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError
            date.fromisoformat(data['checked_at'])
            if not isinstance(data['models'], dict) or not data['models']:
                raise ValueError
            prices = []
            for model, rates in data['models'].items():
                if not isinstance(model, str) or not model or len(model) > 128:
                    raise ValueError
                if set(rates) != {'input_cny_per_million', 'output_cny_per_million'}:
                    raise ValueError
                prices.append(ModelPrice(model, positive_amount(rates['input_cny_per_million']),
                                         positive_amount(rates['output_cny_per_million'])))
            return cls(data['source'], data['checked_at'], hashlib.sha256(raw).hexdigest(), tuple(prices))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise CostError('invalid or unreadable cost profile') from None

    def price(self, model: str) -> ModelPrice:
        for price in self.prices:
            if price.model == model:
                return price
        raise CostError('cost profile has no price for the selected model')


@dataclass(frozen=True)
class CostSnapshot:
    limit_cny: str | None
    max_calls: int | None
    calls: int
    settled_upper_cny: str | None
    pending_upper_cny: str | None
    sealed_reason: str | None
    source: str | None
    checked_at: str | None
    profile_sha256: str | None
    ledger_path: str
    scope: str = 'current-process; restart creates a new allowance'
    prices: tuple[tuple[str, str, str], ...] = ()
    input_token_bound: int = SERVER_INPUT_BOUND


@dataclass(frozen=True)
class _Reservation:
    call: int
    output_cap: int
    price: ModelPrice | None
    amount: int


class CostLedger:
    """一个实际发送在途；预留可靠写入后才允许发送，未知保持封口。"""

    def __init__(self, directory: Path, *, limit: Decimal | None = None,
                 max_calls: int | None = None, profile: CostProfile | None = None):
        if limit is None and max_calls is None:
            raise CostError('cost control requires an amount or call limit')
        if limit is not None and (not _valid_amount(limit) or profile is None):
            raise CostError('cost limit requires a positive amount and an explicit price profile')
        if max_calls is not None and (type(max_calls) is not int or max_calls <= 0):
            raise CostError('max-model-calls must be a positive integer')
        self.limit, self.max_calls, self.profile = limit, max_calls, profile
        self.path = directory / (uuid.uuid4().hex + '.json')
        self._calls = 0
        self._settled = 0
        self._pending: _Reservation | None = None
        self._sealed: str | None = None

    @property
    def snapshot(self) -> CostSnapshot:
        profile = self.profile
        return CostSnapshot(
            str(self.limit) if self.limit is not None else None, self.max_calls, self._calls,
            str(_money(self._settled)) if profile else None,
            str(_money(self._pending.amount if self._pending else 0)) if profile else None,
            self._sealed, profile.source if profile else None, profile.checked_at if profile else None,
            profile.sha256 if profile else None, str(self.path),
            prices=tuple((p.model, str(p.input_rate), str(p.output_rate)) for p in profile.prices) if profile else ())

    def _persist(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open('w', encoding='utf-8', newline='\n') as handle:
                json.dump(asdict(self.snapshot), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            self._sealed = 'cost ledger write failed'
            raise CostError(self._sealed) from None

    def reserve(self, model: str, output_cap: int) -> _Reservation:
        if type(output_cap) is not int or output_cap <= 0:
            raise CostError('model cost output cap must be a positive integer')
        if self._sealed:
            raise CostError(self._sealed)
        if self._pending is not None:
            raise CostError('a model cost reservation is still pending')
        if self.max_calls is not None and self._calls >= self.max_calls:
            raise CostError('model call limit reached; not sent')
        price = self.profile.price(model) if self.profile else None
        amount = price.upper_cents(SERVER_INPUT_BOUND, output_cap) if price else 0
        if self.limit is not None and _money(self._settled + amount) > self.limit:
            raise CostError('model cost upper bound exceeds remaining allowance; not sent')
        reservation = _Reservation(self._calls + 1, output_cap, price, amount)
        self._pending = reservation
        self._calls += 1
        self._persist()
        return reservation

    def finish(self, reservation: _Reservation, metrics: ModelMetrics | None, *, complete: bool) -> None:
        if self._pending is not reservation:
            raise CostError('model cost reservation is not current')
        valid = complete
        if reservation.price:
            valid = valid and metrics is not None and (
                type(metrics.input_tokens) is int and 0 <= metrics.input_tokens <= SERVER_INPUT_BOUND
                and type(metrics.output_tokens) is int and 0 <= metrics.output_tokens <= reservation.output_cap)
        if not valid:
            self._sealed = 'model cost unresolved; restart with a new allowance to continue'
            self._persist()
            raise CostError(self._sealed)
        if reservation.price:
            self._settled += reservation.price.upper_cents(metrics.input_tokens, metrics.output_tokens)
        self._pending = None
        self._persist()


@runtime_checkable
class CostControlPort(Protocol):
    def set_cost_control(self, ledger: CostLedger) -> None: ...


def cost_lines(snapshot: CostSnapshot) -> list[str]:
    """只读显示；费用是配置单价的上界，不是账户扣费。"""
    lines = [f'进程调用: {snapshot.calls}/{snapshot.max_calls if snapshot.max_calls is not None else "不限"}']
    if snapshot.limit_cny is not None:
        lines.append(f'费用上界: 已结算 {snapshot.settled_upper_cny} 元 · 待决预留 {snapshot.pending_upper_cny} 元 · 上限 {snapshot.limit_cny} 元')
        lines.append(f'价格: {snapshot.source} · 核对 {snapshot.checked_at}（本地配置，非账单）')
    if snapshot.sealed_reason:
        lines.append('费用控制已封口: ' + snapshot.sealed_reason)
    return lines
