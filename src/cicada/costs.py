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
from cicada.models import SafeModelIdentity

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
    provider: str = 'deepseek'
    protocol: str = 'deepseek-chat'
    base_url: str = 'https://api.deepseek.com'
    input_token_bound: int = SERVER_INPUT_BOUND

    def __post_init__(self) -> None:
        if not _valid_amount(self.input_rate) or not _valid_amount(self.output_rate):
            raise CostError('model prices require positive amounts with at most 8 decimal places')
        if (not isinstance(self.model, str) or not self.model or len(self.model) > 512
                or not isinstance(self.provider, str) or not self.provider or len(self.provider) > 128
                or any(char.isspace() or ord(char) < 32 for char in self.provider)
                or self.protocol not in ('deepseek-chat', 'openai-chat-completions')):
            raise CostError('invalid model price identity')
        try:
            if not isinstance(self.base_url, str):
                raise ValueError
            endpoint = urlsplit(self.base_url)
            if (endpoint.scheme not in ('https', 'http') or not endpoint.hostname
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.query or endpoint.fragment
                    or (endpoint.scheme == 'http' and endpoint.hostname not in ('localhost', '127.0.0.1', '::1'))):
                raise ValueError
        except (TypeError, ValueError):
            raise CostError('invalid model price endpoint') from None
        if type(self.input_token_bound) is not int or not 0 < self.input_token_bound <= 1_000_000_000:
            raise CostError('invalid model price input bound')

    @property
    def route(self) -> tuple[str, str, str, str]:
        return self.provider, self.protocol, self.base_url, self.model

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
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path) -> CostProfile:
        try:
            with path.open('rb') as handle:
                raw = handle.read(65537)
            if len(raw) > 65536:
                raise ValueError
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_pairs,
                              parse_constant=lambda _: None)
            if (set(data) != {'schema_version', 'source', 'checked_at', 'models'}
                    or type(data['schema_version']) is not int or data['schema_version'] not in (1, 2)):
                raise ValueError
            url = urlsplit(data['source'])
            if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError
            date.fromisoformat(data['checked_at'])
            prices = []
            if data['schema_version'] == 1:
                if not isinstance(data['models'], dict) or not data['models']:
                    raise ValueError
                for model, rates in data['models'].items():
                    if not isinstance(model, str) or not model or len(model) > 128:
                        raise ValueError
                    if set(rates) != {'input_cny_per_million', 'output_cny_per_million'}:
                        raise ValueError
                    prices.append(ModelPrice(model, positive_amount(rates['input_cny_per_million']),
                                             positive_amount(rates['output_cny_per_million'])))
            else:
                if not isinstance(data['models'], list) or not data['models'] or len(data['models']) > 256:
                    raise ValueError
                for rates in data['models']:
                    if set(rates) != {'provider', 'protocol', 'base_url', 'model', 'input_token_bound',
                                      'input_cny_per_million', 'output_cny_per_million'}:
                        raise ValueError
                    prices.append(ModelPrice(
                        rates['model'], positive_amount(rates['input_cny_per_million']),
                        positive_amount(rates['output_cny_per_million']), rates['provider'],
                        rates['protocol'], rates['base_url'], rates['input_token_bound'],
                    ))
                if len({price.route for price in prices}) != len(prices):
                    raise ValueError
            return cls(data['source'], data['checked_at'], hashlib.sha256(raw).hexdigest(),
                       tuple(prices), data['schema_version'])
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise CostError('invalid or unreadable cost profile') from None

    def price(self, model: str | SafeModelIdentity) -> ModelPrice:
        if isinstance(model, SafeModelIdentity):
            options = dict(model.options)
            protocol = options.get('protocol')
            if protocol is None and model.provider == 'deepseek':
                protocol = 'deepseek-chat'
            route = model.provider, protocol, model.base_url, model.model
        elif isinstance(model, str) and self.schema_version == 1:
            # 旧字符串入口只有原 DeepSeek 官方语义，不能匹配 v2 网关价格。
            route = 'deepseek', 'deepseek-chat', 'https://api.deepseek.com', model
        else:
            raise CostError('cost profile requires the full selected model identity')
        if self.schema_version == 1 and route[:3] != ('deepseek', 'deepseek-chat', 'https://api.deepseek.com'):
            raise CostError('legacy cost profile only supports the official DeepSeek endpoint')
        for price in self.prices:
            if price.route == route:
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
    input_token_bound: int | None = SERVER_INPUT_BOUND
    input_token_bounds: tuple[tuple[str, str, str, str, int], ...] = ()


@dataclass(frozen=True)
class _Reservation:
    call: int
    output_cap: int
    price: ModelPrice | None
    amount: int
    input_token_bound: int = SERVER_INPUT_BOUND


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
            prices=tuple((p.model if profile.schema_version == 1 else f'{p.provider}/{p.model}',
                          str(p.input_rate), str(p.output_rate)) for p in profile.prices) if profile else (),
            input_token_bound=SERVER_INPUT_BOUND if profile is None or profile.schema_version == 1 else None,
            input_token_bounds=tuple((*p.route, p.input_token_bound) for p in profile.prices)
                if profile is not None and profile.schema_version == 2 else ())

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

    def _check_available(self) -> None:
        if self._sealed:
            raise CostError(self._sealed)
        if self._pending is not None:
            raise CostError('a model cost reservation is still pending')
        if self.max_calls is not None and self._calls >= self.max_calls:
            raise CostError('model call limit reached; not sent')

    def validate_identity(self, identity: SafeModelIdentity) -> None:
        """切换前只读核对身份和适用上界；不创建账本或消费额度。"""
        if not isinstance(identity, SafeModelIdentity):
            raise CostError('cost validation requires a safe model identity')
        self._check_available()
        if self.profile is None:
            return
        price = self.profile.price(identity)
        options = dict(identity.options)
        context = options.get('num_ctx')
        output = options.get('max_tokens', options.get('num_predict'))
        if type(context) is not int or context <= 0 or context > price.input_token_bound:
            raise CostError('selected model context exceeds the configured cost input bound')
        if type(output) is not int or output <= 0:
            raise CostError('selected model has no cost output cap')
        amount = price.upper_cents(price.input_token_bound, output)
        if self.limit is not None and _money(self._settled + amount) > self.limit:
            raise CostError('model cost upper bound exceeds remaining allowance; not sent')

    def reserve(self, model: str | SafeModelIdentity, output_cap: int) -> _Reservation:
        if not isinstance(model, (str, SafeModelIdentity)):
            raise CostError('model cost reservation requires a model identity')
        if type(output_cap) is not int or output_cap <= 0:
            raise CostError('model cost output cap must be a positive integer')
        self._check_available()
        if isinstance(model, SafeModelIdentity):
            self.validate_identity(model)
        price = self.profile.price(model) if self.profile else None
        input_bound = price.input_token_bound if price else SERVER_INPUT_BOUND
        amount = price.upper_cents(input_bound, output_cap) if price else 0
        if self.limit is not None and _money(self._settled + amount) > self.limit:
            raise CostError('model cost upper bound exceeds remaining allowance; not sent')
        reservation = _Reservation(self._calls + 1, output_cap, price, amount, input_bound)
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
                type(metrics.input_tokens) is int and 0 <= metrics.input_tokens <= reservation.input_token_bound
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
