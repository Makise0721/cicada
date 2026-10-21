"""多供应商价格身份、只读准入和每条预留上界的关键回归。"""

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from cicada.core.ports import ModelMetrics
from cicada.costs import CostError, CostLedger, CostProfile, ModelPrice
from cicada.models import SafeModelIdentity


def _identity(provider='work', *, endpoint='https://gateway.invalid/v1', context=131072):
    return SafeModelIdentity(provider, 'org/shared-model', endpoint, False, (
        ('num_ctx', context), ('max_tokens', 4096), ('protocol', 'openai-chat-completions'),
        ('profile_version', 1), ('model_key', 'coder'), ('stream_usage', True),
        ('output_token_field', 'max_tokens'),
    ), 2097152)


def _profile(tmp_path, *, bound=131072, duplicate=False):
    item = {
        'provider': 'work', 'protocol': 'openai-chat-completions',
        'base_url': 'https://gateway.invalid/v1', 'model': 'org/shared-model',
        'input_token_bound': bound, 'input_cny_per_million': '1', 'output_cny_per_million': '2',
    }
    path = tmp_path / 'prices.json'
    path.write_text(json.dumps({
        'schema_version': 2, 'source': 'https://gateway.invalid/prices',
        'checked_at': '2026-10-21', 'models': [item, item] if duplicate else [item],
    }), encoding='utf-8')
    return CostProfile.load(path)


def test_v2_price_requires_full_route_and_validation_has_no_side_effect(tmp_path):
    profile = _profile(tmp_path)
    identity = _identity()
    ledger = CostLedger(tmp_path / 'cost', limit=Decimal('10'), profile=profile)
    ledger.validate_identity(identity)
    assert ledger.snapshot.calls == 0 and not ledger.path.exists()
    for wrong in (replace(identity, provider='personal'),
                  replace(identity, base_url='https://other.invalid/v1'),
                  replace(identity, options=tuple((k, 'deepseek-chat' if k == 'protocol' else v)
                                                 for k, v in identity.options))):
        with pytest.raises(CostError, match='no price'):
            ledger.validate_identity(wrong)
    with pytest.raises(CostError, match='full selected model identity'):
        ledger.reserve(identity.model, 4096)
    assert ledger.snapshot.calls == 0 and not ledger.path.exists()
    reservation = ledger.reserve(identity, 4096)
    assert ledger.snapshot.calls == 1 and ledger.snapshot.pending_upper_cny == '0.14'
    ledger.finish(reservation, ModelMetrics(input_tokens=100, output_tokens=20), complete=True)
    summary = replace(identity, options=tuple((k, 2048 if k == 'max_tokens' else v)
                                             for k, v in identity.options))
    ledger.reserve(summary, 2048)
    assert ledger.snapshot.calls == 2


def test_price_input_bound_applies_to_reservation_and_unknown_stays_sealed(tmp_path):
    profile = _profile(tmp_path, bound=2_000_000)
    identity = _identity(context=2_000_000)
    ledger = CostLedger(tmp_path / 'cost', limit=Decimal('10'), profile=profile)
    first = ledger.reserve(identity, 4096)
    assert ledger.snapshot.pending_upper_cny == '2.01'
    ledger.finish(first, ModelMetrics(input_tokens=1_500_000, output_tokens=1), complete=True)
    assert ledger.snapshot.settled_upper_cny == '1.51'
    second = ledger.reserve(identity, 4096)
    with pytest.raises(CostError, match='unresolved'):
        ledger.finish(second, ModelMetrics(input_tokens=2_000_001, output_tokens=1), complete=True)
    assert ledger.snapshot.pending_upper_cny == '2.01'
    with pytest.raises(CostError, match='unresolved'):
        ledger.validate_identity(identity)


def test_legacy_prices_cannot_be_used_for_gateway_and_call_only_needs_no_price(tmp_path):
    profile = CostProfile('https://example.invalid/prices', '2026-10-21', 'a' * 64,
                          (ModelPrice('org/shared-model', Decimal('1'), Decimal('1')),))
    with pytest.raises(CostError, match='official DeepSeek'):
        CostLedger(tmp_path / 'money', limit=Decimal('10'), profile=profile).reserve(_identity(), 4096)
    calls = CostLedger(tmp_path / 'calls', max_calls=1)
    calls.validate_identity(_identity())
    reservation = calls.reserve(_identity(), 4096)
    calls.finish(reservation, None, complete=True)
    with pytest.raises(CostError, match='call limit'):
        calls.validate_identity(_identity(provider='personal'))


def test_duplicate_routes_and_too_small_input_bound_are_rejected(tmp_path):
    with pytest.raises(CostError, match='invalid or unreadable'):
        _profile(tmp_path, duplicate=True)
    profile = _profile(tmp_path, bound=65536)
    ledger = CostLedger(tmp_path / 'cost', limit=Decimal('10'), profile=profile)
    with pytest.raises(CostError, match='input bound'):
        ledger.reserve(_identity(), 4096)
    assert ledger.snapshot.calls == 0 and not ledger.path.exists()
