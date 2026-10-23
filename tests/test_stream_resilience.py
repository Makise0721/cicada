"""次数额度下的有限重试：分类、次数、账本与不重放边界。"""

import json
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelRequest, StreamDone, TextDelta
from cicada.costs import CostLedger, CostProfile, ModelPrice
from cicada.plugins.deepseek.model import DeepSeekModel
from cicada.plugins.deepseek.protocol import DeepSeekConfig
from cicada.plugins.openai_chat.model import ChatModel
from cicada.plugins.openai_chat.protocol import ChatConfig

REQUEST = ModelRequest(messages=(UserMessage(text='hi'),), tools=())


def _config():
    return ChatConfig('work', 'model-key', 'org/model',
                      'https://gateway.example/v1', 131072, 4096)


def _frame(text='', finish='stop', usage=None):
    payload = {'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': finish}],
               'usage': usage or {'prompt_tokens': 10, 'completion_tokens': 5}}
    return ('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode()


def _priced_ledger(tmp_path):
    profile = CostProfile('https://gateway.example/prices', '2026-10-21', 'a' * 64, (
        ModelPrice('org/model', Decimal('1'), Decimal('2'), 'work',
                   'openai-chat-completions', 'https://gateway.example/v1'),), schema_version=2)
    return CostLedger(tmp_path / 'priced', limit=Decimal('10'), profile=profile)


async def _collect(model, token=None):
    return [event async for event in model.stream(REQUEST, token or CancelToken())]


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    from cicada.plugins import chat_stream
    monkeypatch.setattr(chat_stream, 'LOCAL_DELAYS', (0.01, 0.02))


async def test_retry_on_429_then_success_keeps_amount_unknown(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after': '0'})
        return httpx.Response(200, content=_frame('ok'))

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert events[-1].stop_reason == 'stop' and len(calls) == 2
    assert ledger.snapshot.calls == 2
    assert ledger._pending is None and ledger.snapshot.sealed_reason is None
    assert ledger.snapshot.settled_upper_cny is None


async def test_three_retryable_failures_end_once_and_release_ledger(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    done = [event for event in events if isinstance(event, StreamDone)]
    assert len(calls) == 3 and len(done) == 1
    assert done[0].stop_reason == 'error' and 'HTTP 503' in done[0].error
    assert ledger.snapshot.calls == 3
    assert ledger._pending is None and ledger.snapshot.sealed_reason is None


async def test_call_limit_one_keeps_first_http_class(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    ledger = CostLedger(tmp_path / 'cost', max_calls=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(calls) == 1 and events[-1].stop_reason == 'error'
    assert 'HTTP 429' in events[-1].error and 'call limit' in events[-1].error
    assert ledger.snapshot.calls == 1


async def test_priced_ledger_never_retries(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    ledger = _priced_ledger(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(calls) == 1 and events[-1].stop_reason == 'error'
    assert ledger.snapshot.sealed_reason is not None


async def test_accepted_text_blocks_retry_after_eof(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        payload = {'choices': [{'index': 0, 'delta': {'content': 'part'}}]}
        return httpx.Response(200, content=b'data: ' + json.dumps(payload).encode() + b'\n\n')

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert [event.text for event in events if isinstance(event, TextDelta)] == ['part']
    assert len(calls) == 1 and events[-1].stop_reason == 'error'
    assert 'without DONE' in events[-1].error


async def test_retry_after_over_limit_and_cancel_during_backoff(tmp_path):
    calls = []

    def slow_retry(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after': '15'})

    ledger = CostLedger(tmp_path / 'wait', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_retry)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(calls) == 1 and events[-1].stop_reason == 'error'
    assert 'HTTP 429' in events[-1].error

    cancelled = []
    token = CancelToken()

    def cancelling(request):
        cancelled.append(request)
        token.cancel()
        return httpx.Response(429)

    ledger = CostLedger(tmp_path / 'cancel', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelling)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger), token)
    assert len(cancelled) == 1 and events[-1].stop_reason == 'aborted'


async def test_deepseek_adapter_shares_the_same_retry_boundary(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, content=_frame('ok'))

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    model = DeepSeekModel.__new__(DeepSeekModel)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(think=False, max_tokens=2048), 'k',
                              cost_control=ledger)
        events = await _collect(model)
    assert len(calls) == 2 and events[-1].stop_reason == 'stop'
    assert ledger.snapshot.calls == 2
