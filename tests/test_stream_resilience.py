"""次数额度下的有限重试：分类、次数、账本与不重放边界。"""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import format_datetime
from types import SimpleNamespace

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


async def test_retry_on_429_then_success_keeps_amount_unknown(tmp_path, monkeypatch):
    calls = []
    trace = []

    class RejectedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b''

        async def aclose(self):
            trace.append('close:1')

    def handler(request):
        calls.append(request)
        trace.append(f'send:{len(calls)}')
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after': '0'}, stream=RejectedStream())
        return httpx.Response(200, content=_frame('ok'))

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    reserve, finish_failed = ledger.reserve, ledger.finish_failed_attempt
    def traced_reserve(*args):
        reservation = reserve(*args)
        trace.append(f'reserve:{ledger.snapshot.calls}')
        return reservation

    def traced_finish(*args, **kwargs):
        finish_failed(*args, **kwargs)
        trace.append('failed-persisted:1')

    monkeypatch.setattr(ledger, 'reserve', traced_reserve)
    monkeypatch.setattr(ledger, 'finish_failed_attempt', traced_finish)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(think=False), 'k', cost_control=ledger)
        observe = model._record_observation
        def before_send(request, body):
            trace.append(f'before-send:{ledger.snapshot.calls}')
            observe(request, body)
        monkeypatch.setattr(model, '_record_observation', before_send)
        events = await _collect(model)
    assert events[-1].stop_reason == 'stop' and len(calls) == 2
    assert sum(isinstance(event, StreamDone) for event in events) == 1
    assert trace == ['reserve:1', 'before-send:1', 'send:1', 'close:1',
                     'failed-persisted:1', 'reserve:2', 'before-send:2', 'send:2']
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
    assert 'attempt 1/3' in events[-1].error and 'admission refused' in events[-1].error
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
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None


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


async def test_retry_after_over_limit_and_cancel_during_backoff(tmp_path, monkeypatch):
    from cicada.plugins import chat_stream
    calls = []

    def slow_retry(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after': '15'})

    ledger = CostLedger(tmp_path / 'wait', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_retry)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(calls) == 1 and events[-1].stop_reason == 'error'
    assert 'HTTP 429' in events[-1].error
    assert 'attempt 1/3' in events[-1].error and 'retry wait exceeds limit' in events[-1].error

    cancelled = []
    token = CancelToken()
    entered = asyncio.Event()
    backoff = chat_stream.sleep_with_cancel

    async def waiting(seconds, cancel):
        entered.set()
        return await backoff(seconds, cancel)

    monkeypatch.setattr(chat_stream, 'LOCAL_DELAYS', (1, 1))
    monkeypatch.setattr(chat_stream, 'sleep_with_cancel', waiting)

    def cancelling(request):
        cancelled.append(request)
        return httpx.Response(429)

    ledger = CostLedger(tmp_path / 'cancel', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelling)) as client:
        owner = asyncio.create_task(_collect(ChatModel(client, _config(), 'k', cost_control=ledger), token))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not owner.done() and ledger._pending is None
            token.cancel()
            events = await asyncio.wait_for(owner, 2)
        finally:
            token.cancel()
            await asyncio.gather(owner, return_exceptions=True)
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


async def test_retryable_response_close_failure_keeps_pending_and_stops(tmp_path):
    class FailedClose(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b''

        async def aclose(self):
            raise RuntimeError('synthetic close failure')

    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(429, stream=FailedClose())

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(requests) == ledger.snapshot.calls == 1
    assert len(events) == 1 and events[0].stop_reason == 'error'
    assert events[0].error.startswith('chat response close failed (attempt 1/3;')
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None


async def test_connect_failure_retries_only_during_send(tmp_path, monkeypatch):
    # A real transport connection failure before receiving a response is eligible.
    sent = []
    def reconnect(request):
        sent.append(request)
        if len(sent) == 1:
            raise httpx.ConnectError('synthetic connection failure', request=request)
        return httpx.Response(200, content=_frame('ok'))

    ledger = CostLedger(tmp_path / 'connect', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reconnect)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 2 and events[-1].stop_reason == 'stop'
    assert ledger._pending is None and ledger.snapshot.sealed_reason is None

    # An observer failure occurs before any send and must retain its reservation.
    observed = []
    def failed_observer(self, request, body):
        observed.append(request)
        raise httpx.ConnectError('synthetic observer failure')

    monkeypatch.setattr(DeepSeekModel, '_record_observation', failed_observer)
    ledger = CostLedger(tmp_path / 'observer', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reconnect)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(think=False), 'k', cost_control=ledger)
        events = await _collect(model)
    assert len(observed) == ledger.snapshot.calls == 1 and len(sent) == 2
    assert events[-1].stop_reason == 'error'
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None


async def test_tool_fragment_then_connect_failure_never_replays(tmp_path):
    from cicada.core.ports import ToolCallEvent

    class PartialTool(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            fragment = {'choices': [{'index': 0, 'delta': {'tool_calls': [
                {'index': 0, 'id': 'call-1', 'type': 'function',
                 'function': {'name': 'read', 'arguments': '{'}}]}}]}
            yield b'data: ' + json.dumps(fragment).encode() + b'\n\n'
            raise httpx.ConnectError('synthetic failure after response')

        async def aclose(self):
            self.closed = True

    stream = PartialTool()
    sent = []
    def handler(request):
        sent.append(request)
        return httpx.Response(200, stream=stream)

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 1 and stream.closed
    assert events[-1].stop_reason == 'error'
    assert not any(isinstance(event, ToolCallEvent) for event in events)
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None


async def test_failed_attempt_persistence_fault_stops_send(tmp_path, monkeypatch):
    from cicada import costs

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    publish = costs.os.replace
    def failed_publish(source, destination):
        if destination == ledger.path and ledger._pending is None:
            raise OSError('synthetic ledger publish failure')
        return publish(source, destination)

    monkeypatch.setattr(costs.os, 'replace', failed_publish)
    sent = []
    def handler(request):
        sent.append(request)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 1
    assert len(events) == 1 and events[0].stop_reason == 'error'
    assert events[0].error.startswith('chat cost ledger failed (attempt 1/3;')
    assert ledger._pending is not None and ledger.snapshot.sealed_reason == 'cost ledger write failed'
    assert json.loads(ledger.path.read_text(encoding='utf-8'))['calls'] == 1


@pytest.mark.parametrize('form', ['seconds', 'date'])
async def test_retry_after_seconds_and_date_use_server_wait(tmp_path, monkeypatch, form):
    from cicada.plugins import chat_stream

    now = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)
    class FixedUTC(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(chat_stream, 'datetime', FixedUTC)
    monkeypatch.setattr(chat_stream, 'LOCAL_DELAYS', (0.5, 1))
    waits, sent = [], []
    async def wait(seconds, cancel):
        waits.append(seconds)
        return True
    monkeypatch.setattr(chat_stream, 'sleep_with_cancel', wait)
    header = '2' if form == 'seconds' else format_datetime(now.replace(second=2), usegmt=True)
    def handler(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(429, headers={'retry-after': header})
        return httpx.Response(200, content=_frame('ok'))

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 2 and waits == [2.0]
    assert events[-1].stop_reason == 'stop'


async def test_retry_attempts_share_one_total_deadline(tmp_path, monkeypatch):
    from cicada.plugins import chat_stream
    from cicada.plugins.openai_chat import model as chat_model

    clock, waits, sent = [100.0], [], []
    monkeypatch.setattr(chat_stream, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(chat_stream, 'LOCAL_DELAYS', (0.5, 1))
    monkeypatch.setattr(chat_model, 'CALL_DEADLINE_S', 1.2)
    async def wait(seconds, cancel):
        waits.append(seconds)
        clock[0] += seconds
        return True
    monkeypatch.setattr(chat_stream, 'sleep_with_cancel', wait)
    def handler(request):
        sent.append(request)
        return httpx.Response(503)

    ledger = CostLedger(tmp_path / 'cost', max_calls=5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 2 and waits == [0.5]
    assert len(events) == 1 and events[0].stop_reason == 'error'
    assert 'HTTP 503' in events[0].error
    assert 'attempt 2/3' in events[0].error and 'remaining deadline' in events[0].error
    assert ledger._pending is None and ledger.snapshot.sealed_reason is None


async def test_deadline_during_observer_prevents_second_send(tmp_path, monkeypatch):
    from cicada.plugins import chat_stream
    from cicada.plugins.deepseek import model as deepseek_model

    clock, sent, observed = [100.0], [], []
    monkeypatch.setattr(chat_stream, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(deepseek_model, 'CALL_DEADLINE_S', 1.0)
    async def wait(seconds, cancel):
        return True
    monkeypatch.setattr(chat_stream, 'sleep_with_cancel', wait)
    def observer(self, request, body):
        observed.append(request)
        if len(observed) == 2:
            clock[0] += 2
    monkeypatch.setattr(DeepSeekModel, '_record_observation', observer)
    def handler(request):
        sent.append(request)
        return httpx.Response(503) if len(sent) == 1 else httpx.Response(200, content=_frame('late'))

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(think=False), 'k', cost_control=ledger)
        events = await _collect(model)
    assert len(sent) == 1 and ledger.snapshot.calls == len(observed) == 2
    assert events[-1].stop_reason == 'error' and 'deadline' in events[-1].error
    assert 'attempt 1/3' in events[-1].error
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None


async def test_oversized_retry_after_never_shortens_server_wait(tmp_path):
    sent = []
    def handler(request):
        sent.append(request)
        return httpx.Response(429, headers={'retry-after': '9' * 400})

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await _collect(ChatModel(client, _config(), 'k', cost_control=ledger))
    assert len(sent) == ledger.snapshot.calls == 1
    assert events[-1].stop_reason == 'error' and 'HTTP 429' in events[-1].error
    assert 'retry wait exceeds limit' in events[-1].error
    assert ledger._pending is None and ledger.snapshot.sealed_reason is None


@pytest.mark.parametrize('failure', ['eof', 'protocol', 'length', 'read-timeout', 'frame-limit'])
async def test_count_only_stream_failures_never_replay(tmp_path, failure):
    from cicada.core.ports import ReasoningDelta, ToolCallEvent

    class FailedStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            payload = {'choices': [{'index': 0, 'delta': {'reasoning_content': 'part'}}]}
            yield b'data: ' + json.dumps(payload).encode() + b'\n\n'
            if failure == 'protocol':
                yield b'data: broken\n\n'
            elif failure == 'length':
                yield _frame(finish='length')
            elif failure == 'read-timeout':
                raise httpx.ReadTimeout('synthetic read timeout')
            elif failure == 'frame-limit':
                yield b'data: ' + b'x' * (256 * 1024)

        async def aclose(self):
            self.closed = True

    sent, stream = [], FailedStream()
    def handler(request):
        sent.append(request)
        return httpx.Response(200, stream=stream)

    ledger = CostLedger(tmp_path / 'cost', max_calls=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = DeepSeekModel(client, DeepSeekConfig(), 'k', cost_control=ledger)
        events = await _collect(model)
    assert len(sent) == ledger.snapshot.calls == 1 and stream.closed
    assert events[0] == ReasoningDelta('part')
    assert events[-1].stop_reason == ('length' if failure == 'length' else 'error')
    assert sum(isinstance(event, StreamDone) for event in events) == 1
    assert not any(isinstance(event, ToolCallEvent) for event in events)
    assert ledger._pending is not None and ledger.snapshot.sealed_reason is not None
