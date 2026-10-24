"""摘要通道经统一发送边界：重试成功只写一条记录，失败不写记录。"""

import json
from dataclasses import replace

import httpx
import pytest

from cicada.application import open_application
from cicada.costs import CostLedger
from cicada.provider_presets import OPENCODE_GO_BASE_URL, OPENCODE_GO_MODEL
from cicada.user_config import ModelProfile, ProviderProfile

from test_application import _repo
from test_deepseek_application import config, response
from test_multi_provider import application_config


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr('cicada.plugins.chat_stream.LOCAL_DELAYS', (0.01, 0.02))


def _provider_config(root, transport, adapter):
    if adapter == 'deepseek':
        return config(root, transport)
    go = adapter == 'go'
    model = ModelProfile('coder', OPENCODE_GO_MODEL if go else 'org/model',
                         max_output_tokens=32768, think=go)
    provider = ProviderProfile(adapter, adapter, 'openai-chat-completions',
        OPENCODE_GO_BASE_URL if go else 'https://chat.example/v1', (model,), model.key)
    return application_config(root, transport, (provider,), adapter + '/' + model.key)


def _summary_handler(attempts, *, statuses):
    index = 0

    def handle(request):
        nonlocal index
        body = json.loads(request.content)
        if 'tools' in body:
            thinking = body.get('thinking', {}).get('type') == 'enabled'
            return httpx.Response(200, content=response('thought' if thinking else None))
        index += 1
        attempts.append(index)
        if index <= len(statuses) and statuses[index - 1] != 200:
            return httpx.Response(statuses[index - 1])
        return httpx.Response(200, content=response(None))

    return handle


@pytest.mark.parametrize('adapter', ['deepseek', 'chat', 'go'])
async def test_summary_retry_success_persists_exactly_one_record(tmp_path, adapter):
    root = _repo(tmp_path)
    attempts = []
    ledger = CostLedger(root / '.cicada' / 'cost', max_calls=10)
    handler = _summary_handler(attempts, statuses=[429])
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(503)
        return handler(request)

    base = _provider_config(root, httpx.MockTransport(handle), adapter)
    session = await open_application(replace(base, cost_control=ledger))
    try:
        await session.execute('first')
        await session.execute('second')
        before = session.history
        compacted = await session.compact_context()
        assert compacted is not None
        assert len(session.summary_records) == 1 and attempts == [1, 2]
        assert ledger.snapshot.calls == 5 and len(sent) == 5
        assert sent[0] == sent[1] and sent[3] == sent[4]
        assert len(session.process_model_calls) == 2
        assert len(session.process_summary_calls) == 1 and session.process_summary_calls[0].ok
        assert ledger._pending is None and ledger.snapshot.sealed_reason is None
        assert session.history == before
        records = session.summary_records
        blocks = session._context_projection.blocks
        assert session._store.recovery_snapshot().summaries == records
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    fresh = CostLedger(root / '.cicada' / 'fresh-cost', max_calls=4)
    restored = await open_application(replace(base, cost_control=fresh,
        persistence=replace(base.persistence, session_id=session_id)))
    try:
        assert restored.summary_records == records
        assert restored._context_projection.blocks == blocks
        assert restored.persistence.restored_model_call_count == 2
        assert restored.process_model_calls == restored.process_summary_calls == ()
        assert restored.budget.anchor is None and fresh.snapshot.calls == 0
        await restored.execute('after restart')
        assert fresh.snapshot.calls == len(restored.process_model_calls) == 1
        assert restored.persistence.restored_model_call_count == 2
        assert ledger.snapshot.calls == 5
    finally:
        await restored.aclose()


async def test_summary_retry_exhaustion_keeps_old_projection(tmp_path):
    root = _repo(tmp_path)
    attempts = []
    ledger = CostLedger(root / '.cicada' / 'cost', max_calls=10)
    base = config(root, httpx.MockTransport(
        _summary_handler(attempts, statuses=[200, 429, 503, 503])))
    session = await open_application(replace(base, cost_control=ledger))
    try:
        await session.execute('first')
        await session.execute('second')
        await session.compact_context()
        await session.execute('third')
        before = session.history
        records = session.summary_records
        blocks = session._context_projection.blocks
        assert len(records) == len(blocks) == 1
        with pytest.raises(RuntimeError, match='summary generation failed'):
            await session.compact_context()
        assert attempts == [1, 2, 3, 4]
        assert session.summary_records == session._store.recovery_snapshot().summaries == records
        assert session._context_projection.blocks == blocks
        assert session.history == before
        assert ledger.snapshot.calls == 7 and ledger._pending is None
        assert ledger.snapshot.sealed_reason is None
        assert len(session.process_model_calls) == 3
        assert [outcome.ok for outcome in session.process_summary_calls] == [True, False]
    finally:
        await session.aclose()
