"""用户配置经过真实启动组合、文件、SQLite 与 HTTP 边界。"""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from cicada import __main__ as cli
from cicada.plugins.deepseek import deepseek_plugin
from cicada.plugins.session_store import read_session_identity
from test_cli import _CaptureStream
from test_deepseek_application import response


def invoke(monkeypatch, args, root):
    from cicada.plugins.user_config import FileUserConfigStore

    out, err = _CaptureStream(), _CaptureStream()
    monkeypatch.setattr(cli.sys, "argv", ["cicada", *args])
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    return cli.main(user_config_store=FileUserConfigStore(root)), out.value, err.value


def write_config(root, defaults='', key='file-sentinel-key'):
    root.mkdir(exist_ok=True)
    ref = str(uuid4())
    (root / 'config.toml').write_text(
        'version = 1\n[defaults]\n' + defaults + '\n[credentials.deepseek]\nref = "' + ref + '"\n',
        encoding='utf-8',
    )
    (root / 'credentials.json').write_text(json.dumps({
        'version': 1, 'entries': {ref: {'provider': 'deepseek', 'api_key': key}},
    }), encoding='utf-8')
    return ref


@pytest.fixture
def requests(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    seen = []

    def handle(req):
        body = json.loads(req.content)
        seen.append((req.headers['authorization'], body))
        return httpx.Response(200, content=response(
            '思考' if body['thinking']['type'] == 'enabled' else None, '离线答复'))

    monkeypatch.setattr(cli, 'deepseek_plugin', lambda config, *, api_key: deepseek_plugin(
        config, api_key=api_key, transport=httpx.MockTransport(handle)))
    return seen


def test_file_defaults_and_credentials_reach_actual_request(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    ref = write_config(root, 'model = "deepseek-v4-pro"\nthink = false\n'
                       'reasoning_effort = "low"\nnum_ctx = 65536\nmax_output_tokens = 4096\n')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '任务'], root)
    assert code == 0, err + out
    auth, body = requests[0]
    assert auth == 'Bearer file-sentinel-key'
    assert body['model'] == 'deepseek-v4-pro'
    assert body['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in body and body['max_tokens'] == 4096
    session_id = out.split('[session id=')[1].split(' ')[0]
    identity = read_session_identity(directory=tmp_path / '.cicada' / 'sessions',
                                     session_id=session_id, workspace=tmp_path).launch
    assert identity.options['num_ctx'] == 65536
    assert identity.sources['model'] == 'user'
    for secret in ('file-sentinel-key', ref):
        assert secret not in out + err + repr(identity)
        assert secret.encode() not in (tmp_path / '.cicada' / 'sessions' / f'{session_id}.sqlite3').read_bytes()


def test_environment_without_user_directory_and_override_unused_bad_credentials(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    monkeypatch.setenv('DEEPSEEK_API_KEY', ' environment-sentinel ')
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '任务'], root)
    assert code == 0, err
    assert requests[-1][0] == 'Bearer environment-sentinel' and not root.exists()
    write_config(root)
    (root / 'credentials.json').write_bytes(b'bad-json-secret-sentinel')
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == root / 'credentials.json':
            pytest.fail('unselected credentials file was read')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '任务'], root)
    assert code == 0 and requests[-1][0] == 'Bearer environment-sentinel', err
    assert 'environment-sentinel' not in out + err


@pytest.mark.parametrize('flags, expected, source', [
    (['--model', 'deepseek-flash'], {'model': 'deepseek-flash'}, 'model'),
    (['--think'], {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}, 'think'),
    (['--reasoning-effort', 'max', '--think'], {'reasoning_effort': 'max'}, 'reasoning_effort'),
    (['--max-output-tokens', '8192'], {'max_tokens': 8192}, 'max_tokens'),
    (['--num-ctx', '131072'], {}, 'num_ctx'),
    (['--provider', 'deepseek'], {}, 'provider'),
])
def test_each_cli_field_overrides_user_defaults(monkeypatch, tmp_path, requests, flags, expected, source):
    root = tmp_path / 'user'
    write_config(root, 'provider = "deepseek"\nmodel = "deepseek-v4-pro"\nthink = false\n'
                 'reasoning_effort = "low"\nnum_ctx = 65536\nmax_output_tokens = 4096\n')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', *flags, '任务'], root)
    assert code == 0, err + out
    for field, value in expected.items():
        assert requests[-1][1][field] == value
    session_id = out.split('[session id=')[1].split(' ')[0]
    identity = read_session_identity(directory=tmp_path / '.cicada' / 'sessions',
                                    session_id=session_id, workspace=tmp_path).launch
    assert identity.sources[source] == 'cli'
    if source == 'num_ctx':
        assert identity.options['num_ctx'] == 131072


def test_omitted_file_fields_use_builtin_defaults(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    write_config(root, 'reasoning_effort = "max"\n')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '任务'], root)
    assert code == 0, err
    body = requests[0][1]
    assert body['model'] == 'deepseek-flash' and body['reasoning_effort'] == 'max'
    assert body['max_tokens'] == 32768 and body['thinking'] == {'type': 'enabled'}
    identity = read_session_identity(directory=tmp_path / '.cicada' / 'sessions',
        session_id=out.split('[session id=')[1].split(' ')[0], workspace=tmp_path).launch
    assert identity.sources['model'] == 'default' and identity.sources['reasoning_effort'] == 'user'


def test_restore_ignores_user_values_and_adopt_uses_them(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    write_config(root, 'model = "deepseek-v4-pro"\nreasoning_effort = "low"\n'
                 'num_ctx = 65536\nmax_output_tokens = 4096\n')
    args = ['--workspace', str(tmp_path)]
    code, out, err = invoke(monkeypatch, [*args, '--save-session', '第一问'], root)
    assert code == 0, err
    session_id = out.split('[session id=')[1].split(' ')[0]
    write_config(root, 'think = false\n', key='replacement-file-sentinel')
    code, out, err = invoke(monkeypatch, [*args, '--resume', session_id, '续问'], root)
    assert code == 0, err
    auth, body = requests[-1]
    assert auth == 'Bearer replacement-file-sentinel'
    assert body['model'] == 'deepseek-v4-pro' and body['reasoning_effort'] == 'low'
    assert body['max_tokens'] == 4096 and body['thinking'] == {'type': 'enabled'}
    identity = read_session_identity(directory=tmp_path / '.cicada' / 'sessions',
                                    session_id=session_id, workspace=tmp_path).launch
    assert identity.sources['model'] == 'archive'
    (root / 'credentials.json').write_bytes(b'broken')
    before = len(requests)
    code, _, err = invoke(monkeypatch, [*args, '--resume', session_id, '--no-think', '冲突'], root)
    assert code == 2 and 'think' in err and 'credentials' not in err and len(requests) == before
    write_config(root, 'think = false\n', key='adopt-file-sentinel')
    code, _, err = invoke(monkeypatch, [*args, '--resume', session_id,
                         '--resume-with-current-config', '采用'], root)
    assert code == 0, err
    assert requests[-1][0] == 'Bearer adopt-file-sentinel'
    assert requests[-1][1]['model'] == 'deepseek-flash'
    assert requests[-1][1]['thinking'] == {'type': 'disabled'}


def test_tui_uses_same_defaults_and_key_for_initial_prompt(monkeypatch, tmp_path, requests):
    import cicada.ui

    root = tmp_path / 'user'
    write_config(root, 'model = "deepseek-v4-pro"\nreasoning_effort = "max"\n')
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)

    async def tui(session, *, initial_prompt, diagnostics):
        try:
            result = await session.execute(initial_prompt)
            assert 'file-sentinel-key' not in repr(session.config) + repr(session.history) + repr(diagnostics())
            return result.exit_code
        finally:
            await session.aclose()

    monkeypatch.setattr(cicada.ui, 'run_tui_session', tui)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '首问'], root)
    assert code == 0, err
    assert len(requests) == 1 and requests[0][0] == 'Bearer file-sentinel-key'
    assert requests[0][1]['model'] == 'deepseek-v4-pro' and requests[0][1]['reasoning_effort'] == 'max'


@pytest.mark.parametrize('raw', [
    b'\xff', b'not TOML secret-sentinel', b'x' * 65537,
    b'version = 2', b'version = true', b'version = 1\nsecret_sentinel = "x"',
    b'version = 1\n[defaults]\napi_key = "secret-sentinel"',
    b'version = 1\n[defaults]\nthink = 1', b'version = 1\n[defaults]\nnum_ctx = true',
    b'version = 1\n[defaults]\nmax_output_tokens = 1.5',
    b'version = 1\n[defaults]\nprovider = "ollama"',
    b'version = 1\n[defaults]\nmodel = "secret-sentinel"',
    b'version = 1\n[defaults]\nreasoning_effort = "medium"',
    b'version = 1\n[defaults]\nnum_ctx = 1048577',
    b'version = 1\n[defaults]\nmax_output_tokens = 393217',
    b'version = 1\n[defaults]\nnum_ctx = 2048\nmax_output_tokens = 4096',
    b'version = 1\n[credentials.deepseek]\nref = "secret-sentinel"',
    b'version = 1\n[credentials.ollama]\nref = "secret-sentinel"',
], ids=[f'config-{i}' for i in range(18)])
def test_bad_config_fails_without_echo_even_with_cli_and_environment(monkeypatch, tmp_path, requests, raw):
    root = tmp_path / 'user'
    root.mkdir()
    (root / 'config.toml').write_bytes(raw)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'environment-secret-sentinel')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--model', 'deepseek-flash', '任务'], root)
    assert code == 2 and requests == [] and 'user configuration config:' in err
    assert 'secret-sentinel' not in out + err


@pytest.mark.parametrize('change', [
    'missing', 'format', 'utf8', 'oversize', 'duplicate', 'version', 'bool-version',
    'unknown', 'missing-ref', 'bad-ref', 'provider', 'unknown-entry', 'bad-key',
    'missing-key', 'long-key', 'control-key', 'wrong-entries-type',
])
def test_bad_selected_credentials_fail_safely(monkeypatch, tmp_path, requests, change):
    root = tmp_path / 'user'
    ref = write_config(root)
    path = root / 'credentials.json'
    data = json.loads(path.read_text())
    if change == 'missing':
        path.unlink()
    elif change in ('format', 'utf8', 'oversize', 'duplicate'):
        raw = {'format': b'secret-sentinel', 'utf8': b'\xff', 'oversize': b'x' * 65537,
               'duplicate': b'{"version":1,"version":1,"entries":{}}'}[change]
        path.write_bytes(raw)
    else:
        if change == 'version': data['version'] = 2
        if change == 'bool-version': data['version'] = True
        if change == 'unknown': data['secret-sentinel'] = 'x'
        if change == 'missing-ref': data['entries'] = {}
        if change == 'bad-ref': data['entries'] = {'secret-sentinel': data['entries'][ref]}
        if change == 'provider': data['entries'][ref]['provider'] = 'ollama'
        if change == 'unknown-entry': data['entries'][ref]['secret-sentinel'] = 'x'
        if change == 'bad-key': data['entries'][ref]['api_key'] = 'secret sentinel'
        if change == 'missing-key': del data['entries'][ref]['api_key']
        if change == 'long-key': data['entries'][ref]['api_key'] = 'x' * 4097
        if change == 'control-key': data['entries'][ref]['api_key'] = 'secret\x00sentinel'
        if change == 'wrong-entries-type': data['entries'] = []
        path.write_text(json.dumps(data), encoding='utf-8')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '任务'], root)
    assert code == 2 and requests == [] and 'credentials:' in err
    assert 'secret-sentinel' not in out + err and ref not in out + err


@pytest.mark.parametrize('key', ['bad key', 'bad\nkey', 'x' * 4097, 'bad\x00key'],
                         ids=['space', 'newline', 'oversize', 'control'])
def test_invalid_nonempty_environment_does_not_fall_back(monkeypatch, tmp_path, requests, key):
    root = tmp_path / 'user'
    write_config(root)
    original_get = cli.os.environ.get
    monkeypatch.setattr(cli.os.environ, 'get', lambda name, *args:
                        key if name == 'DEEPSEEK_API_KEY' else original_get(name, *args))
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '任务'], root)
    assert code == 2 and requests == [] and 'environment: invalid-key' in err
    assert key not in out + err


@pytest.mark.parametrize('changes', [2, 3])
def test_concurrent_publishing_retries_stale_ref_at_most_three_times(monkeypatch, tmp_path, requests, changes):
    root = tmp_path / 'user'
    write_config(root, key='old-file-sentinel')
    original_open = Path.open
    credential_reads = []

    def publishing_open(path, mode='r', *args, **kwargs):
        if path == root / 'credentials.json' and mode == 'rb':
            credential_reads.append(path)
            if len(credential_reads) <= changes:
                # 模拟两次连续可靠发布并裁剪旧ref；这一轮旧config找不到ref，须先复读。
                write_config(root, key=f'new-file-sentinel-{len(credential_reads)}')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', publishing_open)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '任务'], root)
    assert len(credential_reads) == 3
    if changes == 2:
        assert code == 0, err
        assert requests[0][0] == 'Bearer new-file-sentinel-2'
    else:
        assert code == 2 and requests == [] and 'configuration-changing' in err
        assert 'missing-ref' not in err
    assert 'file-sentinel' not in out + err


def test_help_list_fake_ollama_and_selector_cancel_skip_user_config_and_key(monkeypatch, tmp_path):
    import cicada.ui

    from cicada.plugins.user_config import FileUserConfigStore

    def forbidden_load(*args, **kwargs):
        pytest.fail('offline or Ollama entry read user configuration')

    monkeypatch.setattr(FileUserConfigStore, 'load', forbidden_load)
    original_get = cli.os.environ.get
    monkeypatch.setattr(cli.os.environ, 'get', lambda name, *args:
                        pytest.fail('entry read DeepSeek key') if name == 'DEEPSEEK_API_KEY'
                        else original_get(name, *args))
    root = tmp_path / 'absent-user'
    script = tmp_path / 'script.json'
    script.write_text('[{"text":"done","stop":true}]', encoding='utf-8')
    for flags in (['--list-sessions'], ['--script', str(script), '任务']):
        code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), *flags], root)
        assert code == 0, err
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, ['--help'], root)
    assert exc.value.code == 0

    async def send(client, req, *args, **kwargs):
        assert req.url.path in ('/api/version', '/api/chat')
        content = (b'{"version":"mock"}' if req.url.path == '/api/version' else
                   b'{"message":{"role":"assistant","content":"answer"},"done":true,"done_reason":"stop"}\n')
        return httpx.Response(200, content=content, request=req)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--provider', 'ollama', '任务'], root)
    assert code == 0, err
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)

    async def cancelled(workspace):
        return None

    monkeypatch.setattr(cli, '_launch_session_selector', cancelled)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '--resume'], root)
    assert code == 0, err
    assert not root.exists()


@pytest.mark.parametrize('tui', [False, True])
def test_missing_key_exits_without_input_http_or_user_creation(monkeypatch, tmp_path, requests, tui):
    root = tmp_path / 'absent-user'
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    monkeypatch.setattr('builtins.input', lambda *args: pytest.fail('unexpected interaction'))
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), *(['--tui'] if tui else []), '任务'], root)
    assert code == 2 and 'DEEPSEEK_API_KEY' in err and '--tui' in err
    assert requests == [] and not root.exists() and not (tmp_path / '.cicada').exists()


def test_provider_uses_resolved_key_even_if_environment_changes_before_setup(monkeypatch, tmp_path, requests):
    from cicada.plugins.user_config import FileUserConfigStore

    root = tmp_path / 'user'
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'resolved-sentinel')
    store = FileUserConfigStore(root)
    original_load = store.load

    def load(key):
        snapshot = original_load(key)
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'later-sentinel')
        return snapshot

    monkeypatch.setattr(store, 'load', load)
    out, err = _CaptureStream(), _CaptureStream()
    monkeypatch.setattr(cli.sys, 'argv', ['cicada', '--workspace', str(tmp_path), '任务'])
    monkeypatch.setattr(cli.sys, 'stdout', out)
    monkeypatch.setattr(cli.sys, 'stderr', err)
    assert cli.main(user_config_store=store) == 0, err.value
    assert requests[0][0] == 'Bearer resolved-sentinel'


def test_safe_view_and_opaque_secret_have_no_credentials_or_reference(tmp_path):
    from cicada.plugins.user_config import FileUserConfigStore
    from cicada.user_config import SecretValue

    root = tmp_path / 'user'
    ref = write_config(root)
    store = FileUserConfigStore(root)
    view = store.read_view()
    assert view.credential_present and view.credential_source == 'file'
    assert not hasattr(view, 'credential') and not hasattr(view, 'credential_ref')
    assert 'file-sentinel-key' not in repr(view) + repr(store.load()) + str(SecretValue('file-sentinel-key'))
    assert ref not in repr(view) + repr(store.load())


def test_adopt_bad_archive_is_rejected_before_credentials(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', str(uuid4()),
                                      '--resume-with-current-config', '任务'], root)
    assert code == 2 and 'session' in err and 'DEEPSEEK_API_KEY' not in err
    assert requests == [] and not root.exists()


def test_user_context_limit_still_obeys_check_mode_minimum(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    write_config(root, 'num_ctx = 16384\nmax_output_tokens = 4096\n')
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--check-command', 'echo checked', '任务'], root)
    assert code == 2 and '32768' in err and requests == []
