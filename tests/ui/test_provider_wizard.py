"""新向导的凭据提交隔离和受确认的真实配置保存。"""

from dataclasses import replace

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui.login_wizard import make_login_wizard
from cicada.ui.provider_wizard import ProviderWizard
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.user_config import LoginSubmission, ModelProfile, ProviderProfile, SecretValue, UserDefaults

ENTER = KeyPress(Keys.ControlM, '\r')
DOWN = KeyPress(Keys.Down, '')


async def _configured(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    snapshot = store.load()
    provider = ProviderProfile('work', '工作', 'openai-chat-completions', 'https://gateway.invalid/v1',
                               (ModelProfile('coder', 'org/model'),), 'coder')
    result = await store.save(LoginSubmission(
        UserDefaults(provider='work', model='coder', think=False, max_output_tokens=8192),
        'replace', snapshot.revision, credential=SecretValue('old-work-sentinel'),
        provider_profile=provider, set_default=False))
    assert result.defaults_saved
    return store


async def test_provider_key_replace_is_masked_and_keeps_default_and_other_models(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    store = await _configured(tmp_path)
    before_default = store.load().defaults
    view = store.read_view(provider_id='work')
    wizard = make_login_wizard(view, title='Cicada · 登录配置')
    assert isinstance(wizard, ProviderWizard)
    # 保留 coder，并通过菜单增加第二个远端模型；不覆盖同实例原模型。
    for key in (ENTER, ENTER, DOWN, ENTER, DOWN, ENTER):
        wizard.handle(key)
    wizard.handle(KeyPress(Keys.BracketedPaste, 'org/second-model'))
    wizard.handle(ENTER)
    for key in (*([DOWN] * 8), ENTER):
        wizard.handle(key)
    # 工作实例 → 凭据 → 新Key；输入内容从不进入显示。
    for key in (DOWN, DOWN, ENTER, DOWN, ENTER):
        wizard.handle(key)
    wizard.handle(KeyPress(Keys.BracketedPaste, 'replacement-work-sentinel'))
    text = '\n'.join(wizard.render_lines(80, 24))
    assert 'replacement-work-sentinel' not in text and '*' in text
    wizard.handle(ENTER)
    # 设置页在凭据行，跳过默认开关直接到确认。
    for key in (DOWN, DOWN, ENTER):
        wizard.handle(key)
    wizard.render_lines(80, 40)
    assert wizard.handle(ENTER) == 'submit'
    submission = wizard.submission()
    assert not submission.set_default and submission.provider_profile.credential_ref is None
    wizard.clear_secret()
    result = await store.save(submission)
    assert result.defaults_saved
    assert store.load(provider_id='work').credential.reveal() == 'replacement-work-sentinel'
    assert store.load().defaults == before_default
    assert {item.id for item in store.read_profiles()} == {'deepseek', 'work'}
    work = next(item for item in store.read_profiles() if item.id == 'work')
    assert {item.model_id for item in work.models} == {'org/model', 'org/second-model'}


async def test_provider_confirmation_blocks_hidden_connection_and_cancel_discards_secret(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    store = await _configured(tmp_path)
    view = replace(store.read_view(provider_id='work'), actions=('save-and-apply', 'defaults-only'))
    wizard = ProviderWizard(view, title='Cicada · 登录配置')
    for key in (ENTER, DOWN, DOWN, ENTER, DOWN, ENTER):
        wizard.handle(key)
    wizard.handle(KeyPress(Keys.BracketedPaste, 'invalid\nsecret'))
    assert 'invalid' not in '\n'.join(wizard.render_lines(80, 24))
    wizard.handle(KeyPress(Keys.BracketedPaste, 'cancel-work-sentinel'))
    wizard.handle(ENTER)
    for key in (DOWN, DOWN, ENTER):
        wizard.handle(key)
    narrow = wizard.render_lines(18, 6)
    assert all(visible_width(line) <= 18 for line in narrow)
    assert wizard.handle(ENTER) is None and not wizard.submitting
    assert wizard.handle(KeyPress(Keys.Escape, '\x1b')) == 'cancel'
    assert wizard._credential is None
    assert 'cancel-work-sentinel' not in strip_terminal_sequences('\n'.join(wizard.render_lines(80, 40)))
    assert store.load(provider_id='work').credential.reveal() == 'old-work-sentinel'
