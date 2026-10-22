"""内置供应商的非秘密配置事实；不读取环境、文件或网络。"""

from cicada.user_config import ModelProfile, ProviderProfile

OPENCODE_GO_BASE_URL = 'https://opencode.ai/inference/go/openai/v1'
OPENCODE_GO_MODEL = 'kimi-k2.6'


def is_opencode_go_endpoint(base_url: str) -> bool:
    return base_url.rstrip('/') == OPENCODE_GO_BASE_URL


def opencode_go_profile() -> ProviderProfile:
    model = ModelProfile(OPENCODE_GO_MODEL, OPENCODE_GO_MODEL, name='Kimi K2.6',
                         context_window=262144, max_output_tokens=32768)
    return ProviderProfile('opencode-go', 'OpenCode Go', 'openai-chat-completions',
                           OPENCODE_GO_BASE_URL, (model,), model.key)
