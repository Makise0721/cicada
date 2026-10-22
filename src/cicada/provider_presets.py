"""内置供应商的非秘密配置事实；不读取环境、文件或网络。"""

from cicada.user_config import ModelProfile, ProviderProfile

OPENCODE_GO_BASE_URL = 'https://opencode.ai/inference/go/openai/v1'
OPENCODE_GO_MODEL = 'deepseek-v4.1-flash'
OPENCODE_GO_CONTEXT_WINDOW = 1000000


def is_opencode_go_endpoint(base_url: str) -> bool:
    return base_url.rstrip('/') == OPENCODE_GO_BASE_URL


def opencode_go_profile() -> ProviderProfile:
    model = ModelProfile(OPENCODE_GO_MODEL, OPENCODE_GO_MODEL, name='DeepSeek V4.1 Flash',
                         context_window=OPENCODE_GO_CONTEXT_WINDOW, max_output_tokens=32768, think=True)
    return ProviderProfile('opencode-go', 'OpenCode Go', 'openai-chat-completions',
                           OPENCODE_GO_BASE_URL, (model,), model.key)
