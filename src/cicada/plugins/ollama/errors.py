"""Ollama 适配器归一错误文本的上下文超限分类 (03, D7/PA06).

输入是本插件 OllamaModel 归一后的 StreamDone("error").error 文本
("ollama http <code>: ..." / "ollama protocol error: ..."), 只做文本形态判定,
不发任何请求。true 只给确认的上下文超限形态; 取消/网络/超时/其他服务端
错误/坏行一律 false (回放 fixture 逐条钉住, 见 tests/plugins/ollama
/test_context_overflow.py 的逐字段版本化出处)。

确认形态及其上游出处 (均经 /api/chat 流式路径到达客户端):

1. "the input length exceeds the context length"
   - ollama v0.12.9 runner/ollamarunner/runner.go:114
     `var errorInputTooLong = errors.New("the input length exceeds the context length")`;
     runner 以 HTTP 400 纯文本回给 server (runner.go:857-858), server 把 >=400
     响应包成 api.StatusError (llm/server.go:1480-1486), ChatHandler 经
     errors.As 命中 StatusError 分支, 客户端收到带 status 的 NDJSON 错误行
     (`ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}`,
     server/routes.go:2196-2200)。
   - ollama v0.35.1 llm/llama_server.go:2550 (normalizeEmbeddingError) 复用同一
     规范文案, 并在 isEmbeddingInputLimitError 中把 llama.cpp 各上下文形态归一到它。

2. "the prompt is longer than the context length currently available to the model; ..."
   - ollama v0.35.1 llm/llama_server.go:299-302 (completionPromptForRequest,
     ContextShift 关闭且提示长于 num_ctx-1) → api.StatusError{400, ErrorMessage}
     → ChatHandler 流式错误行 `ch <- gin.H{"error": serr.ErrorMessage, "status": ...}`
     (server/routes.go:783-786)。

3. "request (N tokens) exceeds the available context size (M tokens), try increasing it"
   - llama.cpp @ b11232 (ollama v0.35.1 LLAMA_CPP_VERSION 钉定, commit
     6f767fe960c3b97cf37fac4626c86400561ca1e4) tools/server/server-context.cpp:3218-3224
     (ERROR_TYPE_EXCEED_CONTEXT_SIZE); ollama v0.35.1 statusErrorMessage
     (llm/llama_server.go:1888-1900) 原样透传, ollama 自带测试
     llm/llama_server_test.go:2719 记录同一文案。
"""

from __future__ import annotations

import re

# 逐形态匹配: 大小写不敏感 (statusErrorMessage 原样透传, 归一检测在 ollama 侧
# 也按小写包含判定); 第 3 形态的 token 数是服务端运行时数字, 用正则锚定结构。
_AVAILABLE_CONTEXT_PATTERN = re.compile(
    r"request \(\d+ tokens\) exceeds the available context size \(\d+ tokens\)",
    re.IGNORECASE,
)


def classify_context_overflow(error_text: str) -> bool:
    """判定 adapter 归一后的错误文本是否为确认的上下文超限形态 (中性布尔注入接缝)."""
    if isinstance(error_text, bool) or not isinstance(error_text, str):
        raise TypeError("error_text must be a str")
    lowered = error_text.lower()
    return (
        "the input length exceeds the context length" in lowered
        or "the prompt is longer than the context length" in lowered
        or _AVAILABLE_CONTEXT_PATTERN.search(error_text) is not None
    )
