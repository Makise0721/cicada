"""上下文超限回放分类测试 (03, A10/D7/PA06).

fixture 全部为离线录制字节流, 测试不访问网络; 每条 fixture 的逐字段来源钉在
本文件注记中 (上游版本化源码/文档出处), 分类只作用于 OllamaModel 归一后的
StreamDone("error").error 文本。

fixture 逐字段出处
------------------
context-overflow-input-length.ndjson
  - `{"error": "the input length exceeds the context length"}` (无 status 字段):
    error 文本 = ollama v0.12.9 `runner/ollamarunner/runner.go:114`
    `var errorInputTooLong = errors.New("the input length exceeds the context length")`
    (llamarunner 同文 :104); runner 以 HTTP 400 纯文本回给 server
    (`http.Error(w, err.Error(), http.StatusBadRequest)`, runner.go:857), server
    ChatHandler 把 `err.Error()` 作为 NDJSON error 行转发 (`ch <- gin.H{"error":
    err.Error()}`, server/routes.go)。ollama v0.35.1 `llm/llama_server.go:2550`
    (normalizeEmbeddingError) 复用同一规范文案。错误行不带 status 字段是 v0.12.x
    非 StatusError 转发分支的形态。
context-overflow-prompt-longer.ndjson
  - `{"error": "the prompt is longer than the context length currently available "
    "to the model; shorten the prompt, adjust the context length in settings, or "
    "use a model with a longer context length", "status": 400}`:
    error 文本 = ollama v0.35.1 `llm/llama_server.go:299-302`
    (completionPromptForRequest, ContextShift 关闭且提示长于 num_ctx-1 时构造
    `api.StatusError{StatusCode: 400, ErrorMessage: ...}`); status 字段 = 同文件
    StatusError.StatusCode, ChatHandler 经 `ch <- gin.H{"error": serr.ErrorMessage,
    "status": serr.StatusCode}` (server/routes.go:783-786) 作为流式错误行写出。
context-overflow-available-context.ndjson
  - `{"error": "request (302 tokens) exceeds the available context size (256 "
    "tokens), try increasing it", "status": 400}`:
    error 文本 = llama.cpp @ b11232 (ollama v0.35.1 `LLAMA_CPP_VERSION` 钉定;
    完整 commit 6f767fe960c3b97cf37fac4626c86400561ca1e4, 2026-09-28)
    `tools/server/server-context.cpp:3218-3224`
    `string_format("request (%d tokens) exceeds the available context size (%d "
    "tokens), try increasing it", slot.task->n_tokens(), slot.n_ctx)`
    (ERROR_TYPE_EXCEED_CONTEXT_SIZE); 括号内 token 数字为服务端运行时值, fixture
    采用 ollama v0.35.1 自带测试 `llm/llama_server_test.go:2719` 记录的同一组数字;
    ollama `statusErrorMessage` (llm/llama_server.go:1888-1900) 原样透传该文案,
    经同上 ChatHandler StatusError 分支变为带 status 的流式错误行。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cicada.core.cancel import CancelToken
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelRequest, StreamDone
from cicada.plugins.ollama import classify_context_overflow
from cicada.plugins.ollama.model import OllamaModel
from cicada.plugins.ollama.protocol import OllamaConfig

CONFIG = OllamaConfig(base_url="http://ollama.test")
RECORDINGS = Path(__file__).parent / "recordings"


def recording(name: str) -> bytes:
    return (RECORDINGS / f"{name}.ndjson").read_bytes()


def _request() -> ModelRequest:
    return ModelRequest(messages=(UserMessage(text="hello"),), tools=(), system_prompt="sys")


async def _stream_once(body: bytes, status_code: int = 200) -> StreamDone:
    """回放字节流: 用 MockTransport 把 body 作为 /api/chat 响应喂给真实适配器."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )

    model = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), CONFIG)
    events = [event async for event in model.stream(_request(), CancelToken())]
    assert events and isinstance(events[-1], StreamDone)
    return events[-1]


# --- 超限 fixture 逐条: adapter 归一后的错误文本必须分类为 true (A10) ---------------------


@pytest.mark.parametrize(
    "fixture",
    [
        "context-overflow-input-length",
        "context-overflow-prompt-longer",
        "context-overflow-available-context",
    ],
)
async def test_overflow_fixture_replay_classifies_true(fixture: str):
    """流内 NDJSON error 行形态 (HTTP 200): 归一为 'ollama protocol error: ...' → true."""
    done = await _stream_once(recording(fixture))
    assert done.stop_reason == "error"
    assert done.error is not None and done.error.startswith("ollama protocol error: ")
    assert classify_context_overflow(done.error) is True


@pytest.mark.parametrize(
    "fixture",
    [
        "context-overflow-input-length",
        "context-overflow-available-context",
    ],
)
async def test_overflow_fixture_http_body_form_classifies_true(fixture: str):
    """非 2xx + JSON error 体形态: 归一为 'ollama http <code>: ...' → true.

    出处同 fixture 注记: server 在流开始前的 StatusError/校验拒绝会以
    `c.JSON(status, gin.H{"error": msg})` 返回 (同批 v0.35.1 routes.go 分支),
    适配器经 `_error_text` 提取同一 error 文本。
    """
    import json

    line = json.loads(recording(fixture).decode("utf-8"))
    body = json.dumps({"error": line["error"]}).encode("utf-8")
    done = await _stream_once(body, status_code=400)
    assert done.stop_reason == "error"
    assert done.error is not None and done.error.startswith("ollama http 400: ")
    assert classify_context_overflow(done.error) is True


# --- 非超限形态逐类: 一律 false (A10) ------------------------------------------------------


@pytest.mark.parametrize(
    "error_text",
    [
        # 网络失败 (ollama request failed: ...; httpx 连接异常归一, model.py:53/68)
        "ollama request failed: [Errno None] Cannot connect to host ollama.test:11434",
        # 超时 (httpx.TimeoutException 归一, model.py:53/68)
        "ollama request failed: The read operation timed out",
        # 其他服务端错误: HTTP 500/404 与 runner 崩溃 (model.py:91 / llama_server.go:1773)
        "ollama http 500: backend failed",
        "ollama http 404: model 'qwen3.5:9b' not found",
        "ollama protocol error: model runner has unexpectedly stopped, this may be due to "
        "resource limitations or an internal error, check ollama server logs for details",
        # 坏行/流中断 (model.py:112/119; parse_line 归一)
        "ollama protocol error: line is not valid JSON: Expecting value: line 1 column 1 (char 0)",
        "ollama stream ended without a done line",
        "ollama stream failed: peer closed connection without send",
        # P4 §8 wire 字节守卫是独立语义, 不是 token 预算 (A04; model.py:59-63)
        "ollama request too large: 65537 bytes exceeds limit 65536 bytes; not sent",
        # 预算准入拒绝文本与 wire 守卫一样不是服务端超限确认 (context_policy.py)
        "context budget exceeded: estimated 1234 tokens vs num_ctx 1000 (...)",
        # 空文本防御
        "",
    ],
)
def test_non_overflow_forms_classify_false(error_text: str):
    assert classify_context_overflow(error_text) is False


def test_classifier_rejects_non_string():
    with pytest.raises(TypeError):
        classify_context_overflow(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fixture",
    ["context-overflow-input-length", "context-overflow-prompt-longer", "context-overflow-available-context"],
)
def test_fixture_bytes_match_annotated_error_field(fixture: str):
    """fixture 防漂移: 错误字段必须与本文件注记的上游出处逐字一致."""
    import json

    expected = {
        "context-overflow-input-length": (
            "the input length exceeds the context length",
            None,
        ),
        "context-overflow-prompt-longer": (
            "the prompt is longer than the context length currently available to the "
            "model; shorten the prompt, adjust the context length in settings, or use a "
            "model with a longer context length",
            400,
        ),
        "context-overflow-available-context": (
            "request (302 tokens) exceeds the available context size (256 tokens), "
            "try increasing it",
            400,
        ),
    }[fixture]
    payload = json.loads(recording(fixture).decode("utf-8"))
    assert payload["error"] == expected[0]
    assert payload.get("status") == expected[1]
