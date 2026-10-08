"""拥有响应生命周期的 DeepSeek SSE ModelPort；零自动 HTTP 重试。"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx

from cicada.budget import BudgetConfig
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent
from cicada.models import ModelBinding, ModelOption
from cicada.plugins.deepseek.protocol import BASE_URL, MODEL_IDS, DeepSeekConfig, build_request
from cicada.plugins.deepseek.sse import Completion, ProtocolError, SSEDecoder

DEFAULT_TIMEOUT = httpx.Timeout(connect=10, read=60, write=30, pool=10)
CALL_DEADLINE_S = 300.0


async def _wait_operation(operation, cancel: CancelToken, deadline: float):
    """发送和每次读取都参与协作取消与同一个总 deadline。"""
    task = asyncio.ensure_future(operation)
    cancelled = asyncio.create_task(cancel.wait())
    delivered = False
    try:
        done, _ = await asyncio.wait({task, cancelled}, timeout=max(0, deadline - time.monotonic()),
                                     return_when=asyncio.FIRST_COMPLETED)
        if cancelled in done:
            raise asyncio.CancelledError
        if task not in done:
            raise TimeoutError
        result = task.result()
        delivered = True
        return result
    finally:
        cancelled.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, cancelled, return_exceptions=True)
        # send 与取消同一轮就绪时，响应仍归我们所有，不能丢在 task.result 中。
        if not delivered and not task.cancelled() and task.exception() is None:
            result = task.result()
            if isinstance(result, httpx.Response):
                await _close_response(result, deadline)


async def _close_response(response: httpx.Response, deadline: float) -> bool:
    """保留关闭任务直到明确结果；调用方取消不会遗弃已经取得的响应。"""
    close_task = asyncio.create_task(asyncio.wait_for(
        response.aclose(), timeout=max(0.01, min(10, deadline - time.monotonic()))))
    interrupted = False
    while True:
        try:
            await asyncio.shield(close_task)
            return interrupted
        except asyncio.CancelledError:
            if close_task.cancelled():
                raise
            interrupted = True


class DeepSeekModel:
    def __init__(self, client: httpx.AsyncClient, config: DeepSeekConfig, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise ValueError("DEEPSEEK_API_KEY is missing or invalid")
        self.client = client
        self.config = config
        self._api_key = api_key

    @property
    def options(self) -> tuple[ModelOption, ...]:
        return tuple(ModelOption(model, model) for model in MODEL_IDS)

    @property
    def initial_binding(self) -> ModelBinding:
        summary_config = replace(self.config, think=False, max_tokens=2048)
        summary = DeepSeekModel(self.client, summary_config, self._api_key)
        return ModelBinding(self, summary, self.config.identity, summary_config.identity,
                            BudgetConfig(self.config.num_ctx, self.config.max_tokens),
                            BudgetConfig(summary_config.num_ctx, summary_config.max_tokens),
                            self.config.validate_request)

    def prepare_model(self, option_id: str) -> ModelBinding:
        config = replace(self.config, model=option_id)
        return DeepSeekModel(self.client, config, self._api_key).initial_binding

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        if cancel.cancelled:
            yield StreamDone("aborted")
            return
        completion = Completion(think=self.config.think)
        response = None
        chunks = None
        terminal = StreamDone("error", "deepseek request failed")
        calls = []
        task_cancelled = False
        deadline = time.monotonic() + CALL_DEADLINE_S
        try:
            body = build_request(request, self.config)
            built = self.client.build_request("POST", BASE_URL + "/chat/completions", json=body,
                headers={"Authorization": "Bearer " + self._api_key, "Accept-Encoding": "identity"},
                timeout=DEFAULT_TIMEOUT)
            if len(built.content) > self.config.max_request_bytes:
                raise ProtocolError("request exceeds byte limit; not sent")
            response = await _wait_operation(self.client.send(built, stream=True), cancel, deadline)
            if not response.is_success:
                # 不读取错误 body；状态码本身足以分类且不会回显服务提供的秘密。
                terminal = StreamDone("error", f"deepseek HTTP {response.status_code}")
            else:
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ProtocolError("unexpected response content encoding")
                decoder = SSEDecoder()
                # httpx aiter_bytes 在 EOF 内隐式 aclose，读任务取消会打断关闭并
                # 留下 is_closed=True 的未释放响应。直接消费已请求 identity 的
                # transport stream，关闭只在下方受拥有的监督点发生。
                chunks = response.stream.__aiter__()
                while True:
                    try:
                        chunk = await _wait_operation(anext(chunks), cancel, deadline)
                    except StopAsyncIteration:
                        decoder.finish()
                        if completion.done:
                            break
                        raise ProtocolError("stream ended without DONE") from None
                    for frame in decoder.feed(chunk):
                        for event in completion.consume(frame):
                            yield event
                calls, reason, error = completion.terminal()
                terminal = StreamDone(reason, error, completion.metrics)
        except asyncio.CancelledError:
            task_cancelled = asyncio.current_task().cancelling() > 0
            terminal = StreamDone("aborted", metrics=completion.metrics)
            calls = []
        except TimeoutError:
            terminal = StreamDone("error", "deepseek call timed out", completion.metrics)
            calls = []
        except ProtocolError as exc:
            terminal = StreamDone("error", f"deepseek protocol error: {exc}", completion.metrics)
            calls = []
        except Exception:
            # 连接、URL、构造或未知异常均不回显 exc/request/Authorization。
            terminal = StreamDone("error", "deepseek request or stream failed", completion.metrics)
            calls = []
        finally:
            if response is not None:
                try:
                    if await _close_response(response, deadline):
                        task_cancelled = True
                        terminal = StreamDone("aborted", metrics=completion.metrics)
                        calls = []
                except Exception:
                    terminal = StreamDone("error", "deepseek response close failed", completion.metrics)
                    calls = []
            if chunks is not None:
                try:
                    await chunks.aclose()
                except Exception:
                    terminal = StreamDone("error", "deepseek stream close failed", completion.metrics)
                    calls = []
        if cancel.cancelled:
            terminal = StreamDone("aborted", metrics=completion.metrics)
            calls = []
        for call in calls:
            yield call
        yield terminal
        if task_cancelled:
            raise asyncio.CancelledError
