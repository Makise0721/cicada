"""拥有响应生命周期的 DeepSeek SSE ModelPort；零自动 HTTP 重试。"""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import httpx

from cicada.budget import BudgetConfig
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent
from cicada.costs import CostError, CostLedger
from cicada.models import (
    CacheRequestObservation, ConfigurationRequest, ModelBinding, ModelOption, SafeModelIdentity,
)
from cicada.plugins.deepseek.protocol import (
    BASE_URL, MODEL_IDS, REASONING_EFFORTS, REQUEST_LIMIT, DeepSeekConfig, build_request, request_body_bytes,
)
from cicada.plugins.deepseek.sse import Completion, ProtocolError, SSEDecoder

DEFAULT_TIMEOUT = httpx.Timeout(connect=10, read=60, write=30, pool=10)
CALL_DEADLINE_S = 300.0


@dataclass(frozen=True)
class _RequestShape:
    """上次实际发送的不可逆形状摘要 (契约 C7); 不保存正文副本或历史。

    identity 变化在本实现中表现为新普通路由实例的首调用, 不推测服务端身份。
    """

    identity: SafeModelIdentity
    system: str | None
    tools: str | None
    messages: tuple[str, ...]  # 核心映射消息摘要, 不含 leading system 与派生尾部
    tail: str | None


def _digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_shape(request: ModelRequest, identity: SafeModelIdentity, body: dict) -> _RequestShape:
    """最终 provider 映射的形状摘要; 只覆盖实际发送内容 (契约 C7)."""
    messages = body["messages"]
    start = 1 if messages and messages[0].get("role") == "system" else 0
    end = len(messages) - 1 if request.request_context else len(messages)
    return _RequestShape(
        identity=identity,
        system=_digest(messages[0]) if start else None,
        tools=_digest(body["tools"]) if "tools" in body else None,
        messages=tuple(_digest(message) for message in messages[start:end]),
        tail=_digest(messages[-1]) if request.request_context else None,
    )


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
    def __init__(self, client: httpx.AsyncClient, config: DeepSeekConfig, api_key: str, *,
                 cost_control: CostLedger | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise ValueError("DEEPSEEK_API_KEY is missing or invalid")
        self.client = client
        self.config = config
        self._api_key = api_key
        self._cost_control = cost_control
        # 缓存 03 (契约 C7): 每实例一份比较基线与最新观察; 换模型/强度/配置
        # 准备出的新实例从空开始, 不持久、不跨进程恢复。
        self._observation: CacheRequestObservation | None = None
        self._send_sequence = 0
        self._shape_baseline: _RequestShape | None = None
        from .tokenizer import FlashTokenCounter, ProTokenCounter
        counter_type = FlashTokenCounter if config.model == "deepseek-flash" else ProTokenCounter
        self._token_counter = counter_type(config)

    @property
    def cache_observation(self) -> CacheRequestObservation | None:
        """最近一次实际发送相对前次的本地形状观察; 未观察或诊断失败为 None."""
        return self._observation

    @property
    def options(self) -> tuple[ModelOption, ...]:
        return tuple(ModelOption(model, model) for model in MODEL_IDS)

    def set_cost_control(self, ledger: CostLedger) -> None:
        """装配时绑定本进程唯一账本，后继配置不能换额度。"""
        if self._cost_control is not None and self._cost_control is not ledger:
            raise CostError('model already has a different cost allowance')
        self._cost_control = ledger

    @property
    def initial_binding(self) -> ModelBinding:
        summary_config = replace(self.config, think=False, max_tokens=2048)
        summary = DeepSeekModel(self.client, summary_config, self._api_key, cost_control=self._cost_control)
        return ModelBinding(self, summary, self.config.identity, summary_config.identity,
                            BudgetConfig(self.config.num_ctx, self.config.max_tokens),
                            BudgetConfig(summary_config.num_ctx, summary_config.max_tokens),
                            self.config.validate_request, self._token_counter, summary._token_counter)

    def prepare_model(self, option_id: str) -> ModelBinding:
        config = replace(self.config, model=option_id)
        return DeepSeekModel(self.client, config, self._api_key, cost_control=self._cost_control).initial_binding

    @property
    def reasoning_efforts(self) -> tuple[str, ...]:
        return REASONING_EFFORTS

    def prepare_reasoning_effort(self, effort: str) -> ModelBinding:
        """思考强度准备 (03, C5): 纯本地, 复用 client/凭据与普通参数.

        普通模型显式 think=true 并设置档位; initial_binding 的摘要通道保持
        think=false (摘要思考禁用)。不合法档位按 ValueError 拒绝, 零网络。
        """
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"unknown reasoning effort: {effort!r}")
        config = replace(self.config, think=True, reasoning_effort=effort)
        return DeepSeekModel(self.client, config, self._api_key, cost_control=self._cost_control).initial_binding

    def prepare_configuration(self, request: ConfigurationRequest) -> ModelBinding:
        """登录应用准备: 复用既有 client, 零网络/文件 I/O, 旧对象保持不可变。"""
        identity = request.identity
        if identity.provider != "deepseek" or type(identity.think) is not bool:
            raise ValueError("DeepSeek configuration requires a deepseek identity with explicit think")
        if identity.base_url not in (None, BASE_URL):
            raise ValueError("DeepSeek configuration requires the official endpoint")
        options = dict(identity.options)
        config = DeepSeekConfig(
            model=identity.model, think=identity.think,
            reasoning_effort=options.get("reasoning_effort", "high"),
            num_ctx=options.get("num_ctx", 131072),
            max_tokens=options.get("max_tokens", 32768),
            max_request_bytes=identity.max_request_bytes or REQUEST_LIMIT,
        )
        key = self._api_key if request.credential_action == "keep" else request.credential.reveal()
        return DeepSeekModel(self.client, config, key, cost_control=self._cost_control).initial_binding

    def _record_observation(self, request: ModelRequest, body: dict) -> None:
        """缓存 03 (契约 C7): client.send 实际启动的同步接缝调用一次。

        sequence 按本实例实际发送尝试递增; 本地资格拒绝、字节守卫与取消前零
        send 不会到达本方法, 发送后失败仍保留观察。诊断异常不改变发送或终结:
        显示不可得并清空比较基线, 下次可观察发送为 first_request, 不假称前次。
        """
        self._send_sequence += 1
        try:
            shape = _request_shape(request, self.config.identity, body)
            baseline = self._shape_baseline
            if baseline is None or baseline.identity != shape.identity:
                observation = CacheRequestObservation(
                    self._send_sequence, ("first_request",), None)
            else:
                changes: list[str] = []
                if shape.system != baseline.system:
                    changes.append("system_changed")
                if shape.tools != baseline.tools:
                    changes.append("tools_changed")
                if shape.messages != baseline.messages:
                    prefix = len(baseline.messages)
                    if (len(shape.messages) > prefix
                            and shape.messages[:prefix] == baseline.messages):
                        changes.append("messages_extended")
                    else:
                        changes.append("messages_replaced")
                if shape.tail != baseline.tail:
                    changes.append("request_context_changed")
                if not changes:
                    changes = ["unchanged"]
                preserved = 0
                for sent, previous in zip(shape.messages, baseline.messages):
                    if sent != previous:
                        break
                    preserved += 1
                observation = CacheRequestObservation(
                    self._send_sequence, tuple(changes), preserved)
            self._shape_baseline = shape
            self._observation = observation
        except Exception:
            self._shape_baseline = None
            self._observation = None

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
        reservation = None
        deadline = time.monotonic() + CALL_DEADLINE_S
        try:
            body = build_request(request, self.config)
            built = self.client.build_request("POST", BASE_URL + "/chat/completions", content=request_body_bytes(body),
                headers={"Authorization": "Bearer " + self._api_key, "Accept-Encoding": "identity",
                         "Content-Type": "application/json"},
                timeout=DEFAULT_TIMEOUT)
            if len(built.content) > self.config.max_request_bytes:
                raise ProtocolError("request exceeds byte limit; not sent")
            if cancel.cancelled:
                raise asyncio.CancelledError
            if self._cost_control is not None:
                reservation = self._cost_control.reserve(self.config.model, self.config.max_tokens)
            # 缓存 03 (契约 C7): 观察只在通过本地验证与字节守卫且 send 实际启动
            # 的同步接缝更新一次, 不能提前到 build_request; 发送后失败仍保留。
            self._record_observation(request, body)
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
        except CostError as exc:
            terminal = StreamDone('error', str(exc), completion.metrics)
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
            if reservation is not None:
                try:
                    self._cost_control.finish(reservation, terminal.metrics, complete=(
                        terminal.stop_reason in ('stop', 'tool_use') and terminal.error is None
                        and completion.done and not cancel.cancelled and not task_cancelled))
                except CostError as exc:
                    reason = 'error' if terminal.stop_reason in ('stop', 'tool_use') else terminal.stop_reason
                    terminal = StreamDone(reason, terminal.error or str(exc), completion.metrics)
                    calls = []
        if cancel.cancelled:
            terminal = StreamDone("aborted", metrics=completion.metrics)
            calls = []
        for call in calls:
            yield call
        yield terminal
        if task_cancelled:
            raise asyncio.CancelledError
