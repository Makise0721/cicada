"""Chat SSE 的受拥有发送生命周期；协议映射与终止校验由调用者提供。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import httpx

from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent
from cicada.costs import CostError, CostLedger
from cicada.models import SafeModelIdentity


async def wait_operation(operation, cancel: CancelToken, deadline: float):
    """发送与读取共用 deadline；取消与 send 同时完成也保留 response owner。"""
    task = asyncio.ensure_future(operation)
    cancelled = asyncio.create_task(cancel.wait())
    delivered = False
    try:
        done, _ = await asyncio.wait(
            {task, cancelled}, timeout=max(0, deadline - time.monotonic()),
            return_when=asyncio.FIRST_COMPLETED,
        )
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
        cleanup = asyncio.gather(task, cancelled, return_exceptions=True)
        interrupted = False
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                interrupted = True
        if (not delivered or interrupted) and not task.cancelled() and task.exception() is None:
            result = task.result()
            if isinstance(result, httpx.Response):
                interrupted = await close_response(result, deadline) or interrupted
        if interrupted:
            raise asyncio.CancelledError


async def close_response(response: httpx.Response, deadline: float) -> bool:
    """不遗弃已获得响应的关闭任务；返回关闭期间是否被取消。"""
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


async def stream_chat(
    request: ModelRequest, cancel: CancelToken, *, client: httpx.AsyncClient,
    endpoint: str, api_key: str | None, identity: SafeModelIdentity,
    build_body: Callable[[ModelRequest], dict], completion,
    max_request_bytes: int, max_tokens: int, timeout: httpx.Timeout,
    deadline_seconds: float, label: str,
    cost_control: CostLedger | None = None,
    before_send: Callable[[ModelRequest, dict], None] | None = None,
) -> AsyncIterator[StreamEvent]:
    """完整 finish + DONE + EOF + 成功关闭之后才交付可执行工具。"""
    # 延迟读取已有中立 framing 实现，避免 DeepSeek 包导出初始化形成环。
    from cicada.plugins.deepseek.sse import ProtocolError, SSEDecoder

    if cancel.cancelled:
        yield StreamDone("aborted")
        return
    response = None
    chunks = None
    terminal = StreamDone("error", f"{label} request failed")
    calls = []
    task_cancelled = False
    reservation = None
    deadline = time.monotonic() + deadline_seconds
    try:
        body = build_body(request)
        headers = {"Accept-Encoding": "identity", "Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = "Bearer " + api_key
        built = client.build_request("POST", endpoint, json=body, headers=headers, timeout=timeout)
        if len(built.content) > max_request_bytes:
            raise ProtocolError("request exceeds byte limit; not sent")
        if cancel.cancelled:
            raise asyncio.CancelledError
        if cost_control is not None:
            reservation = cost_control.reserve(identity, max_tokens)
        if before_send is not None:
            before_send(request, body)
        response = await wait_operation(
            client.send(built, stream=True, follow_redirects=False), cancel, deadline)
        if not response.is_success:
            terminal = StreamDone("error", f"{label} HTTP {response.status_code}")
        else:
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ProtocolError("unexpected response content encoding")
            decoder = SSEDecoder()
            # 不使用 aiter_bytes：其隐式关闭可能在取消时失去明确的资源终态。
            chunks = response.stream.__aiter__()
            while True:
                try:
                    chunk = await wait_operation(anext(chunks), cancel, deadline)
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
        terminal = StreamDone("error", f"{label} call timed out", completion.metrics)
        calls = []
    except ProtocolError as exc:
        terminal = StreamDone("error", f"{label} protocol error: {exc}", completion.metrics)
        calls = []
    except CostError as exc:
        terminal = StreamDone("error", str(exc), completion.metrics)
        calls = []
    except Exception:
        terminal = StreamDone("error", f"{label} request or stream failed", completion.metrics)
        calls = []
    finally:
        if response is not None:
            try:
                if await close_response(response, deadline):
                    task_cancelled = True
                    terminal = StreamDone("aborted", metrics=completion.metrics)
                    calls = []
            except Exception:
                terminal = StreamDone("error", f"{label} response close failed", completion.metrics)
                calls = []
        if chunks is not None:
            close_chunks = asyncio.ensure_future(chunks.aclose())
            try:
                while True:
                    try:
                        await asyncio.shield(close_chunks)
                        break
                    except asyncio.CancelledError:
                        if close_chunks.cancelled():
                            raise
                        task_cancelled = True
                        terminal = StreamDone("aborted", metrics=completion.metrics)
                        calls = []
            except Exception:
                terminal = StreamDone("error", f"{label} stream close failed", completion.metrics)
                calls = []
        if reservation is not None:
            try:
                cost_control.finish(reservation, terminal.metrics, complete=(
                    terminal.stop_reason in ("stop", "tool_use") and terminal.error is None
                    and completion.done and not cancel.cancelled and not task_cancelled))
            except CostError as exc:
                reason = "error" if terminal.stop_reason in ("stop", "tool_use") else terminal.stop_reason
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
