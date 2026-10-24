"""Chat SSE 的受拥有发送生命周期；有限重试只在次数额度模式下启用。"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent
from cicada.costs import CostError, CostLedger
from cicada.models import SafeModelIdentity

MAX_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 502, 503, 504})
RETRY_WAIT_LIMIT = 10.0
LOCAL_DELAYS = (0.5, 1.0)


def count_only_ledger(ledger) -> bool:
    """仅无价格、无金额上限、设置 max_calls 的账本允许受限重试。"""
    return (ledger is not None and getattr(ledger, 'profile', object()) is None
            and getattr(ledger, 'limit', object()) is None
            and getattr(ledger, 'max_calls', None) is not None)


def local_delay(attempt: int) -> float:
    return LOCAL_DELAYS[min(attempt - 1, len(LOCAL_DELAYS) - 1)]


def retry_after_delay(headers, fallback: float) -> float:
    """Retry-After 秒数或 HTTP 日期；无效/过去/无时区回退到本地间隔。"""
    value = (headers.get('retry-after') or '').strip()
    if not value:
        return fallback
    if value.isdigit():
        return max(fallback, float(int(value)))
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if when.tzinfo is None:
        return fallback
    delta = (when - datetime.now(timezone.utc)).total_seconds()
    return max(fallback, max(0.0, delta))


async def sleep_with_cancel(seconds: float, cancel: CancelToken) -> bool:
    """退避等待与取消竞争；取消返回 False。"""
    task = asyncio.ensure_future(asyncio.sleep(seconds))
    waiter = asyncio.create_task(cancel.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        return waiter not in done
    finally:
        for item in (task, waiter):
            item.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)


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
    build_body: Callable[[ModelRequest], dict], completion_factory: Callable[[], object],
    max_request_bytes: int, max_tokens: int, timeout: httpx.Timeout,
    deadline_seconds: float, label: str,
    cost_control: CostLedger | None = None,
    before_send: Callable[[ModelRequest, dict], None] | None = None,
    request_headers: dict[str, str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """同一逻辑调用的有限重试：完整 finish + DONE + EOF + 成功关闭后才交付工具。

    仅次数额度账本启用；HTTP 429/502/503/504 或连接类异常且尚未接受任何
    成功 SSE 帧时才重试。每次尝试新建 completion、先 reserve 再发送；
    失败尝试经账本受限完成结束占用，calls 不回退、金额保持未知。
    """
    from cicada.plugins.deepseek.sse import ProtocolError, SSEDecoder

    if cancel.cancelled:
        yield StreamDone("aborted")
        return
    deadline = time.monotonic() + deadline_seconds
    limit = MAX_ATTEMPTS if count_only_ledger(cost_control) else 1
    attempt = 0
    prior_failure = None
    while True:
        attempt += 1
        completion = completion_factory()
        response = None
        chunks = None
        accepted = False
        retry_reason = None
        retry_delay = 0.0
        retry_candidate = False
        terminal = StreamDone("error", f"{label} request failed")
        calls = []
        task_cancelled = False
        reservation = None
        send_failed = False
        close_failed = False

        async def send_request():
            nonlocal send_failed
            try:
                return await client.send(built, stream=True, follow_redirects=False)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                send_failed = True
                raise

        try:
            body = build_body(request)
            headers = {"Accept-Encoding": "identity", "Content-Type": "application/json", **(request_headers or {})}
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
            response = await wait_operation(send_request(), cancel, deadline)
            if not response.is_success:
                terminal = StreamDone("error", f"{label} HTTP {response.status_code}")
                if response.status_code in RETRY_STATUSES:
                    retry_reason = f"{label} HTTP {response.status_code}"
                    retry_delay = retry_after_delay(response.headers, local_delay(attempt))
            else:
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ProtocolError("unexpected response content encoding")
                decoder = SSEDecoder()
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
                        accepted = True
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
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            terminal = StreamDone("error", f"{label} connection failed", completion.metrics)
            calls = []
            if send_failed and response is None and not accepted:
                retry_reason = f"{label} connection failed"
                retry_delay = local_delay(attempt)
        except CostError as exc:
            detail = f"{prior_failure}; {exc}" if prior_failure else str(exc)
            terminal = StreamDone("error", detail, completion.metrics)
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
                    close_failed = True
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
                    close_failed = True
                    terminal = StreamDone("error", f"{label} stream close failed", completion.metrics)
                    calls = []
            retryable_failure = (limit > 1 and retry_reason is not None and not accepted
                                 and not close_failed and not task_cancelled and not cancel.cancelled)
            retry_candidate = retryable_failure and attempt < limit
            if reservation is not None:
                if retryable_failure:
                    try:
                        cost_control.finish_failed_attempt(reservation, reason=retry_reason)
                    except CostError:
                        terminal = StreamDone("error", f"{label} cost ledger failed",
                                              completion.metrics)
                        calls = []
                        retry_candidate = False
                else:
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
        if retry_candidate:
            if retry_delay <= RETRY_WAIT_LIMIT and time.monotonic() + retry_delay < deadline:
                if not await sleep_with_cancel(retry_delay, cancel):
                    yield StreamDone("aborted", metrics=completion.metrics)
                    return
                if cancel.cancelled:
                    yield StreamDone("aborted", metrics=completion.metrics)
                    return
                if time.monotonic() >= deadline:
                    yield StreamDone("error", f"{label} call timed out", completion.metrics)
                    return
                prior_failure = retry_reason
                continue
        for call in calls:
            yield call
        if attempt > 1 and terminal.error:
            terminal = StreamDone(terminal.stop_reason,
                                  f"{terminal.error} (attempt {attempt}/{limit})",
                                  terminal.metrics)
        yield terminal
        if task_cancelled:
            raise asyncio.CancelledError
        return
