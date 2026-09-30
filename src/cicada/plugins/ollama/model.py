"""OllamaModel: 冻结 ModelPort 的 Ollama /api/chat NDJSON 流式适配.

已知限制 (Part A.4): 取消延迟上界为一次行读取与取消事件的竞速裁决;
服务端已生成的 token 不撤销 (与 CancelToken "取消是请求" 语义一致).
所有失败路径 (请求构造/HTTP 非 2xx/连接失败/坏行/流中断) 均归一为 StreamDone("error"), 不裸抛.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx

from cicada.core.cancel import CancelToken
from cicada.core.messages import StopReason
from cicada.core.ports import ModelRequest, StreamDone, StreamEvent, ToolCallEvent, TextDelta
from cicada.plugins.ollama.protocol import OllamaConfig, build_request, parse_line

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)


class OllamaModel:
    """实现 ModelPort; 可脱离插件直接构造 (聚焦测试注入 MockTransport client)."""

    def __init__(self, client: httpx.AsyncClient, config: OllamaConfig) -> None:
        self.client = client
        self.config = config

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        if cancel.cancelled:
            # A.4: 进入时已取消 -> 不发请求
            yield StreamDone("aborted")
            return
        try:
            payload = build_request(request, self.config)
        except ValueError as exc:
            yield StreamDone("error", f"ollama request build failed: {exc}")
            return
        timeout = self.config.timeout if self.config.timeout is not None else DEFAULT_TIMEOUT
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        try:
            response = await self.client.send(
                self.client.build_request("POST", url, json=payload, timeout=timeout),
                stream=True,
            )
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError) as exc:
            yield StreamDone("error", f"ollama request failed: {exc}")
            return

        closed = False

        async def close() -> None:
            nonlocal closed
            if not closed:
                closed = True
                await response.aclose()

        try:
            if not response.is_success:
                try:
                    body = await response.aread()
                except httpx.HTTPError as exc:
                    yield StreamDone("error", f"ollama http {response.status_code}: {exc}")
                    return
                yield StreamDone("error", f"ollama http {response.status_code}: {_error_text(body)}")
                return

            lines = _iter_lines(response)
            cancel_wait = asyncio.create_task(cancel.wait())
            tool_calls_seen = False
            try:
                while True:
                    # 下一行读取与取消事件竞速: 取消成立即放弃读取、关闭响应
                    read_task = asyncio.create_task(anext(lines))
                    await asyncio.wait({read_task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED)
                    if cancel_wait.done():
                        read_task.cancel()
                        await asyncio.gather(read_task, return_exceptions=True)
                        await close()
                        yield StreamDone("aborted")
                        return
                    try:
                        line = read_task.result()
                    except StopAsyncIteration:
                        await close()
                        yield StreamDone("error", "ollama stream ended without a done line")
                        return
                    except (httpx.HTTPError, ValueError, TypeError) as exc:
                        await close()
                        yield StreamDone("error", f"ollama stream failed: {exc}")
                        return
                    chunk = parse_line(line)
                    if chunk.error is not None:
                        await close()
                        yield StreamDone("error", f"ollama protocol error: {chunk.error}")
                        return
                    if chunk.text:
                        yield TextDelta(chunk.text)
                    # chunk.thinking: 既定语义丢弃, 不产出事件
                    for call in chunk.tool_calls:
                        tool_calls_seen = True
                        yield ToolCallEvent(call.id, call.name, json.dumps(call.arguments))
                    if chunk.done:
                        # 有 tool_calls 即 tool_use, 不看 done_reason (实测工具轮 done_reason 为 "stop")
                        stop_reason: StopReason = (
                            "tool_use"
                            if tool_calls_seen
                            else "length"
                            if chunk.done_reason == "length"
                            else "stop"
                        )
                        await close()
                        yield StreamDone(stop_reason)
                        return
            finally:
                cancel_wait.cancel()
                await asyncio.gather(cancel_wait, return_exceptions=True)
                await lines.aclose()
        finally:
            await close()


async def _iter_lines(response: httpx.Response) -> AsyncIterator[bytes]:
    """把响应字节流切成 NDJSON 行 (剥离行尾 \\n 与 \\r); 流末残留半行也作为一行产出."""
    buffer = bytearray()
    async with aclosing(response.aiter_bytes()) as chunks:
        async for chunk in chunks:
            buffer += chunk
            while True:
                index = buffer.find(b"\n")
                if index < 0:
                    break
                line = bytes(buffer[:index])
                del buffer[: index + 1]
                yield line[:-1] if line.endswith(b"\r") else line
    if buffer:
        yield bytes(buffer)


def _error_text(body: bytes) -> str:
    """从非 2xx 响应体提取可读错误: 优先 JSON error 字段, 否则截断的原文."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="replace")[:200]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return body.decode("utf-8", errors="replace")[:200]
