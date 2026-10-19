"""会话 HTML 导出: 把规范历史与运行事实渲染成一个自包含 HTML 页面。

边界 (与 diagnostic_export 的区别):
- diagnostic_export 只导元数据白名单, **不含**对话、思考、工具正文, 用于反馈
  时附给维护者; 本模块导的是**给人看的完整会话记录**, 含对话与工具正文。
- 只读 session 的公开出口 (history / config / persistence / tool_names /
  process_model_calls), 不碰私有字段、不执行工具、不重新判定交付。
- 输出单文件 HTML (内联 CSS, 无外部资源), 失败不留半成品。
"""

from __future__ import annotations

import html
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cicada.core.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)

# 单个工具输出进页面的字符上限 (超出给截断标记, 不静默丢弃)
_MAX_TOOL_BODY_CHARS = 20_000


class ExportError(ValueError):
    """导出失败 (路径、写入或会话状态); 文案直接可展示。"""


@dataclass(frozen=True)
class ExportResult:
    """导出结果: 落盘路径与是否成功唤起浏览器。"""

    path: Path
    opened: bool


def default_export_name(*, session_id: str | None, now: datetime | None = None) -> str:
    """自动命名: cicada-session-<本地时间戳>-<会话 id 前 8 位>.html

    无持久会话时用 `nosession`, 不伪造 ID。
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    short = (session_id or "nosession")[:8]
    return f"cicada-session-{stamp}-{short}.html"


def open_in_browser(path: Path) -> bool:
    """尽力打开默认浏览器; 失败不影响导出结果 (返回 False)。"""
    try:
        return bool(webbrowser.open(path.resolve().as_uri()))
    except Exception:
        return False


def export_session_html(session, *, open_browser: bool = True) -> ExportResult:
    """把会话导出为 HTML 并 (可选) 在浏览器打开。

    文件名自动生成在工作区根目录; 同名已存在时追加 -2、-3…, 绝不覆盖。
    会话必须已落定 (空闲或已关闭), 否则拒绝 — 与诊断导出口径一致。
    """
    require_settled(session)
    payload = collect(session)
    workspace = session.config.workspace.expanduser().resolve()
    destination = unique_path(workspace, payload["file_name"])
    write_html(destination, render_html(payload))
    opened = open_in_browser(destination) if open_browser else False
    return ExportResult(destination, opened)


# --- 采集 -------------------------------------------------------------------


def require_settled(session) -> None:
    if session.state not in ("idle", "closed"):
        raise ExportError("导出需要空闲会话；请等待当前请求结束")


def collect(session) -> dict:
    """从公开只读出口取数, 组装渲染用的普通字典 (纯读取, 不改状态)。"""
    config = session.config
    persistence = session.persistence
    return {
        "file_name": default_export_name(
            session_id=persistence.session_id if persistence is not None else None
        ),
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "cicada_version": _package_version(),
        "session": {
            "id": persistence.session_id if persistence is not None else None,
            "name": persistence.user_name if persistence is not None else None,
            "mode": persistence.mode if persistence is not None else None,
            "workspace": str(config.workspace.expanduser().resolve()),
            "model": config.model_label or "不可得",
            "thinking": _thinking_label(session),
            "turns": config.max_turns,
        },
        "tools": list(session.tool_names),
        "system_prompt": config.system_prompt,
        "messages": _messages(session.history, session.process_model_calls),
        "metrics": _aggregate_metrics(session.process_model_calls),
    }


def _package_version() -> str | None:
    try:
        return version("cicada")
    except PackageNotFoundError:
        return None


def _thinking_label(session) -> str:
    """思考强度: 只报已生效值; 不支持时如实说不可得。"""
    state = session.reasoning_effort
    if not state.enabled:
        return "不可得"
    return state.current or "off"


def _aggregate_metrics(calls) -> dict:
    """本次进程的累计计量; 任何字段无已知值时为 None (不可得, 不是 0)。"""
    records = list(calls)
    result: dict[str, object] = {"calls": len(records)}
    for field in ("input_tokens", "output_tokens", "cache_hit_input_tokens", "cache_miss_input_tokens"):
        values = [
            getattr(record.metrics, field)
            for record in records
            if record.metrics is not None and getattr(record.metrics, field) is not None
        ]
        result[field] = sum(values) if values else None
    result["elapsed_s"] = round(sum(record.elapsed_s for record in records), 3) if records else None
    return result


def _messages(history: tuple[Message, ...], calls) -> list[dict]:
    """规范历史 -> 渲染条目; 同轮工具结果合并进对应助手条目。

    计量按"第 N 次模型调用"与"第 N 个助手条目"顺序对齐: 会话历史不保留
    run_id, 因此不做身份配对; 数量不足或超出时不显示该轮计量 (不伪造)。
    """
    call_list = list(calls)
    blocks: list[dict] = []
    assistant_seen = 0
    for message in history:
        if isinstance(message, UserMessage):
            blocks.append({"kind": "user", "text": message.text})
        elif isinstance(message, AssistantMessage):
            entry: dict = {
                "kind": "assistant",
                "text": message.text,
                "reasoning": message.reasoning.text if message.reasoning is not None else None,
                "stop_reason": message.stop_reason,
                "error": message.error,
                "tools": [],
                "metrics": None,
            }
            if assistant_seen < len(call_list):
                call = call_list[assistant_seen]
                entry["metrics"] = {
                    "input_tokens": call.metrics.input_tokens if call.metrics else None,
                    "output_tokens": call.metrics.output_tokens if call.metrics else None,
                    "elapsed_s": call.elapsed_s,
                }
            assistant_seen += 1
            blocks.append(entry)
        elif isinstance(message, ToolResultMessage):
            result = message.result
            tool = {
                "name": result.name,
                "content": _clip(result.content),
                "is_error": result.is_error,
            }
            # 归属最近的助手条目 (内核保证工具结果紧随其调用组)
            owner = next(
                (block for block in reversed(blocks) if block["kind"] == "assistant"), None
            )
            if owner is not None and not _is_answered(owner, blocks):
                owner["tools"].append(tool)
            else:
                blocks.append({"kind": "orphan_tool", **tool})
        else:  # 未知类型: 如实标注, 不猜
            blocks.append({"kind": "unknown", "text": repr(message)[:500]})
    return blocks


def _is_answered(owner: dict, blocks: list[dict]) -> bool:
    """该助手条目之后是否已经出现过用户消息 (即不再是最后一个助手轮次)。"""
    index = blocks.index(owner)
    return any(block["kind"] == "user" for block in blocks[index + 1:])


def _clip(text: str) -> str:
    if len(text) <= _MAX_TOOL_BODY_CHARS:
        return text
    return text[:_MAX_TOOL_BODY_CHARS] + f"\n…[已截断: 保留 {_MAX_TOOL_BODY_CHARS}/{len(text)} 字符]"


# --- 渲染 -------------------------------------------------------------------


_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 64px;
  background: #0f1116; color: #d4d4d4;
  font: 14px/1.65 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 6px; color: #b294bb; font-weight: 600; }
.sub { color: #808080; font-size: 12px; margin-bottom: 28px; }
.card {
  background: #16181f; border: 1px solid #24262f; border-radius: 8px;
  padding: 16px 18px; margin-bottom: 14px;
}
.card > .label {
  color: #808080; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; margin-bottom: 10px;
}
.meta { display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 13px; }
.meta b { color: #81a2be; font-weight: 500; }
pre {
  background: #0d0f14; border: 1px solid #22242d; border-radius: 6px;
  padding: 12px 14px; margin: 8px 0 0; overflow-x: auto; white-space: pre-wrap;
  word-break: break-word;
  font: 12.5px/1.6 "Cascadia Mono", Consolas, monospace; color: #a8a8a8;
}
.msg { margin-bottom: 16px; }
.who {
  display: inline-block; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; padding: 2px 8px; border-radius: 4px;
  margin-bottom: 8px; font-weight: 600;
}
.who.user { background: #1e2733; color: #81a2be; }
.who.assistant { background: #232a20; color: #b5bd68; }
.body { word-break: break-word; }
.body p { margin: 8px 0; white-space: pre-wrap; }
.body h3, .body h4, .body h5, .body h6 {
  margin: 16px 0 8px; font-size: 14px; color: #f0c674; font-weight: 600;
}
.body ul { margin: 8px 0; padding-left: 22px; }
.body li { margin: 3px 0; }
.body strong { color: #e8e8e8; font-weight: 600; }
.body code {
  background: #1d2029; border-radius: 3px; padding: 1px 5px;
  font: 12.5px "Cascadia Mono", Consolas, monospace; color: #8abeb7;
}
.tool { border-left: 2px solid #2d3139; padding-left: 12px; margin-top: 10px; }
.tool.err { border-left-color: #cc6666; }
.tool > .name { font-size: 12px; color: #8abeb7; margin-bottom: 4px; }
.tool > pre { max-height: 420px; overflow-y: auto; }
.reasoning { color: #6f6f6f; font-style: italic; white-space: pre-wrap; margin-bottom: 10px; }
.note { color: #cc6666; font-size: 12.5px; margin-top: 8px; }
.dim { color: #6a6a6a; font-size: 12px; margin-top: 6px; }
.empty { color: #6a6a6a; font-style: italic; }
footer { color: #55575f; font-size: 11.5px; margin-top: 36px; line-height: 1.8; }
"""


def render_html(payload: dict) -> str:
    """把采集结果渲染为完整 HTML 文档 (纯函数, 便于单测)。"""
    session = payload["session"]
    title = f"Cicada Session {session['id']}" if session["id"] else "Cicada Session"
    parts = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">",
        f"<h1>{_e(title)}</h1>",
        f'<div class="sub">生成于 {_e(payload["generated_at"])} · Cicada '
        f'{_e(payload["cicada_version"] or "版本不可得")}</div>',
        _session_card(session),
        _tools_card(payload["tools"]),
        _system_card(payload["system_prompt"]),
        _metrics_card(payload["metrics"]),
        '<div class="card"><div class="label">会话记录</div>',
    ]
    if not payload["messages"]:
        parts.append('<div class="empty">本次会话还没有消息。</div>')
    for block in payload["messages"]:
        parts.append(_message_html(block))
    parts.append("</div>")
    parts.append(
        "<footer>本页由 Cicada 本地导出，内容取自会话规范历史与运行计量。"
        "工具输出超过 20000 字符会截断。导出不重新检查工作区，"
        "不构成当前可交付证明。</footer>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def _session_card(session: dict) -> str:
    rows = [
        ("会话 ID", session["id"] or "未启用保存"),
        ("名称", session["name"] or "未命名"),
        ("模式", session["mode"] or "不落盘"),
        ("模型", session["model"]),
        ("思考强度", session["thinking"]),
        ("轮次上限", str(session["turns"])),
        ("工作区", session["workspace"]),
    ]
    body = "".join(f"<span><b>{_e(label)}</b> {_e(value)}</span>" for label, value in rows)
    return f'<div class="card"><div class="label">会话</div><div class="meta">{body}</div></div>'


def _tools_card(tools: list[str]) -> str:
    if not tools:
        return ""
    body = " ".join(_e(name) for name in tools)
    return (
        f'<div class="card"><div class="label">可用工具 ({len(tools)})</div>'
        f'<div class="body">{body}</div></div>'
    )


def _system_card(text: str) -> str:
    if not text:
        return ""
    return f'<div class="card"><div class="label">系统提示</div><pre>{_e(text)}</pre></div>'


def _metrics_card(metrics: dict) -> str:
    def show(value, suffix: str = "") -> str:
        return "不可得" if value is None else f"{value}{suffix}"

    rows = [
        ("调用次数", show(metrics["calls"])),
        ("输入 tokens", show(metrics["input_tokens"])),
        ("输出 tokens", show(metrics["output_tokens"])),
        ("缓存命中", show(metrics["cache_hit_input_tokens"])),
        ("缓存未命中", show(metrics["cache_miss_input_tokens"])),
        ("累计时长", show(metrics["elapsed_s"], " 秒")),
    ]
    body = "".join(f"<span><b>{_e(k)}</b> {_e(v)}</span>" for k, v in rows)
    return f'<div class="card"><div class="label">本次进程计量</div><div class="meta">{body}</div></div>'


def _message_html(block: dict) -> str:
    kind = block["kind"]
    if kind == "user":
        return (
            '<div class="msg"><div class="who user">user</div>'
            f'<div class="body">{_rich(block["text"])}</div></div>'
        )
    if kind == "assistant":
        out = ['<div class="msg"><div class="who assistant">assistant</div>']
        if block.get("reasoning"):
            out.append(f'<div class="reasoning">{_e(block["reasoning"])}</div>')
        if block["text"]:
            out.append(f'<div class="body">{_rich(block["text"])}</div>')
        elif not block["tools"]:
            out.append('<div class="empty">（本轮无正文）</div>')
        for tool in block["tools"]:
            out.append(_tool_html(tool))
        if block.get("error"):
            out.append(f'<div class="note">错误: {_e(block["error"])}</div>')
        if block.get("stop_reason") and block["stop_reason"] != "stop":
            out.append(f'<div class="note">停止原因: {_e(block["stop_reason"])}</div>')
        bits = _metrics_bits(block.get("metrics"))
        if bits:
            out.append(f'<div class="dim">{" · ".join(bits)}</div>')
        out.append("</div>")
        return "".join(out)
    if kind == "orphan_tool":
        return f'<div class="msg">{_tool_html(block)}</div>'
    return (
        '<div class="msg"><div class="who">unknown</div>'
        f'<div class="body">{_e(block["text"])}</div></div>'
    )


def _metrics_bits(metrics: dict | None) -> list[str]:
    if not metrics:
        return []
    bits: list[str] = []
    if metrics["input_tokens"] is not None:
        bits.append(f"输入 {metrics['input_tokens']}")
    if metrics["output_tokens"] is not None:
        bits.append(f"输出 {metrics['output_tokens']}")
    if metrics["elapsed_s"] is not None:
        bits.append(f"{metrics['elapsed_s']:.2f}s")
    return bits


def _tool_html(tool: dict) -> str:
    cls = "tool err" if tool["is_error"] else "tool"
    marker = " · 失败" if tool["is_error"] else ""
    return (
        f'<div class="{cls}"><div class="name">{_e(tool["name"])}{marker}</div>'
        f'<pre>{_e(tool["content"])}</pre></div>'
    )


_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_INLINE_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")


def _rich(text: str) -> str:
    """正文: 块级只认标题与无序列表, 行内只认 `代码` 与 **粗体**。

    不做完整 Markdown: 表格/图片/链接等按原文显示, 宁可朴素也不误渲染。
    所有内容先转义再套标签, 模型输出里的 HTML 不会生效。
    连续普通行合并为一个段落 (段内保留原始换行), 避免与块级元素叠加出双倍空行。
    """
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append("<p>" + "<br>".join(_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()

    def flush_list() -> None:
        if items:
            blocks.append("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>")
            items.clear()

    for line in text.split("\n"):
        heading = _HEADING.match(line)
        bullet = _BULLET.match(line)
        if heading is not None:
            flush_paragraph()
            flush_list()
            level = min(len(heading.group(1)) + 2, 6)  # # -> h3, 不与 h1 冲突
            blocks.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif bullet is not None:
            flush_paragraph()
            items.append(bullet.group(1))
        elif not line.strip():
            flush_paragraph()
            flush_list()
        else:
            flush_list()
            paragraph.append(line)
    flush_paragraph()
    flush_list()
    return "".join(blocks)


def _inline(text: str) -> str:
    """单行行内标记; 先转义再加标签。"""
    escaped = _e(text)
    escaped = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", escaped)
    return _INLINE_BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", escaped)


def _e(text: object) -> str:
    return html.escape(str(text), quote=False)


# --- 落盘 -------------------------------------------------------------------


def unique_path(directory: Path, name: str) -> Path:
    """同名已存在则追加 -2、-3…; 绝不覆盖已有文件。"""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(2, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ExportError("导出文件名冲突过多；请清理旧导出后重试")


def write_html(path: Path, content: str) -> None:
    """独占创建; 写入失败清除本次新建的不完整文件。"""
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            created = True
            handle.write(content)
    except (OSError, ValueError):
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise ExportError("导出写入失败；请检查目标目录与权限") from None


__all__ = [
    "ExportError",
    "ExportResult",
    "collect",
    "default_export_name",
    "export_session_html",
    "open_in_browser",
    "render_html",
    "unique_path",
    "write_html",
]
