"""DeepSeek Chat Completion 的本地配置、只读资格校验与请求白名单。"""

from dataclasses import dataclass

from cicada.core.messages import AssistantMessage, ToolResultMessage, UserMessage, validate_history_structure
from cicada.core.ports import ModelRequest
from cicada.models import SafeModelIdentity

BASE_URL = "https://api.deepseek.com"
MODEL_IDS = ("deepseek-flash", "deepseek-v4-pro")
REASONING_EFFORTS = ("low", "high", "max")
REQUEST_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class DeepSeekConfig:
    model: str = MODEL_IDS[0]
    think: bool = True
    reasoning_effort: str = "high"
    num_ctx: int = 131072
    max_tokens: int = 32768
    max_request_bytes: int = REQUEST_LIMIT

    def __post_init__(self) -> None:
        if self.model not in MODEL_IDS:
            raise ValueError("DeepSeek model must be deepseek-flash or deepseek-v4-pro")
        if type(self.think) is not bool or self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("invalid DeepSeek thinking configuration")
        for name, limit in (("num_ctx", 1048576), ("max_tokens", 393216),
                            ("max_request_bytes", REQUEST_LIMIT)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= limit:
                raise ValueError(f"DeepSeek {name} must be a positive integer <= {limit}")
        if self.max_tokens > self.num_ctx:
            raise ValueError("DeepSeek max_tokens must not exceed num_ctx")
        if self.num_ctx < 2048:
            raise ValueError("DeepSeek num_ctx must be >= 2048 for the summary channel")

    @property
    def identity(self) -> SafeModelIdentity:
        return SafeModelIdentity("deepseek", self.model, BASE_URL, self.think,
                                 (("num_ctx", self.num_ctx), ("max_tokens", self.max_tokens),
                                  ("reasoning_effort", self.reasoning_effort)), self.max_request_bytes)

    def validate_request(self, request: ModelRequest) -> None:
        validate_history_structure(request.messages)
        # 缓存 02 (契约 C3): 非空 request_context 说明程序状态, 需要工具调用形态
        # 才能映射尾部 user 消息; 无工具且非空本地拒绝, fake 继续支持中立请求捕获。
        if request.request_context and not request.tools:
            raise ValueError(
                "DeepSeek requires tools to carry a nonempty request context"
            )
        if not self.think or not request.tools:
            return
        for message in request.messages:
            if not isinstance(message, AssistantMessage):
                continue
            if message.stop_reason in ("error", "aborted", "length"):
                raise ValueError("DeepSeek thinking history contains a failed assistant; clear or use non-thinking")
            if message.reasoning is not None and not message.reasoning.complete:
                raise ValueError("DeepSeek thinking history contains incomplete reasoning")
            if message.tool_calls and message.reasoning is None:
                raise ValueError("DeepSeek thinking tool history is missing reasoning")


def build_request(request: ModelRequest, config: DeepSeekConfig) -> dict:
    config.validate_request(request)
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for message in request.messages:
        if isinstance(message, UserMessage):
            mapped = {"role": "user", "content": message.text}
        elif isinstance(message, ToolResultMessage):
            mapped = {"role": "tool", "content": message.result.content,
                      "tool_call_id": message.result.call_id}
        else:
            mapped = {"role": "assistant", "content": message.text or None}
            if config.think and message.reasoning is not None:
                mapped["reasoning_content"] = message.reasoning.text
            if message.tool_calls:
                mapped["tool_calls"] = [{"id": c.id, "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments_json}}
                    for c in message.tool_calls]
        messages.append(mapped)
    # 缓存 02 (契约 C3): 非空 context 在所有规范消息之后追加恰好一条带固定
    # 内容的 user 消息 (工具响应配对完成后才追加); 空值不增加消息, 不发送空 user。
    if request.request_context:
        messages.append({"role": "user", "content": request.request_context})
    body = {"model": config.model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": config.max_tokens,
            "thinking": {"type": "enabled" if config.think else "disabled"}}
    if config.think:
        body["reasoning_effort"] = config.reasoning_effort
    if request.tools:
        body["tools"] = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in request.tools]
    return body
