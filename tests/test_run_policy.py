"""P4 05 应用层投影策略测试: 共享 VerificationService fake + 真实 ModelPort/MockTransport."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from cicada.boot import bootstrap
from cicada.core.cancel import CancelToken
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelRequest, StreamDone, TextDelta, ToolSpec
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    CheckReceipt,
    CheckState,
    VerificationPlan,
    VerificationView,
)
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.fake_tools import EchoTool, fake_tools_plugin
from cicada.plugins.ollama.model import OllamaModel
from cicada.plugins.ollama.protocol import OllamaConfig
from cicada.run_policy import (
    CONTROL_SECTION_MAX_BYTES,
    RunPolicy,
    RunPolicyConfig,
    verification_policy,
)
from cicada.runtime.plugin import PluginDefinition

REF = "sha256:snapshot-ref"
SCOPE = "scope-1"
RUN_ID = "verify-run-1"


class FakeVerificationService:
    """只实现共享 VerificationService Protocol; 不引入第二套生产契约."""

    def __init__(self, view: VerificationView) -> None:
        self._view = view
        self.plan = VerificationPlan(
            verification_run_id=view.verification_run_id,
            root=Path("C:/ws"),
            checks=(CheckDefinition("check-1", "python -m pytest -q"),),
        )
        self.initialize_calls: list[CancelToken] = []
        self.run_check_calls: list[str] = []
        self.refresh_calls: list[CancelToken] = []
        self.uncertain_reasons: list[str] = []
        self.view = view

    async def initialize(self, cancel: CancelToken):
        self.initialize_calls.append(cancel)

    async def run_check(self, check_id: str, cancel: CancelToken) -> CheckReceipt:
        self.run_check_calls.append(check_id)
        raise AssertionError("投影不执行检查")

    async def refresh(self, cancel: CancelToken) -> VerificationView:
        self.refresh_calls.append(cancel)
        return self.view

    def mark_process_uncertain(self, reason: str) -> None:
        self.uncertain_reasons.append(reason)
        self.view = replace(
            self.view,
            process_uncertain=True,
            blocking_reasons=self.view.blocking_reasons + (reason,),
        )

    async def finalize(self, cancel: CancelToken):
        raise AssertionError("投影不生成交付证据")


def make_receipt(
    receipt_id: str,
    check_id: str = "check-1",
    status: str = "passed",
    failure_kind: str | None = None,
) -> CheckReceipt:
    return CheckReceipt(
        verification_run_id=RUN_ID,
        receipt_id=receipt_id,
        check_id=check_id,
        command="python -m pytest -q",
        cwd=Path("C:/ws"),
        timeout_s=120.0,
        elapsed_s=1.5,
        execution_status="exited",
        exit_code=0 if status == "passed" else 1,
        timed_out=False,
        cancelled=False,
        output_complete=True,
        snapshot_before=REF,
        snapshot_after=REF,
        verification_status=status,
        failure_kind=failure_kind,
        output_truncated=False,
        artifact_truncated=False,
        output_artifact_path=None,
        output_artifact_sha256=None,
    )


def make_view(
    states: tuple[CheckState, ...] = (),
    receipts: tuple[CheckReceipt, ...] = (),
    snapshot_ref: str | None = REF,
    process_uncertain: bool = False,
    blocking_reasons: tuple[str, ...] = (),
) -> VerificationView:
    return VerificationView(
        verification_run_id=RUN_ID,
        snapshot_ref=snapshot_ref,
        scope_id=SCOPE if snapshot_ref is not None else None,
        checks=states,
        receipts=receipts,
        process_uncertain=process_uncertain,
        blocking_reasons=blocking_reasons,
    )


def check_result(
    call_id: str,
    content: str,
    receipt_id: str | None = "r-1",
    is_error: bool = False,
    extra: dict | None = None,
) -> ToolResultMessage:
    details: dict = {} if extra is None else dict(extra)
    if receipt_id is not None:
        details["receipt_id"] = receipt_id
    return ToolResultMessage(
        result=ToolResult(
            call_id=call_id,
            name="check",
            content=content,
            is_error=is_error,
            details=details,
        )
    )


class RecordingModel:
    """记录投影请求的最小 ModelPort; 不做别的判断."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest, cancel: CancelToken):
        self.requests.append(request)
        yield TextDelta("ok")
        yield StreamDone("stop")


async def projected_request(service: FakeVerificationService, messages):
    model = RecordingModel()
    policy = RunPolicy(model, service)
    events = [event async for event in policy.stream(ModelRequest(tuple(messages), ()), CancelToken())]
    assert isinstance(events[-1], StreamDone)
    return model.requests[0], events


def check_messages(request: ModelRequest) -> list[ToolResultMessage]:
    return [m for m in request.messages if isinstance(m, ToolResultMessage)]


class Qwen:
    """跑一遍完整策略 + 真实 OllamaModel 适配路径, 返回 wire 请求体 (未发送则 None)."""

    def __init__(self, service: FakeVerificationService, limit: int | None = None) -> None:
        self.service = service
        config = OllamaConfig(base_url="http://ollama.test", max_request_bytes=limit)
        self.seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            body = b'{"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}\n'
            return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

        self.model = OllamaModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), config)
        self.policy = RunPolicy(self.model, service)

    async def run(self, messages, tools=(), system_prompt=""):
        request = ModelRequest(tuple(messages), tools, system_prompt)
        events = [event async for event in self.policy.stream(request, CancelToken())]
        await self.model.client.aclose()
        return events

    @property
    def wire(self) -> dict:
        assert len(self.seen) == 1, f"expected exactly one POST, got {len(self.seen)}"
        return json.loads(self.seen[0].content)


# --- 系统状态段: current / stale / unknown -------------------------------------


async def test_system_section_reports_snapshot_and_check_state_within_budget():
    receipt = make_receipt("r-1")
    view = make_view(
        states=(
            CheckState("check-1", receipt=receipt, freshness="current"),
            CheckState("check-2", receipt=make_receipt("r-2", "check-2", "failed", "nonzero_exit"), freshness="stale"),
            CheckState("check-3"),
        ),
        receipts=(receipt,),
        blocking_reasons=("check-2 latest attempt failed",),
    )
    model = RecordingModel()
    policy = RunPolicy(model, FakeVerificationService(view))
    events = [
        event
        async for event in policy.stream(
            ModelRequest((UserMessage(text="go"),), (), system_prompt="rules"), CancelToken()
        )
    ]
    request = model.requests[0]

    assert CONTROL_SECTION_MAX_BYTES == 4096  # spec §6 控制段上限
    section = request.system_prompt
    assert section.startswith("rules\n<cicada-verification-state>")
    assert f"snapshot_ref: {REF}" in section
    assert f"scope_id: {SCOPE}" in section
    assert "check check-1: status=passed freshness=current" in section
    assert "check check-2: status=failed freshness=stale" in section
    assert "check check-3: status=not_run freshness=unknown" in section
    assert "blocking_reasons:" in section
    assert "- check-2 latest attempt failed" in section
    assert len(section.encode("utf-8")) <= CONTROL_SECTION_MAX_BYTES
    assert events[-1] == StreamDone("stop")


async def test_system_section_is_bounded_for_oversized_identifiers_and_reasons():
    # 默认界下靠字段膨胀超过 4096 需要极端输入; 这里用同一路径的收紧界证明截断与关键字段优先
    config = RunPolicyConfig(control_section_max_bytes=512, identifier_max_bytes=32, reason_max_bytes=24)
    receipt = make_receipt("r-" + "x" * 4000, "check-" + "y" * 2000)
    view = make_view(
        states=(CheckState("check-" + "y" * 2000, receipt=receipt, freshness="current"),),
        receipts=(receipt,),
        blocking_reasons=tuple("reason " + "z" * 900 for _ in range(20)),
    )
    model = RecordingModel()
    policy = RunPolicy(model, FakeVerificationService(view), config)
    async for _ in policy.stream(ModelRequest((UserMessage(text="go"),), ()), CancelToken()):
        pass
    section = model.requests[0].system_prompt

    assert len(section.encode("utf-8")) <= 512
    assert "truncated at byte limit" in section
    assert section.startswith("<cicada-verification-state>")
    # 关键字段 (snapshot 身份与 check 状态) 在截断前仍可见
    assert f"snapshot_ref: {REF}" in section
    assert "status=passed freshness=current" in section


async def test_tiny_control_budget_still_stays_within_it():
    receipt = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=receipt, freshness="current"),), receipts=(receipt,))
    model = RecordingModel()
    policy = RunPolicy(model, FakeVerificationService(view), RunPolicyConfig(control_section_max_bytes=16))
    async for _ in policy.stream(ModelRequest((UserMessage(text="go"),), ()), CancelToken()):
        pass
    assert len(model.requests[0].system_prompt.encode("utf-8")) <= 16


async def test_multibyte_fields_never_break_the_byte_budget_or_utf8():
    receipt = make_receipt("r-1", "check-" + "中" * 200, "failed", "非零退出" * 40)
    view = make_view(
        states=(CheckState("check-" + "中" * 200, receipt=receipt, freshness="stale"),),
        receipts=(receipt,),
        blocking_reasons=tuple("原因" * 300 for _ in range(4)),
    )
    for budget in (64, 129, 512, 4096):
        model = RecordingModel()
        policy = RunPolicy(
            model,
            FakeVerificationService(view),
            RunPolicyConfig(control_section_max_bytes=budget),
        )
        async for _ in policy.stream(ModelRequest((UserMessage(text="go"),), ()), CancelToken()):
            pass
        section = model.requests[0].system_prompt
        assert len(section.encode("utf-8")) <= budget
        section.encode("utf-8").decode("utf-8")  # 不产生半个字符


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4096"])
def test_config_rejects_non_positive_budgets(value):
    with pytest.raises(ValueError, match="control_section_max_bytes"):
        RunPolicyConfig(control_section_max_bytes=value)


async def test_unavailable_snapshot_reports_unknown_instead_of_claiming_current():
    view = make_view(snapshot_ref=None)
    request, _ = await projected_request(FakeVerificationService(view), (UserMessage(text="go"),))
    assert "snapshot: unavailable" in request.system_prompt
    assert "freshness: unknown" in request.system_prompt
    assert REF not in request.system_prompt


async def test_empty_original_prompt_yields_exactly_the_control_section():
    request, _ = await projected_request(FakeVerificationService(make_view()), (UserMessage(text="go"),))
    assert request.system_prompt.startswith("<cicada-verification-state>")
    assert request.system_prompt.count("<cicada-verification-state>") == 1


# --- 历史投影: receipt_id 账本关联 ---------------------------------------------


async def test_history_check_text_is_annotated_and_original_bytes_kept():
    fresh = make_receipt("r-current")
    old = make_receipt("r-old")
    view = make_view(
        states=(
            CheckState("check-1", receipt=fresh, freshness="current"),
            CheckState("check-2", receipt=make_receipt("r-2", "check-2"), freshness="stale"),
        ),
        receipts=(fresh, old, make_receipt("r-2", "check-2")),
    )
    original = "PASS: all checks succeeded"
    messages = (
        UserMessage(text="task"),
        AssistantMessage(text="", tool_calls=(ToolCall("c1", "check", '{"action":"run"}'),)),
        check_result("c1", original, receipt_id="r-current"),
        AssistantMessage(text="", tool_calls=(ToolCall("c2", "check", '{"action":"run"}'),)),
        check_result("c2", "PASS: old run", receipt_id="r-old"),
        check_result("c3", "PASS: wrong check", receipt_id="r-ghost"),
        check_result("c4", "PASS: no receipt", receipt_id=None),
    )
    request, _ = await projected_request(FakeVerificationService(view), messages)

    bodies = [m.result for m in check_messages(request)]
    assert [r.call_id for r in bodies] == ["c1", "c2", "c3", "c4"]
    assert [r.name for r in bodies] == ["check"] * 4
    assert bodies[0].content == original + "\nCicada freshness: stale=false freshness=current tool=check"
    assert bodies[1].content.startswith("PASS: old run\nCicada freshness: stale=true freshness=stale tool=check")
    assert bodies[1].details["receipt_id"] == "r-old"
    assert "superseded" in bodies[1].content
    assert "freshness=unknown" in bodies[2].content
    assert "not in the verification ledger" in bodies[2].content
    assert "freshness=unknown" in bodies[3].content
    assert "no receipt_id" in bodies[3].content
    # 每个历史 PASS 都紧邻 freshness 标记, 不依赖模型回忆
    assert all("Cicada freshness" in r.content for r in bodies)
    # 只有 check/powershell 结果被标注
    assert [m.result.name for m in request.messages if isinstance(m, ToolResultMessage)] == ["check"] * 4


async def test_duplicate_model_call_ids_do_not_confuse_ledger_and_do_not_leak_receipt_of_other_attempt():
    first = make_receipt("r-1")
    second = make_receipt("r-2")
    view = make_view(
        states=(CheckState("check-1", receipt=second, freshness="current"),),
        receipts=(first, second),
    )
    same_id = "call_dup"
    messages = (
        check_result(same_id, "old attempt output", receipt_id="r-1"),
        check_result(same_id, "new attempt output", receipt_id="r-2"),
    )
    request, _ = await projected_request(FakeVerificationService(view), messages)

    bodies = [m.result for m in check_messages(request)]
    assert [r.call_id for r in bodies] == [same_id, same_id]
    assert "stale=true freshness=stale" in bodies[0].content
    assert "freshness=current stale=false" not in bodies[0].content
    assert "freshness=current" in bodies[1].content
    assert "stale=true" not in bodies[1].content


async def test_original_message_objects_are_not_rewritten():
    first = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=first, freshness="current"),), receipts=(first,))
    message = check_result("c1", "original content", receipt_id="r-1")
    request = ModelRequest((message,), (), "sys")
    policy = RunPolicy(RecordingModel(), FakeVerificationService(view))
    async for _ in policy.stream(request, CancelToken()):
        pass
    assert message.result.content == "original content"
    assert message.result.details == {"receipt_id": "r-1"}
    assert request.system_prompt == "sys"


async def test_missing_structured_receipt_is_unknown_and_latched_conservatively():
    view = make_view()
    detail_less = ToolResultMessage(
        result=ToolResult("c1", "check", "check failed", is_error=True, details=None)
    )
    request, _ = await projected_request(FakeVerificationService(view), (detail_less,))
    body = check_messages(request)[0].result
    assert "freshness=unknown" in body.content
    assert "no structured receipt" in body.content


async def test_powershell_history_is_annotated_as_unknown_and_non_check_results_untouched():
    other = ToolResultMessage(result=ToolResult("c9", "read", "file body"))
    shell = ToolResultMessage(
        result=ToolResult("c8", "powershell", "out", details={"exit_code": 0, "output_complete": True})
    )
    request, _ = await projected_request(FakeVerificationService(make_view()), (other, shell))
    results = [m.result for m in request.messages if isinstance(m, ToolResultMessage)]
    assert results[0].content == "file body"
    assert "freshness=unknown" in results[1].content
    assert results[1].content.startswith("out\nCicada freshness:")


async def test_projection_is_idempotent_for_repeated_view():
    first = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=first, freshness="current"),), receipts=(first,))
    service = FakeVerificationService(view)
    message = check_result("c1", "body", receipt_id="r-1")
    once, _ = await projected_request(service, (message,))
    twice, _ = await projected_request(service, tuple(once.messages))
    assert check_messages(twice)[0].result.content == check_messages(once)[0].result.content


# --- 终结点事实与保守锁存 -------------------------------------------------------


async def test_timeout_and_cancel_results_latch_process_uncertain():
    service = FakeVerificationService(make_view())
    messages = (
        check_result(
            "c1",
            "check blocked: timed out",
            receipt_id="r-1",
            is_error=True,
            extra={"timed_out": True, "output_complete": False},
        ),
        ToolResultMessage(
            result=ToolResult(
                "c2",
                "powershell",
                "cancelled",
                is_error=True,
                details={"cancelled": True, "output_complete": False},
            )
        ),
    )
    request, _ = await projected_request(service, messages)
    assert len(service.uncertain_reasons) == 2
    assert "timed out" in service.uncertain_reasons[0]
    assert "cancelled" in service.uncertain_reasons[1]
    # 锁存经 view 出现在系统状态段
    assert "process_uncertain: true" in request.system_prompt
    assert "timed out before EOF" in request.system_prompt


async def test_non_eof_and_missing_details_errors_latch_but_plain_errors_do_not():
    service = FakeVerificationService(make_view())
    messages = (
        ToolResultMessage(result=ToolResult("c1", "powershell", "boom", is_error=True, details=None)),
        ToolResultMessage(
            result=ToolResult(
                "c2", "powershell", "tool raised: RuntimeError", is_error=True, details={"exit_code": 3}
            )
        ),
        ToolResultMessage(result=ToolResult("c3", "powershell", "no such tool", is_error=True, details={"exit_code": None, "output_complete": False})),
        # 正常非零且终结完整: 可以修复后重检, 不锁存
        ToolResultMessage(
            result=ToolResult(
                "c4", "powershell", "exit 1", is_error=True, details={"exit_code": 1, "output_complete": True}
            )
        ),
        ToolResultMessage(result=ToolResult("c5", "read", "unrelated", is_error=True)),
    )
    await projected_request(service, messages)
    assert len(service.uncertain_reasons) == 3
    assert "no structured details" in service.uncertain_reasons[0]
    assert "termination failure" in service.uncertain_reasons[1]
    assert "complete terminal fact" in service.uncertain_reasons[2]


async def test_success_text_does_not_clear_latched_uncertainty_and_successes_are_not_latched():
    service = FakeVerificationService(make_view())
    good = ToolResultMessage(
        result=ToolResult("c1", "powershell", "all good", details={"exit_code": 0, "output_complete": True})
    )
    await projected_request(service, (good,))
    assert service.uncertain_reasons == []

    service.mark_process_uncertain("earlier cancelled process")
    request, _ = await projected_request(service, (good,))
    assert service.uncertain_reasons == ["earlier cancelled process"]  # 单调: 不被 PASS 清除
    assert "process_uncertain: true" in request.system_prompt


async def test_check_receipt_identity_is_not_treated_as_unknown_termination():
    service = FakeVerificationService(make_view())
    result = check_result("c1", "check blocked", receipt_id="r-1", is_error=True, extra={"output_complete": False})
    await projected_request(service, (result,))
    assert service.uncertain_reasons == []  # 事实由回执账本承载


async def test_policy_scans_before_refresh_and_uses_one_view_per_stream():
    service = FakeVerificationService(make_view())
    result = ToolResultMessage(
        result=ToolResult("c1", "powershell", "x", is_error=True, details={"output_complete": False})
    )
    model = RecordingModel()
    policy = RunPolicy(model, service)
    async for _ in policy.stream(ModelRequest((result,), ()), CancelToken()):
        pass
    assert len(service.refresh_calls) == 1
    # 锁存发生在 refresh 之前: 同一轮系统状态段已含 process_uncertain
    assert "process_uncertain: true" in model.requests[0].system_prompt


async def test_already_cancelled_stream_sends_nothing_and_skips_refresh():
    service = FakeVerificationService(make_view())
    model = RecordingModel()
    policy = RunPolicy(model, service)
    cancel = CancelToken()
    cancel.cancel()
    events = [event async for event in policy.stream(ModelRequest((UserMessage(text="hi"),), ()), cancel)]
    assert events == [StreamDone("aborted")]
    assert model.requests == []
    assert service.refresh_calls == []
    assert service.uncertain_reasons == []


async def test_refresh_cancellation_propagates_to_caller():
    class CancellingService(FakeVerificationService):
        async def refresh(self, cancel: CancelToken) -> VerificationView:
            raise KeyboardInterrupt

    policy = RunPolicy(RecordingModel(), CancellingService(make_view()))
    with pytest.raises(KeyboardInterrupt):
        async for _ in policy.stream(ModelRequest((UserMessage(text="hi"),), ()), CancelToken()):
            pass


def test_policy_rejects_non_model_and_non_service():
    with pytest.raises(TypeError, match="ModelPort"):
        RunPolicy(object(), FakeVerificationService(make_view()))
    with pytest.raises(TypeError, match="VerificationService"):
        RunPolicy(RecordingModel(), object())


# --- wire 边界与真实适配路径 ----------------------------------------------------


async def test_wire_request_carries_control_section_and_freshness_markers():
    fresh = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=fresh, freshness="current"),), receipts=(fresh,))
    qwen = Qwen(FakeVerificationService(view))
    messages = (
        UserMessage(text="任务"),
        check_result("c1", "PASS", receipt_id="r-1"),
    )
    events = await qwen.run(messages, (ToolSpec("check", "run a check", {"type": "object"}),), system_prompt="rules")

    payload = qwen.wire
    assert payload["messages"][0]["role"] == "system"
    assert "<cicada-verification-state>" in payload["messages"][0]["content"]
    assert f"snapshot_ref: {REF}" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "任务"}
    tool_message = payload["messages"][2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "c1"
    assert tool_message["name"] == "check"
    assert tool_message["content"] == "PASS\nCicada freshness: stale=false freshness=current tool=check"
    assert [e.stop_reason for e in events if isinstance(e, StreamDone)] == ["stop"]


async def test_wire_request_cap_counts_projection_and_refuses_one_byte_over():
    fresh = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=fresh, freshness="current"),), receipts=(fresh,))
    messages = (
        UserMessage(text="开始"),
        check_result("c1", "PASS" + "中" * 200, receipt_id="r-1"),
    )
    tools = (ToolSpec("check", "run a check", {"type": "object"}),)
    service = FakeVerificationService(view)

    at_limit = Qwen(service, limit=4096)
    # 先量出同一投影的真实 wire 字节, 再按差值把上限设成 恰好 / 恰好-1
    reference = Qwen(service)
    await reference.run(messages, tools, system_prompt="rules")
    size = len(reference.seen[0].content)

    qwen_ok = Qwen(service, limit=size)
    events_ok = await qwen_ok.run(messages, tools, system_prompt="rules")
    assert len(qwen_ok.seen) == 1
    assert len(qwen_ok.seen[0].content) == size
    assert [e.stop_reason for e in events_ok if isinstance(e, StreamDone)] == ["stop"]
    assert "<cicada-verification-state>" in qwen_ok.wire["messages"][0]["content"]

    qwen_over = Qwen(service, limit=size - 1)
    events_over = await qwen_over.run(messages, tools, system_prompt="rules")
    assert qwen_over.seen == []  # 超限不发 POST
    dones = [e for e in events_over if isinstance(e, StreamDone)]
    assert dones[0].stop_reason == "error"
    assert f"{size} bytes" in dones[0].error
    assert f"{size - 1} bytes" in dones[0].error
    assert at_limit is not None


# --- bootstrap 公开组装入口 ----------------------------------------------------


async def test_bootstrap_policy_wires_service_into_model_port():
    fresh = make_receipt("r-1")
    service = FakeVerificationService(
        make_view(states=(CheckState("check-1", receipt=fresh, freshness="current"),), receipts=(fresh,))
    )
    model = FakeModel(
        [
            [TextDelta("done"), StreamDone("stop")],
        ]
    )
    seen: list = []

    def verify_plugin():
        def setup(ctx):
            ctx.provide("coding.verification", service)

        return PluginDefinition(
            name="fake-verification", setup=setup, provides=frozenset({"coding.verification"})
        )

    def policy(original, runtime):
        assert runtime.capability("model") is original
        seen.append(runtime.capability("coding.verification"))
        return verification_policy(original, runtime.capability("coding.verification"))

    app = await bootstrap(
        [fake_model_plugin(model), fake_tools_plugin([EchoTool()]), verify_plugin()],
        tool_capabilities=("tools",),
        system_prompt="rules",
        model_policy=policy,
    )
    result = await app.agent.run("任务")
    assert result.stop_reason == "stop"
    assert seen == [service]
    sent = model.requests[0]
    assert sent.system_prompt.startswith("rules\n<cicada-verification-state>")
    assert f"snapshot_ref: {REF}" in sent.system_prompt
    assert sent.tools[0].name == "echo"
    assert len(service.refresh_calls) == 1
    # 内核会话与 RunResult 保留原始 system prompt; 投影只存在于发给模型的请求
    assert [m.text for m in result.messages if isinstance(m, UserMessage)] == ["任务"]
    assert not any(
        getattr(m, "text", "").startswith("rules\n<cicada-verification-state>") for m in result.messages
    )
    assert "cicada-verification-state" not in repr(result.messages)
    await app.aclose()
