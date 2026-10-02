"""P4 05 应用层投影策略测试: 共享 VerificationService fake + 真实 ModelPort/MockTransport."""

import json
import subprocess
import sys
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
from cicada.core.ports import ModelRequest, StreamDone, TextDelta, ToolContext, ToolSpec
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.process import BoundedText, PowerShellRunner, ProcessResult
from cicada.plugins.coding.snapshot import Snapshotter
from cicada.plugins.coding.tool_check import CheckTool
from cicada.plugins.coding.verification import Verifier, generate_verification_run_id
from cicada.plugins.coding.verification_contracts import (
    CheckDefinition,
    CheckReceipt,
    CheckState,
    VerificationPlan,
    VerificationView,
)
from cicada.plugins.coding.workspace import Workspace
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

    def mark_process_uncertain(self, reason: str) -> None:
        if reason in self.uncertain_reasons:  # 真实服务同样按 reason 去重 (单调)
            return
        self.uncertain_reasons.append(reason)
        # 真实服务由 mark 后的 refresh 反映锁存; fake 同样在 refresh 时重建 view
        self.view = replace(
            self.view,
            process_uncertain=True,
            blocking_reasons=self.view.blocking_reasons + (reason,),
        )

    async def refresh(self, cancel: CancelToken) -> VerificationView:
        self.refresh_calls.append(cancel)
        return self.view

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


async def test_latching_projection_keeps_ids_names_order_and_original_bodies():
    """锁存路径同样不得改写原消息: id/name/顺序/正文一字不改, 只追加投影 marker."""
    view = make_view()  # 空账本: 所有 check 回执都是 unknown
    messages = (
        UserMessage(text="task"),
        AssistantMessage(text="", tool_calls=(ToolCall("c1", "powershell", '{"command":"x"}'),)),
        ToolResultMessage(
            result=ToolResult(
                "c1",
                "powershell",
                "out\n[exit_code=0 timed_out=False cancelled=False output_complete=False]",
                details={"exit_code": 0, "timed_out": False, "cancelled": False, "output_complete": False},
            )
        ),
        check_result("c2", "check body", receipt_id="ghost"),
    )
    service = FakeVerificationService(view)
    model = RecordingModel()
    policy = RunPolicy(model, service)
    request = ModelRequest(messages=messages, tools=(), system_prompt="rules")
    async for _ in policy.stream(request, CancelToken()):
        pass
    assert len(service.uncertain_reasons) == 2  # 两条都保守锁存

    projected = model.requests[0]
    bodies = [m.result for m in projected.messages if isinstance(m, ToolResultMessage)]
    assert [r.call_id for r in bodies] == ["c1", "c2"]
    assert [r.name for r in bodies] == ["powershell", "check"]
    assert bodies[0].content.startswith(
        "out\n[exit_code=0 timed_out=False cancelled=False output_complete=False]"
        "\nCicada freshness: stale=true freshness=unknown tool=powershell"
    )
    assert bodies[1].content.startswith(
        "check body\nCicada freshness: stale=true freshness=unknown tool=check"
    )
    assert bodies[1].details == {"receipt_id": "ghost"}
    # 原消息对象与传入请求都不被重写
    assert messages[2].result.content == "out\n[exit_code=0 timed_out=False cancelled=False output_complete=False]"
    assert messages[2].result.details == {
        "exit_code": 0,
        "timed_out": False,
        "cancelled": False,
        "output_complete": False,
    }
    assert messages[3].result.content == "check body"
    assert request.messages[2].result.content == messages[2].result.content
    assert request.system_prompt == "rules"


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
    assert "without reaching EOF" in service.uncertain_reasons[2]


async def test_reliable_non_eof_latches_even_with_exit_code_zero():
    """S3 反例: exit_code=0 不能抵消可靠的 output_complete=false."""
    service = FakeVerificationService(make_view())
    result = ToolResultMessage(
        result=ToolResult(
            "c1",
            "powershell",
            "out\n[exit_code=0 timed_out=False cancelled=False output_complete=False]",
            details={"exit_code": 0, "timed_out": False, "cancelled": False, "output_complete": False},
        )
    )
    request, _ = await projected_request(service, (result,))
    assert service.uncertain_reasons == [
        "powershell ended without reaching EOF; termination unproven"
    ]
    # 锁存发生在 refresh 之前, 同一轮系统状态段已含 process_uncertain
    assert "process_uncertain: true" in request.system_prompt


async def test_unknown_receipt_id_never_skips_ledger_check():
    """S3 反例: 只要 details 里有 receipt_id 字符串就跳过终结核对是不成立的."""
    ghost = ToolResultMessage(
        result=ToolResult(
            "c1",
            "check",
            "check output",
            is_error=True,
            details={"receipt_id": "ghost", "timed_out": False, "output_complete": False},
        )
    )
    service = FakeVerificationService(make_view())
    request, _ = await projected_request(service, (ghost,))
    reason = "check result receipt_id is not in the verification ledger; termination unproven"
    assert service.uncertain_reasons == [reason]
    # 账本核对结果必须在**同一轮**发给模型的请求里可见, 不能只保证下一轮
    assert "process_uncertain: true" in request.system_prompt
    assert f"- {reason}" in request.system_prompt
    assert "none reported by the verification view" not in request.system_prompt
    # 不为此重复整仓快照: 每轮仍只有一次 refresh
    assert len(service.refresh_calls) == 1


async def test_structured_generic_error_without_terminal_fields_latches_conservatively():
    """F13 反例: 存在 details 字典不是终结证明; 缺可靠字段的 error 保守锁存."""
    service = FakeVerificationService(make_view())
    messages = tuple(
        ToolResultMessage(result=ToolResult(f"c{i}", "powershell", "boom", is_error=True, details=details))
        for i, details in enumerate(
            (
                {},
                {"exit_code": None},
                {"exit_code": True, "output_complete": True},
                {"exit_code": "0", "output_complete": True},
                {"exit_code": 0},
            ),
            start=1,
        )
    ) + (
        ToolResultMessage(
            result=ToolResult("c6", "check", "blocked", is_error=True, details={"receipt_id": "ghost"})
        ),
    )
    request, _ = await projected_request(service, messages)
    # 每条都锁存; 服务按 reason 去重, 因此同一原因在账本里只留一条
    assert service.uncertain_reasons == [
        "powershell error carries no trustworthy terminal fact",
        "check result receipt_id is not in the verification ledger; termination unproven",
    ]
    assert "process_uncertain: true" in request.system_prompt


async def test_trustworthy_terminal_facts_stay_clean():
    """保留既定行为: 正常0/正常非零完整/合法启动失败/执行前拒绝都不锁存."""
    service = FakeVerificationService(make_view())
    messages = (
        ToolResultMessage(
            result=ToolResult(
                "c1", "powershell", "ok", details={"exit_code": 0, "timed_out": False, "cancelled": False, "output_complete": True}
            )
        ),
        ToolResultMessage(
            result=ToolResult(
                "c2", "powershell", "exit 1", is_error=True, details={"exit_code": 1, "timed_out": False, "cancelled": False, "output_complete": True}
            )
        ),
        ToolResultMessage(
            result=ToolResult("c3", "powershell", "pwsh executable not found on PATH", is_error=True, details=None)
        ),
        ToolResultMessage(result=ToolResult("c4", "powershell", "unknown tool: powershell", is_error=True, details=None)),
        ToolResultMessage(result=ToolResult("c5", "powershell", "cancelled before execution", is_error=True, details=None)),
        ToolResultMessage(result=ToolResult("c6", "powershell", "arguments failed validation: nope", is_error=True, details=None)),
    )
    request, _ = await projected_request(service, messages)
    assert service.uncertain_reasons == []
    assert "process_uncertain: true" not in request.system_prompt


async def test_known_receipt_in_ledger_resolves_check_identity():
    """正例: 账本里能找到且自身终结可信的回执不需要锁存."""
    passed = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=passed, freshness="current"),), receipts=(passed,))
    service = FakeVerificationService(view)
    result = check_result("c1", "PASS", receipt_id="r-1")
    request, _ = await projected_request(service, (result,))
    assert service.uncertain_reasons == []
    assert "process_uncertain: true" not in request.system_prompt
    assert "- none reported by the verification view" in request.system_prompt
    assert "freshness=current" in check_messages(request)[0].result.content
    assert len(service.refresh_calls) == 1


async def test_powershell_result_with_ledger_receipt_is_resolved_not_latched():
    """通用 powershell 结果同样按账本核对回执身份, 不因工具名不同而跳过."""
    passed = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=passed, freshness="current"),), receipts=(passed,))
    service = FakeVerificationService(view)
    known = ToolResultMessage(
        result=ToolResult("c1", "powershell", "out", details={"receipt_id": "r-1"})
    )
    request, _ = await projected_request(service, (known,))
    assert service.uncertain_reasons == []
    assert "process_uncertain: true" not in request.system_prompt
    assert check_messages(request)[0].result.content.startswith("out\nCicada freshness: stale=false freshness=current")

    ghost = ToolResultMessage(
        result=ToolResult("c2", "powershell", "out", details={"receipt_id": "ghost"})
    )
    service_ghost = FakeVerificationService(make_view())
    ghost_request, _ = await projected_request(service_ghost, (ghost,))
    assert len(service_ghost.uncertain_reasons) == 1
    assert "not in the verification ledger" in service_ghost.uncertain_reasons[0]
    assert "freshness=unknown" in check_messages(ghost_request)[0].result.content
    # 同一轮的 system/control 段也必须是锁存后的状态
    assert "process_uncertain: true" in ghost_request.system_prompt
    assert f"- {service_ghost.uncertain_reasons[0]}" in ghost_request.system_prompt


async def test_receipt_in_ledger_without_terminal_fact_still_latches():
    """账本里有回执, 但回执自身记录非 EOF: 仍然锁存, 不被 receipt_id 抵消."""
    incomplete = replace(
        make_receipt("r-1"),
        output_complete=False,
        verification_status="blocked",
        failure_kind="output_incomplete",
    )
    view = make_view(states=(CheckState("check-1", receipt=incomplete, freshness="current"),), receipts=(incomplete,))
    service = FakeVerificationService(view)
    result = check_result("c1", "check blocked", receipt_id="r-1", is_error=True)
    request, _ = await projected_request(service, (result,))
    assert len(service.uncertain_reasons) == 1
    assert "trustworthy terminal fact" in service.uncertain_reasons[0]
    # 同轮可见, 不推迟到下一轮
    assert "process_uncertain: true" in request.system_prompt
    assert f"- {service.uncertain_reasons[0]}" in request.system_prompt
    assert len(service.refresh_calls) == 1
    # 原始工具结果不被改写 (投影只加 freshness marker)
    assert result.result.details == {"receipt_id": "r-1"}
    assert check_messages(request)[0].result.call_id == "c1"


async def test_launch_failed_receipt_is_a_known_failure_not_unknown_termination():
    """合法启动失败由程序记录, 不属于"进程终结未知"."""
    launched = replace(
        make_receipt("r-1", status="blocked"),
        execution_status="launch_failed",
        exit_code=None,
        output_complete=False,
        failure_kind="launch_failed",
    )
    view = make_view(states=(CheckState("check-1", receipt=launched, freshness="current"),), receipts=(launched,))
    service = FakeVerificationService(view)
    result = check_result("c1", "check blocked", receipt_id="r-1", is_error=True)
    request, _ = await projected_request(service, (result,))
    assert service.uncertain_reasons == []
    assert "process_uncertain: true" not in request.system_prompt


async def test_static_service_view_still_yields_same_round_latched_control_section():
    """即使 refresh 的 view 未反映本轮锁存, 交给 inner 的控制段也必须带上已核对的锁存事实."""

    class StaticViewService(FakeVerificationService):
        def mark_process_uncertain(self, reason: str) -> None:
            if reason not in self.uncertain_reasons:
                self.uncertain_reasons.append(reason)  # 故意不更新 view: 只有 service state 变

    ghost = ToolResultMessage(
        result=ToolResult("c1", "check", "check output", is_error=True, details={"receipt_id": "ghost"})
    )
    service = StaticViewService(make_view())
    request, _ = await projected_request(service, (ghost,))
    reason = "check result receipt_id is not in the verification ledger; termination unproven"
    assert service.uncertain_reasons == [reason]
    assert "process_uncertain: true" in request.system_prompt
    assert f"- {reason}" in request.system_prompt
    assert "none reported by the verification view" not in request.system_prompt
    assert len(service.refresh_calls) == 1


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


async def test_check_receipt_identity_alone_does_not_decide_termination():
    """receipt_id 只是待核对身份: 账本里没有它就不能跳过终结核对."""
    service = FakeVerificationService(make_view())
    result = check_result("c1", "check blocked", receipt_id="r-1", is_error=True, extra={"output_complete": False})
    await projected_request(service, (result,))
    assert len(service.uncertain_reasons) == 1
    assert "not in the verification ledger" in service.uncertain_reasons[0]


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


# --- 真实 Verifier 服务 + ModelPort 聚焦集成 ------------------------------------


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input_bytes, capture_output=True, check=True
    ).stdout


def _real_repo(tmp_path: Path) -> Workspace:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "import sample\n\n\ndef test_value():\n    assert sample.VALUE == 1\n", encoding="utf-8"
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "baseline")
    return Workspace.create(root)


def _real_verifier(workspace: Workspace, *checks: CheckDefinition, runner=None) -> Verifier:
    snapshotter = Snapshotter(workspace, GitInventory(workspace))
    plan = VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=workspace.root,
        checks=tuple(checks),
    )
    return Verifier(workspace, snapshotter, runner or PowerShellRunner(), plan)


def _scripted_result(exit_code: int, output_complete: bool) -> ProcessResult:
    return ProcessResult(
        exit_code=exit_code,
        timed_out=False,
        cancelled=False,
        output=BoundedText(
            text="out\n", truncated=False, total_bytes=4, total_lines=1, full_output_path=None
        ),
        output_complete=output_complete,
    )


class _ScriptedRunner:
    """受控 runner: 只替代低层采集结果, 检查分类/回执/账本仍由真实服务计算."""

    def __init__(self, result: ProcessResult | None = None, raises: bool = False) -> None:
        self._result = result
        self._raises = raises

    async def run(self, *, command, cwd, timeout, cancel, output_dir) -> ProcessResult:
        if self._raises:
            raise KeyboardInterrupt
        assert self._result is not None
        return self._result


async def _real_check_result(workspace: Workspace, verifier: Verifier, check_id: str) -> ToolResult:
    capture = await verifier.initialize(CancelToken())
    assert capture.available, (capture.failure_kind, capture.error)
    return await CheckTool(verifier).execute(
        {"action": "run", "check_id": check_id}, ToolContext(call_id="c1", cancel=CancelToken())
    )


async def _stream_with(policy: RunPolicy, model: RecordingModel, messages) -> tuple[str, list]:
    events = [event async for event in policy.stream(ModelRequest(messages, ()), CancelToken())]
    return model.requests[0].system_prompt, events


async def test_real_service_receipt_survives_projection_without_latching(tmp_path):
    """真实服务回执 id 经公开 ModelPort 投影: 账本核对通过, 不锁存且 marker 为 current."""
    workspace = _real_repo(tmp_path)
    verifier = _real_verifier(workspace, CheckDefinition("check-1", "exit 0"))
    result = await _real_check_result(workspace, verifier, "check-1")
    assert result.details["output_complete"] is True
    assert result.details["execution_status"] == "exited"
    assert verifier.receipts[-1].verification_status == "passed"

    model = RecordingModel()
    section, _ = await _stream_with(
        RunPolicy(model, verifier), model, (ToolResultMessage(result=result),)
    )
    assert verifier.process_uncertain is False
    assert "process_uncertain: true" not in section
    assert f"verification_run_id: {verifier.plan.verification_run_id}" in section
    assert f"check check-1: status=passed freshness=current" in section
    projected = model.requests[0].messages[0].result
    assert result.details["receipt_id"] in projected.content
    assert "freshness=current" in projected.content


async def test_real_service_non_eof_check_negates_exit_zero_and_latches(tmp_path):
    """真实服务低层 non-EOF 且 exit_code=0: 回执 blocked, 服务与投影都锁存且下一轮仍在."""
    workspace = _real_repo(tmp_path)
    non_eof = _ScriptedRunner(_scripted_result(exit_code=0, output_complete=False))
    verifier = _real_verifier(workspace, CheckDefinition("check-1", "exit 0"), runner=non_eof)
    result = await _real_check_result(workspace, verifier, "check-1")
    receipt = verifier.receipts[-1]
    assert receipt.verification_status == "blocked"
    assert receipt.failure_kind == "output_incomplete"
    assert receipt.exit_code == 0 and receipt.output_complete is False
    # 服务先锁存 (spec §5); 投影仍独立从原始 details 得到同一保守事实 (spec §6)
    assert verifier.process_uncertain is True

    first_model = RecordingModel()
    first_section, _ = await _stream_with(
        RunPolicy(first_model, verifier), first_model, (ToolResultMessage(result=result),)
    )
    # 投影按账本核对回执: 回执自身 output_complete=False, 因此保守锁存 (与服务的锁存并存)
    ledger_reason = next(
        reason
        for reason in verifier._uncertain_reasons
        if "does not record a trustworthy terminal fact" in reason
    )
    assert first_section.count("process_uncertain: true") == 1
    # 账本核对产生的新锁存必须在同一轮可见, 不能只保证下一轮
    assert f"- {ledger_reason}" in first_section
    assert "none reported by the verification view" not in first_section

    # 下一轮同一历史: 锁存仍在, 不会因为已记录过而消失
    second_model = RecordingModel()
    second_section, _ = await _stream_with(
        RunPolicy(second_model, verifier), second_model, (ToolResultMessage(result=result),)
    )
    assert "process_uncertain: true" in second_section
    assert sum(
        "does not record a trustworthy terminal fact" in reason for reason in verifier._uncertain_reasons
    ) == 1
    # 未在账本里的回执身份保守锁存, 不会被当成终结证明
    ghost = ToolResultMessage(
        result=ToolResult("c9", "check", "PASS", details={"receipt_id": "chk-ghost"})
    )
    third_model = RecordingModel()
    third_section, _ = await _stream_with(RunPolicy(third_model, verifier), third_model, (ghost,))
    assert any("not in the verification ledger" in reason for reason in verifier._uncertain_reasons)
    assert "process_uncertain: true" in third_section
    assert "- check result receipt_id is not in the verification ledger; termination unproven" in third_section


async def test_real_service_execution_cancel_propagates_and_projection_latches(tmp_path):
    """执行阶段取消: 服务传播取消 (不加回执), 投影对内核归一结果保守锁存."""
    workspace = _real_repo(tmp_path)
    verifier = _real_verifier(
        workspace,
        CheckDefinition("check-1", "exit 0"),
        runner=_ScriptedRunner(raises=True),
    )
    capture = await verifier.initialize(CancelToken())
    assert capture.available, (capture.failure_kind, capture.error)
    with pytest.raises(KeyboardInterrupt):
        await verifier.run_check("check-1", CancelToken())  # 取消仍协作传播
    assert verifier.receipts == ()  # 取消收尾没有回执事实

    # core 把 CancelledError 归一为无 details 的 error 结果 (agent.py: "cancelled during execution")
    normalized = ToolResultMessage(
        result=ToolResult("c1", "powershell", "cancelled during execution", is_error=True, details=None)
    )
    model = RecordingModel()
    section, _ = await _stream_with(RunPolicy(model, verifier), model, (normalized,))
    assert "process_uncertain: true" in section
    assert any("termination failure" in reason for reason in verifier._uncertain_reasons)
