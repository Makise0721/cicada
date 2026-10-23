"""缓存 02 应用层投影策略测试: 固定信任规则 + 独立程序验证上下文尾部.

共享 VerificationService fake + 真实 ModelPort/MockTransport。历史逐字透传,
不再追加 freshness 标记; 当前验证观察经 request_context 尾部发送, 核心身份
超限拒绝本轮 (零 inner/provider)。
"""

import hashlib
import json
import subprocess
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
    isolate_message,
)
from cicada.core.ports import ModelRequest, StreamDone, TextDelta, ToolContext, ToolSpec
from cicada.persistence.codec import encode_messages
from cicada.plugins.coding import process as coding_process
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
from cicada.plugins.deepseek import DeepSeekModel, DeepSeekConfig
from cicada.tool_outcomes import check_status_details, process_details
from cicada.run_policy import (
    CONTEXT_CLOSE_TAG,
    CONTEXT_OPEN_TAG,
    REQUEST_CONTEXT_MAX_BYTES,
    HistoricalPrefixProof,
    HistoricalProjection,
    PrefixEntry,
    PrefixViolation,
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

    async def finalize_current(self, cancel: CancelToken):
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


def parse_tail(request: ModelRequest) -> dict:
    """尾部文本必须是固定标签包住的完整可解析 JSON."""
    text = request.request_context
    assert text.startswith(CONTEXT_OPEN_TAG + "\n"), text[:80]
    assert text.endswith("\n" + CONTEXT_CLOSE_TAG)
    body = text[len(CONTEXT_OPEN_TAG) + 1:len(text) - len(CONTEXT_CLOSE_TAG) - 1]
    return json.loads(body)


class ApiWire:
    """跑一遍完整策略 + 真实 DeepSeekModel 适配路径, 返回 wire 请求体 (未发送则 None)."""

    def __init__(self, service: FakeVerificationService, limit: int | None = None) -> None:
        self.service = service
        config = DeepSeekConfig(think=False, max_request_bytes=limit or 2 * 1024 * 1024)
        self.seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            body = b'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        self.model = DeepSeekModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), config, "offline-key")
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


# --- 程序验证上下文尾部: current / stale / unknown --------------------------------


async def test_tail_reports_full_identities_and_fixed_rules():
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

    assert REQUEST_CONTEXT_MAX_BYTES == 8192  # 契约 C2 尾部上限
    # 固定信任规则在 system; 当前状态不在 system
    assert request.system_prompt.startswith("rules\n<cicada-verification-rules>")
    assert "cicada-verification-context" not in request.system_prompt
    assert "cicada-verification-rules" not in request.request_context

    payload = parse_tail(request)
    assert payload["verification_run_id"] == RUN_ID
    assert payload["snapshot_ref"] == REF
    assert payload["scope_id"] == SCOPE
    assert payload["process_uncertain"] is False
    assert payload["blocked"] is True, "stale/not_run/失败原因任一即 blocked"
    assert payload["explanations_omitted"] is False
    assert payload["blocking_reasons"] == ["check-2 latest attempt failed"]
    assert payload["checks"] == [
        {
            "check_id": "check-1",
            "latest_receipt_id": "r-1",
            "receipt_run_id": RUN_ID,
            "receipt_snapshot_after": REF,
            "status": "passed",
            "freshness": "current",
        },
        {
            "check_id": "check-2",
            "latest_receipt_id": "r-2",
            "receipt_run_id": RUN_ID,
            "receipt_snapshot_after": REF,
            "status": "failed",
            "freshness": "stale",
        },
        {
            "check_id": "check-3",
            "latest_receipt_id": None,
            "receipt_run_id": None,
            "receipt_snapshot_after": None,
            "status": "not_run",
            "freshness": "unknown",
        },
    ]
    assert len(request.request_context.encode("utf-8")) <= REQUEST_CONTEXT_MAX_BYTES
    assert events[-1] == StreamDone("stop")

    # 空原始 prompt: 固定规则单独构成 system_prompt, 恰好追加一次
    request2, _ = await projected_request(FakeVerificationService(make_view()), (UserMessage(text="go"),))
    assert request2.system_prompt.startswith("<cicada-verification-rules>")
    assert request2.system_prompt.count("<cicada-verification-rules>") == 1


async def test_tail_carries_full_latest_receipt_so_new_failure_overrides_old_success():
    """同一检查旧 PASS 之后新失败: 尾部列最新 receipt 身份与失败状态, 不回退旧成功."""
    old_pass = make_receipt("r-old", "check-1", "passed")
    new_fail = make_receipt("r-new", "check-1", "failed", "nonzero_exit")
    view = make_view(
        states=(CheckState("check-1", receipt=new_fail, freshness="current"),),
        receipts=(old_pass, new_fail),
    )
    messages = (
        check_result("c1", "PASS: old run", receipt_id="r-old"),
        check_result("c2", "FAIL: new run", receipt_id="r-new", is_error=True),
    )
    request, _ = await projected_request(FakeVerificationService(view), messages)
    payload = parse_tail(request)
    assert payload["checks"] == [{
        "check_id": "check-1",
        "latest_receipt_id": "r-new",
        "receipt_run_id": RUN_ID,
        "receipt_snapshot_after": REF,
        "status": "failed",
        "freshness": "current",
    }]
    assert payload["blocked"] is True
    # 历史两轮正文原样保留, 旧 PASS 文字不被清洗或标注
    assert [m.result.content for m in check_messages(request)] == ["PASS: old run", "FAIL: new run"]


async def test_full_identities_are_never_truncated_and_explanations_are_omitted_with_flag():
    """核心身份不截断不规整; 可缩短的解释按剩余空间容纳, 超出显式 explanations_omitted."""
    receipt = make_receipt("r-" + "x" * 4000, "check-" + "y" * 2000)
    view = make_view(
        states=(CheckState("check-" + "y" * 2000, receipt=receipt, freshness="current"),),
        receipts=(receipt,),
        blocking_reasons=tuple("reason " + "z" * 900 for _ in range(20)),
    )
    request, _ = await projected_request(FakeVerificationService(view), (UserMessage(text="go"),))
    payload = parse_tail(request)
    # 核心身份完整原样 (默认 8192 上限内), 无截断/hash/空白规整
    assert payload["checks"][0]["check_id"] == "check-" + "y" * 2000
    assert payload["checks"][0]["latest_receipt_id"] == "r-" + "x" * 4000
    assert len(request.request_context.encode("utf-8")) <= REQUEST_CONTEXT_MAX_BYTES
    # 解释被截短 (reason_max_bytes) 且部分装不下: 显式标记, blocked 不受影响
    assert payload["explanations_omitted"] is True
    assert 1 <= len(payload["blocking_reasons"]) < 20
    assert all(len(reason.encode("utf-8")) <= 512 + 3 for reason in payload["blocking_reasons"])
    assert payload["blocked"] is True, "存在 blocking reasons 即 blocked, 与解释缩短无关"


async def test_core_overflow_refuses_round_with_zero_inner_calls_and_keeps_locks():
    """核心加标签超限: 本轮一次 StreamDone(error), 零 inner 调用, 已发生锁存保留."""
    receipt = make_receipt("r-" + "x" * 4000, "check-" + "y" * 2000)
    view = make_view(
        states=(CheckState("check-" + "y" * 2000, receipt=receipt, freshness="current"),),
        receipts=(receipt,),
    )
    ghost = ToolResultMessage(
        result=ToolResult("c1", "check", "check output", is_error=True, details={"receipt_id": "ghost"})
    )
    service = FakeVerificationService(view)
    model = RecordingModel()
    policy = RunPolicy(model, service, RunPolicyConfig(control_section_max_bytes=512))
    events = [event async for event in policy.stream(ModelRequest((ghost,), ()), CancelToken())]
    assert len(events) == 1
    done = events[0]
    assert isinstance(done, StreamDone)
    assert done.stop_reason == "error"
    assert "byte limit" in done.error and "512" in done.error
    # 错误只说明 context size/limit, 不泄露身份/正文
    assert REF not in done.error and "ghost" not in done.error
    assert model.requests == [], "零 inner/provider 调用"
    assert service.uncertain_reasons, "已发生的锁存保留"
    assert "not in the verification ledger" in service.uncertain_reasons[0]


async def test_multibyte_tail_stays_within_budget_and_valid_utf8():
    """多字节身份原样进 JSON; 超限拒绝不产生半个字符, 默认上限内完整可解析."""
    receipt = make_receipt("r-1", "check-" + "中" * 200, "failed", "非零退出")
    view = make_view(
        states=(CheckState("check-" + "中" * 200, receipt=receipt, freshness="stale"),),
        receipts=(receipt,),
        blocking_reasons=tuple("原因" * 300 for _ in range(4)),
    )
    request, _ = await projected_request(FakeVerificationService(view), (UserMessage(text="go"),))
    assert len(request.request_context.encode("utf-8")) <= REQUEST_CONTEXT_MAX_BYTES
    payload = parse_tail(request)  # 完整可解析即无半个字符
    assert payload["checks"][0]["check_id"] == "check-" + "中" * 200
    assert payload["checks"][0]["status"] == "failed"
    assert payload["explanations_omitted"] is True, "解释按 reason 上限截短须显式标记"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4096"])
def test_config_rejects_non_positive_budgets(value):
    with pytest.raises(ValueError, match="control_section_max_bytes"):
        RunPolicyConfig(control_section_max_bytes=value)


async def test_unavailable_snapshot_reports_null_and_blocks():
    view = make_view(snapshot_ref=None)
    request, _ = await projected_request(FakeVerificationService(view), (UserMessage(text="go"),))
    payload = parse_tail(request)
    assert payload["snapshot_ref"] is None
    assert payload["scope_id"] is None
    assert payload["blocked"] is True, "snapshot/scope 不可得即 blocked"
    assert REF not in request.request_context


def test_preflight_system_prompt_is_pure_sync_and_stable():
    policy = RunPolicy(RecordingModel(), FakeVerificationService(make_view()))
    assert policy.request_context_max_bytes == REQUEST_CONTEXT_MAX_BYTES
    once = policy.preflight_system_prompt("base")
    assert once.startswith("base\n<cicada-verification-rules>")
    assert once == policy.preflight_system_prompt("base"), "同 base 同字节"
    assert once.count("base") == 1
    assert policy.preflight_system_prompt("") == policy.preflight_system_prompt("").strip()
    assert "<cicada-verification-rules>" in policy.preflight_system_prompt("")


# --- 历史逐字透传: 不再追加任何 freshness 标记 ------------------------------------


async def test_history_bodies_are_passed_through_without_annotations():
    fresh = make_receipt("r-current")
    old = make_receipt("r-old")
    view = make_view(
        states=(
            CheckState("check-1", receipt=fresh, freshness="current"),
            CheckState("check-2", receipt=make_receipt("r-2", "check-2"), freshness="stale"),
        ),
        receipts=(fresh, old, make_receipt("r-2", "check-2")),
    )
    messages = (
        UserMessage(text="task"),
        AssistantMessage(text="", tool_calls=(ToolCall("c1", "check", '{"action":"run"}'),)),
        check_result("c1", "PASS: all checks succeeded", receipt_id="r-current"),
        AssistantMessage(text="", tool_calls=(ToolCall("c2", "check", '{"action":"run"}'),)),
        check_result("c2", "PASS: old run", receipt_id="r-old"),
        check_result("c3", "PASS: wrong check", receipt_id="r-ghost"),
        check_result("c4", "PASS: no receipt", receipt_id=None),
    )
    request, _ = await projected_request(FakeVerificationService(view), messages)

    bodies = [m.result for m in check_messages(request)]
    # 原文一字不改; 旧标记/历史说明都不再追加
    assert [r.content for r in bodies] == [
        "PASS: all checks succeeded",
        "PASS: old run",
        "PASS: wrong check",
        "PASS: no receipt",
    ]
    assert [r.call_id for r in bodies] == ["c1", "c2", "c3", "c4"]
    assert all("Cicada freshness" not in r.content for r in bodies)
    assert "Cicada freshness" not in request.system_prompt
    assert "Cicada freshness" not in request.request_context
    # 只有结构化扫描需要 check/powershell 结果; 非检查结果不受影响
    assert [m.result.name for m in request.messages if isinstance(m, ToolResultMessage)] == ["check"] * 4


async def test_projection_keeps_ids_names_order_and_original_bodies_when_latching():
    """锁存路径同样不得改写原消息: id/name/顺序/正文一字不改."""
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
    assert projected.messages == messages, "规范历史逐字逐对象透传"
    bodies = [m.result for m in projected.messages if isinstance(m, ToolResultMessage)]
    assert [r.call_id for r in bodies] == ["c1", "c2"]
    assert [r.name for r in bodies] == ["powershell", "check"]
    assert "Cicada freshness" not in projected.messages[2].result.content
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
    # 同一轮锁存经尾部可见
    payload = parse_tail(projected)
    assert payload["process_uncertain"] is True
    assert len(payload["blocking_reasons"]) == 2

    # 非锁存路径 (已知回执) 同样不改写原消息对象与传入请求
    resolved = make_receipt("r-1")
    resolved_view = make_view(
        states=(CheckState("check-1", receipt=resolved, freshness="current"),), receipts=(resolved,)
    )
    known = check_result("c9", "original content", receipt_id="r-1")
    known_request = ModelRequest((known,), (), "sys")
    async for _ in RunPolicy(RecordingModel(), FakeVerificationService(resolved_view)).stream(
        known_request, CancelToken()
    ):
        pass
    assert known.result.content == "original content"
    assert known.result.details == {"receipt_id": "r-1"}
    assert known_request.system_prompt == "sys"


async def test_structured_receipt_absence_latches_but_leaves_body_untouched():
    service = FakeVerificationService(make_view())
    detail_less = ToolResultMessage(
        result=ToolResult("c1", "check", "check failed", is_error=True, details=None)
    )
    request, _ = await projected_request(service, (detail_less,))
    assert check_messages(request)[0].result.content == "check failed", "正文不再追加标记"
    assert service.uncertain_reasons == [
        "check result has no structured details; termination unproven"
    ]


async def test_powershell_results_pass_through_and_non_check_results_untouched():
    other = ToolResultMessage(result=ToolResult("c9", "read", "file body"))
    shell = ToolResultMessage(
        result=ToolResult("c8", "powershell", "out", details={"exit_code": 0, "output_complete": True})
    )
    request, _ = await projected_request(FakeVerificationService(make_view()), (other, shell))
    results = [m.result for m in request.messages if isinstance(m, ToolResultMessage)]
    assert results[0].content == "file body"
    assert results[1].content == "out", "check/powershell 正文同样逐字透传"


async def test_projection_is_idempotent_for_repeated_view():
    first = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=first, freshness="current"),), receipts=(first,))
    service = FakeVerificationService(view)
    message = check_result("c1", "body", receipt_id="r-1")
    once, _ = await projected_request(service, (message,))
    twice, _ = await projected_request(service, tuple(once.messages))
    assert check_messages(twice)[0].result.content == check_messages(once)[0].result.content
    assert twice.request_context == once.request_context


# --- 终结点事实与保守锁存 -------------------------------------------------------


@pytest.mark.parametrize("marker,value", [("contract_version", 2), ("contract", "future.tool-outcome")])
async def test_unknown_contract_latches_without_misclassifying_status_or_legacy(marker, value):
    service = FakeVerificationService(make_view())
    details = process_details({"exit_code": 0, "output_complete": True})
    details[marker] = value
    messages = (
        ToolResultMessage(ToolResult("status", "check", "status", details=check_status_details({}))),
        ToolResultMessage(ToolResult("legacy", "powershell", "ok", details={"exit_code": 0, "output_complete": True})),
        ToolResultMessage(ToolResult("unknown", "powershell", "ok", details=details)),
    )
    request, _ = await projected_request(service, messages)
    assert len(service.uncertain_reasons) == 1
    assert "unknown tool outcome contract" in service.uncertain_reasons[0]
    assert parse_tail(request)["process_uncertain"] is True


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
    # 锁存经尾部可见
    payload = parse_tail(request)
    assert payload["process_uncertain"] is True
    assert any("timed out before EOF" in reason for reason in payload["blocking_reasons"])


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
    # 锁存发生在 refresh 之前, 同一轮尾部已含 process_uncertain;
    # 每轮一次 refresh, 扫描先于 refresh
    assert parse_tail(request)["process_uncertain"] is True
    assert len(service.refresh_calls) == 1


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
    # 账本核对结果必须在**同一轮**发给模型的尾部里可见, 不能只保证下一轮
    payload = parse_tail(request)
    assert payload["process_uncertain"] is True
    assert reason in payload["blocking_reasons"]
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
    assert parse_tail(request)["process_uncertain"] is True


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
    assert parse_tail(request)["process_uncertain"] is False


async def test_powershell_result_with_ledger_receipt_is_resolved_not_latched():
    """账本可核对且自身终结可信的回执不需要锁存 (check 与 powershell 同一分支); 幽灵回执锁存."""
    passed = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=passed, freshness="current"),), receipts=(passed,))
    service = FakeVerificationService(view)
    check_known = check_result("c1", "PASS", receipt_id="r-1")
    request, _ = await projected_request(service, (check_known,))
    assert service.uncertain_reasons == []
    payload = parse_tail(request)
    assert payload["process_uncertain"] is False
    assert payload["checks"][0]["status"] == "passed" and payload["checks"][0]["freshness"] == "current"

    service2 = FakeVerificationService(view)
    known = ToolResultMessage(
        result=ToolResult("c2", "powershell", "out", details={"receipt_id": "r-1"})
    )
    request2, _ = await projected_request(service2, (known,))
    assert service2.uncertain_reasons == []
    assert parse_tail(request2)["process_uncertain"] is False

    ghost = ToolResultMessage(
        result=ToolResult("c3", "powershell", "out", details={"receipt_id": "ghost"})
    )
    service_ghost = FakeVerificationService(make_view())
    ghost_request, _ = await projected_request(service_ghost, (ghost,))
    assert len(service_ghost.uncertain_reasons) == 1
    assert "not in the verification ledger" in service_ghost.uncertain_reasons[0]
    # 同一轮的尾部也必须是锁存后的状态
    ghost_payload = parse_tail(ghost_request)
    assert ghost_payload["process_uncertain"] is True
    assert service_ghost.uncertain_reasons[0] in ghost_payload["blocking_reasons"]


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
    payload = parse_tail(request)
    assert payload["process_uncertain"] is True
    assert service.uncertain_reasons[0] in payload["blocking_reasons"]
    assert len(service.refresh_calls) == 1
    # 原始工具结果不被改写
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
    payload = parse_tail(request)
    assert payload["process_uncertain"] is False
    assert payload["blocked"] is True, "status=blocked 是观察的保守摘要"


async def test_model_stream_failed_closure_is_pre_execution_sentinel_not_latch():
    """T03: 模型错误后未执行调用的固定关闭文本是执行前哨兵, 不锁存进程不确定.

    即使结果带 details 或错误形态 (通常会导致保守锁存), 进程根本未启动.
    """
    service = FakeVerificationService(make_view())
    messages = (
        ToolResultMessage(
            result=ToolResult(
                "c1",
                "powershell",
                "model stream failed; tool call not executed",
                is_error=True,
                details={"receipt_id": "r-1", "output_complete": False},
            )
        ),
        ToolResultMessage(
            result=ToolResult(
                "c2",
                "check",
                "model stream failed; tool call not executed",
                is_error=True,
            )
        ),
    )
    request, _ = await projected_request(service, messages)
    assert service.uncertain_reasons == []
    assert parse_tail(request)["process_uncertain"] is False


async def test_static_service_view_still_yields_same_round_latched_tail():
    """即使 refresh 的 view 未反映本轮锁存, 交给 inner 的尾部也必须带上已核对的锁存事实."""

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
    payload = parse_tail(request)
    assert payload["process_uncertain"] is True
    assert reason in payload["blocking_reasons"]
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
    assert parse_tail(request)["process_uncertain"] is True


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


async def test_policy_rejects_incoming_nonempty_request_context():
    """request_context 只能由 RunPolicy 写入: 入口已非空即明确错误, 不覆盖不拼接."""
    model = RecordingModel()
    service = FakeVerificationService(make_view())
    policy = RunPolicy(model, service)
    poisoned = ModelRequest(
        (UserMessage(text="hi"),), (), system_prompt="rules", request_context="<fake/>"
    )
    with pytest.raises(ValueError, match="only the RunPolicy may write it"):
        async for _ in policy.stream(poisoned, CancelToken()):
            pass
    assert model.requests == []
    assert service.refresh_calls == [], "拒绝发生在刷新之前"


# --- wire 边界与真实适配路径 ----------------------------------------------------


async def test_wire_request_maps_context_to_trailing_user_message():
    fresh = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=fresh, freshness="current"),), receipts=(fresh,))
    api_wire = ApiWire(FakeVerificationService(view))
    messages = (
        UserMessage(text="任务"),
        AssistantMessage("", tool_calls=(ToolCall("c1", "check", "{}"),), stop_reason="tool_use"),
        check_result("c1", "PASS", receipt_id="r-1"),
    )
    events = await api_wire.run(messages, (ToolSpec("check", "run a check", {"type": "object"}),), system_prompt="rules")

    payload = api_wire.wire
    # system 只含基础提示与固定规则, 不含当前状态
    assert payload["messages"][0]["role"] == "system"
    assert "<cicada-verification-rules>" in payload["messages"][0]["content"]
    assert "cicada-verification-context" not in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "任务"}
    tool_message = payload["messages"][3]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "c1"
    assert tool_message["content"] == "PASS", "工具正文原样映射"
    # 尾部恰好一条 user 消息, 内容为固定标签包住的完整 JSON
    assert len(payload["messages"]) == 5
    tail = payload["messages"][-1]
    assert tail["role"] == "user"
    assert tail["content"].startswith(CONTEXT_OPEN_TAG)
    tail_payload = json.loads(
        tail["content"][len(CONTEXT_OPEN_TAG) + 1:len(tail["content"]) - len(CONTEXT_CLOSE_TAG) - 1]
    )
    assert tail_payload["checks"][0]["latest_receipt_id"] == "r-1"
    assert [e.stop_reason for e in events if isinstance(e, StreamDone)] == ["stop"]


async def test_wire_request_cap_counts_context_and_refuses_one_byte_over():
    fresh = make_receipt("r-1")
    view = make_view(states=(CheckState("check-1", receipt=fresh, freshness="current"),), receipts=(fresh,))
    messages = (
        UserMessage(text="开始"),
        AssistantMessage("", tool_calls=(ToolCall("c1", "check", "{}"),), stop_reason="tool_use"),
        check_result("c1", "PASS" + "中" * 200, receipt_id="r-1"),
    )
    tools = (ToolSpec("check", "run a check", {"type": "object"}),)
    service = FakeVerificationService(view)

    # 先量出同一投影的真实 wire 字节, 再按差值把上限设成 恰好 / 恰好-1
    reference = ApiWire(service)
    await reference.run(messages, tools, system_prompt="rules")
    size = len(reference.seen[0].content)

    api_wire_ok = ApiWire(service, limit=size)
    events_ok = await api_wire_ok.run(messages, tools, system_prompt="rules")
    assert len(api_wire_ok.seen) == 1
    assert len(api_wire_ok.seen[0].content) == size
    assert [e.stop_reason for e in events_ok if isinstance(e, StreamDone)] == ["stop"]
    assert api_wire_ok.wire["messages"][-1]["content"].startswith(CONTEXT_OPEN_TAG)

    api_wire_over = ApiWire(service, limit=size - 1)
    events_over = await api_wire_over.run(messages, tools, system_prompt="rules")
    assert api_wire_over.seen == []  # 超限不发 POST
    dones = [e for e in events_over if isinstance(e, StreamDone)]
    assert dones[0].stop_reason == "error"
    assert "request exceeds byte limit; not sent" in dones[0].error


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
    assert sent.system_prompt.startswith("rules\n<cicada-verification-rules>")
    assert parse_tail(sent)["checks"][0]["latest_receipt_id"] == "r-1"
    assert sent.tools[0].name == "echo"
    assert len(service.refresh_calls) == 1
    # 内核会话与 RunResult 保留原始 system prompt; 投影只存在于发给模型的请求
    assert [m.text for m in result.messages if isinstance(m, UserMessage)] == ["任务"]
    assert not any(
        getattr(m, "text", "").startswith("rules\n<cicada-verification-rules>") for m in result.messages
    )
    assert "cicada-verification-rules" not in repr(result.messages)
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


async def _stream_tail(policy: RunPolicy, model: RecordingModel, messages) -> tuple[dict, list]:
    events = [event async for event in policy.stream(ModelRequest(messages, ()), CancelToken())]
    return parse_tail(model.requests[0]), events


async def test_real_service_receipt_survives_projection_without_latching(tmp_path):
    """真实服务回执 id 经公开 ModelPort 投影: 账本核对通过, 不锁存且尾部为 current."""
    workspace = _real_repo(tmp_path)
    verifier = _real_verifier(workspace, CheckDefinition("check-1", "exit 0"))
    result = await _real_check_result(workspace, verifier, "check-1")
    assert result.details["output_complete"] is True
    assert result.details["execution_status"] == "exited"
    assert verifier.receipts[-1].verification_status == "passed"

    model = RecordingModel()
    payload, _ = await _stream_tail(RunPolicy(model, verifier), model, (ToolResultMessage(result=result),))
    assert verifier.process_uncertain is False
    assert payload["process_uncertain"] is False
    assert payload["verification_run_id"] == verifier.plan.verification_run_id
    assert payload["checks"][0]["status"] == "passed"
    assert payload["checks"][0]["freshness"] == "current"
    assert payload["checks"][0]["latest_receipt_id"] == result.details["receipt_id"]
    # 原始工具结果正文逐字透传
    assert model.requests[0].messages[0].result.content == result.content


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
    first_payload, _ = await _stream_tail(
        RunPolicy(first_model, verifier), first_model, (ToolResultMessage(result=result),)
    )
    # 投影按账本核对回执: 回执自身 output_complete=False, 因此保守锁存 (与服务的锁存并存)
    ledger_reason = next(
        reason
        for reason in verifier._uncertain_reasons
        if "does not record a trustworthy terminal fact" in reason
    )
    assert first_payload["process_uncertain"] is True
    # 账本核对产生的新锁存必须在同一轮可见, 不能只保证下一轮
    assert ledger_reason in first_payload["blocking_reasons"]

    # 下一轮同一历史: 锁存仍在, 不会因为已记录过而消失
    second_model = RecordingModel()
    second_payload, _ = await _stream_tail(
        RunPolicy(second_model, verifier), second_model, (ToolResultMessage(result=result),)
    )
    assert second_payload["process_uncertain"] is True
    assert sum(
        "does not record a trustworthy terminal fact" in reason for reason in verifier._uncertain_reasons
    ) == 1
    # 未在账本里的回执身份保守锁存, 不会被当成终结证明
    ghost = ToolResultMessage(
        result=ToolResult("c9", "check", "PASS", details={"receipt_id": "chk-ghost"})
    )
    third_model = RecordingModel()
    third_payload, _ = await _stream_tail(RunPolicy(third_model, verifier), third_model, (ghost,))
    assert any("not in the verification ledger" in reason for reason in verifier._uncertain_reasons)
    assert third_payload["process_uncertain"] is True
    assert "check result receipt_id is not in the verification ledger; termination unproven" in third_payload["blocking_reasons"]


async def test_real_service_launch_failure_projects_blocked_without_new_latch(tmp_path, monkeypatch):
    """S/V 发布明确启动失败后: 真实服务给 blocked/launch_failed 回执, 投影不新增不确定锁存."""
    workspace = _real_repo(tmp_path)
    verifier = _real_verifier(workspace, CheckDefinition("check-1", "exit 0"))
    missing = tmp_path / "missing-pwsh.exe"
    monkeypatch.setattr(coding_process, "resolve_pwsh", lambda: str(missing))
    result = await _real_check_result(workspace, verifier, "check-1")

    receipt = verifier.receipts[-1]
    assert receipt.execution_status == "launch_failed"
    assert receipt.verification_status == "blocked"
    assert receipt.failure_kind == "launch_failed"
    assert receipt.exit_code is None and receipt.timed_out is False and receipt.cancelled is False
    assert verifier.process_uncertain is False  # 已知未启动不是"进程终结未知"

    model = RecordingModel()
    payload, _ = await _stream_tail(RunPolicy(model, verifier), model, (ToolResultMessage(result=result),))
    assert verifier.process_uncertain is False
    assert payload["process_uncertain"] is False
    assert payload["checks"][0]["status"] == "blocked"
    assert not any("not in the verification ledger" in reason for reason in payload["blocking_reasons"])
    assert not any("trustworthy terminal fact" in reason for reason in payload["blocking_reasons"])


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
    payload, _ = await _stream_tail(RunPolicy(model, verifier), model, (normalized,))
    assert payload["process_uncertain"] is True
    assert any("termination failure" in reason for reason in verifier._uncertain_reasons)


# --- 03: 恢复前缀证明与历史投影 ---------------------------------------------------------------


def _prefix_entry(message, *, epoch_id="e1", request_id="q1", seq=2, receipt_id=None, run_id=None):
    return PrefixEntry(
        epoch_id=epoch_id, request_id=request_id, seq=seq,
        message_sha256=hashlib.sha256(encode_messages((message,)).encode("utf-8")).hexdigest(),
        receipt_id=receipt_id, verification_run_id=run_id,
    )


def test_proof_validate_rejects_missing_reorder_and_modification():
    prefix = (
        UserMessage(text="旧问"),
        check_result("c1", "旧检查输出", receipt_id="r-old"),
        AssistantMessage(text="旧答"),
    )
    proof = HistoricalPrefixProof(entries=(
        _prefix_entry(prefix[0]),
        _prefix_entry(prefix[1], receipt_id="r-old", run_id="run-old"),
        _prefix_entry(prefix[2]),
    ))
    assert proof.length == 3

    # 消息拷贝不改变证明 (isolate 复制后仍通过)
    copied = tuple(isolate_message(m) for m in prefix)
    proof.validate(copied + (UserMessage(text="新问"),))
    proof.validate(prefix)

    with pytest.raises(PrefixViolation, match="missing restored prefix"):
        proof.validate(prefix[:2])
    with pytest.raises(PrefixViolation, match="changed since the proof was built"):
        proof.validate((UserMessage(text="被改写的旧问"),) + prefix[1:])
    with pytest.raises(PrefixViolation, match="changed since the proof was built"):
        proof.validate((prefix[1], prefix[0], prefix[2]))  # 重排
    # receipt 交换会先撞结构指纹; 专用分支用"指纹匹配但来源不符"的构造直测
    swapped = check_result("c1", "旧检查输出", receipt_id="r-other")
    mismatch_proof = HistoricalPrefixProof(entries=(
        _prefix_entry(prefix[0]),
        _prefix_entry(swapped, receipt_id="r-old", run_id="run-old"),
        _prefix_entry(prefix[2]),
    ))
    with pytest.raises(PrefixViolation, match="different receipt"):
        mismatch_proof.validate((prefix[0], swapped, prefix[2]))
    impostor = AssistantMessage(text="顶替")
    lost_proof = HistoricalPrefixProof(entries=(
        _prefix_entry(prefix[0]),
        _prefix_entry(impostor, receipt_id="r-old", run_id="run-old"),
        _prefix_entry(prefix[2]),
    ))
    with pytest.raises(PrefixViolation, match="lost its receipt identity"):
        lost_proof.validate((prefix[0], impostor, prefix[2]))


async def test_prefix_walks_historical_path_without_current_ledger_latch():
    """A09: 前缀内旧回执不走当前账本核对 (不 ghost 锁存), 正文一字不改."""
    prefix = (UserMessage(text="旧问"), check_result("c1", "旧检查通过", receipt_id="r-old"))
    proof_entries = (
        _prefix_entry(prefix[0]),
        _prefix_entry(prefix[1], receipt_id="r-old", run_id="run-old"),
    )
    history = HistoricalProjection()
    history.proof = type(history.proof)(entries=proof_entries)

    model = RecordingModel()
    service = FakeVerificationService(make_view())
    policy = RunPolicy(model, service, history=history)
    events = [
        event
        async for event in policy.stream(
            ModelRequest(tuple(prefix) + (UserMessage(text="新问"),), ()), CancelToken(),
        )
    ]
    assert isinstance(events[-1], StreamDone)
    assert service.uncertain_reasons == [], "前缀不按当前账本锁存"
    sent = model.requests[0]
    projected_prefix = [m for m in sent.messages if isinstance(m, ToolResultMessage)][0]
    assert projected_prefix.result.content == "旧检查通过", "恢复旧文字不清洗不标注"
    # 原消息不改: 投影只发生在发给模型的副本上
    assert "旧检查通过" in prefix[1].result.content


async def test_suffix_copying_old_receipt_field_by_field_still_ghost_latches():
    """A09: 新后缀逐字段复制旧回执也不获历史豁免 —— 仍按当前账本锁存."""
    prefix = (UserMessage(text="旧问"), check_result("c1", "旧检查通过", receipt_id="r-old"))
    history = HistoricalProjection()
    history.proof = type(history.proof)(entries=(
        _prefix_entry(prefix[0]),
        _prefix_entry(prefix[1], receipt_id="r-old", run_id="run-old"),
    ))
    # 逐字段复制: 同 call_id/content/details/receipt_id 的独立副本 (仅对象不同)
    copied_suffix = isolate_message(prefix[1])
    assert copied_suffix is not prefix[1]
    model = RecordingModel()
    service = FakeVerificationService(make_view())
    policy = RunPolicy(model, service, history=history)
    events = [
        event
        async for event in policy.stream(
            ModelRequest(tuple(prefix) + (UserMessage(text="新问"), copied_suffix), ()),
            CancelToken(),
        )
    ]
    assert isinstance(events[-1], StreamDone)
    assert any("not in the verification ledger" in r for r in service.uncertain_reasons), (
        "后缀复制旧回执必须按当前账本 ghost 锁存"
    )


async def test_prefix_violation_refuses_before_any_request():
    """A09: 前缀被改动 -> 发送前拒绝, 内层模型零请求."""
    prefix = (UserMessage(text="旧问"),)
    history = HistoricalProjection()
    history.proof = type(history.proof)(entries=(_prefix_entry(prefix[0]),))
    model = RecordingModel()
    service = FakeVerificationService(make_view())
    policy = RunPolicy(model, service, history=history)
    with pytest.raises(PrefixViolation, match="changed since the proof was built"):
        async for _ in policy.stream(
            ModelRequest((UserMessage(text="被改写"),), ()), CancelToken(),
        ):
            pass
    assert model.requests == []
    assert service.refresh_calls == []


async def test_projection_reset_returns_all_messages_to_current_rules():
    """清空重置后前缀为空: 全部消息按当前规则扫描/核对."""
    history = HistoricalProjection()
    history.proof = type(history.proof)(entries=(
        _prefix_entry(UserMessage(text="旧问")),),
    )
    history.reset()
    assert history.proof.length == 0
    ghost = check_result("c1", "旧回执现在按当前账本核对", receipt_id="r-old")
    model = RecordingModel()
    service = FakeVerificationService(make_view())
    policy = RunPolicy(model, service, history=history)
    events = [
        event
        async for event in policy.stream(ModelRequest((UserMessage(text="问"), ghost), ()), CancelToken())
    ]
    assert isinstance(events[-1], StreamDone)
    assert any("not in the verification ledger" in r for r in service.uncertain_reasons)
