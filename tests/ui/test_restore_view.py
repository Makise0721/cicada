"""票据 05: 恢复历史的只读显示投影聚焦检查 (A13 来源定位/计量重建/归属可读)."""

from __future__ import annotations

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelMetrics
from cicada.persistence.contracts import (
    InterruptedRequest,
    RequestOutcomeFacts,
    RestoredRequest,
)
from cicada.ui.restore_view import CLEAR_BOUNDARY_TEXT, project_restored_history
from cicada.ui.tool_view import ToolDocument
from cicada.ui.transcript import Transcript


def _request(
    *, request_id: str, epoch_id: str = "e" * 36, seq: int = 1, generation: int = 0,
    messages: tuple = (), stop_reason: str = "stop", error: str | None = None,
    model_calls: tuple = (),
) -> RestoredRequest:
    return RestoredRequest(
        request_id=request_id,
        epoch_id=epoch_id,
        seq=seq,
        context_generation=generation,
        prompt="第一问",
        messages=messages,
        outcome=RequestOutcomeFacts(
            run_id="run-1", stop_reason=stop_reason, error=error, exit_code=0,
            finalized=True, model_calls=model_calls,
        ),
        committed_at=1.0,
    )


def test_empty_snapshot_projects_nothing():
    transcript, tools = Transcript(), ToolDocument()
    project_restored_history(transcript, tools, requests=(), interrupted=())
    assert transcript.entries() == []


def test_restored_request_projects_user_assistant_and_metrics():
    transcript, tools = Transcript(), ToolDocument()
    request = _request(
        request_id="r-1",
        messages=(
            UserMessage(text="第一问"),
            AssistantMessage(text="恢复后的回答", stop_reason="stop"),
        ),
        model_calls=(ModelCallCompleted(
            run_id="run-1", turn_index=0, stop_reason="stop",
            metrics=ModelMetrics(input_tokens=10, output_tokens=5), elapsed_s=0.1,
        ),),
    )
    project_restored_history(transcript, tools, requests=(request,), interrupted=())
    rendered = "\n".join(transcript.render(60))
    assert "已恢复历史：1 条请求" in rendered
    assert "历史请求 #1 · epoch" in rendered
    assert "恢复后的回答" in rendered
    assert "累计输入 10" in transcript.metrics_summary(), "历史计量一次重建"
    assert "不可得" not in transcript.metrics_summary()


def test_restored_metrics_none_stays_unknown():
    transcript, tools = Transcript(), ToolDocument()
    request = _request(
        request_id="r-1",
        messages=(UserMessage(text="第一问"), AssistantMessage(text="答")),
        model_calls=(ModelCallCompleted(
            run_id="run-1", turn_index=0, stop_reason="stop", metrics=None, elapsed_s=0.1,
        ),),
    )
    project_restored_history(transcript, tools, requests=(request,), interrupted=())
    assert "不可得" in transcript.metrics_summary(), "None 计量不当作 0"


def test_tool_cards_namespaced_and_check_attribution():
    transcript, tools = Transcript(), ToolDocument()
    details = {"receipt_id": "chk-abc123", "verification_run_id": "vr-xyz789"}
    request = _request(
        request_id="r-1",
        messages=(
            UserMessage(text="问"),
            AssistantMessage(text="", tool_calls=(
                ToolCall(id="c1", name="check", arguments_json='{"check_id": "check-1"}'),
            )),
            ToolResultMessage(result=ToolResult(
                "c1", "check", "PASS", details=details,
            )),
        ),
    )
    project_restored_history(transcript, tools, requests=(request,), interrupted=())
    entries = tools.entries
    assert len(entries) == 1
    assert entries[0].run_id.startswith("restored:r-1"), "工具卡键命名空间化"
    assert entries[0].status == "success"
    rendered = "\n".join(transcript.render(80))
    assert "历史检查回执 chk-abc123" in rendered, "旧检查归属可读"
    assert "不适用于当前验证运行" in rendered


def test_same_call_id_across_requests_does_not_merge():
    transcript, tools = Transcript(), ToolDocument()
    messages = (
        UserMessage(text="问"),
        AssistantMessage(text="", tool_calls=(ToolCall(id="c1", name="read", arguments_json="{}"),)),
        ToolResultMessage(result=ToolResult("c1", "read", "正文")),
    )
    request_one = _request(request_id="r-1", seq=1, messages=messages)
    request_two = _request(request_id="r-2", seq=2, messages=messages)
    project_restored_history(transcript, tools, requests=(request_one, request_two), interrupted=())
    entries = tools.entries
    assert len(entries) == 2, "跨组同 call_id 不合并"
    assert entries[0].run_id != entries[1].run_id


def test_generation_change_inserts_clear_boundary():
    transcript, tools = Transcript(), ToolDocument()
    request_one = _request(
        request_id="r-1", generation=0,
        messages=(UserMessage(text="一"), AssistantMessage(text="答一")),
    )
    request_two = _request(
        request_id="r-2", seq=2, generation=1,
        messages=(UserMessage(text="二"), AssistantMessage(text="答二")),
    )
    project_restored_history(transcript, tools, requests=(request_one, request_two), interrupted=())
    rendered = "\n".join(transcript.render(60))
    assert CLEAR_BOUNDARY_TEXT in rendered
    assert rendered.index("答一") < rendered.index(CLEAR_BOUNDARY_TEXT) < rendered.index("答二")


def test_aborted_and_error_outcomes_projected():
    transcript, tools = Transcript(), ToolDocument()
    aborted = _request(
        request_id="r-1", stop_reason="aborted",
        messages=(UserMessage(text="问"),),
    )
    failed = _request(
        request_id="r-2", seq=2, stop_reason="error", error="boom",
        messages=(UserMessage(text="问二"),),
    )
    project_restored_history(transcript, tools, requests=(aborted, failed), interrupted=())
    rendered = "\n".join(transcript.render(60))
    assert "已取消" in rendered and "boom" in rendered


def test_interrupted_requests_listed_with_bounded_prompt():
    transcript, tools = Transcript(), ToolDocument()
    interrupted = (InterruptedRequest(
        request_id="r-i", epoch_id="e" * 36, prompt="未完成" * 40, status="pending",
    ),)
    project_restored_history(transcript, tools, requests=(), interrupted=interrupted)
    rendered = "\n".join(transcript.render(60))
    assert "1 条未完成" in rendered
    assert "中断请求（未完成）" in rendered
    assert all(len(line) <= 60 for line in rendered.splitlines()), "显示行有界"
