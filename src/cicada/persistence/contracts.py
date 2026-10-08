"""持久化契约: 版本/限额常量、错误分类、不可变 DTO 与存储 port.

本模块是中立契约层: 只描述"存什么、错误怎么分", 不执行 SQL, 不导入
boot/runtime/插件实现. 实际签名以第一票共享交接回执为准.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import Message, StopReason
from cicada.core.ports import ModelMetrics
from cicada.models import SafeModelIdentity

# --- 版本与限额 (contracts §2; 调整须新版本, 不在同一版本默默改变语义) -----------------

SESSION_SCHEMA_VERSION = 3
# SQLite application_id: "CIC1" 的 32 位编码, 区分 Cicada 会话库
SESSION_APPLICATION_ID = 0x43494331

MAX_DB_BYTES = 256 * 1024 * 1024          # 数据库主文件; journal 额外空间不含在内
MAX_REQUEST_PAYLOAD_BYTES = 8 * 1024 * 1024   # 单个完整请求有效载荷
MAX_METADATA_BYTES = 64 * 1024            # 单条元数据/intent/配置记录
MAX_COMMITTED_REQUESTS = 10_000           # 完整请求数上限
MAX_MESSAGES = 200_000                    # 会话内消息总数上限
MAX_JSON_DEPTH = 32                       # 嵌套 JSON 深度上限

SHORT_TITLE_MAX_CHARS = 80                # 列表短标题的有界文字

# 03: 回执行标识/说明字段的有界长度; 写侧构造与加载侧解码对称拒绝
RECEIPT_ID_MAX_CHARS = 256               # receipt_id/verification_run_id/epoch_id/request_id
RECEIPT_FAILURE_KIND_MAX_CHARS = 512     # failure_kind 说明文本

# 02 (schema v2): 摘要记录限额 (spec D8)
SUMMARY_TEXT_MAX_BYTES = 16 * 1024        # 摘要正文有界长度
MAX_SUMMARIES_PER_GENERATION = 128        # 每个上下文代次的摘要记录数上限
SUMMARY_ID_MAX_CHARS = 256                # summary_id/摘要提示版本等标识有界长度


# --- 错误分类 ---------------------------------------------------------------------------


class PersistenceError(RuntimeError):
    """持久化层错误基类; 异常文本面向用户显示."""


class StorageUnavailableError(PersistenceError):
    """无法打开或占用存储: 文件缺失、无法创建、IO 打开失败."""


class LeaseHeldError(StorageUnavailableError):
    """单写入者冲突: 另一进程仍持有会话 lease."""


class SessionNotFoundError(StorageUnavailableError):
    """按 ID 恢复时存档不存在."""


class ArchiveFormatError(PersistenceError):
    """存档坏格式/关系损坏: 拒绝恢复, 不改写/截尾/删除原文件."""


class UnknownArchiveVersionError(ArchiveFormatError):
    """存档版本高于或不同于本程序支持的版本."""


class ArchiveLimitError(PersistenceError):
    """超出会话限额: 打开或写入事务前拒绝, 不静默截断."""


class WorkspaceMismatchError(PersistenceError):
    """存档绑定的工作区与当前解析路径不同."""


class ConfigConflictError(PersistenceError):
    """恢复配置与存档不一致; fields 列出冲突字段."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = tuple(fields)
        super().__init__(f"restored session config conflicts with the current launch: "
                         f"{', '.join(self.fields)}")


class StorageWriteError(PersistenceError):
    """begin/commit/clear 落盘失败: 调用方据此锁存 storage_failed."""


# --- 启动身份 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchIdentity:
    """启动层给出的有效配置身份; 04 前恢复要求与存档完全一致.

    只含 CLI 可表达的有效值 (默认值已解析, 不含 argparse 原始旗标);
    sources 记录显式来源 (如 num_ctx=cli/default), 供 04 的差异记录, 不参与 01 比对.
    options/sources 在构造时复制为普通 dict, 编码时按严格 JSON 校验.
    """

    provider: str
    model: str
    base_url: str | None
    think: bool | None
    options: Mapping[str, object] = field(default_factory=dict)
    max_request_bytes: int | None = None
    max_turns: int = 50
    prompt_version: str = ""
    prompt_sha256: str = ""
    instructions_sha256: str | None = None
    checks_signature: str | None = None
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("provider", "model", "prompt_version", "prompt_sha256"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"LaunchIdentity.{name} must be a str")
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ValueError("LaunchIdentity.base_url must be a str or None")
        if self.think is not None and not isinstance(self.think, bool):
            raise ValueError("LaunchIdentity.think must be a bool or None")
        if not isinstance(self.options, Mapping):
            raise ValueError("LaunchIdentity.options must be a Mapping")
        if not isinstance(self.sources, Mapping):
            raise ValueError("LaunchIdentity.sources must be a Mapping")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.sources.items()):
            raise ValueError("LaunchIdentity.sources must map str to str")
        for name in ("max_request_bytes", "max_turns"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"LaunchIdentity.{name} must be an int or None")
        if self.max_turns <= 0:
            raise ValueError("LaunchIdentity.max_turns must be positive")
        for name in ("instructions_sha256", "checks_signature"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"LaunchIdentity.{name} must be a str or None")

    def comparable_dict(self) -> dict[str, object]:
        """参与同配置比对的规范化视图; options 展开为可比较的普通 dict."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "think": self.think,
            "options": dict(self.options),
            "max_request_bytes": self.max_request_bytes,
            "max_turns": self.max_turns,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "instructions_sha256": self.instructions_sha256,
            "checks_signature": self.checks_signature,
        }


@dataclass(frozen=True)
class StoredIdentity:
    """写入 epoch 的完整身份: 启动身份 + bootstrap 后的实际工具声明指纹."""

    launch: LaunchIdentity
    tool_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.launch, LaunchIdentity):
            raise ValueError("StoredIdentity.launch must be a LaunchIdentity")
        if not isinstance(self.tool_fingerprint, str) or not self.tool_fingerprint:
            raise ValueError("StoredIdentity.tool_fingerprint must be a nonempty str")


def conflicting_launch_fields(
    saved: LaunchIdentity, current: LaunchIdentity
) -> tuple[str, ...]:
    """逐字段比较两份启动身份, 返回冲突字段名; 空元组表示完全一致.

    options 按整体 dict 比较; sources 只作来源记录, 不参与比对 (04 再裁决来源语义).
    """
    saved_view = saved.comparable_dict()
    current_view = current.comparable_dict()
    return tuple(
        name for name in saved_view if saved_view[name] != current_view[name]
    )


# --- 请求与恢复事实 ---------------------------------------------------------------------


@dataclass(frozen=True)
class RequestOutcomeFacts:
    """单个已提交请求的内核终结事实与应用收尾描述; 可序列化, 不含检查回执结构."""

    run_id: str
    stop_reason: StopReason
    error: str | None
    exit_code: int
    finalized: bool                     # 应用收尾 (交付判定段) 是否完成
    finalization_notes: tuple[str, ...] = ()
    model_calls: tuple[ModelCallCompleted, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("RequestOutcomeFacts.run_id must be a nonempty str")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise ValueError("RequestOutcomeFacts.exit_code must be an int")
        if not isinstance(self.finalized, bool):
            raise ValueError("RequestOutcomeFacts.finalized must be a bool")
        for call in self.model_calls:
            if not isinstance(call, ModelCallCompleted):
                raise ValueError("RequestOutcomeFacts.model_calls must hold ModelCallCompleted")


@dataclass(frozen=True)
class RestoredRequest:
    """恢复快照中的一条已提交完整请求; messages 为闭合后缀 (含首条用户消息)."""

    request_id: str
    epoch_id: str
    seq: int
    context_generation: int
    prompt: str
    messages: tuple[Message, ...]
    outcome: RequestOutcomeFacts
    committed_at: float


@dataclass(frozen=True)
class InterruptedRequest:
    """未完成 intent (pending/interrupted); 只作诊断与显示, 不进入可续写历史."""

    request_id: str
    epoch_id: str
    prompt: str
    status: Literal["pending", "interrupted"]


# --- 进程活动 (02; 契约 §4) ----------------------------------------------------------------


TERMINAL_KIND = Literal["proven_terminal", "not_started", "unproven_terminal"]


@dataclass(frozen=True)
class ProcessTerminalFacts:
    """命令终结的结构化事实; 包装器按既有 P4 分类写入, 存档只存事实不重判."""

    kind: TERMINAL_KIND
    exit_code: int | None = None
    output_complete: bool | None = None
    timed_out: bool = False
    cancelled: bool = False
    receipt_id: str | None = None      # check(action=run) 保留身份; 03 用真实账本核对
    reason: str | None = None          # not_started/unproven 的来源说明

    def __post_init__(self) -> None:
        if self.kind not in ("proven_terminal", "not_started", "unproven_terminal"):
            raise ValueError("ProcessTerminalFacts.kind must be a known terminal kind")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("ProcessTerminalFacts.exit_code must be an int or None")
        if self.output_complete is not None and not isinstance(self.output_complete, bool):
            raise ValueError("ProcessTerminalFacts.output_complete must be a bool or None")
        if not isinstance(self.timed_out, bool) or not isinstance(self.cancelled, bool):
            raise ValueError("ProcessTerminalFacts.timed_out/cancelled must be bools")
        if self.receipt_id is not None and (not isinstance(self.receipt_id, str) or not self.receipt_id):
            raise ValueError("ProcessTerminalFacts.receipt_id must be a nonempty str or None")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("ProcessTerminalFacts.reason must be a str or None")


@dataclass(frozen=True)
class ProcessActivityFacts:
    """恢复快照中的一条进程活动; terminal=None 表示已登记但没有可靠终结记录."""

    activity_id: str
    request_id: str
    tool_name: str
    call_id: str
    arguments_digest: str
    description: str
    started_at: float
    terminal: ProcessTerminalFacts | None


# --- 历史检查回执 (03; 契约 §5) -------------------------------------------------------------


@dataclass(frozen=True)
class StoredReceipt:
    """历史检查回执的持久事实; 只作来源核对, 不构成当前验证权威.

    definition_fingerprint 是检查定义 (check_id/command/timeout) 的稳定摘要;
    终结字段与 P4 CheckReceipt 同语义, 经严格 codec 存取.
    """

    verification_run_id: str
    receipt_id: str
    epoch_id: str
    request_id: str
    definition_fingerprint: str
    execution_status: str          # exited/launch_failed/timed_out/cancelled/error
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    output_complete: bool
    failure_kind: str | None
    activity_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("verification_run_id", "receipt_id", "epoch_id", "request_id",
                     "definition_fingerprint", "execution_status"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"StoredReceipt.{name} must be a nonempty str")
        for name in ("verification_run_id", "receipt_id", "epoch_id", "request_id"):
            if len(getattr(self, name)) > RECEIPT_ID_MAX_CHARS:
                raise ValueError(
                    f"StoredReceipt.{name} exceeds the limit ({RECEIPT_ID_MAX_CHARS} chars)"
                )
        if self.execution_status not in (
            "exited", "launch_failed", "timed_out", "cancelled", "error",
        ):
            raise ValueError("StoredReceipt.execution_status must be a known status")
        if len(self.definition_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in self.definition_fingerprint
        ):
            raise ValueError("StoredReceipt.definition_fingerprint must be a sha256 hex digest")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("StoredReceipt.exit_code must be an int or None")
        for name in ("timed_out", "cancelled", "output_complete"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"StoredReceipt.{name} must be a bool")
        if self.failure_kind is not None and not isinstance(self.failure_kind, str):
            raise ValueError("StoredReceipt.failure_kind must be a str or None")
        if self.failure_kind is not None and len(self.failure_kind) > RECEIPT_FAILURE_KIND_MAX_CHARS:
            raise ValueError(
                "StoredReceipt.failure_kind exceeds the limit "
                f"({RECEIPT_FAILURE_KIND_MAX_CHARS} chars)"
            )
        if self.activity_id is not None and not isinstance(self.activity_id, str):
            raise ValueError("StoredReceipt.activity_id must be a str or None")


def receipt_terminal_trustworthy(receipt: StoredReceipt) -> bool:
    """回执自身是否记录可靠终结事实; 判定与 run_policy._receipt_terminates 同语义.

    存档中存在不可信终结的回执 = 存档已记录 process_uncertain 的证据 (契约 D3),
    恢复为查看状态.
    """
    if receipt.cancelled or receipt.timed_out:
        return False
    if receipt.execution_status == "launch_failed":
        return receipt.exit_code is None
    return (
        receipt.execution_status == "exited"
        and receipt.output_complete
        and receipt.exit_code is not None
    )


# --- 摘要记录 (02, schema v2; spec D6/D8) ----------------------------------------------------


@dataclass(frozen=True)
class SummaryRecord:
    """一条已提交摘要的持久事实; 投影切换的可靠提交凭据, 不携带验证权威.

    覆盖范围: 本块自身覆盖规范历史 [start, end) 的消息 (message_count 条),
    覆盖的完整请求 seq 范围 [first_request_seq, last_request_seq];
    prefix_sha256 是规范历史 [0:end) 全前缀的规范编码 sha256 (恢复逐块核对);
    end_message_index 即保留边界 (保留后缀的规范历史起点)。
    """

    summary_id: str
    epoch_id: str
    context_generation: int
    first_request_seq: int
    last_request_seq: int
    message_count: int
    end_message_index: int
    prefix_sha256: str
    text: str
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    metrics: ModelMetrics | None
    reason: Literal["manual", "auto"]
    created_at: float
    previous_projection_identity: str | None = None
    call_identity: SafeModelIdentity | None = None

    def __post_init__(self) -> None:
        if self.call_identity is not None and not isinstance(self.call_identity, SafeModelIdentity):
            raise ValueError("SummaryRecord.call_identity must be SafeModelIdentity or None")
        for name in ("summary_id", "epoch_id", "provider", "model", "prompt_version", "prompt_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"SummaryRecord.{name} must be a nonempty str")
            if len(value) > SUMMARY_ID_MAX_CHARS:
                raise ValueError(
                    f"SummaryRecord.{name} exceeds the limit ({SUMMARY_ID_MAX_CHARS} chars)"
                )
        for name in ("context_generation", "first_request_seq", "last_request_seq",
                     "message_count", "end_message_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"SummaryRecord.{name} must be a nonnegative int")
        if self.first_request_seq > self.last_request_seq:
            raise ValueError("SummaryRecord.first_request_seq must not exceed last_request_seq")
        if self.message_count <= 0:
            raise ValueError("SummaryRecord.message_count must be positive")
        if self.end_message_index < self.message_count:
            raise ValueError(
                "SummaryRecord.end_message_index must be at least the covered message count"
            )
        if (
            not isinstance(self.prefix_sha256, str)
            or len(self.prefix_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.prefix_sha256)
        ):
            raise ValueError("SummaryRecord.prefix_sha256 must be a sha256 hex digest")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("SummaryRecord.text must be a nonempty str")
        if len(self.text.encode("utf-8")) > SUMMARY_TEXT_MAX_BYTES:
            raise ValueError(
                f"SummaryRecord.text exceeds the limit ({SUMMARY_TEXT_MAX_BYTES} bytes)"
            )
        if self.metrics is not None and not isinstance(self.metrics, ModelMetrics):
            raise ValueError("SummaryRecord.metrics must be ModelMetrics or None")
        if self.reason not in ("manual", "auto"):
            raise ValueError("SummaryRecord.reason must be 'manual' or 'auto'")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, (int, float)):
            raise ValueError("SummaryRecord.created_at must be a number")
        if self.previous_projection_identity is not None and (
            not isinstance(self.previous_projection_identity, str)
            or len(self.previous_projection_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.previous_projection_identity)
        ):
            raise ValueError(
                "SummaryRecord.previous_projection_identity must be a sha256 hex digest or None"
            )


@dataclass(frozen=True)
class RecoverySnapshot:
    """打开存档后的隔离恢复快照; 不含 live 插件对象, 嵌套 details 已复制."""

    session_id: str
    canonical_workspace: str
    created_at: float
    short_title: str
    identity: StoredIdentity | None
    epoch_count: int
    context_generation: int
    requests: tuple[RestoredRequest, ...]          # 已提交完整请求 (全部代次)
    interrupted: tuple[InterruptedRequest, ...]
    canonical_history: tuple[Message, ...]          # 当前代次的闭合历史 (已过结构校验)
    process_activities: tuple[ProcessActivityFacts, ...] = ()   # 全部代次的进程活动
    restored_receipts: tuple[StoredReceipt, ...] = ()           # 历史检查回执 (03)
    summaries: tuple[SummaryRecord, ...] = ()                   # 当前代次摘要记录 (02, v2)

    @property
    def recorded_process_uncertain(self) -> bool:
        """存档是否记录了不可信终结的检查回执 (跨恢复 process_uncertain 证据)."""
        return any(not receipt_terminal_trustworthy(r) for r in self.restored_receipts)

    @property
    def has_unproven_process_activity(self) -> bool:
        """存在已登记但终结未证 (或完全缺失终结记录) 的活动 -> 恢复为查看状态 (契约 D3)."""
        return any(
            activity.terminal is None or activity.terminal.kind == "unproven_terminal"
            for activity in self.process_activities
        )

    def unproven_process_activities(self) -> tuple[ProcessActivityFacts, ...]:
        """列出阻断续写的活动, 供拒绝理由点名; 顺序与存档加载顺序一致."""
        return tuple(
            activity for activity in self.process_activities
            if activity.terminal is None or activity.terminal.kind == "unproven_terminal"
        )


@dataclass(frozen=True)
class RequestPersistenceFacts:
    """单请求持久结果; RequestOutcome 的尾随字段, 默认 None 保持既有语义."""

    request_id: str
    saved: bool
    seq: int | None = None
    error: str | None = None


# --- 会话列表 (05; 契约 §3 list_sessions) ---------------------------------------------------

LIST_SESSIONS_PAGE_SIZE = 25


@dataclass(frozen=True)
class SessionSummary:
    """列表中单个会话的有界事实; error 非空 = 坏档/不可读 (其余字段仅部分可信).

    blocked 是恢复阻断的证据聚合: 未证终结/未闭合活动或不可信终结回执
    (与恢复准入 view-only 派生同源), 列表只作标注, 不替代打开时校验.
    """

    session_id: str
    short_title: str
    created_at: float
    updated_at: float | None          # 最近提交记录时间; 无记录为 None
    committed_requests: int
    interrupted_requests: int         # pending/interrupted intent 总数
    occupied: bool                    # lease 疑似被其他进程持有
    unproven_activities: int
    untrusted_receipts: int
    error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.unproven_activities > 0 or self.untrusted_receipts > 0


@dataclass(frozen=True)
class SessionPage:
    """有界分页列表; next_cursor 非空表示还有更多条目 (不声称其余不存在)."""

    directory: str
    entries: tuple[SessionSummary, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class PersistenceState:
    """应用暴露的只读持久状态; 字段为打开/保存到当前时刻的事实快照."""

    session_id: str
    epoch_id: str
    mode: Literal["save", "resume"]
    context_generation: int
    restored_request_count: int
    restored_model_call_count: int
    saved_request_count: int
    last_committed_seq: int | None
    storage_failed: str | None       # 非空 = 已锁存 storage_failed, 拒绝后续编码请求


# --- 存储 port --------------------------------------------------------------------------


@runtime_checkable
class SessionStorePort(Protocol):
    """单会话存储 port; 实现由 session_store 能力插件提供, 应用持直接引用.

    阻塞操作为 async: 实现在自有串行 worker 中执行 SQLite; 调用方取消不取消
    已派发事务, 每个事务必得明确结果 (成功或 PersistenceError).
    """

    @property
    def session_id(self) -> str: ...

    def recovery_snapshot(self) -> RecoverySnapshot:
        """隔离的恢复快照; 打开后可随时读取, 反映已提交事实."""
        ...

    async def start_epoch(self, identity: StoredIdentity) -> str:
        """可靠写入新的启动代次; 同时把此前 pending intent 标为 interrupted."""
        ...

    async def begin_request(self, epoch_id: str, prompt: str) -> str:
        """可靠提交请求开始 intent; 返回 request_id. 失败零模型调用."""
        ...

    async def commit_request(self, request_id: str, *, messages: tuple[Message, ...],
                             outcome: RequestOutcomeFacts,
                             receipts: tuple[StoredReceipt, ...] = ()) -> int:
        """完整请求单事务提交; 成功返回 records 序号. 重复提交 (含同内容) 一律拒绝.

        03 起 receipts 在同一事务写入 historical_receipts (检查模式); 既有调用
        不传则行为不变. 全有或全无: 回执行与请求记录、intent 状态同事务落库.
        """
        ...

    async def begin_process(self, request_id: str, *, tool_name: str, call_id: str,
                            arguments: Mapping[str, object]) -> str:
        """命令真正执行前可靠登记活动开始, 返回 activity_id.

        失败时调用方零命令执行并锁存 storage_failed; request_id 必须是当前
        pending intent. 02 起属于 port; 01 存档 (v1 schema) 已含 process_activity 表.
        """
        ...

    async def end_process(self, activity_id: str, terminal: ProcessTerminalFacts) -> None:
        """登记结构化终结事实; 开始/终结关联错误拒绝; 写入失败保留真实工具结果."""
        ...

    async def commit_clear(self, epoch_id: str) -> int:
        """可靠提交清空边界; 成功返回新的上下文代次."""
        ...

    async def commit_summary(self, record: SummaryRecord) -> None:
        """可靠提交摘要记录 (02, schema v2); 成功即 store 已接受 (明确结果).

        关系校验: epoch/generation 必须是当前打开值; summary_id 重复拒绝;
        seq 范围必须引用当前代次已提交完整请求; 每代次摘要数限额。失败抛
        PersistenceError, 调用方按真实存储故障处理; 与请求提交同一 worker 纪律
        (调用方取消不取消事务, 每次派发必得明确结果)。
        """
        ...

    async def aclose(self) -> None:
        """排空已派发事务后关闭连接并释放 lease; 不得提前释放仍在使用的资源."""
        ...
