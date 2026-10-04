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

# --- 版本与限额 (contracts §2; 调整须新版本, 不在同一版本默默改变语义) -----------------

SESSION_SCHEMA_VERSION = 1
# SQLite application_id: "CIC1" 的 32 位编码, 区分 Cicada 会话库
SESSION_APPLICATION_ID = 0x43494331

MAX_DB_BYTES = 256 * 1024 * 1024          # 数据库主文件; journal 额外空间不含在内
MAX_REQUEST_PAYLOAD_BYTES = 8 * 1024 * 1024   # 单个完整请求有效载荷
MAX_METADATA_BYTES = 64 * 1024            # 单条元数据/intent/配置记录
MAX_COMMITTED_REQUESTS = 10_000           # 完整请求数上限
MAX_MESSAGES = 200_000                    # 会话内消息总数上限
MAX_JSON_DEPTH = 32                       # 嵌套 JSON 深度上限

SHORT_TITLE_MAX_CHARS = 80                # 列表短标题的有界文字


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
    """存档版本高于或不同于本程序支持的 v1."""


class ArchiveLimitError(PersistenceError):
    """超出 v1 限额: 打开或写入事务前拒绝, 不静默截断."""


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


@dataclass(frozen=True)
class RequestPersistenceFacts:
    """单请求持久结果; RequestOutcome 的尾随字段, 默认 None 保持既有语义."""

    request_id: str
    saved: bool
    seq: int | None = None
    error: str | None = None


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
                             outcome: RequestOutcomeFacts) -> int:
        """完整请求单事务提交; 成功返回 records 序号. 重复提交 (含同内容) 一律拒绝."""
        ...

    async def commit_clear(self, epoch_id: str) -> int:
        """可靠提交清空边界; 成功返回新的上下文代次."""
        ...

    async def aclose(self) -> None:
        """排空已派发事务后关闭连接并释放 lease; 不得提前释放仍在使用的资源."""
        ...
