"""应用内配置、结果、事件和异常；不拥有运行中的会话。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cicada.budget import BudgetConfig
from cicada.core.agent import RunResult
from cicada.core.cancel import CancelToken
from cicada.core.events import Event
from cicada.costs import CostLedger
from cicada.models import ReasoningEffortState, SafeModelIdentity
from cicada.user_config import LoginResult
from cicada.persistence.contracts import (
    LaunchIdentity,
    PersistenceState,
    RecoverySnapshot,
    RequestPersistenceFacts,
)
from cicada.restore_points import RewindRestoreResult
from cicada.plugins.coding.verification_contracts import (
    ChangeEvidence,
    VerificationPlan,
    VerificationView,
)
from cicada.prompting import PromptInfo
from cicada.reporting import DeliveryDecision
from cicada.runtime.plugin import PluginDefinition
from cicada.skills import SkillCatalogSnapshot


ApplicationPhase = Literal["executing", "cancelling", "finalizing", "idle"]


SessionStateName = Literal[
    "idle", "executing", "cancelling", "finalizing", "closing", "closed"
]


@dataclass(frozen=True)
class ApplicationConfig:
    """不可变启动配置; CLI 保留参数解析与 fake JSON 转换, open 接收已校验结构."""

    workspace: Path
    system_prompt: str
    prompt_info: PromptInfo
    model_plugin: PluginDefinition
    plan: VerificationPlan | None = None
    max_turns: int = 50
    model_label: str = ""
    persistence: PersistenceConfig | None = None
    # 01 尾随扩展: 显式预算配置才启用预算层 (D9); fake 默认 None, CLI 由 num_ctx 实填
    budget: BudgetConfig | None = None
    # 02 尾随扩展: 启用 Skills 的冻结快照; None = 旧行为逐位保持
    skills: SkillCatalogSnapshot | None = None
    # 请求级快照与撤回 01 (C2): restore_capture 默认开启; 关闭时不安装捕获汇,
    # 编辑照常工作且不产生还原点 (受限环境出口)
    restore_capture: bool = True
    # 执行安全边界 P0-1: 只读档只注册 read/glob/grep, 不提供写入或任意命令
    read_only: bool = False
    # P0-1 审批档: 仅 powershell 在启动进程前等待用户决定; 默认仍为 unrestricted
    approval: bool = False
    cost_control: CostLedger | None = None
    runtime_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise ValueError("workspace must be a Path")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a str")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if self.plan is not None and not isinstance(self.plan, VerificationPlan):
            raise ValueError("plan must be a VerificationPlan or None")
        if not isinstance(self.restore_capture, bool):
            raise ValueError("restore_capture must be a bool")
        if not isinstance(self.read_only, bool):
            raise ValueError("read_only must be a bool")
        if not isinstance(self.approval, bool):
            raise ValueError("approval must be a bool")
        if self.read_only and self.plan is not None:
            raise ValueError("read_only cannot be combined with a verification plan")
        if self.read_only and self.approval:
            raise ValueError("read_only cannot be combined with approval mode")
        if self.cost_control is not None and not isinstance(self.cost_control, CostLedger):
            raise ValueError('cost_control must be a CostLedger or None')
        if self.runtime_dir is not None and not isinstance(self.runtime_dir, Path):
            raise ValueError('runtime_dir must be a Path or None')
        if self.persistence is not None:
            if not isinstance(self.persistence, PersistenceConfig):
                raise ValueError("persistence must be a PersistenceConfig or None")
            # 03 起检查模式 + 持久会话组合放开 (票据 03); 恢复要求 checks_signature 一致
        if self.budget is not None and not isinstance(self.budget, BudgetConfig):
            raise ValueError("budget must be a BudgetConfig or None")
        if self.skills is not None and not isinstance(self.skills, SkillCatalogSnapshot):
            raise ValueError("skills must be a SkillCatalogSnapshot or None")


@dataclass(frozen=True)
class PersistenceConfig:
    """持久会话入口配置; session_id=None 表示创建新会话, 否则按精确 ID 恢复.

    identity 是启动层给出的有效配置身份 (默认值已解析)。默认恢复要求 identity
    与存档一致; adopt_current_config=True (--resume-with-current-config) 显式
    采用当前配置: 身份/工具指纹差异不拒绝, 差异记入新 epoch 的身份来源, 不解除
    view-only/process_uncertain 阻断 (04)。
    """

    session_id: str | None
    identity: LaunchIdentity
    adopt_current_config: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LaunchIdentity):
            raise ValueError("PersistenceConfig.identity must be a LaunchIdentity")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise ValueError("PersistenceConfig.session_id must be a str or None")
        if not isinstance(self.adopt_current_config, bool):
            raise ValueError("PersistenceConfig.adopt_current_config must be a bool")


@dataclass(frozen=True)
class RequestOutcome:
    """单个用户请求的权威终结; 复用现有记录, 不另造第二份检查回执."""

    result: RunResult
    view: VerificationView | None
    evidence: ChangeEvidence | None
    decision: DeliveryDecision | None
    exit_code: int
    persistence: RequestPersistenceFacts | None = None


@dataclass(frozen=True)
class AgentEventReceived:
    """包装原内核事件; 不改写."""

    event: Event


@dataclass(frozen=True)
class RequestPhaseChanged:
    """应用阶段变化; 首个 run ID 由 RunStarted 确认, 此前允许 run_id=None."""

    run_id: str | None
    phase: ApplicationPhase


@dataclass(frozen=True)
class RequestCompleted:
    """验证收尾后的唯一请求终结; 正常状态下收到时 session 已回 idle."""

    outcome: RequestOutcome


@dataclass(frozen=True)
class HistoryCleared:
    """已清空模型历史; 验证基线/账本/锁存继续保留."""


@dataclass(frozen=True)
class OperationBusy:
    """非请求操作进行中 (当前: 持久清空/压缩提交).

    只读补充原请求 phase: 请求生命周期已有 RequestPhaseChanged, 清空/压缩等
    操作经本事件让 UI 输入屏障可消费 (05/02 接线); 完成分别由 HistoryCleared
    与 ContextCompacted 标记.
    """

    operation: Literal["clear", "compact", "model", "login", "session"]


@dataclass(frozen=True)
class SessionNameChanged:
    session_id: str
    user_name: str | None
    session_generation: int


@dataclass(frozen=True)
class SessionChanged:
    session_generation: int
    persistence: PersistenceState
    snapshot: RecoverySnapshot


@dataclass(frozen=True)
class SessionManagementResult:
    operation: Literal["name", "new", "resume"]
    session_id: str
    session_generation: int
    changed: bool


class SessionManagementError(RuntimeError):
    def __init__(self, code: str, message: str, *, published: bool = False) -> None:
        self.code = code
        self.published = published
        super().__init__(message[:512])


class SessionManagementSubmission:
    """应用拥有事务任务；wait 取消仅取消本次等待，cancel 发合作取消。"""

    def __init__(self, task: asyncio.Task, token: CancelToken) -> None:
        self._task = task
        self._token = token

    async def wait(self) -> SessionManagementResult:
        return await asyncio.shield(self._task)

    def cancel(self) -> None:
        self._token.cancel()


@dataclass(frozen=True)
class LoginCompleted:
    """仅携带安全保存事实；None 表示编辑取消，无模型轮或历史写入。"""

    result: LoginResult | None


@dataclass(frozen=True)
class ModelChanged:
    previous_label: str
    model_label: str
    epoch_id: str | None


@dataclass(frozen=True)
class ConfigurationChanged:
    """登录配置已应用到当前会话 (04, C7); 只含安全模型身份与 epoch, 无凭据/ref."""

    identity: SafeModelIdentity
    epoch_id: str | None


@dataclass(frozen=True)
class ReasoningEffortChanged:
    """思考强度切换已生效 (03, C5); previous/current 为已生效状态, 非请求目标.

    只表达强度变化: 不触发登录收尾 (ConfigurationChanged 保留), 不宣称模型
    名称变化 (ModelChanged 保留), 也不向 history 添加 UserMessage。
    """

    previous: ReasoningEffortState
    current: ReasoningEffortState
    epoch_id: str | None


class ModelSelectionError(RuntimeError):
    """本地模型选择被拒绝; 有界原因, 无请求历史或凭据."""


@dataclass(frozen=True)
class ContextCompacted:
    """压缩切换成功 (02, D6): 摘要已持久并成为模型可见投影的一部分.

    payload 携带覆盖范围与保留边界 (规范历史下标); 原始记录不变, UI 据此
    追加确认条目并释放屏障。失败/取消不产生本事件。
    """

    summary_id: str
    first_request_seq: int
    last_request_seq: int
    message_count: int
    end_message_index: int
    reason: str  # manual / auto


@dataclass(frozen=True)
class RewindRestoreCompleted:
    result: RewindRestoreResult


@dataclass(frozen=True)
class ApprovalRequested:
    """审批档待决命令 (P0-1); 决定经 ApplicationSession.resolve_approval 回传.

    command 是有界预览, 不是执行依据; 同一时刻至多一个待决请求。
    """

    request_id: str
    tool_name: str
    command: str


@dataclass(frozen=True)
class ExecutionModeChanged:
    """执行档已在会话内切换 (P0-1 v2); mode 为 read-only/approval/full."""

    mode: str


@dataclass(frozen=True)
class ApplicationClosed:
    """关闭完成或明确清理失败; error 非空不代表所有资源均已成功释放."""

    error: str | None = None


ApplicationUpdate = (
    AgentEventReceived | RequestPhaseChanged | RequestCompleted | HistoryCleared
    | OperationBusy | ContextCompacted | ModelChanged | ConfigurationChanged
    | ReasoningEffortChanged | LoginCompleted | ApplicationClosed
    | SessionNameChanged | SessionChanged
    | RewindRestoreCompleted | ApprovalRequested | ExecutionModeChanged
)


class ApplicationOpenError(RuntimeError):
    """启动组装或检查基线建立失败; 显示文本即异常文本."""


class CompactionSealedError(RuntimeError):
    """close 已封住未接受的摘要提交 (PA05); 归关闭取消, 非存储故障."""


def _runtime_dir(config: ApplicationConfig) -> Path:
    # 直接调用旧应用 API 的调用方继续使用原位置；CLI 显式传用户级目录。
    return config.runtime_dir if config.runtime_dir is not None else Path(config.workspace) / ".cicada"
