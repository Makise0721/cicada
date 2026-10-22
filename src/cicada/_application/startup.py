"""应用启动装配、初始恢复和失败清理。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from typing import Callable

from cicada.boot import App, BootError, bootstrap
from cicada.context_policy import BudgetPolicy, ContextProjection
from cicada.core.cancel import CancelToken
from cicada.costs import CostControlPort
from cicada.models import ModelRoute, ModelSelectionPort
from cicada.user_config import UserConfigStore
from cicada.persistence.routing import SessionStoreRoute
from cicada.persistence.contracts import (
    PersistenceError,
    RecoverySnapshot,
    SessionCatalogPort,
    StoredIdentity,
    conflicting_launch_fields,
)
from cicada.process_activity import ActivityCoordinator, ProcessActivityWrapper
from cicada.run_policy import HistoricalProjection, verification_policy
from cicada.plugins.session_store import (
    SESSION_CATALOG_CAPABILITY,
    SessionCatalogReader,
    SQLiteSessionStore,
    open_session_store,
    session_store_plugin,
)
from cicada.plugins.session_store.catalog import CombinedSessionCatalogReader
from cicada.plugins.session_store.migration import migrate_legacy_session
from cicada.state_paths import legacy_session_directory
from cicada.plugins.coding.file_completion import file_completion_plugin
from cicada.plugins.coding.inventory import inventory_plugin
from cicada.plugins.coding.change_review import CHANGE_REVIEW_CAPABILITY, change_review_plugin
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.restore_store import RESTORE_CAPABILITY, RestoreStore, restore_plugin
from cicada.plugins.coding.tool_check import check_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_glob import glob_plugin
from cicada.plugins.coding.tool_grep import grep_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.verification import verification_plugin
from cicada.plugins.coding.verification_contracts import VERIFICATION_CAPABILITY, VerificationPlan
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.skills import CATALOG_CAPABILITY, skills_plugin
from cicada.runtime.plugin import PluginDefinition
from cicada.skills import (
    AUTO_SKILLS_PROMPT_VERSION,
    LOAD_SKILL_TOOL_NAME,
    SKILLS_PROMPT_VERSION,
    SkillCatalogSnapshot,
    catalog_prompt_section,
)

from .contracts import (
    ApplicationConfig,
    ApplicationOpenError,
    _runtime_dir,
)
from . import recovery
from .session import (
    ApplicationSession,
)


TOOL_CAPABILITIES = (
    "tool.read", "tool.edit", "tool.write", "tool.powershell", "tool.glob", "tool.grep",
)


CHECK_TOOL_CAPABILITIES = TOOL_CAPABILITIES + ("tool.check",)

# 执行安全边界 P0-1: 只读档只暴露检查类工具; load_skill 只读取本地提示, 不授予动作
READ_ONLY_TOOL_CAPABILITIES = ("tool.read", "tool.glob", "tool.grep")
READ_ONLY_TOOL_NAMES = frozenset({"read", "glob", "grep", LOAD_SKILL_TOOL_NAME})


def make_session_catalog_reader(
    *, directory: Path, workspace: str | None = None, legacy_directory: Path | None = None,
) -> SessionCatalogPort:
    """组合层创建 reader；None 仅供旧任意目录启动调用兼容。"""
    if legacy_directory is not None and legacy_directory != directory:
        return CombinedSessionCatalogReader(
            directory=directory, legacy_directory=legacy_directory, workspace=workspace)
    return SessionCatalogReader(directory=directory, workspace=workspace)


def default_definitions(
    workspace: Path, model_plugin: PluginDefinition, plan: VerificationPlan | None = None,
    *, skills: SkillCatalogSnapshot | None = None, restore: bool = False,
    runtime_dir: Path | None = None,
) -> list[PluginDefinition]:
    """内置 Coding 能力 + 模型插件 (+ 检查模式验证服务 + Skills 能力) 的启动名单.

    skills 为启用 Skills 的冻结快照: 追加目录能力与 load_skill 工具插件;
    None 时名单与未启用逐位一致。restore 为真时追加还原点引擎 (C2: 仅持久会话
    且配置开启捕获的组合才装配), 为假时不安装捕获汇, edit/write 行为逐位不变。
    """
    definitions: list[PluginDefinition] = [
        workspace_plugin(workspace, data_dir=runtime_dir),
        inventory_plugin(),
        file_completion_plugin(),
        change_review_plugin(),
        process_plugin(),
        read_plugin(),
        edit_plugin(),
        write_plugin(),
        powershell_plugin(),
        glob_plugin(),
        grep_plugin(),
    ]
    if plan is not None:
        definitions.append(verification_plugin(plan))
        definitions.append(check_plugin())
    if skills is not None:
        if not isinstance(skills, SkillCatalogSnapshot):
            raise ValueError("skills must be a SkillCatalogSnapshot or None")
        definitions.append(skills_plugin(skills))
    if restore:
        definitions.append(restore_plugin())
    definitions.append(model_plugin)
    return definitions


def _validate_skills_prompt_identity(config: "ApplicationConfig") -> None:
    """Skills 启用时的纯配置一致性校验 (F1 处置; 不做任何 I/O).

    实际 system_prompt 的 hash 必须对应 prompt_info, 目录段必须由同一快照构造,
    版本必须是 Skills 版本; 有 PersistenceConfig 时持久身份的 prompt 字段必须与
    实际 PromptInfo 相等 (adopt_current_config 不豁免)。错配 = 启动错误。
    """
    info = config.prompt_info
    expected_version = AUTO_SKILLS_PROMPT_VERSION if config.skills.roots else SKILLS_PROMPT_VERSION
    if info.version != expected_version:
        raise ValueError(
            f"skills enabled requires prompt version {expected_version}, got {info.version}"
        )
    actual = hashlib.sha256(config.system_prompt.encode("utf-8")).hexdigest()
    if actual != info.prompt_sha256:
        raise ValueError("prompt_info.prompt_sha256 does not match the actual system_prompt")
    if catalog_prompt_section(config.skills) not in config.system_prompt:
        raise ValueError(
            "system_prompt does not contain the catalog section of the configured skills snapshot"
        )
    persistence = config.persistence
    if persistence is None:
        return
    identity = persistence.identity
    mismatches = []
    if identity.prompt_version != info.version:
        mismatches.append("prompt_version")
    if identity.prompt_sha256 != info.prompt_sha256:
        mismatches.append("prompt_sha256")
    if identity.instructions_sha256 != info.instructions_sha256:
        mismatches.append("instructions_sha256")
    if mismatches:
        raise ValueError(
            "persistence identity prompt fields do not match the actual PromptInfo: "
            + ", ".join(mismatches)
        )


def _verify_skills_assembly(app: App, config: "ApplicationConfig") -> None:
    """bootstrap 后核对实际装配: 同一快照/工具注册/指纹含 load_skill (契约 §5)."""
    try:
        catalog = app.runtime.capability(CATALOG_CAPABILITY)
    except Exception as exc:
        raise ApplicationOpenError(f"skills catalog capability missing: {exc}") from exc
    if catalog != config.skills:
        raise ApplicationOpenError(
            "actual skills catalog capability does not match the configured snapshot "
            "(root/manifest/content identity)"
        )
    tools = getattr(app.agent, "_tools", {})
    if LOAD_SKILL_TOOL_NAME not in tools:
        raise ApplicationOpenError(
            f"the {LOAD_SKILL_TOOL_NAME} tool is not registered while skills are enabled"
        )


def _verify_read_only_assembly(app: App, config: "ApplicationConfig") -> None:
    """只读档装配核对: 实际注册工具必须全部在只读名单内 (含 load_skill).

    注册名单已经收窄; 这里再核对 bootstrap 之后的真实工具表, 防止组合层把写入或
    任意命令能力重新带回只读会话。
    """
    if not config.read_only:
        return
    unexpected = sorted(set(getattr(app.agent, "_tools", {})) - READ_ONLY_TOOL_NAMES)
    if unexpected:
        raise ApplicationOpenError(
            f"read-only assembly exposed non-read-only tools: {unexpected}"
        )


def _bind_store_last_cleanup(
    definitions: list[PluginDefinition],
) -> list[PluginDefinition]:
    """持久拓扑: 每个资源插件显式 require session.store (契约 §1)。

    store 最先激活 -> runtime 逆序清理时模型/进程等资源先清理、store 最后释放;
    依赖方向显式可核对, 不靠注册顺序推断。
    """
    return [
        definition if "session.store" in definition.provides
        else replace(definition, requires=definition.requires | {"session.store"})
        for definition in definitions
    ]


async def _open_restore_store(app: App, enabled: bool) -> tuple[RestoreStore | None, tuple[str, ...]]:
    """取得本会话的还原点引擎并做启动机会式清理 (C3); 未装配时返回 (None, ()).

    清理失败只作为告警事实返回 (不阻断启动, 不换方式绕过); 能力缺失说明装配与
    config.restore_capture 不一致, 属组装错误, fail-loud 而不是静默禁用捕获。
    """
    if not enabled:
        return None, ()
    store = app.runtime.capability(RESTORE_CAPABILITY)
    if not isinstance(store, RestoreStore):
        raise ApplicationOpenError(
            f"capability {RESTORE_CAPABILITY!r} did not provide the restore store"
        )
    worker = asyncio.create_task(asyncio.to_thread(store.maintenance))
    cancelled = False
    while True:
        try:
            warnings = await asyncio.shield(worker)
            break
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError()
    return store, warnings


def _open_switch_store(config: ApplicationConfig, session_id: str) -> SQLiteSessionStore:
    if config.runtime_dir is not None:
        migrate_legacy_session(config.workspace, session_id, runtime_dir=config.runtime_dir)
    return open_session_store(
        directory=_runtime_dir(config) / "sessions", session_id=session_id,
        workspace=config.workspace,
    )


def _open_persistent_store(
    config: ApplicationConfig,
) -> tuple[SQLiteSessionStore, RecoverySnapshot]:
    """打开或新建会话存储; 失败时已取得资源由本函数释放, 统一映射为启动错误."""
    persistence = config.persistence
    assert persistence is not None
    try:
        store = open_session_store(
            directory=_runtime_dir(config) / "sessions",
            session_id=persistence.session_id,
            workspace=config.workspace,
        )
    except PersistenceError as exc:
        raise ApplicationOpenError(f"session store open failed: {exc}") from exc
    return store, store.recovery_snapshot()


async def _run_preflight(preflight: Callable[[], object] | None) -> None:
    """恢复路径的模型可达性探测: 必须在格式/配置预检之后、bootstrap 之前执行."""
    if preflight is None:
        return
    if not callable(preflight):
        raise TypeError("preflight must be a callable or None")
    outcome = preflight()
    if inspect.isawaitable(outcome):
        await outcome


async def open_application(
    config: ApplicationConfig, *, preflight: Callable[[], object] | None = None,
    login_store_factory: Callable[[], UserConfigStore] | None = None,
    login_environment: Callable[[], str | None] | None = None,
) -> ApplicationSession:
    """输入/preflight/boot/initialize 成功才返回; 部分启动失败关闭已取得资源.

    持久模式次序: 校验配置及 workspace -> 独占 lease -> 打开/校验存档 ->
    恢复准入与已知配置冲突 -> preflight -> bootstrap (store 插件注册首位, 最后清理) ->
    核对实际工具指纹 -> 写入新的启动代次 -> 注入闭合历史并发布会话.
    与契约 §1 的记载差异: 契约要求注入先于启动代次, 实现中 epoch_id 是会话构造
    参数故 start_epoch 先行, 语义等价 (会话发布前完成全部写入); 偏差记录见
    shared-boundary-handoff.md §4.
    """
    plan = config.plan
    budget_config = config.budget
    policy_state: dict[str, object] = {}
    # 02/F1: Skills 启用时的纯配置一致性校验 (提示/身份绑定) 先于一切应用发起 I/O
    if config.skills is not None:
        try:
            _validate_skills_prompt_identity(config)
        except ValueError as exc:
            raise ApplicationOpenError(f"skills config invalid: {exc}") from exc
    # 恢复前缀证明容器 (03): 检查模式 + 持久会话时由 RunPolicy 每次发送前核对
    history_projection: HistoricalProjection | None = None
    # 模型可见投影容器 (02, D4): 预算启用即存在 (空 = 恒等投影); 压缩/clear/恢复
    # 由应用驱动, BudgetPolicy 与会话持同一实例
    context_projection: ContextProjection | None = None
    if budget_config is not None:
        context_projection = ContextProjection()

    def _make_budget_policy(model, binding=None):
        # D1 次序: BudgetPolicy 在 RunPolicy 下游、适配器上游;
        # 计量因此覆盖验证投影 (固定规则 system 与 request_context 尾部) 的最终形态
        budget = BudgetPolicy(
            model,
            budget_config,
            # D5: 摘要通道注入同一无状态适配器实例; 仅持久会话可压缩 (D9)
            summarize=(policy_state["summary_route"] if binding is not None else model) if store is not None else None,
            summary_budget=binding.summary_budget if binding is not None else None,
            summary_identity=binding.summary_identity if binding is not None else None,
            validate_request=binding.validate_request if binding is not None else None,
            token_counter=binding.token_counter if binding is not None else None,
            summary_token_counter=binding.summary_token_counter if binding is not None else None,
            projection=context_projection,
        )
        policy_state["budget_policy"] = budget
        return budget

    def _model_policy(model, runtime):
        nonlocal config, budget_config, context_projection
        if config.cost_control is not None:
            if not isinstance(model, CostControlPort):
                raise ValueError('selected model does not support cost control')
            model.set_cost_control(config.cost_control)
        binding = model.initial_binding if isinstance(model, ModelSelectionPort) else None
        if binding is not None:
            if config.cost_control is not None:
                config.cost_control.validate_identity(binding.identity)
                config.cost_control.validate_identity(binding.summary_identity)
            policy_state["model_selection"] = model
            policy_state["model_route"] = ModelRoute(binding)
            policy_state["summary_route"] = ModelRoute(binding, summary=True)
            budget_config = binding.budget
            config = replace(config, budget=budget_config, model_label=binding.label or binding.identity.model)
            if context_projection is None:
                context_projection = ContextProjection()
            model = policy_state["model_route"]
        if budget_config is not None:
            model = _make_budget_policy(model, binding)
        if plan is not None:
            service = runtime.capability(VERIFICATION_CAPABILITY)
            model = verification_policy(model, service, history=history_projection)
            policy_state["policy"] = model
            policy_state["service"] = service
        return model

    # model_policy 无条件安装: 检测 ModelSelectionPort 能力并建立稳定路由与两通道
    # 绑定 (D7), 无该能力的 provider 透传。预算/检查包装按 D1 四态在 _model_policy
    # 内叠加: 检查+预算 = verification_policy(budget_policy(inner));
    # 非检查+预算 = budget_policy(inner); 仅检查 = verification_policy(inner);
    # 皆无 = 仅路由检测, 不加包装
    model_policy = _model_policy

    store: SessionStoreRoute | None = None
    snapshot: RecoverySnapshot | None = None
    if config.persistence is not None:
        concrete_store, snapshot = _open_persistent_store(config)
        store = SessionStoreRoute(concrete_store)
    if store is not None and plan is not None:
        history_projection = HistoricalProjection()

    # 活动包装器经 tool_policy 装入; 协调器在会话构造后绑定 (契约 §3)
    coordinator = ActivityCoordinator()
    tool_policy = None
    if store is not None:
        assert store is not None

        def tool_policy(tool, runtime, _store=store, _coordinator=coordinator):
            return ProcessActivityWrapper(tool, _store, _coordinator)

    # 02: Skills 启用时实际注册工具在既有名单上追加 load_skill
    if config.read_only:
        tool_capabilities: tuple[str, ...] = READ_ONLY_TOOL_CAPABILITIES
    else:
        tool_capabilities = CHECK_TOOL_CAPABILITIES if plan is not None else TOOL_CAPABILITIES
    if config.skills is not None:
        tool_capabilities = tool_capabilities + ("tool.load_skill",)

    definitions: list[PluginDefinition] = []
    catalog = None
    # C2: 仅持久会话 + 配置开启捕获才装配还原点引擎; 关闭或单次模式不安装捕获汇
    restore_enabled = store is not None and config.restore_capture
    if store is not None:
        catalog = make_session_catalog_reader(
            directory=_runtime_dir(config) / "sessions", workspace=str(config.workspace),
            legacy_directory=(legacy_session_directory(config.workspace)
                              if config.runtime_dir is not None else None),
        )
        definitions.append(session_store_plugin(store, catalog))
        definitions.extend(_bind_store_last_cleanup(
            default_definitions(config.workspace, config.model_plugin, plan,
                                skills=config.skills, restore=restore_enabled,
                                runtime_dir=config.runtime_dir)
        ))
    else:
        definitions.extend(
            default_definitions(config.workspace, config.model_plugin, plan,
                                skills=config.skills, restore=restore_enabled,
                                runtime_dir=config.runtime_dir)
        )

    if store is not None:
        assert snapshot is not None and config.persistence is not None
        try:
            view_only_reason = recovery._admit_persistent_resume(config.persistence, snapshot)
            await _run_preflight(preflight)
            app: App = await bootstrap(
                definitions,
                tool_capabilities=tool_capabilities,
                system_prompt=config.system_prompt,
                model_policy=model_policy,
                tool_policy=tool_policy,
                max_turns=config.max_turns,
            )
        except BootError as exc:
            await store.aclose()
            raise ApplicationOpenError(f"boot failed: {exc}") from exc
        except BaseException:
            await store.aclose()
            raise
    else:
        try:
            app = await bootstrap(
                definitions,
                tool_capabilities=tool_capabilities,
                system_prompt=config.system_prompt,
                model_policy=model_policy,
                max_turns=config.max_turns,
            )
        except BootError as exc:
            raise ApplicationOpenError(f"boot failed: {exc}") from exc
    if config.skills is not None:
        # 02/契约 §5: 实际装配核对 — 同一快照/工具注册/工具指纹含 load_skill
        try:
            _verify_skills_assembly(app, config)
        except ApplicationOpenError:
            await app.aclose()
            raise
    try:
        _verify_read_only_assembly(app, config)
    except ApplicationOpenError:
        await app.aclose()
        raise

    if store is not None and snapshot is not None and config.persistence is not None:
        # 指纹核对/启动代次/会话构造的每条失败路径都必须释放 app (store 随 runtime 清理)
        try:
            restore_store, restore_warnings = await _open_restore_store(app, restore_enabled)
            fingerprint = recovery._tool_fingerprint(app)
            saved_identity = snapshot.identity
            adopt = config.persistence.adopt_current_config
            legacy_tool_upgrade = False
            if saved_identity is not None and not adopt:
                if saved_identity.tool_fingerprint != fingerprint:
                    legacy_tool_upgrade = (
                        saved_identity.tool_fingerprint ==
                        recovery._tool_fingerprint(app, legacy_output_paths=True))
                    if not legacy_tool_upgrade:
                        raise ApplicationOpenError(
                            "restored session tool declarations differ from the current launch "
                            "(tool_fingerprint)"
                        )
            epoch_launch = config.persistence.identity
            if legacy_tool_upgrade:
                epoch_launch = replace(epoch_launch, sources={
                    **epoch_launch.sources, "tool_output_location": "user-workspace-state",
                })
            if (
                saved_identity is not None
                and adopt
                and config.persistence.session_id is not None
            ):
                # 04: 显式采用当前配置——身份/指纹差异不拒绝, 差异记入新 epoch 的
                # 身份来源 (sources 只作记录, 不参与同配置比对)
                changed = conflicting_launch_fields(saved_identity.launch, epoch_launch)
                sources = {**epoch_launch.sources, "resume_config": "current"}
                if changed:
                    sources["changed_fields"] = ",".join(changed)
                epoch_launch = replace(epoch_launch, sources=sources)
            try:
                epoch_id = await store.start_epoch(
                    StoredIdentity(launch=epoch_launch, tool_fingerprint=fingerprint)
                )
            except PersistenceError as exc:
                raise ApplicationOpenError(f"session epoch start failed: {exc}") from exc
            # start_epoch 已把 pending intent 标中断: 重取点时快照, 会话交接事实与存档一致
            snapshot = store.recovery_snapshot()
            if history_projection is not None:
                try:
                    built = recovery._build_prefix_proof(snapshot)
                except ApplicationOpenError:
                    raise
                except Exception as exc:
                    raise ApplicationOpenError(
                        f"restored prefix proof build failed: {exc}"
                    ) from exc
                history_projection.proof = built.proof
            if context_projection is not None:
                # 02/A08: 按摘要记录确定性重建投影; 覆盖/指纹不符属恢复关系错误
                try:
                    rebuilt_blocks = recovery._rebuild_context_projection(snapshot)
                except ApplicationOpenError:
                    raise
                except Exception as exc:
                    raise ApplicationOpenError(
                        f"restored context projection rebuild failed: {exc}"
                    ) from exc
                context_projection.adopt(rebuilt_blocks)
            session = ApplicationSession(
                app=app,
                config=config,
                service=policy_state.get("service"),  # type: ignore[arg-type]
                policy=policy_state.get("policy"),  # type: ignore[arg-type]
                store=store,
                epoch_id=epoch_id,
                persistence_mode="resume" if config.persistence.session_id is not None else "save",
                restored_history=snapshot.canonical_history,
                restored_requests=snapshot.requests,
                restored_context_generation=snapshot.context_generation,
                restored_interrupted=snapshot.interrupted,
                restored_receipts=snapshot.restored_receipts,
                restored_summaries=snapshot.summaries,
                view_only_reason=view_only_reason,
                projection=history_projection,
                budget_policy=policy_state.get("budget_policy"),  # type: ignore[arg-type]
                model_selection=policy_state.get("model_selection"),
                model_route=policy_state.get("model_route"),
                summary_route=policy_state.get("summary_route"),
                context_projection=context_projection,
                login_store_factory=login_store_factory, login_environment=login_environment,
                review_reader=app.runtime.capability(CHANGE_REVIEW_CAPABILITY),
                catalog_reader=app.runtime.capability(SESSION_CATALOG_CAPABILITY),
                new_store_factory=lambda: open_session_store(
                    directory=_runtime_dir(config) / "sessions",
                    session_id=None, workspace=config.workspace),
                resume_store_factory=lambda session_id: _open_switch_store(config, session_id),
                restore_store=restore_store,
                restore_warnings=restore_warnings,
            )
            # 03/D7: 自动压缩回调在会话构造后绑定 (回调是会话的边界方法);
            # 仅持久会话接线 (D9), 非持久 auto_compact 保持 None=无自动行为
            _wired = policy_state.get("budget_policy")
            if _wired is not None:
                _wired.set_auto_compact(session._auto_compact_for_request)
            coordinator.bind(session)
        except ApplicationOpenError:
            await app.aclose()
            raise
        except BaseException as exc:
            await app.aclose()
            if isinstance(exc, Exception):
                raise ApplicationOpenError(f"session bootstrap failed: {exc}") from exc
            raise
    else:
        session = ApplicationSession(
            app=app,
            config=config,
            service=policy_state.get("service"),  # type: ignore[arg-type]
            policy=policy_state.get("policy"),  # type: ignore[arg-type]
            budget_policy=policy_state.get("budget_policy"),  # type: ignore[arg-type]
            model_selection=policy_state.get("model_selection"),
            model_route=policy_state.get("model_route"),
            summary_route=policy_state.get("summary_route"),
            context_projection=context_projection,
            login_store_factory=login_store_factory, login_environment=login_environment,
                review_reader=app.runtime.capability(CHANGE_REVIEW_CAPABILITY),
        )
    if plan is None:
        return session
    try:
        capture = await session._service.initialize(CancelToken())  # type: ignore[union-attr]
    except BaseException:
        await app.aclose()
        raise
    if not capture.available:
        await app.aclose()
        detail = f" ({capture.failure_kind}): {capture.error}" if capture.error else ""
        raise ApplicationOpenError(f"verification initialize failed{detail}")
    if (
        store is not None
        and snapshot is not None
        and snapshot.recorded_process_uncertain
        and session._service is not None
    ):
        # 跨恢复 process_uncertain: 应用准入与新验证服务控制段一致 (A15);
        # 清空不解除, 交付判断沿用 P4 既有锁存语义
        session._process_uncertain = True
        session._service.mark_process_uncertain(
            "restored archive records process_uncertain "
            "(check receipts without trustworthy terminal facts)"
        )
    return session
