"""工具结果载荷契约: 带版本的类型化子集 + 旧记录解码.

`ToolResult.details` 仍是冻结接口的存储形态; 本模块只定义其中**参与判定**的字段,
并给出唯一读写入口:

- 生产者经 `process_details` / `mutation_details` / `check_status_details` 标注
  `contract` / `contract_version` / `kind`, 类型不符即抛 TypeError (生产方缺陷).
- 消费者只经 `process_facts` / `mutation_facts` 读取; 未类型化的其它字段保留给
  展示与诊断, 不参与任何判定.
- 无标记的旧记录按字段推断并标 `legacy=True`, 行为与标注前一致; 标记存在但
  contract/版本/kind 不认识时解码返回 None, 调用方必须按"事实未知"保守处理.
  (旧 Cicada 写的新版本记录不会被误读成可信终结事实.)

版本变更规则: 判定字段增减/语义变化时提升 `CONTRACT_VERSION`; 旧版本记录只由
对应版本的解码器处理, 当前解码器对未知版本一律返回 None.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CONTRACT = "cicada.tool-outcome"
CONTRACT_VERSION = 1

CONTRACT_KEY = "contract"
VERSION_KEY = "contract_version"
KIND_KEY = "kind"

KIND_PROCESS = "process"
KIND_MUTATION = "mutation"
KIND_CHECK_STATUS = "check-status"

_MARKER_KEYS = frozenset({CONTRACT_KEY, VERSION_KEY, KIND_KEY})


@dataclass(frozen=True)
class ProcessFacts:
    """一次进程类工具结果的可判定事实; None 表示该字段缺失或类型不合法.

    `timed_out` / `cancelled` 缺失时为 None 而非 False: 恢复比对只比较显式事实
    (旧行为), 判定方按 `is True` 取用.
    """

    receipt_id: str | None = None
    verification_run_id: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    cancelled: bool | None = None
    output_complete: bool | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    legacy: bool = False

    @property
    def termination_known(self) -> bool:
        """可靠的终结事实: 记录了退出码且采集到 EOF, 未超时/取消."""
        return (
            self.output_complete is True
            and self.timed_out is not True
            and self.cancelled is not True
            and self.exit_code is not None
        )


@dataclass(frozen=True)
class MutationFacts:
    """一次文件修改结果的可判定事实; 只读结构化字段, 不认文字推断."""

    state: str
    reason: str | None = None
    canonical_path: str | None = None
    current_matches: bool | None = None
    caller_cancelled: bool | None = None
    winerror: int | None = None
    retained_paths: tuple[str, ...] = ()
    cleanup_warning: str | None = None
    artifact_path: str | None = None
    legacy: bool = False


# --- 生产者入口 ---------------------------------------------------------------------------


def process_details(
    payload: Mapping[str, Any], *, extras: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """标注一个进程类结果载荷; 判定字段类型不合法即 TypeError.

    `payload` 里允许保留展示字段 (如 truncated/full_output_path), 它们原样保存但
    不进入 `ProcessFacts`. `extras` 与 payload 合并, 但不得覆盖契约标记.
    """
    _optional_int(payload, "exit_code")
    _optional_bool(payload, "timed_out")
    _optional_bool(payload, "cancelled")
    _optional_bool(payload, "output_complete")
    for key in ("receipt_id", "verification_run_id", "artifact_path", "artifact_sha256",
                "output_artifact_path", "output_artifact_sha256", "full_output_path"):
        _optional_str(payload, key)
    return _labeled(KIND_PROCESS, payload, extras)


def mutation_details(
    mutation: Mapping[str, Any], *, extras: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """标注一次文件修改结果; 内层 mutation 必须是结构化事实, 深拷贝后保存."""
    if not isinstance(mutation, Mapping):
        raise TypeError("mutation must be a mapping")
    state = mutation.get("state")
    if not isinstance(state, str) or not state:
        raise TypeError("mutation.state must be a nonempty string")
    _optional_str(mutation, "reason")
    _optional_str(mutation, "canonical_path")
    _optional_bool(mutation, "current_matches")
    _optional_bool(mutation, "caller_cancelled")
    _optional_int(mutation, "winerror")
    _optional_str(mutation, "cleanup_warning")
    retained = mutation.get("retained_paths", ())
    if not isinstance(retained, (list, tuple)) or not all(isinstance(item, str) for item in retained):
        raise TypeError("mutation.retained_paths must be a sequence of strings")
    return _labeled(KIND_MUTATION, {"mutation": copy.deepcopy(dict(mutation))}, extras)

def check_status_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    """标注 check action=status 的观察载荷: 不参与终结判定, 只避免被误读成进程事实."""
    return _labeled(KIND_CHECK_STATUS, payload, None)


# --- 消费者入口 ---------------------------------------------------------------------------


def is_check_status(details: Any) -> bool:
    """仅识别当前版本的检查观察载荷; 它没有启动进程."""
    return isinstance(details, Mapping) and _is_legacy(details, KIND_CHECK_STATUS) is False


def process_facts(details: Any) -> ProcessFacts | None:
    """解码进程类事实; 无标记旧记录按字段推断, 未知契约/版本/kind 返回 None."""
    if not isinstance(details, Mapping):
        return None
    legacy = _is_legacy(details, KIND_PROCESS)
    if legacy is None:
        return None
    return ProcessFacts(
        receipt_id=_str_or_none(details.get("receipt_id")),
        verification_run_id=_str_or_none(details.get("verification_run_id")),
        exit_code=_int_or_none(details.get("exit_code")),
        timed_out=_bool_or_none(details.get("timed_out")),
        cancelled=_bool_or_none(details.get("cancelled")),
        output_complete=_bool_or_none(details.get("output_complete")),
        artifact_path=_first_str(details, ("artifact_path", "output_artifact_path", "full_output_path")),
        artifact_sha256=_first_str(details, ("artifact_sha256", "output_artifact_sha256")),
        legacy=legacy,
    )


def mutation_facts(details: Any) -> MutationFacts | None:
    """解码文件修改事实; 无标记旧记录按内层 mutation 推断, 未知契约/版本/kind 返回 None."""
    if not isinstance(details, Mapping):
        return None
    legacy = _is_legacy(details, KIND_MUTATION)
    if legacy is None:
        return None
    mutation = details.get("mutation")
    if not isinstance(mutation, Mapping):
        return None
    state = mutation.get("state")
    if not isinstance(state, str) or not state:
        return None
    retained = mutation.get("retained_paths", ())
    return MutationFacts(
        state=state,
        reason=_str_or_none(mutation.get("reason")),
        canonical_path=_str_or_none(mutation.get("canonical_path")),
        current_matches=_bool_or_none(mutation.get("current_matches")),
        caller_cancelled=_bool_or_none(mutation.get("caller_cancelled")),
        winerror=_int_or_none(mutation.get("winerror")),
        retained_paths=tuple(item for item in retained if isinstance(item, str))
        if isinstance(retained, (list, tuple)) else (),
        cleanup_warning=_str_or_none(mutation.get("cleanup_warning")),
        artifact_path=_first_str(details, ("artifact_path", "full_diff_path")),
        legacy=legacy,
    )


# --- 内部 ---------------------------------------------------------------------------------


def _is_legacy(details: Mapping[str, Any], kind: str) -> bool | None:
    """返回 legacy 标记; None 表示该载荷不属于当前契约 (调用方按未知处理)."""
    if CONTRACT_KEY not in details:
        # 无标记旧记录: 除显式 kind 外都按 legacy 解码 (与契约前行为逐位一致)
        return None if KIND_KEY in details else True
    if details.get(CONTRACT_KEY) != CONTRACT:
        return None
    if details.get(VERSION_KEY) != CONTRACT_VERSION:
        return None
    if details.get(KIND_KEY) != kind:
        return None
    return False


def _labeled(
    kind: str, payload: Mapping[str, Any], extras: Mapping[str, Any] | None
) -> dict[str, Any]:
    if extras is not None and not isinstance(extras, Mapping):
        raise TypeError("extras must be a mapping")
    data: dict[str, Any] = dict(extras or {})
    for key in data:
        if not isinstance(key, str):
            raise TypeError("details keys must be strings")
        if key in _MARKER_KEYS:
            raise ValueError(f"extras may not override the contract marker {key!r}")
    data.update(payload)
    data[CONTRACT_KEY] = CONTRACT
    data[VERSION_KEY] = CONTRACT_VERSION
    data[KIND_KEY] = kind
    return data


def _optional_str(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None and not isinstance(payload[key], str):
        raise TypeError(f"{key} must be str or None, got {type(payload[key]).__name__}")


def _optional_bool(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None and not isinstance(payload[key], bool):
        raise TypeError(f"{key} must be bool or None, got {type(payload[key]).__name__}")


def _optional_int(payload: Mapping[str, Any], key: str) -> None:
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be int or None, got {type(value).__name__}")


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_str(details: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    """按顺序取第一个非空字符串: 规范键优先, 兼容旧工具键."""
    for key in keys:
        value = _str_or_none(details.get(key))
        if value is not None:
            return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
