"""请求还原点的不可变显示投影；不携带文件访问或验证权威。"""

from __future__ import annotations

from dataclasses import dataclass

SCOPE_NOTE = "仅文件工具改动；PowerShell/check/进程外改动不在还原范围"
PREVIEW_LIMIT = 256 * 1024
FILE_LIMIT = 16 * 1024 * 1024
REWIND_TIMEOUT_S = 10.0


class RewindError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message[:512])


@dataclass(frozen=True)
class RewindPointSummary:
    point_id: str
    kind: str
    request_id: str | None
    seq: int | None
    epoch_id: str | None
    created_at: str
    source: str
    availability: str
    entry_count: int
    total_bytes: int
    note: str | None = None


@dataclass(frozen=True)
class RewindListView:
    observed_at: str
    workspace_label: str
    points: tuple[RewindPointSummary, ...]
    scope_note: str = SCOPE_NOTE


@dataclass(frozen=True)
class RewindEntry:
    entry_id: str
    relative_path: str
    op: str
    size_bytes: int
    before_sha256: str | None
    after_sha256: str | None
    current_relation: str
    later_seq: int | None = None


@dataclass(frozen=True)
class RewindPointView:
    point_id: str
    observed_at: str
    availability: str
    entries: tuple[RewindEntry, ...]
    scope_note: str = SCOPE_NOTE


@dataclass(frozen=True)
class RewindPreview:
    point_id: str
    entry_id: str
    state: str
    text: str
    notice: str | None
    observed_at: str
