"""只读审阅的中立显示投影；不携带模型历史或验证权威。"""

from __future__ import annotations

from dataclasses import dataclass, field


class ReviewError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message[:512])


@dataclass(frozen=True)
class ReviewGroupState:
    group: str
    state: str
    reason: str | None = None


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    group: str
    relative_path: str
    kind: str


@dataclass(frozen=True)
class ReviewSnapshot:
    review_id: str
    workspace_label: str
    observed_at: str
    groups: tuple[ReviewGroupState, ...]
    items: tuple[ReviewItem, ...]
    checks_state: str = "not_enabled"
    checks_target_id: str | None = None
    # Reader 内部观察依据，调用方不解析或构造它。
    _basis: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _filter_paths: tuple[str, ...] = field(default=(), repr=False, compare=False)
    scope: str = "Git root；沿用文件清单排除；忽略文件和子模块正文不在范围内"


@dataclass(frozen=True)
class ReviewPreview:
    review_id: str
    target_id: str
    title: str
    comparison: str
    state: str
    text: str = ""
    notice: str | None = None
    observed_at: str | None = None
    integrity: str = "not_applicable"


@dataclass(frozen=True)
class ReviewCheckRow:
    check_id: str
    receipt_id: str | None
    command: str
    execution_status: str | None
    verification_status: str | None
    freshness: str
    artifact_integrity: str
    reason: str | None = None


@dataclass(frozen=True)
class ReviewCheckDetails:
    review_id: str
    target_id: str
    verification_run_id: str
    snapshot_ref: str | None
    scope_id: str | None
    rows: tuple[ReviewCheckRow, ...]
    process_uncertain: bool
    blocking_reasons: tuple[str, ...]
    artifact_target_id: str | None = None
    artifact_label: str | None = None
