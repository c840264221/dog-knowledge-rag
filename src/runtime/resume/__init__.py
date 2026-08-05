"""主图中断、恢复和任务关系判断的统一导入入口。"""

from src.runtime.resume.task_relation import (
    TaskRelation,
    TaskRelationDecision,
    classify_pending_task_relation,
)
from src.runtime.resume.state_adapter import (
    PendingTaskKind,
    resolve_pending_task_relation,
)

__all__ = [
    "TaskRelation",
    "TaskRelationDecision",
    "classify_pending_task_relation",
    "PendingTaskKind",
    "resolve_pending_task_relation",
]
