"""长任务与步骤的最小纯状态机。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskPendingInteraction,
    LongTaskStatus,
    LongTaskStep,
    LongTaskStepStatus,
    LongTaskWaitingReason,
    utc_now,
)


TASK_STATUS_TRANSITIONS: Mapping[
    LongTaskStatus,
    frozenset[LongTaskStatus],
] = {
    "created": frozenset({"queued", "running", "cancelled"}),
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset(
        {
            "queued",
            "awaiting_input",
            "completed",
            "failed",
            "cancelled",
        }
    ),
    "awaiting_input": frozenset(
        {"queued", "running", "cancelled"}
    ),
    "failed": frozenset({"queued", "cancelled"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}

STEP_STATUS_TRANSITIONS: Mapping[
    LongTaskStepStatus,
    frozenset[LongTaskStepStatus],
] = {
    "pending": frozenset({"ready", "skipped", "cancelled"}),
    "ready": frozenset({"running", "skipped", "cancelled"}),
    "running": frozenset(
        {"awaiting_input", "completed", "failed", "cancelled"}
    ),
    "awaiting_input": frozenset({"ready", "running", "cancelled"}),
    "failed": frozenset({"ready", "cancelled"}),
    "completed": frozenset(),
    "skipped": frozenset(),
    "cancelled": frozenset(),
}


class InvalidLongTaskTransitionError(ValueError):
    """表示 Task 或 Step 尝试执行状态机不允许的跳转。"""


def transition_long_task(
    task: LongTask,
    *,
    target_status: LongTaskStatus,
    pending_interaction: LongTaskPendingInteraction | None = None,
    active_step_ids: Sequence[str] | None = None,
) -> LongTask:
    """
    校验并返回状态、版本和更新时间已变化的新任务快照。

    参数含义：
        task:
            调用方当前读取到的任务快照。
        target_status:
            任务准备进入的目标状态。
        pending_interaction:
            目标状态为 awaiting_input 时必须提供的结构化等待交互。
        active_step_ids:
            可选的最新活动步骤编号；未提供时沿用旧值，终态自动清空。

    返回值含义：
        LongTask:
            不修改原对象、版本递增 1 的新任务快照。
    """

    allowed_targets = TASK_STATUS_TRANSITIONS[task.status]
    if target_status not in allowed_targets:
        raise InvalidLongTaskTransitionError(
            f"非法任务状态跳转: {task.status} -> {target_status}"
        )

    resolved_active_step_ids = (
        list(active_step_ids)
        if active_step_ids is not None
        else list(task.active_step_ids)
    )
    if target_status in {"completed", "cancelled"}:
        resolved_active_step_ids = []

    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "status": target_status,
            "pending_interaction": pending_interaction,
            "active_step_ids": resolved_active_step_ids,
            "version": task.version + 1,
            "updated_at": utc_now(),
        }
    )
    return LongTask.model_validate(task_data)


def transition_long_task_step(
    step: LongTaskStep,
    *,
    target_status: LongTaskStepStatus,
    waiting_reason: LongTaskWaitingReason | None = None,
) -> LongTaskStep:
    """
    校验并返回状态、版本和更新时间已变化的新步骤快照。

    参数含义：
        step:
            调用方当前读取到的步骤快照。
        target_status:
            步骤准备进入的目标状态。
        waiting_reason:
            目标状态为 awaiting_input 时必须提供的等待原因。

    返回值含义：
        LongTaskStep:
            不修改原对象、版本递增 1 的新步骤快照。
    """

    allowed_targets = STEP_STATUS_TRANSITIONS[step.status]
    if target_status not in allowed_targets:
        raise InvalidLongTaskTransitionError(
            f"非法步骤状态跳转: {step.status} -> {target_status}"
        )

    step_data = step.model_dump(mode="python")
    next_attempt_count = step.attempt_count
    if target_status == "running":
        next_attempt_count += 1
    step_data.update(
        {
            "status": target_status,
            "waiting_reason": waiting_reason,
            "attempt_count": next_attempt_count,
            "version": step.version + 1,
            "updated_at": utc_now(),
        }
    )
    if target_status != "running":
        step_data.update(
            {
                "claimed_by": None,
                "claim_id": None,
                "lease_expires_at": None,
            }
        )
    if target_status != "failed":
        step_data.update(
            {
                "last_error_code": None,
                "last_error_message": None,
            }
        )
    return LongTaskStep.model_validate(step_data)


def promote_long_task_to_durable(task: LongTask) -> LongTask:
    """
    把非终态内联任务单向升级为可持久化执行模式。

    参数含义：
        task:
            当前任务快照。

    返回值含义：
        LongTask:
            已经是 durable 时原样返回；否则返回执行模式已升级且版本递增
            的新任务快照。终态任务不允许再升级。
    """

    if task.status in {"completed", "cancelled"}:
        raise InvalidLongTaskTransitionError("终态任务不能升级执行模式")
    if task.execution_mode == "durable":
        return task

    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "execution_mode": "durable",
            "version": task.version + 1,
            "updated_at": utc_now(),
        }
    )
    return LongTask.model_validate(task_data)


def revise_long_task_runtime_state(
    task: LongTask,
    *,
    steps: Sequence[LongTaskStep],
    active_step_ids: Sequence[str],
) -> LongTask:
    """
    在 Task 总状态不变时提交步骤结果并递增任务快照版本。

    功能：
        用于一个执行批次结束后更新 Step、清理旧活动批次或暴露新的
        Ready Step。函数不允许借机修改 Task 状态、执行模式或等待交互。

    参数含义：
        task:
            批次提交前的完整任务快照。
        steps:
            已通过步骤状态机迁移的新步骤集合。
        active_step_ids:
            提交后仍然处于活动批次的步骤编号。

    返回值含义：
        LongTask:
            Task 总状态保持不变、版本递增 1 的新快照。
    """

    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "steps": list(steps),
            "active_step_ids": list(active_step_ids),
            "version": task.version + 1,
            "updated_at": utc_now(),
        }
    )
    return LongTask.model_validate(task_data)
