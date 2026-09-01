"""长任务最小状态机单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.runtime.long_tasks import (
    InvalidLongTaskTransitionError,
    LongTask,
    LongTaskGoal,
    LongTaskPendingInteraction,
    LongTaskStep,
    promote_long_task_to_durable,
    transition_long_task,
    transition_long_task_step,
)


def build_task(*, status: str = "created") -> LongTask:
    """
    构建状态机测试使用的两步骤人在回路任务。

    参数含义：
        status:
            整体任务初始状态。

    返回值含义：
        LongTask:
            包含一个 Ready Step 和一个 Pending Step 的任务快照。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="分步骤制定狗狗健康计划。",
            objective="制定狗狗健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="读取档案",
                assigned_agent="profile_agent",
                status="ready",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
            ),
        ],
        status=status,
        progression_mode="guided",
    )


def test_task_transition_should_return_new_versioned_snapshot() -> None:
    """验证状态机不修改原任务，并让新快照版本递增。"""

    original_task = build_task()

    running_task = transition_long_task(
        original_task,
        target_status="running",
        active_step_ids=["step_1"],
    )

    assert original_task.status == "created"
    assert original_task.version == 1
    assert running_task.status == "running"
    assert running_task.active_step_ids == ["step_1"]
    assert running_task.version == 2
    assert running_task.updated_at >= original_task.updated_at


def test_guided_task_should_wait_for_approval_then_resume() -> None:
    """验证人在回路任务可以运行、等待批准并恢复运行。"""

    running_task = transition_long_task(
        build_task(),
        target_status="running",
        active_step_ids=["step_1"],
    )
    running_step_1 = transition_long_task_step(
        running_task.steps[0],
        target_status="running",
    )
    completed_step_1 = transition_long_task_step(
        running_step_1,
        target_status="completed",
    )
    ready_step_2 = transition_long_task_step(
        running_task.steps[1],
        target_status="ready",
    )
    boundary_data = running_task.model_dump(mode="python")
    boundary_data.update(
        {
            "steps": [completed_step_1, ready_step_2],
            "active_step_ids": [],
        }
    )
    boundary_task = LongTask.model_validate(boundary_data)
    pending_interaction = LongTaskPendingInteraction(
        interaction_id="interaction_001",
        interaction_type="approval",
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="Step 1 已完成，是否继续 Step 2？",
        allowed_actions=["continue", "cancel"],
    )
    awaiting_task = transition_long_task(
        boundary_task,
        target_status="awaiting_input",
        pending_interaction=pending_interaction,
        active_step_ids=[],
    )
    resumed_task = transition_long_task(
        awaiting_task,
        target_status="running",
        active_step_ids=["step_2"],
    )

    assert awaiting_task.active_step_ids == []
    assert awaiting_task.pending_interaction == pending_interaction
    assert resumed_task.status == "running"
    assert resumed_task.pending_interaction is None
    assert resumed_task.active_step_ids == ["step_2"]
    assert resumed_task.version == 4


def test_task_transition_should_reject_illegal_jump() -> None:
    """验证任务不能从 created 直接伪造为 completed。"""

    with pytest.raises(
        InvalidLongTaskTransitionError,
        match="created -> completed",
    ):
        transition_long_task(
            build_task(),
            target_status="completed",
        )


def test_terminal_task_should_reject_follow_up_transition() -> None:
    """验证完成状态是终态，不能重新进入 running。"""

    running_task = transition_long_task(
        build_task(),
        target_status="running",
    )
    completed_task = transition_long_task(
        running_task,
        target_status="completed",
    )

    assert completed_task.active_step_ids == []
    with pytest.raises(InvalidLongTaskTransitionError):
        transition_long_task(
            completed_task,
            target_status="running",
        )


def test_awaiting_input_transition_should_require_interaction() -> None:
    """验证状态机不能生成没有等待交互的 awaiting_input 快照。"""

    running_task = transition_long_task(
        build_task(),
        target_status="running",
    )

    with pytest.raises(ValidationError, match="pending_interaction"):
        transition_long_task(
            running_task,
            target_status="awaiting_input",
        )


def test_step_transition_should_increment_attempt_when_running() -> None:
    """验证 Step 每次真正进入 running 都递增尝试次数。"""

    step = build_task().steps[0]

    running_step = transition_long_task_step(
        step,
        target_status="running",
    )
    failed_step = transition_long_task_step(
        running_step,
        target_status="failed",
    )
    failed_step = LongTaskStep.model_validate(
        {
            **failed_step.model_dump(mode="python"),
            "last_error_code": "TOOL_TIMEOUT",
            "last_error_message": "工具调用超时",
        }
    )
    ready_retry_step = transition_long_task_step(
        failed_step,
        target_status="ready",
    )
    retrying_step = transition_long_task_step(
        ready_retry_step,
        target_status="running",
    )

    assert step.attempt_count == 0
    assert running_step.attempt_count == 1
    assert ready_retry_step.last_error_code is None
    assert ready_retry_step.last_error_message is None
    assert retrying_step.attempt_count == 2
    assert retrying_step.version == 5


def test_step_should_wait_for_approval_then_return_to_ready() -> None:
    """验证步骤可以等待用户批准，再回到可调度状态。"""

    running_step = transition_long_task_step(
        build_task().steps[0],
        target_status="running",
    )
    awaiting_step = transition_long_task_step(
        running_step,
        target_status="awaiting_input",
        waiting_reason="approval",
    )
    ready_step = transition_long_task_step(
        awaiting_step,
        target_status="ready",
    )

    assert awaiting_step.waiting_reason == "approval"
    assert ready_step.waiting_reason is None
    assert ready_step.status == "ready"


def test_inline_task_should_promote_to_durable_only_once() -> None:
    """验证执行模式只能从 inline 单向、幂等地升级为 durable。"""

    original_task = build_task()

    durable_task = promote_long_task_to_durable(original_task)
    repeated_task = promote_long_task_to_durable(durable_task)

    assert original_task.execution_mode == "inline"
    assert durable_task.execution_mode == "durable"
    assert durable_task.version == 2
    assert repeated_task is durable_task
