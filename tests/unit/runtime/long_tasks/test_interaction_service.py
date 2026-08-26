"""长任务人工交互纯业务服务单元测试。"""

from __future__ import annotations

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskGoal,
    LongTaskStep,
)
from src.runtime.long_tasks.interaction_service import (
    InvalidLongTaskInteractionError,
    LongTaskInteractionService,
)


def build_boundary_task(
    *,
    active_step_ids: list[str] | None = None,
) -> LongTask:
    """
    构建 Step 1 已完成、Step 2 已解锁的批准边界任务。

    参数含义：
        active_step_ids:
            可选活动步骤，用于验证尚有步骤运行时不能进入批准边界。

    返回值含义：
        LongTask:
            当前处于 running 的两步骤人在回路任务快照。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="分步骤生成狗狗健康计划，每步让我批准。",
            objective="生成狗狗健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="读取档案",
                assigned_agent="profile_agent",
                status="completed",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="ready",
            ),
        ],
        status="running",
        progression_mode="guided",
        active_step_ids=active_step_ids or [],
    )


def build_service() -> LongTaskInteractionService:
    """
    构建使用固定交互编号的测试服务。

    返回值含义：
        LongTaskInteractionService:
            每次创建交互都返回可预测编号的无状态服务。
    """

    return LongTaskInteractionService(
        interaction_id_factory=lambda: "interaction_001"
    )


def wait_for_step_2_approval() -> LongTask:
    """
    构建已经暂停并等待批准 Step 2 的任务。

    返回值含义：
        LongTask:
            携带 interaction_001 等待交互的 awaiting_input 快照。
    """

    return build_service().wait_for_approval(
        build_boundary_task(),
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="Step 1 已完成，是否继续 Step 2？",
    )


def wait_for_step_2_input() -> LongTask:
    """
    构建 Step 2 已执行但缺少用户输入的等待任务。

    返回值含义：
        LongTask:
            携带 missing_input 交互、Step 2 为 awaiting_input 的任务快照。
    """

    task = build_boundary_task()
    step_data = task.steps[1].model_dump(mode="python")
    step_data.update(
        {
            "status": "awaiting_input",
            "waiting_reason": "missing_input",
            "output_summary": "还缺少狗狗年龄。",
        }
    )
    task_data = task.model_dump(mode="python")
    task_data["steps"] = [
        task.steps[0],
        LongTaskStep.model_validate(step_data),
    ]
    return build_service().wait_for_step_input(
        LongTask.model_validate(task_data),
        waiting_step_ids=["step_2"],
        interaction_type="missing_input",
        prompt="请补充狗狗年龄。",
    )


def test_wait_for_approval_should_build_consistent_boundary() -> None:
    """验证一个业务动作能同步设置状态、活动步骤和等待交互。"""

    original_task = build_boundary_task()

    awaiting_task = build_service().wait_for_approval(
        original_task,
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="Step 1 已完成，是否继续 Step 2？",
    )

    assert original_task.status == "running"
    assert awaiting_task.status == "awaiting_input"
    assert awaiting_task.active_step_ids == []
    assert awaiting_task.version == original_task.version + 1
    assert awaiting_task.pending_interaction is not None
    assert awaiting_task.pending_interaction.interaction_id == (
        "interaction_001"
    )
    assert awaiting_task.pending_interaction.source_step_ids == [
        "step_1"
    ]
    assert awaiting_task.pending_interaction.target_step_ids == [
        "step_2"
    ]


def test_wait_for_step_input_should_preserve_waiting_step_boundary() -> None:
    """验证步骤内等待会保存提示并保留当前等待 Step 的活动语义。"""

    task = build_boundary_task()
    step_data = task.steps[1].model_dump(mode="python")
    step_data.update(
        {
            "status": "awaiting_input",
            "waiting_reason": "missing_input",
        }
    )
    task_data = task.model_dump(mode="python")
    task_data["steps"] = [
        task.steps[0],
        LongTaskStep.model_validate(step_data),
    ]
    waiting_draft = LongTask.model_validate(task_data)

    awaiting_task = build_service().wait_for_step_input(
        waiting_draft,
        waiting_step_ids=["step_2"],
        interaction_type="missing_input",
        prompt="请补充狗狗年龄。",
        input_contract={
            "items": [
                {
                    "step_id": "step_2",
                    "prompt": "请补充狗狗年龄。",
                }
            ]
        },
    )

    assert awaiting_task.status == "awaiting_input"
    assert awaiting_task.active_step_ids == ["step_2"]
    assert awaiting_task.pending_interaction is not None
    assert awaiting_task.pending_interaction.prompt == "请补充狗狗年龄。"
    assert awaiting_task.pending_interaction.allowed_actions == [
        "submit_input",
        "cancel",
    ]


def test_continue_should_activate_interaction_target_steps() -> None:
    """验证用户批准后只激活交互中声明的目标步骤。"""

    awaiting_task = wait_for_step_2_approval()

    resumed_task = build_service().resume_after_approval(
        awaiting_task,
        interaction_id="interaction_001",
        action="continue",
    )

    assert resumed_task.status == "running"
    assert resumed_task.active_step_ids == ["step_2"]
    assert resumed_task.pending_interaction is None
    assert resumed_task.version == awaiting_task.version + 1


def test_submit_input_should_prepare_step_without_marking_it_active() -> None:
    """验证回答只写入目标 Step，真正领取前保持 ready 和非活动状态。"""

    awaiting_task = wait_for_step_2_input()

    resumed_task = build_service().respond_to_missing_input(
        awaiting_task,
        interaction_id="interaction_001",
        action="submit_input",
        answers={"step_2": {"dog_age": 6}},
    )

    resumed_step = resumed_task.steps[1]
    assert resumed_task.status == "running"
    assert resumed_task.active_step_ids == []
    assert resumed_task.pending_interaction is None
    assert resumed_step.status == "ready"
    assert resumed_step.waiting_reason is None
    assert resumed_step.input_data["multi_agent_resume_input"] == {
        "dog_age": 6
    }
    assert resumed_step.input_data["multi_agent_is_resuming"] is True


def test_submit_input_should_require_exact_waiting_step_answers() -> None:
    """验证缺少或额外 Step 回答时系统拒绝猜测回答归属。"""

    with pytest.raises(
        InvalidLongTaskInteractionError,
        match="完整对应",
    ):
        build_service().respond_to_missing_input(
            wait_for_step_2_input(),
            interaction_id="interaction_001",
            action="submit_input",
            answers={"step_unknown": "6岁"},
        )


def test_cancel_should_finish_task_without_active_steps() -> None:
    """验证用户取消后清除等待交互和活动步骤。"""

    awaiting_task = wait_for_step_2_approval()

    cancelled_task = build_service().resume_after_approval(
        awaiting_task,
        interaction_id="interaction_001",
        action="cancel",
    )

    assert cancelled_task.status == "cancelled"
    assert cancelled_task.active_step_ids == []
    assert cancelled_task.pending_interaction is None


def test_resume_should_reject_stale_interaction_id() -> None:
    """验证旧页面或重复请求不能恢复已经变化的等待交互。"""

    with pytest.raises(
        InvalidLongTaskInteractionError,
        match="已过期",
    ):
        build_service().resume_after_approval(
            wait_for_step_2_approval(),
            interaction_id="interaction_old",
            action="continue",
        )


def test_wait_should_reject_unfinished_active_batch() -> None:
    """验证仍有活动步骤时不能把整个任务伪装成等待批准。"""

    with pytest.raises(
        InvalidLongTaskInteractionError,
        match="结束当前活动批次",
    ):
        build_service().wait_for_approval(
            build_boundary_task(active_step_ids=["step_2"]),
            source_step_ids=["step_1"],
            target_step_ids=["step_2"],
            prompt="是否继续？",
        )


def test_wait_should_require_completed_source_and_ready_target() -> None:
    """验证批准边界不能指向状态含义不正确的步骤。"""

    with pytest.raises(
        InvalidLongTaskInteractionError,
        match="来源步骤状态必须是 completed",
    ):
        build_service().wait_for_approval(
            build_boundary_task(),
            source_step_ids=["step_2"],
            target_step_ids=["step_1"],
            prompt="是否继续？",
        )
