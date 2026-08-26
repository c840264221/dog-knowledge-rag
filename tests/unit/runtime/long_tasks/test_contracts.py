"""长任务统一数据契约单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskEvent,
    LongTaskGoal,
    LongTaskPendingInteraction,
    LongTaskQueueMessage,
    LongTaskStep,
)


def build_step(
    step_id: str,
    *,
    task_id: str = "task_001",
    depends_on: list[str] | None = None,
    status: str = "pending",
) -> LongTaskStep:
    """
    构建契约测试使用的最小长任务步骤。

    参数含义：
        step_id:
            当前测试步骤编号。
        task_id:
            步骤所属任务编号。
        depends_on:
            可选前置步骤编号。
        status:
            当前步骤初始状态。

    返回值含义：
        LongTaskStep:
            已通过 Pydantic 校验的测试步骤。
    """

    return LongTaskStep(
        step_id=step_id,
        task_id=task_id,
        title=f"步骤 {step_id}",
        assigned_agent="test_agent",
        depends_on=depends_on or [],
        status=status,
    )


def build_task(
    *,
    steps: list[LongTaskStep] | None = None,
    status: str = "created",
    pending_interaction: LongTaskPendingInteraction | None = None,
    active_step_ids: list[str] | None = None,
) -> LongTask:
    """
    构建契约测试使用的最小长任务。

    参数含义：
        steps:
            可选任务步骤列表。
        status:
            整体任务状态。
        pending_interaction:
            可选的结构化等待交互。
        active_step_ids:
            可选活动步骤编号。

    返回值含义：
        LongTask:
            已通过拓扑和字段校验的测试任务。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="帮我制定狗狗减重计划，每一步让我确认。",
            objective="制定一份分步骤的狗狗减重计划",
            constraints={"requires_step_approval": True},
            expected_output="可执行的减重计划",
        ),
        steps=steps or [build_step("step_1", status="ready")],
        status=status,
        pending_interaction=pending_interaction,
        active_step_ids=active_step_ids or [],
        progression_mode="guided",
    )


def test_long_task_should_preserve_raw_and_normalized_goal() -> None:
    """验证任务同时保留用户原话和规范化执行目标。"""

    task = build_task()

    assert "每一步让我确认" in task.goal.original_request
    assert task.goal.objective == "制定一份分步骤的狗狗减重计划"
    assert task.goal.constraints == {"requires_step_approval": True}
    assert task.execution_mode == "inline"
    assert task.progression_mode == "guided"


def test_long_task_should_round_trip_through_json() -> None:
    """验证完整任务可以安全写入并从 JSON 恢复。"""

    task = build_task()

    restored_task = LongTask.model_validate_json(task.model_dump_json())

    assert restored_task == task
    assert restored_task.steps[0].task_id == task.task_id


def test_long_task_should_reject_mismatched_step_task_id() -> None:
    """验证步骤不能冒充属于另一个任务。"""

    with pytest.raises(ValidationError, match="task_id 与所属任务不一致"):
        build_task(steps=[build_step("step_1", task_id="task_002")])


def test_long_task_should_reject_duplicate_step_ids() -> None:
    """验证同一任务不能包含重复步骤编号。"""

    with pytest.raises(ValidationError, match="step_id 不能重复"):
        build_task(steps=[build_step("step_1"), build_step("step_1")])


def test_long_task_should_reject_missing_dependency() -> None:
    """验证拓扑依赖必须指向任务内真实步骤。"""

    with pytest.raises(ValidationError, match="不存在的依赖步骤"):
        build_task(
            steps=[build_step("step_2", depends_on=["step_missing"])]
        )


def test_long_task_should_reject_dependency_cycle() -> None:
    """验证 Task Step 必须组成有向无环图。"""

    with pytest.raises(ValidationError, match="循环依赖"):
        build_task(
            steps=[
                build_step("step_1", depends_on=["step_2"]),
                build_step("step_2", depends_on=["step_1"]),
            ]
        )


def test_awaiting_input_task_should_require_pending_interaction() -> None:
    """验证等待状态必须提供包含目标步骤的结构化交互。"""

    with pytest.raises(ValidationError, match="pending_interaction"):
        build_task(status="awaiting_input")


def test_awaiting_input_step_should_require_waiting_reason() -> None:
    """验证步骤自身等待输入时仍需说明等待原因。"""

    with pytest.raises(ValidationError, match="waiting_reason"):
        build_step("step_1", status="awaiting_input")


def test_guided_boundary_should_target_next_step_without_active_step() -> None:
    """
    验证 Step 1 完成、Step 2 待批准时没有活动步骤，恢复目标是 Step 2。
    """

    interaction = LongTaskPendingInteraction(
        interaction_id="interaction_001",
        interaction_type="approval",
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="Step 1 已完成，是否继续 Step 2？",
        allowed_actions=["continue", "cancel"],
    )
    task = build_task(
        steps=[
            build_step("step_1", status="completed"),
            build_step("step_2", depends_on=["step_1"], status="ready"),
        ],
        status="awaiting_input",
        pending_interaction=interaction,
        active_step_ids=[],
    )

    assert task.active_step_ids == []
    assert task.steps[0].status == "completed"
    assert task.steps[1].status == "ready"
    assert task.pending_interaction is not None
    assert task.pending_interaction.source_step_ids == ["step_1"]
    assert task.pending_interaction.target_step_ids == ["step_2"]


def test_pending_interaction_should_reject_unknown_step_reference() -> None:
    """验证等待交互不能把用户恢复到任务之外的步骤。"""

    interaction = LongTaskPendingInteraction(
        interaction_id="interaction_001",
        interaction_type="approval",
        source_step_ids=["step_1"],
        target_step_ids=["step_missing"],
        prompt="是否继续？",
        allowed_actions=["continue", "cancel"],
    )

    with pytest.raises(ValidationError, match="不存在的步骤"):
        build_task(
            status="awaiting_input",
            pending_interaction=interaction,
        )


def test_terminal_task_should_not_keep_active_steps() -> None:
    """验证完成或取消后的任务不能继续声明当前执行步骤。"""

    with pytest.raises(ValidationError, match="终态任务"):
        build_task(
            status="completed",
            active_step_ids=["step_1"],
        )


def test_active_steps_should_reject_completed_step() -> None:
    """验证已完成步骤不能被误标记为当前活动步骤。"""

    with pytest.raises(ValidationError, match="只能引用"):
        build_task(
            steps=[build_step("step_1", status="completed")],
            status="running",
            active_step_ids=["step_1"],
        )


def test_long_task_event_should_store_small_structured_fact() -> None:
    """验证 Event 保存结构化事实、序号和 Artifact 引用。"""

    event = LongTaskEvent(
        event_id="event_001",
        task_id="task_001",
        step_id="step_1",
        sequence=1,
        event_type="step_completed",
        actor_type="worker",
        actor_id="worker_A",
        payload={"artifact_id": "artifact_001"},
    )

    assert event.sequence == 1
    assert event.payload["artifact_id"] == "artifact_001"
    with pytest.raises(ValidationError):
        LongTaskEvent(
            event_id="event_002",
            task_id="task_001",
            sequence=0,
            event_type="task_created",
            actor_type="system",
        )


def test_queue_message_should_be_lightweight_and_versioned() -> None:
    """验证 Redis 消息只携带任务引用、版本和 Ready Step 提示。"""

    message = LongTaskQueueMessage(
        task_id="task_001",
        task_version=3,
        reason="continued",
        ready_step_ids=["step_2", "step_3"],
        correlation_id="trace_001",
    )

    restored_message = LongTaskQueueMessage.model_validate_json(
        message.model_dump_json()
    )

    assert restored_message == message
    assert "steps" not in message.model_dump(mode="json")

    with pytest.raises(ValidationError, match="ready_step_ids"):
        LongTaskQueueMessage(
            task_id="task_001",
            task_version=3,
            reason="continued",
            ready_step_ids=["step_2", "step_2"],
        )


def test_batch_result_should_support_single_and_multiple_steps() -> None:
    """验证单 Agent 和多 Agent 批次使用同一个结果契约。"""

    single_agent_batch = LongTaskBatchResult(
        batch_id="batch_001",
        task_id="task_001",
        step_results=[
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                output_summary="已完成犬种资料检索。",
                output_ref="artifact_001",
            )
        ],
    )
    multi_agent_batch = LongTaskBatchResult(
        batch_id="batch_002",
        task_id="task_001",
        step_results=[
            LongTaskBatchStepResult(
                step_id="step_2",
                status="completed",
                output_summary="已完成健康风险分析。",
            ),
            LongTaskBatchStepResult(
                step_id="step_3",
                status="completed",
                output_summary="已完成训练建议整理。",
            ),
        ],
    )

    assert len(single_agent_batch.step_results) == 1
    assert [
        result.step_id for result in multi_agent_batch.step_results
    ] == ["step_2", "step_3"]


def test_batch_step_result_should_require_waiting_details() -> None:
    """验证等待用户的步骤结果必须说明原因并提供用户提示。"""

    with pytest.raises(ValidationError, match="waiting_reason"):
        LongTaskBatchStepResult(
            step_id="step_2",
            status="awaiting_input",
            user_prompt="是否继续生成饮食计划？",
        )

    with pytest.raises(ValidationError, match="user_prompt"):
        LongTaskBatchStepResult(
            step_id="step_2",
            status="awaiting_input",
            waiting_reason="approval",
        )

    result = LongTaskBatchStepResult(
        step_id="step_2",
        status="awaiting_input",
        waiting_reason="approval",
        user_prompt="是否继续生成饮食计划？",
    )

    assert result.waiting_reason == "approval"


def test_batch_step_result_should_require_failure_details() -> None:
    """验证失败步骤必须提供错误说明，成功步骤不能残留错误信息。"""

    with pytest.raises(ValidationError, match="error_message"):
        LongTaskBatchStepResult(
            step_id="step_2",
            status="failed",
        )

    with pytest.raises(ValidationError, match="只有 failed"):
        LongTaskBatchStepResult(
            step_id="step_2",
            status="completed",
            error_message="不应残留的旧错误",
        )

    result = LongTaskBatchStepResult(
        step_id="step_2",
        status="failed",
        error_message="健康数据服务暂时不可用",
    )

    assert "暂时不可用" in result.error_message


def test_batch_result_should_reject_duplicate_step_results() -> None:
    """验证同一批次不能为一个步骤保存两份冲突结果。"""

    with pytest.raises(ValidationError, match="重复的 step_id"):
        LongTaskBatchResult(
            batch_id="batch_001",
            task_id="task_001",
            step_results=[
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="completed",
                ),
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="failed",
                    error_message="重复执行",
                ),
            ],
        )
