"""统一 LongTask Runtime Driver / PEP 单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskExecutionContext,
    LongTaskGoal,
    LongTaskRuntimeDriver,
    LongTaskStep,
    LongTaskStepClaimConflictError,
    LongTaskVersionConflictError,
    UnsupportedLongTaskDriverActionError,
)
from src.runtime.long_tasks.interaction_service import (
    LongTaskInteractionService,
)


class InMemoryDriverStore:
    """为 Runtime Driver 测试提供带乐观锁语义的内存 Store。"""

    def __init__(self, task: LongTask) -> None:
        self.task = task
        self.save_count = 0

    async def create(self, task: LongTask) -> LongTask:
        """保存测试初始任务并返回原快照。"""

        self.task = task
        return task

    async def load(self, task_id: str) -> LongTask | None:
        """按编号返回测试任务，不匹配时返回 None。"""

        return self.task if self.task.task_id == task_id else None

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        校验版本后保存 Driver 产生的新任务快照。

        参数含义：
            task:
                Driver 已应用决策的新任务。
            expected_version:
                Driver 开始处理时读取到的旧版本号。

        返回值含义：
            LongTask:
                乐观锁校验成功后保存的任务快照。
        """

        if self.task.version != expected_version:
            raise LongTaskVersionConflictError("测试任务版本冲突")
        if task.version != expected_version + 1:
            raise ValueError("新任务版本必须只递增 1")
        self.task = task
        self.save_count += 1
        return task


def build_running_task(
    *,
    include_next_step: bool = True,
) -> LongTask:
    """
    构建第一个步骤运行中、可选第二个依赖步骤的测试任务。

    参数含义：
        include_next_step:
            是否创建依赖第一个步骤的后续步骤。

    返回值含义：
        LongTask:
            可以接收一个执行批次结果的版本 1 任务。
    """

    steps = [
        LongTaskStep(
            step_id="step_1",
            task_id="task_001",
            title="读取档案",
            assigned_agent="profile_agent",
            status="running",
            attempt_count=1,
        )
    ]
    if include_next_step:
        steps.append(
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            )
        )
    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="制定健康计划",
            objective="生成分步骤健康计划",
        ),
        steps=steps,
        status="running",
        active_step_ids=["step_1"],
    )


def build_batch_result(
    result: LongTaskBatchStepResult,
) -> LongTaskBatchResult:
    """
    构建只包含一个步骤结果的测试批次。

    参数含义：
        result:
            当前批次唯一的步骤结果。

    返回值含义：
        LongTaskBatchResult:
            可以交给 Runtime Driver 的批次结果。
    """

    return LongTaskBatchResult(
        batch_id="batch_001",
        task_id="task_001",
        step_results=[result],
    )


def build_claimed_running_task() -> LongTask:
    """
    构建持有有效后台领取令牌的运行中任务。

    返回值含义：
        LongTask：可用于校验批次结果领取令牌的 durable 任务。
    """

    task = build_running_task(include_next_step=False)
    step_data = task.steps[0].model_dump(mode="python")
    step_data.update(
        {
            "claimed_by": "worker-1",
            "claim_id": "claim-current",
            "lease_expires_at": datetime.now(timezone.utc)
            + timedelta(seconds=30),
        }
    )
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "execution_mode": "durable",
            "steps": [LongTaskStep.model_validate(step_data)],
        }
    )
    return LongTask.model_validate(task_data)


def execution_context(*, exhausted: bool = False) -> LongTaskExecutionContext:
    """
    构建同步预算充足或已经耗尽的测试运行上下文。

    参数含义：
        exhausted:
            是否让已用时间达到同步预算。

    返回值含义：
        LongTaskExecutionContext:
            可用于触发 advance 或 durable 决策的计时信息。
    """

    return LongTaskExecutionContext(
        elapsed_ms=1000 if exhausted else 100,
        inline_budget_ms=1000,
    )


@pytest.mark.asyncio
async def test_driver_should_commit_batch_and_expose_next_ready_step() -> None:
    """验证 advance 动作会提交结果、清空旧批次并暴露下一 Ready Step。"""

    store = InMemoryDriverStore(build_running_task())
    task = await LongTaskRuntimeDriver(store).handle_batch_result(
        task_id="task_001",
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                output_summary="档案读取完成",
                output_ref="artifact_profile_001",
            )
        ),
        execution_context=execution_context(),
    )

    assert task.version == 2
    assert task.status == "running"
    assert task.active_step_ids == []
    assert task.steps[0].status == "completed"
    assert task.steps[0].output_summary == "档案读取完成"
    assert task.steps[0].output_ref == "artifact_profile_001"
    assert task.steps[1].status == "ready"
    assert store.save_count == 1


@pytest.mark.asyncio
async def test_driver_should_keep_trace_refs_without_diagnostic_metadata(
) -> None:
    """验证权威 Step 只保存 Trace 引用，不复制批次诊断详情。"""

    store = InMemoryDriverStore(build_running_task())
    task = await LongTaskRuntimeDriver(store).handle_batch_result(
        task_id="task_001",
        batch_result=LongTaskBatchResult(
            batch_id="batch_trace_001",
            task_id="task_001",
            actor_type="agent",
            actor_id="profile_agent",
            trace_id="trace_001",
            step_results=[
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="completed",
                    output_summary="档案读取完成",
                    span_id="span_step_1",
                    metadata={
                        "duration_ms": 1200,
                        "runtime_host": "worker-node-1",
                    },
                )
            ],
            metadata={"token_usage": {"input": 800, "output": 200}},
        ),
        execution_context=execution_context(),
    )

    committed_step = task.steps[0]
    assert committed_step.last_trace_id == "trace_001"
    assert committed_step.last_span_id == "span_step_1"
    assert "last_batch_result" not in committed_step.metadata
    assert "duration_ms" not in committed_step.metadata


@pytest.mark.asyncio
async def test_driver_should_complete_last_step() -> None:
    """验证 complete_task 动作会通过状态机结束整份任务。"""

    store = InMemoryDriverStore(
        build_running_task(include_next_step=False)
    )
    task = await LongTaskRuntimeDriver(store).handle_batch_result(
        task_id="task_001",
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
        execution_context=execution_context(exhausted=True),
    )

    assert task.status == "completed"
    assert task.execution_mode == "inline"
    assert task.active_step_ids == []
    assert task.version == 2


@pytest.mark.asyncio
async def test_driver_should_accept_current_claim_and_clear_lease() -> None:
    """验证当前领取令牌可以提交结果，并在 Step 完成后清除租约。"""

    store = InMemoryDriverStore(build_claimed_running_task())
    task = await LongTaskRuntimeDriver(store).handle_batch_result(
        task_id="task_001",
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                claim_id="claim-current",
            )
        ),
        execution_context=execution_context(),
    )

    step = task.steps[0]
    assert task.status == "completed"
    assert step.status == "completed"
    assert step.claimed_by is None
    assert step.claim_id is None
    assert step.lease_expires_at is None


@pytest.mark.asyncio
async def test_driver_should_reject_stale_claim_result() -> None:
    """验证旧 Worker 的迟到结果不能覆盖当前领取者的执行事实。"""

    original_task = build_claimed_running_task()
    store = InMemoryDriverStore(original_task)

    with pytest.raises(
        LongTaskStepClaimConflictError,
        match="领取令牌不一致",
    ):
        await LongTaskRuntimeDriver(store).handle_batch_result(
            task_id="task_001",
            batch_result=build_batch_result(
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="completed",
                    claim_id="claim-old",
                )
            ),
            execution_context=execution_context(),
        )

    assert store.task == original_task
    assert store.save_count == 0


@pytest.mark.asyncio
async def test_driver_should_promote_unfinished_task_to_durable() -> None:
    """验证同步预算耗尽时会提交结果并单向升级执行模式。"""

    store = InMemoryDriverStore(build_running_task())
    task = await LongTaskRuntimeDriver(store).handle_batch_result(
        task_id="task_001",
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
        execution_context=execution_context(exhausted=True),
    )

    assert task.execution_mode == "durable"
    assert task.status == "running"
    assert task.steps[1].status == "ready"
    assert task.version == 2


@pytest.mark.asyncio
async def test_driver_should_persist_single_waiting_interaction() -> None:
    """验证单个等待结果会保存为前端可查询的 pending_interaction。"""

    store = InMemoryDriverStore(build_claimed_running_task())
    driver = LongTaskRuntimeDriver(
        store,
        interaction_service=LongTaskInteractionService(
            interaction_id_factory=lambda: "interaction_001"
        ),
    )
    task = await driver.handle_batch_result(
        task_id="task_001",
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="awaiting_input",
                waiting_reason="missing_input",
                user_prompt="请补充狗狗年龄。",
                claim_id="claim-current",
            )
        ),
        execution_context=execution_context(exhausted=True),
    )

    assert task.status == "awaiting_input"
    assert task.active_step_ids == ["step_1"]
    assert task.version == 2
    assert task.steps[0].status == "awaiting_input"
    assert task.steps[0].claim_id is None
    assert task.pending_interaction is not None
    assert task.pending_interaction.interaction_id == "interaction_001"
    assert task.pending_interaction.prompt == "请补充狗狗年龄。"
    assert task.pending_interaction.source_step_ids == ["step_1"]
    assert task.pending_interaction.target_step_ids == ["step_1"]
    assert store.save_count == 1


@pytest.mark.asyncio
async def test_driver_should_aggregate_multiple_waiting_steps() -> None:
    """验证同一批次多个等待提示会形成一个结构化交互。"""

    original_task = LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="生成健康计划",
            objective="并发检查档案和授权",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="检查年龄",
                assigned_agent="profile_agent",
                status="running",
                attempt_count=1,
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="检查档案授权",
                assigned_agent="profile_agent",
                status="running",
                attempt_count=1,
            ),
        ],
        status="running",
        active_step_ids=["step_1", "step_2"],
    )
    store = InMemoryDriverStore(original_task)
    driver = LongTaskRuntimeDriver(
        store,
        interaction_service=LongTaskInteractionService(
            interaction_id_factory=lambda: "interaction_002"
        ),
    )
    task = await driver.handle_batch_result(
        task_id="task_001",
        batch_result=LongTaskBatchResult(
            batch_id="batch_002",
            task_id="task_001",
            step_results=[
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="awaiting_input",
                    waiting_reason="missing_input",
                    user_prompt="请补充狗狗年龄。",
                ),
                LongTaskBatchStepResult(
                    step_id="step_2",
                    status="awaiting_input",
                    waiting_reason="approval",
                    user_prompt="是否允许读取健康档案？",
                ),
            ],
        ),
        execution_context=execution_context(),
    )

    assert task.status == "awaiting_input"
    assert task.active_step_ids == ["step_1", "step_2"]
    assert task.pending_interaction is not None
    assert task.pending_interaction.interaction_type == "missing_input"
    assert task.pending_interaction.prompt == (
        "继续任务前，请处理以下信息：\n"
        "1. 检查年龄：请补充狗狗年龄。\n"
        "2. 检查档案授权：是否允许读取健康档案？"
    )
    assert len(
        task.pending_interaction.input_contract["items"]
    ) == 2
    assert store.save_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step_result", "expected_message"),
    [
        (
            LongTaskBatchStepResult(
                step_id="step_1",
                status="failed",
                error_message="档案服务不可用",
            ),
            "重试与降级策略",
        ),
    ],
)
async def test_driver_should_reject_unhandled_action_without_saving(
    step_result: LongTaskBatchStepResult,
    expected_message: str,
) -> None:
    """验证尚未接入的失败治理不会产生半成品快照。"""

    original_task = build_running_task()
    store = InMemoryDriverStore(original_task)

    with pytest.raises(
        UnsupportedLongTaskDriverActionError,
        match=expected_message,
    ):
        await LongTaskRuntimeDriver(store).handle_batch_result(
            task_id="task_001",
            batch_result=build_batch_result(step_result),
            execution_context=execution_context(exhausted=True),
        )

    assert store.task == original_task
    assert store.save_count == 0
