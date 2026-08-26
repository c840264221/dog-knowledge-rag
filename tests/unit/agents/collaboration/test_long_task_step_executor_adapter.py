"""LongTask Step 到现有多 Agent Worker 的执行适配测试。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.agents.collaboration.adapters import (
    LongTaskStepExecutionAdapterError,
    LongTaskStepExecutorAdapter,
)
from src.agents.collaboration.contracts import (
    AgentTaskResult,
    AgentTaskStep,
)
from src.runtime.long_tasks import (
    LongTask,
    LongTaskGoal,
    LongTaskStep,
)
from src.runtime.long_tasks.contracts import utc_now


def build_claimed_task() -> LongTask:
    """
    构建一个依赖已完成步骤的 running 后台任务。

    返回值含义：
        LongTask：当前 step_2 已由 worker-1 领取的任务快照。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="制定健康计划",
            objective="结合档案制定狗狗健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="读取档案",
                assigned_agent="profile_agent",
                status="completed",
                output_summary="体重 10kg",
                output_ref="artifact://profile-001",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成建议",
                description="根据档案生成健康建议",
                assigned_agent="health_agent",
                depends_on=["step_1"],
                input_data={"question": "生成健康建议"},
                status="running",
                claimed_by="worker-1",
                claim_id="claim-001",
                lease_expires_at=utc_now() + timedelta(seconds=30),
            ),
        ],
        status="running",
        execution_mode="durable",
        active_step_ids=["step_2"],
    )


@pytest.mark.asyncio
async def test_adapter_should_execute_registered_worker() -> None:
    """验证适配器保留目标、依赖摘要、Artifact 引用和领取令牌。"""

    observed_step: AgentTaskStep | None = None
    observed_dependencies: dict[str, AgentTaskResult] = {}

    async def health_worker(
        step: AgentTaskStep,
        dependencies: dict[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """记录转换后的 Worker 输入并返回完成结果。"""

        nonlocal observed_step, observed_dependencies
        observed_step = step
        observed_dependencies = dict(dependencies)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary="健康建议已生成",
            output={"output_ref": "artifact://health-001"},
            evidence_ids=["profile-001"],
        )

    task = build_claimed_task()
    result = await LongTaskStepExecutorAdapter(
        {"health_agent": health_worker}
    )(task, task.steps[1], "claim-001")

    assert observed_step is not None
    assert observed_step.status == "running"
    assert observed_step.input_data["long_task_goal"]["objective"] == (
        "结合档案制定狗狗健康计划"
    )
    assert observed_dependencies["step_1"].summary == "体重 10kg"
    assert observed_dependencies["step_1"].output == {
        "output_ref": "artifact://profile-001"
    }
    assert result.status == "completed"
    assert result.output_ref == "artifact://health-001"
    assert result.claim_id == "claim-001"


@pytest.mark.asyncio
async def test_adapter_should_convert_awaiting_input_result() -> None:
    """验证现有 Worker 的确认请求会转换成长任务等待事实。"""

    async def health_worker(
        step: AgentTaskStep,
        _: dict[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回需要用户确认的协作结果。"""

        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="awaiting_input",
            summary="等待确认健康计划",
            requires_user_input=True,
            clarification_prompt="是否采用这份健康计划？",
            metadata={"waiting_reason": "confirmation"},
        )

    task = build_claimed_task()
    result = await LongTaskStepExecutorAdapter(
        {"health_agent": health_worker}
    )(task, task.steps[1], "claim-001")

    assert result.status == "awaiting_input"
    assert result.waiting_reason == "confirmation"
    assert result.user_prompt == "是否采用这份健康计划？"


@pytest.mark.asyncio
async def test_adapter_should_reject_unregistered_worker() -> None:
    """验证 assigned_agent 没有注册时不会猜测执行器。"""

    async def another_worker(
        step: AgentTaskStep,
        _: dict[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """测试中不应被调用的其他 Worker。"""

        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
        )

    task = build_claimed_task()
    with pytest.raises(
        LongTaskStepExecutionAdapterError,
        match="没有注册对应 Worker",
    ):
        await LongTaskStepExecutorAdapter(
            {"general_agent": another_worker}
        )(task, task.steps[1], "claim-001")


@pytest.mark.asyncio
async def test_adapter_should_reject_wrong_worker_result() -> None:
    """验证 Worker 不能返回其他 Step 的结果。"""

    async def health_worker(
        step: AgentTaskStep,
        _: dict[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """模拟错误返回其他步骤编号。"""

        return AgentTaskResult(
            step_id="step-wrong",
            assigned_agent=step.assigned_agent,
            status="completed",
        )

    task = build_claimed_task()
    with pytest.raises(
        LongTaskStepExecutionAdapterError,
        match="错误 step_id",
    ):
        await LongTaskStepExecutorAdapter(
            {"health_agent": health_worker}
        )(task, task.steps[1], "claim-001")
