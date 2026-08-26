"""长任务预测执行事实与 PDP 决策策略单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskExecutionContext,
    LongTaskExecutionDecision,
    LongTaskGoal,
    LongTaskProjectedExecutionFacts,
    LongTaskStep,
)
from src.runtime.long_tasks.execution_policy import (
    decide_long_task_execution,
    project_long_task_execution_facts,
)


def build_task(
    *,
    steps: list[LongTaskStep] | None = None,
    active_step_ids: list[str] | None = None,
    execution_mode: str = "inline",
) -> LongTask:
    """
    构建执行策略测试使用的最小长任务。

    参数含义：
        steps:
            可选的完整拓扑步骤列表。
        active_step_ids:
            当前批次已经启动的步骤编号。
        execution_mode:
            当前任务是在请求内执行还是后台持久化执行。

    返回值含义：
        LongTask:
            可以参与预测事实计算的完整任务快照。
    """

    resolved_steps = steps or [
        LongTaskStep(
            step_id="step_1",
            task_id="task_001",
            title="检索资料",
            assigned_agent="dog_knowledge_agent",
            status="running",
        )
    ]
    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="制定狗狗健康计划",
            objective="生成分步骤健康计划",
        ),
        steps=resolved_steps,
        status="running",
        execution_mode=execution_mode,
        active_step_ids=active_step_ids or ["step_1"],
    )


def build_batch_result(
    *step_results: LongTaskBatchStepResult,
    task_id: str = "task_001",
) -> LongTaskBatchResult:
    """
    构建执行策略测试使用的最小批次结果。

    参数含义：
        step_results:
            当前批次包含的一个或多个步骤结果。
        task_id:
            当前批次所属的长任务编号。

    返回值含义：
        LongTaskBatchResult:
            可以参与预测事实计算的统一批次结果。
    """

    return LongTaskBatchResult(
        batch_id="batch_001",
        task_id=task_id,
        step_results=list(step_results),
    )


def build_facts(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
    elapsed_ms: float = 100,
    inline_budget_ms: float = 1000,
) -> LongTaskProjectedExecutionFacts:
    """
    使用测试计时信息构建一次预测执行事实。

    参数含义：
        task:
            执行当前批次前的完整任务。
        batch_result:
            当前批次产生的步骤结果。
        elapsed_ms:
            当前请求已经执行的毫秒数。
        inline_budget_ms:
            当前请求允许使用的同步执行毫秒数。

    返回值含义：
        LongTaskProjectedExecutionFacts:
            可直接交给 PDP 的预测事实。
    """

    return project_long_task_execution_facts(
        task=task,
        batch_result=batch_result,
        execution_context=LongTaskExecutionContext(
            elapsed_ms=elapsed_ms,
            inline_budget_ms=inline_budget_ms,
        ),
    )


def test_projection_should_find_remaining_and_ready_steps() -> None:
    """验证当前批次完成后可以计算剩余步骤和下一批 Ready Step。"""

    task = build_task(
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="检索资料",
                assigned_agent="dog_knowledge_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            ),
        ]
    )
    facts = build_facts(
        task=task,
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
    )

    assert facts.remaining_step_ids == ["step_2"]
    assert facts.ready_step_ids == ["step_2"]
    assert facts.all_steps_successfully_terminal is False
    assert facts.inline_budget_exhausted is False


def test_all_successful_steps_should_complete_task_before_budget_check() -> None:
    """验证最后一步完成后即使预算耗尽也应该优先完成任务。"""

    facts = build_facts(
        task=build_task(),
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
        elapsed_ms=1000,
        inline_budget_ms=1000,
    )
    decision = decide_long_task_execution(facts)

    assert facts.all_steps_successfully_terminal is True
    assert decision.action == "complete_task"
    assert decision.remaining_step_ids == []


def test_exhausted_inline_budget_should_promote_unfinished_task() -> None:
    """验证同步预算耗尽且仍有工作时应该升级为后台持久化执行。"""

    task = build_task(
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="检索资料",
                assigned_agent="dog_knowledge_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            ),
        ]
    )
    facts = build_facts(
        task=task,
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
        elapsed_ms=1200,
        inline_budget_ms=1000,
    )
    decision = decide_long_task_execution(facts)

    assert decision.action == "promote_to_durable"
    assert decision.remaining_step_ids == ["step_2"]
    assert decision.ready_step_ids == ["step_2"]


def test_durable_task_should_advance_even_when_inline_budget_is_exhausted() -> None:
    """验证已经在后台执行的任务不会重复执行 inline 升级。"""

    task = build_task(
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="检索资料",
                assigned_agent="dog_knowledge_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            ),
        ],
        execution_mode="durable",
    )
    facts = build_facts(
        task=task,
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
            )
        ),
        elapsed_ms=1200,
        inline_budget_ms=1000,
    )

    assert decide_long_task_execution(facts).action == "advance_task"


def test_waiting_step_should_request_user_input_before_budget_check() -> None:
    """验证等待用户属于硬阻塞，即使预算耗尽也不能直接转入后台。"""

    facts = build_facts(
        task=build_task(),
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="awaiting_input",
                waiting_reason="approval",
                user_prompt="是否继续生成健康计划？",
            )
        ),
        elapsed_ms=1200,
        inline_budget_ms=1000,
    )
    decision = decide_long_task_execution(facts)

    assert decision.action == "await_user_input"
    assert decision.awaiting_step_ids == ["step_1"]
    assert decision.inline_budget_exhausted is True


def test_failure_should_take_priority_without_losing_waiting_facts() -> None:
    """验证失败优先处理，同时保留同一批次中的等待步骤事实。"""

    task = build_task(
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="健康分析",
                assigned_agent="health_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="训练计划",
                assigned_agent="training_agent",
                status="running",
            ),
        ],
        active_step_ids=["step_1", "step_2"],
    )
    facts = build_facts(
        task=task,
        batch_result=build_batch_result(
            LongTaskBatchStepResult(
                step_id="step_1",
                status="failed",
                error_message="健康服务暂时不可用",
            ),
            LongTaskBatchStepResult(
                step_id="step_2",
                status="awaiting_input",
                waiting_reason="missing_input",
                user_prompt="请补充狗狗年龄。",
            ),
        ),
    )
    decision = decide_long_task_execution(facts)

    assert decision.action == "handle_failure"
    assert decision.failed_step_ids == ["step_1"]
    assert decision.awaiting_step_ids == ["step_2"]


def test_projection_should_reject_foreign_task_or_step() -> None:
    """验证事实投影不能接收其他任务或任务外步骤的批次结果。"""

    with pytest.raises(ValueError, match="task_id 不一致"):
        build_facts(
            task=build_task(),
            batch_result=build_batch_result(
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="completed",
                ),
                task_id="task_002",
            ),
        )

    with pytest.raises(ValueError, match="任务外步骤"):
        build_facts(
            task=build_task(),
            batch_result=build_batch_result(
                LongTaskBatchStepResult(
                    step_id="step_missing",
                    status="completed",
                )
            ),
        )


def test_decision_contract_should_reject_invalid_durable_promotion() -> None:
    """验证后台升级决策必须同时存在剩余工作和预算耗尽事实。"""

    with pytest.raises(ValidationError, match="剩余步骤"):
        LongTaskExecutionDecision(
            task_id="task_001",
            batch_id="batch_001",
            action="promote_to_durable",
            reason="错误的后台升级决策",
            inline_budget_exhausted=True,
        )

    with pytest.raises(ValidationError, match="预算已经耗尽"):
        LongTaskExecutionDecision(
            task_id="task_001",
            batch_id="batch_001",
            action="promote_to_durable",
            reason="错误的后台升级决策",
            remaining_step_ids=["step_2"],
        )
