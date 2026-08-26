"""长任务批次完成后的事实投影与纯执行决策策略。"""

from __future__ import annotations

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskExecutionContext,
    LongTaskExecutionDecision,
    LongTaskProjectedExecutionFacts,
)


def project_long_task_execution_facts(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
    execution_context: LongTaskExecutionContext,
) -> LongTaskProjectedExecutionFacts:
    """
    把当前批次结果投影到完整任务并计算 PDP 需要的派生事实。

    功能：
        在内存中覆盖当前批次涉及步骤的状态，计算剩余步骤、下一批 Ready
        Step、成功终态和同步预算。函数不会修改传入的不可变 Task。

    参数含义：
        task:
            执行当前批次前加载的完整长任务快照。
        batch_result:
            单 Agent 或多 Agent 返回的当前批次事实。
        execution_context:
            当前请求已用时间和允许的同步执行预算。

    返回值含义：
        LongTaskProjectedExecutionFacts:
            假设当前批次结果生效后得到的不可变策略事实。
    """

    if batch_result.task_id != task.task_id:
        raise ValueError("BatchResult 与 LongTask 的 task_id 不一致")

    steps_by_id = {step.step_id: step for step in task.steps}
    result_step_ids = {
        result.step_id for result in batch_result.step_results
    }
    unknown_step_ids = sorted(result_step_ids - set(steps_by_id))
    if unknown_step_ids:
        raise ValueError(
            f"BatchResult 引用了任务外步骤: {unknown_step_ids}"
        )

    projected_status_by_step_id = {
        step.step_id: step.status for step in task.steps
    }
    for result in batch_result.step_results:
        projected_status_by_step_id[result.step_id] = result.status

    completed_step_ids = [
        step.step_id
        for step in task.steps
        if projected_status_by_step_id[step.step_id] == "completed"
    ]
    skipped_step_ids = [
        step.step_id
        for step in task.steps
        if projected_status_by_step_id[step.step_id] == "skipped"
    ]
    awaiting_step_ids = [
        step.step_id
        for step in task.steps
        if projected_status_by_step_id[step.step_id] == "awaiting_input"
    ]
    failed_step_ids = [
        step.step_id
        for step in task.steps
        if projected_status_by_step_id[step.step_id] == "failed"
    ]
    successfully_terminal_step_ids = {
        *completed_step_ids,
        *skipped_step_ids,
    }
    remaining_step_ids = [
        step.step_id
        for step in task.steps
        if step.step_id not in successfully_terminal_step_ids
    ]
    ready_step_ids = [
        step.step_id
        for step in task.steps
        if projected_status_by_step_id[step.step_id]
        in {"pending", "ready"}
        and set(step.depends_on).issubset(
            successfully_terminal_step_ids
        )
    ]

    return LongTaskProjectedExecutionFacts(
        task_id=task.task_id,
        batch_id=batch_result.batch_id,
        execution_mode=task.execution_mode,
        remaining_step_ids=remaining_step_ids,
        ready_step_ids=ready_step_ids,
        completed_step_ids=completed_step_ids,
        awaiting_step_ids=awaiting_step_ids,
        failed_step_ids=failed_step_ids,
        skipped_step_ids=skipped_step_ids,
        all_steps_successfully_terminal=not remaining_step_ids,
        inline_budget_exhausted=(
            execution_context.elapsed_ms
            >= execution_context.inline_budget_ms
        ),
    )


def decide_long_task_execution(
    facts: LongTaskProjectedExecutionFacts,
) -> LongTaskExecutionDecision:
    """
    根据预测执行事实计算外层下一步应该采取的动作。

    功能：
        按失败、等待用户、成功完成、转后台和正常推进的优先级返回决策。
        本函数不修改 LongTask、不访问 Redis，也不调用 Agent 或工具。

    参数含义：
        facts:
            完整任务、当前批次和运行预算共同计算出的预测执行事实。

    返回值含义：
        LongTaskExecutionDecision:
            包含下一动作、原因和关键事实摘要的不可变决策对象。
    """

    if facts.failed_step_ids:
        action = "handle_failure"
        reason = "任务包含失败步骤，需要外层执行重试、降级或失败处理。"
    elif facts.awaiting_step_ids:
        action = "await_user_input"
        reason = "任务包含等待步骤，需要外层构建交互并暂停任务。"
    elif facts.all_steps_successfully_terminal:
        action = "complete_task"
        reason = "全部步骤已经完成或跳过，可以由外层完成整份任务。"
    elif (
        facts.execution_mode == "inline"
        and facts.inline_budget_exhausted
    ):
        action = "promote_to_durable"
        reason = "任务仍有剩余步骤且同步预算耗尽，需要转入后台执行。"
    else:
        action = "advance_task"
        reason = "任务仍有可处理步骤且没有阻塞，可以检查并调度下一批。"

    return LongTaskExecutionDecision(
        task_id=facts.task_id,
        batch_id=facts.batch_id,
        action=action,
        reason=reason,
        completed_step_ids=facts.completed_step_ids,
        awaiting_step_ids=facts.awaiting_step_ids,
        failed_step_ids=facts.failed_step_ids,
        skipped_step_ids=facts.skipped_step_ids,
        remaining_step_ids=facts.remaining_step_ids,
        ready_step_ids=facts.ready_step_ids,
        inline_budget_exhausted=facts.inline_budget_exhausted,
    )
