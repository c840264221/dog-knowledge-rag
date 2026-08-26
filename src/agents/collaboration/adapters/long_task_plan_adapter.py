"""把尚未执行的多 Agent 计划适配为统一 LongTask 契约。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.collaboration.contracts.schemas import (
    AgentTaskPlan,
    AgentTaskResult,
    MultiAgentTaskResult,
)
from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskExecutionMode,
    LongTaskGoal,
    LongTaskProgressionMode,
    LongTaskStep,
)


class UnsupportedCollaborationPlanError(ValueError):
    """表示协作计划已经进入本 MVP 不负责迁移的运行或等待状态。"""


def adapt_collaboration_plan_to_long_task(
    *,
    plan: AgentTaskPlan,
    user_id: str,
    thread_id: str,
    original_request: str,
    task_id: str | None = None,
    execution_mode: LongTaskExecutionMode = "inline",
    progression_mode: LongTaskProgressionMode = "automatic",
) -> LongTask:
    """
    把一份尚未执行的多 Agent 计划转换为长任务初始快照。

    功能：
        保留目标、Agent 分工和拓扑依赖，为步骤编号增加 Task 命名空间，
        并把无依赖根步骤初始化为 ready。函数不修改原计划、不保存 Redis，
        也不调用现有 Scheduler。

    参数含义：
        plan:
            Planner 已生成、但尚未开始执行且无需用户澄清的协作计划。
        user_id:
            长任务所属用户编号，用于权限和数据隔离。
        thread_id:
            创建长任务的对话线程编号。
        original_request:
            触发当前计划的用户原始输入，用于审计和重新理解需求。
        task_id:
            可选长任务编号；未提供时复用 plan_id。
        execution_mode:
            初始采用请求内 inline 还是后台 durable 执行。
        progression_mode:
            批次完成后自动推进还是等待人工确认。

    返回值含义：
        LongTask:
            保留原计划拓扑、可以交给长任务 Runtime 的 created 状态快照。
    """

    _validate_new_collaboration_plan(plan)
    resolved_task_id = task_id or plan.plan_id
    long_step_id_by_plan_step_id = {
        step.step_id: f"{resolved_task_id}:{step.step_id}"
        for step in plan.steps
    }
    long_task_steps = [
        LongTaskStep(
            step_id=long_step_id_by_plan_step_id[step.step_id],
            task_id=resolved_task_id,
            title=step.title,
            description=step.description,
            assigned_agent=step.assigned_agent,
            depends_on=[
                long_step_id_by_plan_step_id[dependency_id]
                for dependency_id in step.depends_on
            ],
            input_data=dict(step.input_data),
            expected_output=step.expected_output,
            status="pending" if step.depends_on else "ready",
            metadata={
                **step.metadata,
                "source_plan_step_id": step.step_id,
                "allow_failure": step.allow_failure,
            },
        )
        for step in plan.steps
    ]

    return LongTask(
        task_id=resolved_task_id,
        user_id=user_id,
        thread_id=thread_id,
        goal=LongTaskGoal(
            original_request=original_request,
            objective=plan.objective,
        ),
        steps=long_task_steps,
        status="created",
        execution_mode=execution_mode,
        progression_mode=progression_mode,
        metadata={
            **plan.metadata,
            "source_type": "collaboration_plan",
            "source_plan_id": plan.plan_id,
            "plan_reason": plan.reason,
        },
    )


def adapt_paused_collaboration_result_to_long_task(
    *,
    task_result: MultiAgentTaskResult,
    user_id: str,
    thread_id: str,
    original_request: str,
    task_id: str | None = None,
    progression_mode: LongTaskProgressionMode = "automatic",
) -> LongTask:
    """
    把批次预算耗尽的多 Agent 中间结果转换为 durable LongTask。

    功能：
        保留已完成或跳过步骤的摘要、Artifact 引用和执行次数，为剩余步骤
        重新计算 ready/pending，并保持原 Plan 的 Agent 分工与拓扑。本函数
        只创建版本 1 快照，不保存 Redis，也不发布 Stream 消息。

    参数含义：
        task_result:
            Scheduler 在完整批次边界返回、携带 execution_paused 事实的
            MultiAgentTaskResult。
        user_id:
            新 LongTask 所属用户编号。
        thread_id:
            新 LongTask 所属对话线程编号。
        original_request:
            触发本次多智能体计划的原始用户输入。
        task_id:
            可选持久化任务编号；为空时复用 collaboration_id。
        progression_mode:
            后台批次完成后自动推进还是等待人工确认。

    返回值含义：
        LongTask:
            已保留请求内历史、剩余根步骤为 ready、执行模式为 durable 的
            版本 1 运行中任务快照。
    """

    pause_facts = _validate_paused_collaboration_result(task_result)
    plan = task_result.plan
    resolved_task_id = task_id or task_result.collaboration_id
    long_step_id_by_plan_step_id = {
        step.step_id: f"{resolved_task_id}:{step.step_id}"
        for step in plan.steps
    }
    results_by_step_id = {
        result.step_id: result
        for result in task_result.task_results
    }
    long_task_steps: list[LongTaskStep] = []
    for plan_step in plan.steps:
        result = results_by_step_id.get(plan_step.step_id)
        if result is not None:
            attempt_count = int(
                result.metadata.get("scheduler_attempt_count", 0) or 0
            )
            long_task_steps.append(
                LongTaskStep(
                    step_id=long_step_id_by_plan_step_id[plan_step.step_id],
                    task_id=resolved_task_id,
                    title=plan_step.title,
                    description=plan_step.description,
                    assigned_agent=plan_step.assigned_agent,
                    depends_on=[
                        long_step_id_by_plan_step_id[dependency_id]
                        for dependency_id in plan_step.depends_on
                    ],
                    input_data=dict(plan_step.input_data),
                    expected_output=plan_step.expected_output,
                    status=result.status,
                    output_ref=_extract_result_output_ref(result),
                    output_summary=result.summary,
                    attempt_count=attempt_count,
                    max_attempts=max(2, attempt_count),
                    metadata=_build_migrated_step_metadata(
                        plan_step_metadata=plan_step.metadata,
                        allow_failure=plan_step.allow_failure,
                        source_step_id=plan_step.step_id,
                        result=result,
                    ),
                )
            )
            continue

        dependencies_completed = all(
            dependency_id in results_by_step_id
            and results_by_step_id[dependency_id].status
            in {"completed", "skipped"}
            for dependency_id in plan_step.depends_on
        )
        long_task_steps.append(
            LongTaskStep(
                step_id=long_step_id_by_plan_step_id[plan_step.step_id],
                task_id=resolved_task_id,
                title=plan_step.title,
                description=plan_step.description,
                assigned_agent=plan_step.assigned_agent,
                depends_on=[
                    long_step_id_by_plan_step_id[dependency_id]
                    for dependency_id in plan_step.depends_on
                ],
                input_data=dict(plan_step.input_data),
                expected_output=plan_step.expected_output,
                status="ready" if dependencies_completed else "pending",
                metadata={
                    **plan_step.metadata,
                    "source_plan_step_id": plan_step.step_id,
                    "allow_failure": plan_step.allow_failure,
                },
            )
        )

    return LongTask(
        task_id=resolved_task_id,
        user_id=user_id,
        thread_id=thread_id,
        goal=LongTaskGoal(
            original_request=original_request,
            objective=plan.objective,
        ),
        steps=long_task_steps,
        status="running",
        execution_mode="durable",
        progression_mode=progression_mode,
        metadata={
            **plan.metadata,
            "source_type": "collaboration_budget_handoff",
            "source_plan_id": plan.plan_id,
            "source_collaboration_id": task_result.collaboration_id,
            "plan_reason": plan.reason,
            "execution_paused": dict(pause_facts),
            "ready_batches": [
                list(batch)
                for batch in task_result.metadata.get("ready_batches", [])
                if isinstance(batch, list)
            ],
            "worker_step_trace": [
                dict(item)
                for item in task_result.metadata.get(
                    "worker_step_trace",
                    [],
                )
                if isinstance(item, Mapping)
            ],
        },
    )


def _validate_paused_collaboration_result(
    task_result: MultiAgentTaskResult,
) -> Mapping[str, Any]:
    """
    校验多 Agent 结果位于当前 MVP 可以安全迁移的预算暂停边界。

    参数含义：
        task_result:
            准备转换为 LongTask 的多 Agent 中间结果。

    返回值含义：
        Mapping[str, Any]:
            校验通过后的 execution_paused 事实映射。
    """

    pause_facts = task_result.metadata.get("execution_paused")
    if (
        not isinstance(pause_facts, Mapping)
        or pause_facts.get("reason") != "inline_budget_exhausted"
    ):
        raise UnsupportedCollaborationPlanError(
            "只支持 inline_budget_exhausted 的批次暂停结果"
        )
    if task_result.status != "running" or task_result.plan.status != "running":
        raise UnsupportedCollaborationPlanError(
            "预算暂停迁移要求 TaskResult 和 Plan 均为 running"
        )
    if task_result.plan.requires_user_input:
        raise UnsupportedCollaborationPlanError(
            "等待用户输入的计划不能按预算暂停迁移"
        )
    unsupported_result_ids = [
        result.step_id
        for result in task_result.task_results
        if result.status not in {"completed", "skipped"}
    ]
    if unsupported_result_ids:
        raise UnsupportedCollaborationPlanError(
            "预算暂停迁移只接受 completed/skipped 历史步骤: "
            f"{unsupported_result_ids}"
        )
    planned_step_ids = [step.step_id for step in task_result.plan.steps]
    result_step_ids = {result.step_id for result in task_result.task_results}
    expected_remaining_step_ids = [
        step_id
        for step_id in planned_step_ids
        if step_id not in result_step_ids
    ]
    raw_remaining_step_ids = pause_facts.get("remaining_step_ids")
    if not isinstance(raw_remaining_step_ids, list):
        raise UnsupportedCollaborationPlanError(
            "预算暂停结果缺少 remaining_step_ids"
        )
    actual_remaining_step_ids = [
        str(step_id)
        for step_id in raw_remaining_step_ids
    ]
    if actual_remaining_step_ids != expected_remaining_step_ids:
        raise UnsupportedCollaborationPlanError(
            "预算暂停剩余步骤与 Plan 历史不一致"
        )
    if not expected_remaining_step_ids:
        raise UnsupportedCollaborationPlanError(
            "没有剩余步骤的计划不需要升级为 LongTask"
        )
    return pause_facts


def _extract_result_output_ref(result: AgentTaskResult) -> str | None:
    """
    从已完成协作结果中提取可供后续 Worker 召回的大型输出引用。

    参数含义：
        result:
            请求内 Worker 已经产生的步骤结果。

    返回值含义：
        str | None:
            规范化 Artifact 引用；结果没有持久化大型输出时返回 None。
    """

    raw_output_ref = result.output.get("output_ref")
    if raw_output_ref is None:
        raw_output_ref = result.metadata.get("output_ref")
    if raw_output_ref is None:
        return None
    normalized_output_ref = str(raw_output_ref).strip()
    return normalized_output_ref or None


def _build_migrated_step_metadata(
    *,
    plan_step_metadata: Mapping[str, Any],
    allow_failure: bool,
    source_step_id: str,
    result: AgentTaskResult,
) -> dict[str, Any]:
    """
    构建已执行 Step 迁移到 LongTask 后的审计元数据。

    参数含义：
        plan_step_metadata:
            Planner为原步骤生成的扩展信息。
        allow_failure:
            原计划是否允许该步骤失败后继续。
        source_step_id:
            添加 LongTask 命名空间前的原始 Step ID。
        result:
            请求内 Scheduler 保存的实际步骤结果。

    返回值含义：
        dict[str, Any]:
            不复制大型 output、但保留结果来源和执行事实的元数据。
    """

    return {
        **plan_step_metadata,
        "source_plan_step_id": source_step_id,
        "allow_failure": allow_failure,
        "migrated_from_inline_result": True,
        "source_result_metadata": dict(result.metadata),
        "source_result_evidence_ids": list(result.evidence_ids),
        "source_result_latency_ms": result.latency_ms,
    }


def _validate_new_collaboration_plan(plan: AgentTaskPlan) -> None:
    """
    检查计划仍处于可以安全创建新 LongTask 的初始边界。

    参数含义：
        plan:
            准备转换的多 Agent 计划。

    返回值含义：
        None。计划已经运行、等待或包含非 pending 步骤时抛出异常。
    """

    if plan.status != "planned":
        raise UnsupportedCollaborationPlanError(
            "只支持 status=planned 的全新协作计划"
        )
    if plan.requires_user_input:
        raise UnsupportedCollaborationPlanError(
            "需要用户澄清的协作计划应先完成交互，再创建 LongTask"
        )
    non_pending_step_ids = [
        step.step_id
        for step in plan.steps
        if step.status != "pending"
    ]
    if non_pending_step_ids:
        raise UnsupportedCollaborationPlanError(
            "只支持全部步骤为 pending 的全新协作计划: "
            f"{non_pending_step_ids}"
        )
