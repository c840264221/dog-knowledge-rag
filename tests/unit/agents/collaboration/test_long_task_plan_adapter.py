"""多 Agent 计划到 LongTask 的适配器单元测试。"""

from __future__ import annotations

import pytest

from src.agents.collaboration.adapters import (
    UnsupportedCollaborationPlanError,
    adapt_collaboration_plan_to_long_task,
    adapt_paused_collaboration_result_to_long_task,
)
from src.agents.collaboration.contracts.schemas import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)


def build_plan(
    *,
    status: str = "planned",
    requires_user_input: bool = False,
    first_step_status: str = "pending",
) -> AgentTaskPlan:
    """
    构建适配器测试使用的两步骤多 Agent 计划。

    参数含义：
        status:
            整份协作计划的状态。
        requires_user_input:
            当前计划是否仍需用户补充信息。
        first_step_status:
            第一个步骤的状态，用于覆盖非法迁移场景。

    返回值含义：
        AgentTaskPlan:
            包含一个根步骤和一个依赖步骤的测试计划。
    """

    return AgentTaskPlan(
        plan_id="plan_001",
        objective="结合档案和犬种知识生成照护建议",
        status=status,
        requires_user_input=requires_user_input,
        clarification_prompt=(
            "请补充狗狗年龄。" if requires_user_input else ""
        ),
        reason="需要先读取档案，再检索犬种知识。",
        metadata={"planner_version": "v1"},
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取档案",
                assigned_agent="memory_agent",
                status=first_step_status,
                expected_output="狗狗年龄和体重",
                metadata={"source": "profile"},
            ),
            AgentTaskStep(
                step_id="query_knowledge",
                title="检索犬种知识",
                description="根据档案检索相关照护知识。",
                assigned_agent="dog_knowledge_agent",
                depends_on=["load_profile"],
                input_data={"question": "如何照护幼犬？"},
                expected_output="带证据的照护建议",
                allow_failure=True,
            ),
        ],
    )


def build_paused_result(
    *,
    first_result_status: str = "completed",
    remaining_step_ids: list[str] | None = None,
) -> MultiAgentTaskResult:
    """
    构建第一批完成后因请求内预算耗尽的三步骤协作结果。

    参数含义：
        first_result_status:
            第一批步骤结果状态，用于覆盖安全历史和拒绝失败历史。
        remaining_step_ids:
            可选暂停事实中的剩余步骤编号，默认与计划真实剩余步骤一致。

    返回值含义：
        MultiAgentTaskResult:
            Plan/TaskResult 均为 running、包含一条历史结果的暂停快照。
    """

    first_result = AgentTaskResult(
        step_id="load_profile",
        assigned_agent="memory_agent",
        status=first_result_status,
        summary=(
            "档案已读取：6岁，28公斤。"
            if first_result_status == "completed"
            else "档案读取失败。"
        ),
        output={
            "output_ref": "artifact://profile/result-001",
            "large_private_payload": "不应复制进 LongTask metadata",
        },
        error_message=(
            "档案服务不可用"
            if first_result_status == "failed"
            else None
        ),
        evidence_ids=["profile:user_001:dog_001"],
        latency_ms=25.0,
        metadata={"scheduler_attempt_count": 1},
    )
    plan = AgentTaskPlan(
        plan_id="plan_paused_001",
        objective="读取档案并生成照护建议",
        status="running",
        reason="先读取档案，再检索知识并汇总。",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取档案",
                assigned_agent="memory_agent",
                status=first_result_status,
            ),
            AgentTaskStep(
                step_id="query_knowledge",
                title="检索知识",
                assigned_agent="dog_knowledge_agent",
                depends_on=["load_profile"],
                status="pending",
            ),
            AgentTaskStep(
                step_id="build_answer",
                title="生成回答",
                assigned_agent="general_agent",
                depends_on=["query_knowledge"],
                status="pending",
            ),
        ],
    )
    return MultiAgentTaskResult(
        collaboration_id="multi_agent_task_paused_001",
        plan=plan,
        status="running",
        task_results=[first_result],
        metadata={
            "ready_batches": [["load_profile"]],
            "worker_step_trace": [
                {
                    "step_id": "load_profile",
                    "status": first_result_status,
                }
            ],
            "execution_paused": {
                "reason": "inline_budget_exhausted",
                "elapsed_seconds": 2.0,
                "budget_seconds": 1.0,
                "remaining_step_ids": (
                    remaining_step_ids
                    if remaining_step_ids is not None
                    else ["query_knowledge", "build_answer"]
                ),
            },
            "awaiting_result_aggregation": False,
        },
    )


def test_adapter_should_preserve_plan_topology_and_business_fields() -> None:
    """验证适配器保留目标、Agent 分工、输入、依赖和兼容元数据。"""

    plan = build_plan()
    task = adapt_collaboration_plan_to_long_task(
        plan=plan,
        user_id="user_001",
        thread_id="thread_001",
        original_request="根据我家狗狗的档案给出照护建议",
    )

    assert task.task_id == "plan_001"
    assert task.goal.objective == plan.objective
    assert task.goal.original_request.startswith("根据我家狗狗")
    assert task.status == "created"
    assert task.execution_mode == "inline"
    assert task.progression_mode == "automatic"
    assert [step.step_id for step in task.steps] == [
        "plan_001:load_profile",
        "plan_001:query_knowledge",
    ]
    assert task.steps[0].status == "ready"
    assert task.steps[1].status == "pending"
    assert task.steps[1].depends_on == ["plan_001:load_profile"]
    assert task.steps[1].input_data == {
        "question": "如何照护幼犬？"
    }
    assert task.steps[1].metadata["source_plan_step_id"] == (
        "query_knowledge"
    )
    assert task.steps[1].metadata["allow_failure"] is True
    assert task.metadata["source_plan_id"] == "plan_001"
    assert plan.steps[0].status == "pending"


def test_adapter_should_accept_explicit_runtime_modes_and_task_id() -> None:
    """验证调用方可以显式选择长任务编号和两种运行模式。"""

    task = adapt_collaboration_plan_to_long_task(
        plan=build_plan(),
        user_id="user_001",
        thread_id="thread_001",
        original_request="生成计划",
        task_id="long_task_001",
        execution_mode="durable",
        progression_mode="guided",
    )

    assert task.task_id == "long_task_001"
    assert task.execution_mode == "durable"
    assert task.progression_mode == "guided"
    assert task.steps[0].step_id == "long_task_001:load_profile"


def test_adapter_should_reject_waiting_or_started_plan() -> None:
    """验证 MVP 不伪造已经等待或开始执行的计划迁移历史。"""

    with pytest.raises(
        UnsupportedCollaborationPlanError,
        match="status=planned",
    ):
        adapt_collaboration_plan_to_long_task(
            plan=build_plan(status="awaiting_input"),
            user_id="user_001",
            thread_id="thread_001",
            original_request="生成计划",
        )

    with pytest.raises(
        UnsupportedCollaborationPlanError,
        match="完成交互",
    ):
        adapt_collaboration_plan_to_long_task(
            plan=build_plan(requires_user_input=True),
            user_id="user_001",
            thread_id="thread_001",
            original_request="生成计划",
        )

    with pytest.raises(
        UnsupportedCollaborationPlanError,
        match="pending",
    ):
        adapt_collaboration_plan_to_long_task(
            plan=build_plan(first_step_status="running"),
            user_id="user_001",
            thread_id="thread_001",
            original_request="生成计划",
        )


def test_paused_adapter_should_preserve_history_and_rebuild_readiness() -> None:
    """验证暂停迁移保留完成历史并只激活依赖已经满足的剩余根步骤。"""

    paused_result = build_paused_result()

    task = adapt_paused_collaboration_result_to_long_task(
        task_result=paused_result,
        user_id="user_001",
        thread_id="thread_001",
        original_request="请结合档案生成照护建议",
    )

    assert task.task_id == "multi_agent_task_paused_001"
    assert task.status == "running"
    assert task.execution_mode == "durable"
    assert task.version == 1
    assert [step.status for step in task.steps] == [
        "completed",
        "ready",
        "pending",
    ]
    completed_step, ready_step, pending_step = task.steps
    assert completed_step.step_id == (
        "multi_agent_task_paused_001:load_profile"
    )
    assert completed_step.output_summary == "档案已读取：6岁，28公斤。"
    assert completed_step.output_ref == "artifact://profile/result-001"
    assert completed_step.attempt_count == 1
    assert completed_step.metadata["migrated_from_inline_result"] is True
    assert "large_private_payload" not in str(completed_step.metadata)
    assert ready_step.depends_on == [completed_step.step_id]
    assert pending_step.depends_on == [ready_step.step_id]
    assert task.metadata["source_type"] == (
        "collaboration_budget_handoff"
    )
    assert task.metadata["execution_paused"]["remaining_step_ids"] == [
        "query_knowledge",
        "build_answer",
    ]
    assert paused_result.plan.steps[1].status == "pending"


def test_paused_adapter_should_reject_failed_or_inconsistent_history() -> None:
    """验证失败批次和剩余步骤不一致时不会伪造可恢复 LongTask。"""

    with pytest.raises(
        UnsupportedCollaborationPlanError,
        match="completed/skipped",
    ):
        adapt_paused_collaboration_result_to_long_task(
            task_result=build_paused_result(first_result_status="failed"),
            user_id="user_001",
            thread_id="thread_001",
            original_request="生成建议",
        )

    with pytest.raises(
        UnsupportedCollaborationPlanError,
        match="剩余步骤",
    ):
        adapt_paused_collaboration_result_to_long_task(
            task_result=build_paused_result(
                remaining_step_ids=["build_answer"]
            ),
            user_id="user_001",
            thread_id="thread_001",
            original_request="生成建议",
        )
