"""
V1.14 多 Agent 协作 Schema 单元测试。

功能：
    验证任务步骤、完整计划、步骤执行结果和最终协作结果的默认值、依赖校验
    与字典序列化，确保后续 PlannerAgent 和主图接入使用稳定契约。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)


def build_valid_plan(*, status: str = "planned") -> AgentTaskPlan:
    """
    为测试准备一份合法的两步骤协作计划。

    功能：
        第一项由 Memory 读取宠物档案，第二项等待档案完成后由狗狗知识
        Agent 查询犬种知识，方便其他测试复用统一输入。

    参数含义：
        status:
            希望测试计划使用的整体状态。

    返回值含义：
        AgentTaskPlan:
            步骤编号唯一、依赖存在且不存在循环的测试计划。
    """

    return AgentTaskPlan(
        plan_id="plan_dog_care_001",
        objective="根据宠物档案准备狗狗照护建议",
        status=status,
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取宠物档案",
                assigned_agent="memory",
                expected_output="狗狗年龄、犬种和体重",
            ),
            AgentTaskStep(
                step_id="query_knowledge",
                title="查询犬种知识",
                assigned_agent="dog_knowledge_agent",
                depends_on=["load_profile"],
                input_data={"question": "金毛幼犬需要多少运动？"},
                expected_output="带有证据的犬种照护知识",
            ),
        ],
        reason="问题需要结合用户宠物档案和犬种知识。",
    )


def test_task_step_should_use_safe_defaults() -> None:
    """
    检查任务步骤是否使用彼此独立的安全默认值。

    功能：
        确认新步骤默认等待执行，并且可变列表和字典不会在不同对象之间共享。

    参数含义：
        无。

    返回值含义：
        None。
    """

    first = AgentTaskStep(
        step_id="step_1",
        title="第一步",
        assigned_agent="general_agent",
    )
    second = AgentTaskStep(
        step_id="step_2",
        title="第二步",
        assigned_agent="general_agent",
    )

    first.depends_on.append("pre_step")
    first.input_data["question"] = "测试问题"

    assert first.status == "pending"
    assert second.depends_on == []
    assert second.input_data == {}


def test_task_step_should_reject_self_and_duplicate_dependencies() -> None:
    """
    检查任务步骤是否拒绝依赖自己和重复依赖。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="不能依赖自己"):
        AgentTaskStep(
            step_id="step_1",
            title="错误步骤",
            assigned_agent="general_agent",
            depends_on=["step_1"],
        )

    with pytest.raises(ValidationError, match="不能包含重复编号"):
        AgentTaskStep(
            step_id="step_2",
            title="重复依赖步骤",
            assigned_agent="general_agent",
            depends_on=["step_1", "step_1"],
        )


def test_task_plan_should_reject_duplicate_and_missing_step_ids() -> None:
    """
    检查完整计划是否拒绝重复步骤和不存在的依赖目标。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="step_id 不能重复"):
        AgentTaskPlan(
            plan_id="duplicate_plan",
            objective="测试重复步骤",
            steps=[
                AgentTaskStep(
                    step_id="same_step",
                    title="步骤一",
                    assigned_agent="general_agent",
                ),
                AgentTaskStep(
                    step_id="same_step",
                    title="步骤二",
                    assigned_agent="tool_agent",
                ),
            ],
        )

    with pytest.raises(ValidationError, match="不存在的前置步骤"):
        AgentTaskPlan(
            plan_id="missing_dependency_plan",
            objective="测试缺失依赖",
            steps=[
                AgentTaskStep(
                    step_id="step_1",
                    title="依赖缺失步骤",
                    assigned_agent="general_agent",
                    depends_on=["unknown_step"],
                )
            ],
        )


def test_task_plan_should_reject_dependency_cycle() -> None:
    """
    检查完整计划是否拒绝步骤之间的循环等待。

    功能：
        step_1 等待 step_2，而 step_2 又等待 step_1 时，没有任何一步能够
        开始，计划必须在进入执行器前失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="循环依赖"):
        AgentTaskPlan(
            plan_id="cycle_plan",
            objective="测试循环依赖",
            steps=[
                AgentTaskStep(
                    step_id="step_1",
                    title="步骤一",
                    assigned_agent="general_agent",
                    depends_on=["step_2"],
                ),
                AgentTaskStep(
                    step_id="step_2",
                    title="步骤二",
                    assigned_agent="tool_agent",
                    depends_on=["step_1"],
                ),
            ],
        )


def test_task_plan_should_require_prompt_when_waiting_for_user() -> None:
    """
    检查等待用户输入的计划是否提供明确问题。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="clarification_prompt"):
        AgentTaskPlan(
            plan_id="clarification_plan",
            objective="制定幼犬喂养计划",
            requires_user_input=True,
            steps=[
                AgentTaskStep(
                    step_id="collect_weight",
                    title="收集狗狗体重",
                    assigned_agent="general_agent",
                )
            ],
        )


def test_failed_task_result_should_require_error_message() -> None:
    """
    检查失败的步骤结果是否必须说明原因。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="error_message"):
        AgentTaskResult(
            step_id="query_knowledge",
            assigned_agent="dog_knowledge_agent",
            status="failed",
        )


def test_completed_collaboration_should_dump_to_plain_dict() -> None:
    """
    检查完整协作结果是否可以转换成 checkpoint 友好的普通字典。

    参数含义：
        无。

    返回值含义：
        None。
    """

    plan = build_valid_plan(status="completed")
    result = MultiAgentTaskResult(
        collaboration_id="collaboration_001",
        plan=plan,
        status="completed",
        task_results=[
            AgentTaskResult(
                step_id="load_profile",
                assigned_agent="memory",
                status="completed",
                summary="已读取三个月大金毛的档案。",
                output={"age_months": 3, "breed": "Golden Retriever"},
            ),
            AgentTaskResult(
                step_id="query_knowledge",
                assigned_agent="dog_knowledge_agent",
                status="completed",
                summary="已取得幼犬运动知识。",
                output={"exercise_advice": "采用短时、低强度活动。"},
                evidence_ids=["rag_chunk_001"],
            ),
        ],
        final_answer="三个月大的金毛应采用短时、低强度活动。",
    )

    dumped = result.model_dump(mode="python")

    assert dumped["status"] == "completed"
    assert dumped["plan"]["steps"][1]["depends_on"] == ["load_profile"]
    assert dumped["task_results"][1]["evidence_ids"] == ["rag_chunk_001"]
    assert "短时" in dumped["final_answer"]


def test_completed_collaboration_should_require_all_planned_results() -> None:
    """
    检查已完成协作是否覆盖计划中的全部步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="仍缺少步骤结果"):
        MultiAgentTaskResult(
            collaboration_id="incomplete_collaboration",
            plan=build_valid_plan(status="completed"),
            status="completed",
            task_results=[
                AgentTaskResult(
                    step_id="load_profile",
                    assigned_agent="memory",
                    status="completed",
                )
            ],
            final_answer="不完整回答",
        )


def test_collaboration_should_reject_unknown_step_and_wrong_agent() -> None:
    """
    检查协作结果是否拒绝计划外步骤和错误 Worker。

    参数含义：
        无。

    返回值含义：
        None。
    """

    plan = build_valid_plan()
    with pytest.raises(ValidationError, match="计划外步骤"):
        MultiAgentTaskResult(
            collaboration_id="unknown_step_collaboration",
            plan=plan,
            status="running",
            task_results=[
                AgentTaskResult(
                    step_id="unknown_step",
                    assigned_agent="general_agent",
                    status="completed",
                )
            ],
        )

    with pytest.raises(ValidationError, match="执行 Agent 与计划不一致"):
        MultiAgentTaskResult(
            collaboration_id="wrong_agent_collaboration",
            plan=plan,
            status="running",
            task_results=[
                AgentTaskResult(
                    step_id="load_profile",
                    assigned_agent="tool_agent",
                    status="completed",
                )
            ],
        )
