"""多智能体步骤内部 Skill 暂停与恢复集成测试。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    GraphAgentWorkerAdapter,
    MultiAgentTaskScheduler,
)
from src.skills import build_default_skill_runtime


@pytest.mark.asyncio
async def test_scheduler_should_resume_skill_inside_waiting_step() -> None:
    """
    测试 Scheduler 通过等待 Step 恢复其内部 Skill 并继续依赖步骤。

    功能：
        第一轮训练步骤因 Skill 缺少行为基础和训练目标而暂停，且不会调用
        Worker runner；第二轮用户补充后，Scheduler 只恢复等待步骤，Worker
        在调用 runner 前恢复 Skill，完成后再启动依赖它的汇总步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    # 保存真正进入训练子 Agent 的 state；第一轮等待时这个列表必须保持为空。
    training_runner_states: list[Mapping[str, Any]] = []

    # 保存后续汇总步骤收到的 state，用于证明依赖步骤只在训练完成后启动。
    summary_runner_states: list[Mapping[str, Any]] = []

    async def training_runner(
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        模拟使用 Skill 上下文生成训练计划的子 Agent。

        参数含义：
            state:
                已包含 Skill 输入和执行说明的 Worker state。

        返回值含义：
            Mapping[str, Any]:
                带固定训练方案答案的完成状态。
        """

        training_runner_states.append(state)
        return {
            **state,
            "final_answer": "训练步骤已生成分阶段计划。",
        }

    async def summary_runner(
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        模拟读取前置训练结果的后续汇总 Agent。

        参数含义：
            state:
                包含前置步骤结果的 Worker state。

        返回值含义：
            Mapping[str, Any]:
                带固定综合答案的完成状态。
        """

        summary_runner_states.append(state)
        return {
            **state,
            "final_answer": "综合方案整理完成。",
        }

    # 两个 Worker 共享无会话状态的 Runtime，具体 Skill 输入仍保存在各自 Step 中。
    skill_runtime = build_default_skill_runtime()
    scheduler = MultiAgentTaskScheduler(
        workers={
            "dog_knowledge_agent": GraphAgentWorkerAdapter(
                agent_name="dog_knowledge_agent",
                runner=training_runner,
                skill_runtime=skill_runtime,
            ),
            "general_agent": GraphAgentWorkerAdapter(
                agent_name="general_agent",
                runner=summary_runner,
            ),
        }
    )
    plan = AgentTaskPlan(
        plan_id="step_skill_resume_plan",
        objective="为6岁的金毛生成训练和综合方案",
        steps=[
            AgentTaskStep(
                step_id="step_training",
                title="制定训练计划",
                description="为6岁的金毛制定训练计划",
                assigned_agent="dog_knowledge_agent",
            ),
            AgentTaskStep(
                step_id="step_summary",
                title="整理综合方案",
                assigned_agent="general_agent",
                depends_on=["step_training"],
            ),
        ],
    )

    paused_result = await scheduler.execute(
        plan,
        collaboration_id="step_skill_resume_task",
    )

    assert paused_result.status == "awaiting_input"
    assert paused_result.plan.status == "awaiting_input"
    assert training_runner_states == []
    assert summary_runner_states == []
    assert paused_result.task_results[0].step_id == "step_training"
    assert paused_result.task_results[0].status == "awaiting_input"
    assert paused_result.task_results[0].output["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }

    resumed_result = await scheduler.resume(
        paused_result,
        user_inputs={
            "step_training": (
                "它目前会坐下，希望学习等待和召回。"
            ),
        },
    )

    assert resumed_result.status == "running"
    assert resumed_result.plan.status == "completed"
    assert len(training_runner_states) == 1
    assert len(summary_runner_states) == 1

    # 恢复后的训练 Worker 得到合并输入，Skill 说明不再写入检索问题。
    training_state = training_runner_states[0]
    assert training_state["skill_status"] == "ready"
    assert training_state["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }
    assert "技能：狗狗训练计划" in training_state["question"]
    assert "技能：狗狗训练计划" in training_state["skill_context"]
    assert "技能：狗狗训练计划" not in training_state["retrieval_question"]
    assert training_state["retrieval_question"] == (
        "为6岁的金毛制定训练计划"
    )

    # 依赖步骤由 Scheduler 在训练步骤 completed 后启动，并收到标准前置结果。
    summary_state = summary_runner_states[0]
    dependency_result = summary_state[
        "multi_agent_dependency_results"
    ]["step_training"]
    assert dependency_result["status"] == "completed"
    assert dependency_result["summary"] == (
        "训练步骤已生成分阶段计划。"
    )

    results_by_step_id = {
        result.step_id: result
        for result in resumed_result.task_results
    }
    assert results_by_step_id["step_training"].status == "completed"
    assert results_by_step_id["step_summary"].status == "completed"
    assert results_by_step_id["step_training"].metadata[
        "skill_runtime"
    ]["status"] == "ready"
    assert resumed_result.metadata["resume_count"] == 1
