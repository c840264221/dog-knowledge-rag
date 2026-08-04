"""多智能体总编排器内部 Skill 暂停与恢复集成测试。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    GraphAgentWorkerAdapter,
    MultiAgentOrchestrator,
    MultiAgentTaskScheduler,
    PlannerAgent,
    ResultAggregator,
)
from src.skills import build_default_skill_runtime


class FixedOrchestrationMessage:
    """保存总编排集成测试中一条固定的 LLM 文本响应。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FixedOrchestrationLLMProvider:
    """
    按顺序返回规划和结果聚合所需的固定 LLM 响应。

    功能：
        避免真实网络和模型随机性影响测试，同时记录 Planner 与 Aggregator
        实际构建的提示词。

    参数含义：
        responses:
            按调用顺序提供给 Planner 和 Aggregator 的文本响应。

    返回值含义：
        FixedOrchestrationLLMProvider:
            可以代替真实 LLM Provider 的确定性测试对象。
    """

    def __init__(self, responses: list[str]) -> None:
        self.main_llm = object()

        # 上游预先准备、尚未被 Planner 或 Aggregator 取走的固定响应。
        self.responses = list(responses)

        # 保存真实组件生成的提示词，用于确认规划和聚合各调用一次 LLM。
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        *,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
    ) -> FixedOrchestrationMessage:
        """
        记录本次提示词并返回下一条固定响应。

        参数含义：
            llm:
                Planner 或 Aggregator 选择的模型对象，本测试不会真正调用。
            prompt:
                真实组件为本次规划或聚合生成的完整提示词。
            fallback_response:
                真实 Provider 调用失败时使用的兜底文本，本测试不会使用。

        返回值含义：
            FixedOrchestrationMessage:
                包含下一条固定文本响应的消息对象。
        """

        _ = llm, fallback_response
        self.prompts.append(prompt)
        if not self.responses:
            raise ValueError("总编排 Skill 测试缺少固定 LLM 响应")
        return FixedOrchestrationMessage(self.responses.pop(0))


@pytest.mark.asyncio
async def test_orchestrator_should_resume_skill_and_aggregate_result() -> None:
    """
    测试总编排器恢复 Step 内部 Skill 后是否继续调度并聚合回答。

    功能：
        首次运行时，训练 Step 因 Skill 缺少必要输入而暂停；用户补充资料后，
        总编排器不重新规划，而是恢复原 Scheduler、完成后续步骤并调用真实
        ResultAggregator 生成最终回答。

    参数含义：
        无。

    返回值含义：
        None。
    """

    # 第一条响应交给 Planner，第二条响应只在恢复成功后交给 Aggregator。
    provider = FixedOrchestrationLLMProvider(
        [
            json.dumps(
                {
                    "plan_id": "orchestrator_skill_resume_plan",
                    "objective": "为6岁的金毛生成训练和综合方案",
                    "steps": [
                        {
                            "step_id": "step_training",
                            "title": "制定训练计划",
                            "description": "为6岁的金毛制定训练计划",
                            "assigned_agent": "dog_knowledge_agent",
                        },
                        {
                            "step_id": "step_summary",
                            "title": "整理综合方案",
                            "assigned_agent": "general_agent",
                            "depends_on": ["step_training"],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "final_answer": "已生成金毛训练与综合方案。",
                    "used_step_ids": [
                        "step_training",
                        "step_summary",
                    ],
                    "limitations": [],
                },
                ensure_ascii=False,
            ),
        ]
    )

    # 保存真正进入训练子 Agent 的 state，暂停阶段不应该产生任何记录。
    training_runner_states: list[Mapping[str, Any]] = []

    # 保存汇总子 Agent 收到的 state，证明它只在训练步骤完成后运行。
    summary_runner_states: list[Mapping[str, Any]] = []

    async def training_runner(
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        模拟根据准备完成的 Skill 上下文生成训练步骤结果。

        参数含义：
            state:
                Worker 组装并注入 Skill 输入和说明后的子图状态。

        返回值含义：
            Mapping[str, Any]:
                包含固定训练回答的完成状态。
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
        模拟读取训练结果并生成步骤级综合内容。

        参数含义：
            state:
                包含前置训练步骤标准结果的子图状态。

        返回值含义：
            Mapping[str, Any]:
                包含固定综合内容的完成状态。
        """

        summary_runner_states.append(state)
        return {
            **state,
            "final_answer": "步骤级综合内容整理完成。",
        }

    # SkillRuntime 本身不保存会话数据，各步骤的恢复资料由 Scheduler 保存。
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
    orchestrator = MultiAgentOrchestrator(
        planner=PlannerAgent(
            llm_provider=provider,
            available_agents={
                "dog_knowledge_agent": "生成狗狗训练方案。",
                "general_agent": "整理多个步骤的综合内容。",
            },
            maximum_plan_attempts=1,
        ),
        scheduler=scheduler,
        result_aggregator=ResultAggregator(
            llm_provider=provider,
            maximum_aggregation_attempts=1,
        ),
    )

    paused_result = await orchestrator.run(
        "为6岁的金毛生成训练和综合方案",
        plan_id="orchestrator_skill_resume_plan",
        multi_agent_task_id="orchestrator_skill_resume_task",
    )

    assert paused_result.status == "awaiting_input"
    assert paused_result.plan.status == "awaiting_input"
    assert training_runner_states == []
    assert summary_runner_states == []
    assert len(provider.prompts) == 1
    assert paused_result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
    ]

    final_result = await orchestrator.resume(
        paused_result,
        user_inputs={
            "step_training": "它目前会坐下，希望学习等待和召回。",
        },
    )

    assert final_result.status == "completed"
    assert final_result.plan.status == "completed"
    assert final_result.final_answer == "已生成金毛训练与综合方案。"
    assert len(training_runner_states) == 1
    assert len(summary_runner_states) == 1
    assert len(provider.prompts) == 2

    # 恢复时 Planner 没有重新运行，原有阶段后只新增恢复调度和结果聚合。
    assert final_result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
        "resume_scheduling",
        "aggregation",
    ]

    # 训练 Worker 收到合并后的必要输入，Skill 上下文与检索问题保持分离。
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

    # 后续步骤确实收到了已经完成的训练步骤结果。
    dependency_result = summary_runner_states[0][
        "multi_agent_dependency_results"
    ]["step_training"]
    assert dependency_result["status"] == "completed"
    assert dependency_result["summary"] == "训练步骤已生成分阶段计划。"

    results_by_step_id = {
        result.step_id: result
        for result in final_result.task_results
    }
    assert results_by_step_id["step_training"].status == "completed"
    assert results_by_step_id["step_summary"].status == "completed"
    assert results_by_step_id["step_training"].metadata[
        "skill_runtime"
    ]["status"] == "ready"
    assert final_result.metadata["resume_count"] == 1
