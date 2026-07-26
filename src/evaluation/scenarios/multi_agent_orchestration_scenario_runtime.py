from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.agents.collaboration.aggregator import ResultAggregator
from src.agents.collaboration.contracts import MultiAgentTaskResult
from src.agents.collaboration.orchestrator import MultiAgentOrchestrator
from src.agents.collaboration.planner import PlannerAgent
from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
from src.evaluation.schemas import AgentEvaluationCase
from src.evaluation.scenarios.multi_agent_scenario_runtime import (
    EvaluationMultiAgentWorker,
)


class EvaluationOrchestrationMessage:
    """保存总编排评估中一条确定性的 LLM 文本响应。"""

    def __init__(self, content: str) -> None:
        self.content = content


class EvaluationOrchestrationLLMProvider:
    """
    为 Planner 或 Aggregator 提供确定性的 LLM 响应。

    功能：
        按调用顺序返回预设文本并记录提示词，使真实 Planner 和 Aggregator
        可以参加评估，同时避免网络和模型随机性影响质量门禁。

    参数含义：
        responses:
            每次 safe_ainvoke 调用依次取得的文本响应。

    返回值含义：
        EvaluationOrchestrationLLMProvider:
            可注入 PlannerAgent 或 ResultAggregator 的评估 Provider。
    """

    def __init__(self, responses: list[str]) -> None:
        self.main_llm = object()
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        *,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
    ) -> EvaluationOrchestrationMessage:
        """
        返回下一条预设响应并记录当前提示词。

        参数含义：
            llm:
                Planner 或 Aggregator 选择的模型对象。
            prompt:
                当前组件构建的完整提示词。
            fallback_response:
                真实 Provider 失败时使用的兜底文本；评估替身不会使用。

        返回值含义：
            EvaluationOrchestrationMessage:
                包含确定性文本的消息对象。
        """

        _ = llm, fallback_response
        self.prompts.append(prompt)
        if not self.responses:
            raise ValueError("总编排评估 Provider 缺少预设响应")
        return EvaluationOrchestrationMessage(self.responses.pop(0))


@dataclass
class MultiAgentOrchestrationScenarioRuntime:
    """
    保存一条总编排评估所需的真实组件和确定性依赖。

    参数含义：
        orchestrator:
            真实 MultiAgentOrchestrator 总编排器。
        worker:
            根据黄金用例稳定返回步骤结果的确定性 Worker。
        planning_provider:
            为真实 Planner 提供固定计划输出的评估 Provider。
        aggregation_provider:
            为真实 Aggregator 提供固定汇总输出的评估 Provider。
        objective:
            本次多 Agent 任务的原始目标。
        plan_id:
            黄金用例指定的稳定计划编号。
        multi_agent_task_id:
            本次评估使用的稳定协作任务编号。
        resume_user_inputs:
            Worker 等待输入后用于恢复任务的步骤回答。

    返回值含义：
        MultiAgentOrchestrationScenarioRuntime:
            可以执行完整规划、调度、恢复和聚合链路的评估工具箱。
    """

    orchestrator: MultiAgentOrchestrator
    worker: EvaluationMultiAgentWorker
    planning_provider: EvaluationOrchestrationLLMProvider
    aggregation_provider: EvaluationOrchestrationLLMProvider
    objective: str
    plan_id: str
    multi_agent_task_id: str
    resume_user_inputs: dict[str, Any]

    async def run(self) -> MultiAgentTaskResult:
        """
        执行一次总编排流程，并在配置回答时恢复等待任务。

        参数含义：
            无。

        返回值含义：
            MultiAgentTaskResult:
                总编排器完成、等待、失败或恢复后生成的最新标准结果。
        """

        result = await self.orchestrator.run(
            self.objective,
            plan_id=self.plan_id,
            multi_agent_task_id=self.multi_agent_task_id,
        )
        if result.status == "awaiting_input" and self.resume_user_inputs:
            result = await self.orchestrator.resume(
                result,
                user_inputs=self.resume_user_inputs,
            )
        return result


def build_multi_agent_orchestration_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> MultiAgentOrchestrationScenarioRuntime:
    """
    根据黄金用例组装真实总编排器评估环境。

    功能：
        使用真实 Planner、Scheduler、Aggregator 和 Orchestrator，只把 LLM
        响应与 Worker 输出替换为确定性数据，稳定验证完整阶段调用链。

    参数含义：
        eval_case:
            category 为 multi_agent_orchestration 的统一评估用例。

    返回值含义：
        MultiAgentOrchestrationScenarioRuntime:
            已装配完整总编排链路和调用记录器的评估工具箱。
    """

    raw_plan = eval_case.input_state.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("总编排评估 input_state 缺少 plan")
    plan_data = dict(raw_plan)
    plan_id = str(plan_data.get("plan_id") or "").strip()
    if not plan_id:
        raise ValueError("总编排评估 plan 缺少 plan_id")

    raw_behaviors = eval_case.input_state.get("worker_behaviors", {})
    if not isinstance(raw_behaviors, Mapping):
        raise ValueError("worker_behaviors 必须是步骤编号到行为的映射")
    worker = EvaluationMultiAgentWorker(raw_behaviors)

    raw_steps = plan_data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("总编排评估 plan.steps 必须是非空列表")
    available_agents = {
        str(step.get("assigned_agent") or "").strip(): (
            f"执行步骤：{str(step.get('title') or '').strip()}"
        )
        for step in raw_steps
        if isinstance(step, Mapping)
        and str(step.get("assigned_agent") or "").strip()
    }
    if not available_agents:
        raise ValueError("总编排评估 plan 缺少可用 Agent")

    planning_provider = EvaluationOrchestrationLLMProvider(
        [json.dumps(plan_data, ensure_ascii=False)]
    )
    raw_aggregation = eval_case.input_state.get(
        "aggregation_response",
        {
            "final_answer": "多 Agent 总编排评估完成。",
            "used_step_ids": [
                str(step.get("step_id"))
                for step in raw_steps
                if isinstance(step, Mapping)
            ],
            "limitations": [],
        },
    )
    if not isinstance(raw_aggregation, Mapping):
        raise ValueError("aggregation_response 必须是映射")
    aggregation_provider = EvaluationOrchestrationLLMProvider(
        [json.dumps(dict(raw_aggregation), ensure_ascii=False)]
    )

    planner = PlannerAgent(
        llm_provider=planning_provider,
        available_agents=available_agents,
        maximum_plan_attempts=1,
    )
    scheduler = MultiAgentTaskScheduler(
        workers={
            agent_name: worker
            for agent_name in available_agents
        },
        maximum_step_attempts=1,
    )
    aggregator = ResultAggregator(
        llm_provider=aggregation_provider,
        maximum_aggregation_attempts=1,
    )

    raw_resume_inputs = eval_case.input_state.get(
        "resume_user_inputs",
        {},
    )
    if not isinstance(raw_resume_inputs, Mapping):
        raise ValueError("resume_user_inputs 必须是步骤编号到回答的映射")

    return MultiAgentOrchestrationScenarioRuntime(
        orchestrator=MultiAgentOrchestrator(
            planner=planner,
            scheduler=scheduler,
            result_aggregator=aggregator,
        ),
        worker=worker,
        planning_provider=planning_provider,
        aggregation_provider=aggregation_provider,
        objective=eval_case.question,
        plan_id=plan_id,
        multi_agent_task_id=(
            f"evaluation_orchestration_{eval_case.case_id}"
        ),
        resume_user_inputs=dict(raw_resume_inputs),
    )
