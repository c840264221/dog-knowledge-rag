"""
多 Agent 总编排器测试。

功能：
    使用真实 Planner、Scheduler 和 Aggregator 配合固定 LLM 输出，验证完整
    成功、等待用户、关键步骤失败和规划异常，不访问真实外部服务。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentOrchestrationError,
    MultiAgentOrchestrator,
    MultiAgentTaskScheduler,
    PlannerAgent,
    ResultAggregator,
)


class FakeOrchestrationMessage:
    """保存总编排测试中一条固定 LLM 文本。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeOrchestrationLLMProvider:
    """
    按调用顺序提供规划和结果聚合所需的固定 LLM 输出。

    参数含义：
        responses:
            Planner 和 Aggregator 每次调用依次取得的文本。

    返回值含义：
        FakeOrchestrationLLMProvider:
            不访问外部 API、可以记录提示词的测试 Provider。
    """

    def __init__(self, responses: list[str]) -> None:
        self.main_llm = object()
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
    ) -> FakeOrchestrationMessage:
        """
        记录提示词并返回下一条固定文本。

        参数含义：
            llm:
                Planner 或 Aggregator 选择的模型对象。
            prompt:
                当前规划或聚合提示词。
            fallback_response:
                真实调用失败时的兜底文本，本替身不会使用。

        返回值含义：
            FakeOrchestrationMessage:
                包含下一条预设文本的消息对象。
        """

        _ = llm, fallback_response
        self.prompts.append(prompt)
        return FakeOrchestrationMessage(self.responses.pop(0))


def build_orchestration_plan_json(
    *,
    awaiting_input: bool = False,
    allow_failure: bool = False,
) -> str:
    """
    构建总编排器测试使用的任务计划 JSON。

    参数含义：
        awaiting_input:
            是否让计划停下来等待用户补充狗狗年龄。
        allow_failure:
            查询健康知识失败后是否仍允许任务继续。

    返回值含义：
        str:
            可以作为 PlannerAgent 固定输出的 JSON 文本。
    """

    return json.dumps(
        {
            "plan_id": "plan_orchestration_001",
            "objective": "为幼犬制定健康建议",
            "steps": [
                {
                    "step_id": "load_profile",
                    "title": "读取幼犬资料",
                    "assigned_agent": "profile_agent",
                    "depends_on": [],
                    "status": "pending",
                },
                {
                    "step_id": "query_health",
                    "title": "查询健康知识",
                    "assigned_agent": "health_agent",
                    "depends_on": ["load_profile"],
                    "status": "pending",
                    "allow_failure": allow_failure,
                },
            ],
            "status": (
                "awaiting_input"
                if awaiting_input
                else "planned"
            ),
            "requires_user_input": awaiting_input,
            "clarification_prompt": (
                "请补充狗狗年龄。"
                if awaiting_input
                else ""
            ),
        },
        ensure_ascii=False,
    )


def build_orchestration_answer_json() -> str:
    """
    构建总编排器测试使用的合法聚合回答 JSON。

    参数含义：
        无。

    返回值含义：
        str:
            包含最终回答和全部成功步骤编号的 JSON 文本。
    """

    return json.dumps(
        {
            "final_answer": "幼犬应规律体检并按计划完成免疫。",
            "used_step_ids": ["load_profile", "query_health"],
            "limitations": [],
        },
        ensure_ascii=False,
    )


def build_success_worker(calls: list[str]):
    """
    构建记录步骤编号并返回成功结果的 Worker。

    参数含义：
        calls:
            用于保存 Worker 实际执行过的步骤编号。

    返回值含义：
        Callable:
            返回 completed AgentTaskResult 的异步 Worker。
    """

    async def worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        记录当前步骤并返回测试成功结果。

        参数含义：
            step:
                调度器当前交给 Worker 的完整步骤。
            dependency_results:
                当前步骤已经得到的前置结果。

        返回值含义：
            AgentTaskResult:
                与当前步骤编号和 Agent 一致的成功结果。
        """

        _ = dependency_results
        calls.append(step.step_id)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary=f"{step.title}完成",
            output={"result": f"{step.title}结果"},
        )

    return worker


def build_orchestrator(
    *,
    provider: FakeOrchestrationLLMProvider,
    profile_worker: Any,
    health_worker: Any,
) -> MultiAgentOrchestrator:
    """
    使用真实三个阶段组件构建测试总编排器。

    参数含义：
        provider:
            同时为 Planner 和 Aggregator 提供固定输出的测试 Provider。
        profile_worker:
            执行读取资料步骤的 Worker。
        health_worker:
            执行查询健康知识步骤的 Worker。

    返回值含义：
        MultiAgentOrchestrator:
            已注入真实 Planner、Scheduler 和 Aggregator 的总编排器。
    """

    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "profile_agent": "读取狗狗资料。",
            "health_agent": "查询狗狗健康知识。",
        },
        maximum_plan_attempts=1,
    )
    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": profile_worker,
            "health_agent": health_worker,
        }
    )
    aggregator = ResultAggregator(
        llm_provider=provider,
        maximum_aggregation_attempts=1,
    )
    return MultiAgentOrchestrator(
        planner=planner,
        scheduler=scheduler,
        result_aggregator=aggregator,
    )


def test_orchestrator_should_run_complete_multi_agent_flow() -> None:
    """
    检查总编排器是否依次完成规划、步骤执行和最终结果聚合。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(
        [
            build_orchestration_plan_json(),
            build_orchestration_answer_json(),
        ]
    )
    calls: list[str] = []
    worker = build_success_worker(calls)
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=worker,
        health_worker=worker,
    )

    result = asyncio.run(
        orchestrator.run(
            "为幼犬制定健康建议",
            plan_id="plan_orchestration_001",
            multi_agent_task_id="task_orchestration_001",
        )
    )

    assert result.status == "completed"
    assert result.collaboration_id == "task_orchestration_001"
    assert calls == ["load_profile", "query_health"]
    assert len(provider.prompts) == 2
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
        "aggregation",
    ]


def test_orchestrator_should_stop_when_plan_awaits_user() -> None:
    """
    检查计划等待用户输入时是否停止在调度阶段且不调用 Worker 和聚合器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(
        [build_orchestration_plan_json(awaiting_input=True)]
    )
    calls: list[str] = []
    worker = build_success_worker(calls)
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=worker,
        health_worker=worker,
    )

    result = asyncio.run(
        orchestrator.run(
            "为幼犬制定健康建议",
            plan_id="plan_orchestration_001",
        )
    )

    assert result.status == "awaiting_input"
    assert calls == []
    assert len(provider.prompts) == 1
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
    ]


def test_orchestrator_should_stop_when_worker_awaits_user() -> None:
    """
    检查 Worker 执行中等待确认时是否暂停总流程且不调用聚合器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(
        [build_orchestration_plan_json()]
    )

    async def waiting_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回等待用户授权读取资料的标准结果。"""

        _ = dependency_results
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="是否允许读取宠物档案？",
        )

    calls: list[str] = []
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=waiting_profile_worker,
        health_worker=build_success_worker(calls),
    )

    result = asyncio.run(
        orchestrator.run(
            "为幼犬制定健康建议",
            plan_id="plan_orchestration_001",
        )
    )

    assert result.status == "awaiting_input"
    assert result.plan.steps[0].status == "awaiting_input"
    assert result.plan.steps[1].status == "pending"
    assert calls == []
    assert len(provider.prompts) == 1
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
    ]


def test_orchestrator_should_resume_and_aggregate_result() -> None:
    """
    检查总编排器是否能恢复等待步骤并继续完成结果聚合。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(
        [
            build_orchestration_plan_json(),
            build_orchestration_answer_json(),
        ]
    )
    profile_calls: list[str] = []

    async def resumable_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """首次等待授权，恢复后返回资料读取结果。"""

        _ = dependency_results
        profile_calls.append(step.step_id)
        if not step.input_data.get("multi_agent_is_resuming"):
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="是否允许读取宠物档案？",
            )
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary="资料读取完成",
        )

    health_calls: list[str] = []
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=resumable_profile_worker,
        health_worker=build_success_worker(health_calls),
    )

    paused_result = asyncio.run(
        orchestrator.run(
            "为幼犬制定健康建议",
            plan_id="plan_orchestration_001",
        )
    )
    final_result = asyncio.run(
        orchestrator.resume(
            paused_result,
            user_inputs={"load_profile": "允许读取"},
        )
    )

    assert final_result.status == "completed"
    assert profile_calls == ["load_profile", "load_profile"]
    assert health_calls == ["query_health"]
    assert len(provider.prompts) == 2
    assert final_result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
        "resume_scheduling",
        "aggregation",
    ]


def test_orchestrator_should_stop_after_blocking_worker_failure() -> None:
    """
    检查关键 Worker 失败后是否返回 failed 且不再调用结果聚合器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(
        [build_orchestration_plan_json()]
    )

    async def failing_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """模拟资料服务发生异常。"""

        _ = step, dependency_results
        raise RuntimeError("资料服务不可用")

    calls: list[str] = []
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=failing_profile_worker,
        health_worker=build_success_worker(calls),
    )

    result = asyncio.run(
        orchestrator.run(
            "为幼犬制定健康建议",
            plan_id="plan_orchestration_001",
        )
    )

    assert result.status == "failed"
    assert calls == []
    assert len(provider.prompts) == 1
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
    ]


def test_orchestrator_should_mark_planning_error_stage() -> None:
    """
    检查规划失败时统一异常是否明确标记 planning 阶段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeOrchestrationLLMProvider(["不是合法计划"])
    calls: list[str] = []
    worker = build_success_worker(calls)
    orchestrator = build_orchestrator(
        provider=provider,
        profile_worker=worker,
        health_worker=worker,
    )

    with pytest.raises(MultiAgentOrchestrationError) as exc_info:
        asyncio.run(
            orchestrator.run(
                "为幼犬制定健康建议",
                plan_id="plan_orchestration_001",
            )
        )

    assert exc_info.value.stage == "planning"
    assert calls == []
