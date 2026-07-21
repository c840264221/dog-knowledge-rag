"""
多 Agent 结果聚合器测试。

功能：
    使用固定 LLM 输出验证完整结果聚合、部分失败说明、输出修复和状态边界，
    不访问真实模型或外部 API。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
    ResultAggregationError,
    ResultAggregator,
)


class FakeAggregationMessage:
    """保存测试 LLM 返回文本的简单消息对象。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeAggregationLLMProvider:
    """
    为结果聚合器测试提供固定 LLM 输出。

    参数含义：
        responses:
            每次 safe_ainvoke 调用需要依次返回的文本。

    返回值含义：
        FakeAggregationLLMProvider:
            记录模型和提示词但不访问真实服务的 Provider 替身。
    """

    def __init__(self, responses: list[str]) -> None:
        self.main_llm = object()
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.received_llms: list[Any] = []

    async def safe_ainvoke(
        self,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
    ) -> FakeAggregationMessage:
        """
        记录调用参数并返回下一条预设消息。

        参数含义：
            llm:
                聚合器本轮选择的模型对象。
            prompt:
                第一次聚合或修复时使用的提示词。
            fallback_response:
                真实 Provider 调用失败时使用的兜底文本，本测试不会使用。

        返回值含义：
            FakeAggregationMessage:
                包含下一条预设 LLM 文本的消息对象。
        """

        _ = fallback_response
        self.received_llms.append(llm)
        self.prompts.append(prompt)
        return FakeAggregationMessage(self.responses.pop(0))


def build_scheduled_task_result(
    *,
    partial: bool = False,
) -> MultiAgentTaskResult:
    """
    构建等待结果聚合器处理的调度结果。

    参数含义：
        partial:
            是否让训练知识步骤以允许失败的方式结束。

    返回值含义：
        MultiAgentTaskResult:
            完整成功时状态为 running，部分失败时状态为 partial 的测试结果。
    """

    training_status = "failed" if partial else "completed"
    plan = AgentTaskPlan(
        plan_id="aggregation_plan_001",
        objective="为幼犬制定健康和训练建议",
        status="partial" if partial else "completed",
        steps=[
            AgentTaskStep(
                step_id="query_health",
                title="查询健康知识",
                assigned_agent="health_agent",
                status="completed",
            ),
            AgentTaskStep(
                step_id="query_training",
                title="查询训练知识",
                assigned_agent="training_agent",
                status=training_status,
                allow_failure=partial,
            ),
        ],
    )
    task_results = [
        AgentTaskResult(
            step_id="query_health",
            assigned_agent="health_agent",
            status="completed",
            summary="幼犬需要规律体检和按计划免疫。",
            output={
                "health_advice": "规律体检并按计划免疫",
            },
            evidence_ids=["health_chunk_001"],
        ),
        AgentTaskResult(
            step_id="query_training",
            assigned_agent="training_agent",
            status=training_status,
            summary=(
                "训练查询失败"
                if partial
                else "使用短时、正向奖励训练。"
            ),
            output=(
                {}
                if partial
                else {"training_advice": "短时正向奖励训练"}
            ),
            error_message=(
                "训练知识服务暂时不可用"
                if partial
                else None
            ),
        ),
    ]
    return MultiAgentTaskResult(
        collaboration_id="multi_task_001",
        plan=plan,
        status="partial" if partial else "running",
        task_results=task_results,
        metadata={"scheduler": "MultiAgentTaskScheduler"},
    )


def build_aggregation_json(
    *,
    used_step_ids: list[str] | None = None,
    limitations: list[str] | None = None,
) -> str:
    """
    构建一条合法或可调整的聚合器 JSON 输出。

    参数含义：
        used_step_ids:
            声明最终回答使用了哪些成功步骤。
        limitations:
            需要告诉用户的失败影响或数据限制。

    返回值含义：
        str:
            可以作为 Fake LLM 返回值的 JSON 文本。
    """

    return json.dumps(
        {
            "final_answer": (
                "健康方面应规律体检并按计划免疫；训练方面采用短时正向奖励。"
            ),
            "used_step_ids": (
                used_step_ids
                if used_step_ids is not None
                else ["query_health", "query_training"]
            ),
            "limitations": limitations or [],
        },
        ensure_ascii=False,
    )


def test_result_aggregator_should_complete_running_task() -> None:
    """
    检查全部 Worker 成功时是否生成最终回答并完成整次任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider([build_aggregation_json()])
    aggregator = ResultAggregator(llm_provider=provider)
    original_result = build_scheduled_task_result()

    result = asyncio.run(aggregator.aggregate(original_result))

    assert result.status == "completed"
    assert "规律体检" in result.final_answer
    assert original_result.status == "running"
    assert original_result.final_answer == ""
    assert result.metadata["result_aggregation"] == {
        "used_step_ids": ["query_health", "query_training"],
        "limitations": [],
        "attempt_count": 1,
    }


def test_result_aggregator_should_use_injected_model() -> None:
    """
    检查显式传入聚合模型时是否覆盖 Provider 默认模型。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider([build_aggregation_json()])
    aggregation_llm = object()
    aggregator = ResultAggregator(
        llm_provider=provider,
        aggregation_llm=aggregation_llm,
    )

    asyncio.run(aggregator.aggregate(build_scheduled_task_result()))

    assert provider.received_llms == [aggregation_llm]


def test_result_aggregator_should_repair_missing_step() -> None:
    """
    检查第一次遗漏成功步骤时是否反馈错误并要求 LLM 修复。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider(
        [
            build_aggregation_json(used_step_ids=["query_health"]),
            build_aggregation_json(),
        ]
    )
    aggregator = ResultAggregator(
        llm_provider=provider,
        maximum_aggregation_attempts=2,
    )

    result = asyncio.run(
        aggregator.aggregate(build_scheduled_task_result())
    )

    assert result.status == "completed"
    assert result.metadata["result_aggregation"]["attempt_count"] == 2
    assert "遗漏=['query_training']" in provider.prompts[1]
    assert "JSON Schema" in provider.prompts[1]


def test_result_aggregator_should_keep_partial_status() -> None:
    """
    检查允许失败步骤存在时是否保留 partial 并记录限制说明。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider(
        [
            build_aggregation_json(
                used_step_ids=["query_health"],
                limitations=["训练知识服务暂时不可用，未提供训练建议。"],
            )
        ]
    )
    aggregator = ResultAggregator(llm_provider=provider)

    result = asyncio.run(
        aggregator.aggregate(build_scheduled_task_result(partial=True))
    )

    assert result.status == "partial"
    assert result.metadata["result_aggregation"]["limitations"] == [
        "训练知识服务暂时不可用，未提供训练建议。"
    ]


def test_result_aggregator_should_reject_missing_limitations() -> None:
    """
    检查部分失败但未说明限制时是否拒绝不完整回答。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider(
        [build_aggregation_json(used_step_ids=["query_health"])]
    )
    aggregator = ResultAggregator(
        llm_provider=provider,
        maximum_aggregation_attempts=1,
    )

    with pytest.raises(
        ResultAggregationError,
        match="limitations 不能为空",
    ):
        asyncio.run(
            aggregator.aggregate(
                build_scheduled_task_result(partial=True)
            )
        )


def test_result_aggregator_should_reject_failed_task_before_llm() -> None:
    """
    检查整体失败的任务是否在调用 LLM 前被拒绝。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider([build_aggregation_json()])
    aggregator = ResultAggregator(llm_provider=provider)
    result_data = build_scheduled_task_result(partial=True).model_dump(
        mode="python"
    )
    result_data.update(
        {
            "status": "failed",
            "error_message": "关键步骤失败",
        }
    )
    failed_result = MultiAgentTaskResult.model_validate(result_data)

    with pytest.raises(ValueError, match="只接受 status=running 或 partial"):
        asyncio.run(aggregator.aggregate(failed_result))

    assert provider.prompts == []


def test_result_aggregator_should_reject_missing_step_result() -> None:
    """
    检查计划步骤结果不齐时是否在调用 LLM 前停止聚合。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider([build_aggregation_json()])
    aggregator = ResultAggregator(llm_provider=provider)
    result_data = build_scheduled_task_result().model_dump(mode="python")
    result_data["status"] = "partial"
    result_data["task_results"] = result_data["task_results"][:1]
    incomplete_result = MultiAgentTaskResult.model_validate(result_data)

    with pytest.raises(ValueError, match="仍缺少步骤结果"):
        asyncio.run(aggregator.aggregate(incomplete_result))

    assert provider.prompts == []


def test_result_aggregator_should_reject_awaiting_step() -> None:
    """
    检查仍有 Worker 等待用户输入时是否在调用 LLM 前停止聚合。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakeAggregationLLMProvider([build_aggregation_json()])
    aggregator = ResultAggregator(llm_provider=provider)
    result_data = build_scheduled_task_result(partial=True).model_dump(
        mode="python"
    )
    result_data["task_results"][1].update(
        {
            "status": "awaiting_input",
            "error_message": None,
            "requires_user_input": True,
            "clarification_prompt": "请补充训练目标。",
        }
    )
    awaiting_result = MultiAgentTaskResult.model_validate(result_data)

    with pytest.raises(ValueError, match="不能进行结果聚合"):
        asyncio.run(aggregator.aggregate(awaiting_result))

    assert provider.prompts == []
