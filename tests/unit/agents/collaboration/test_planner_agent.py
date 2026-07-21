"""
PlannerAgent 任务计划生成测试。

功能：
    使用确定性 Fake LLM Provider 验证计划生成、结构修复、Agent 白名单和
    拓扑排序，不访问真实模型或外部 API。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.agents.collaboration import (
    PlannerAgent,
    PlannerGenerationError,
)


class FakeLLMMessage:
    """保存测试 LLM 返回文本的简单消息对象。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakePlannerLLMProvider:
    """
    为 PlannerAgent 测试提供固定 LLM 输出。

    参数含义：
        responses:
            每次 safe_ainvoke 调用需要依次返回的文本。

    返回值含义：
        FakePlannerLLMProvider:
            记录提示词且不会访问真实模型的 Provider 替身。
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
    ) -> FakeLLMMessage:
        """
        记录提示词并返回下一条固定消息。

        参数含义：
            llm:
                PlannerAgent 传入的主模型占位对象。
            prompt:
                本轮生成或修复计划的提示词。
            fallback_response:
                真实 Provider 失败时使用的兜底文本，本替身不会使用。

        返回值含义：
            FakeLLMMessage:
                包含预设文本的测试消息。
        """

        _ = fallback_response
        self.prompts.append(prompt)
        self.received_llms.append(llm)
        return FakeLLMMessage(self.responses.pop(0))


def build_plan_json(
    *,
    assigned_agent: str = "dog_knowledge_agent",
    plan_id: str = "plan_001",
) -> str:
    """
    构建一份步骤顺序故意打乱的合法测试计划。

    参数含义：
        assigned_agent:
            知识查询步骤需要分配的 Agent 名称。
        plan_id:
            测试计划编号。

    返回值含义：
        str:
            可以作为 Fake LLM 输出的 JSON 字符串。
    """

    return json.dumps(
        {
            "plan_id": plan_id,
            "objective": "为三个月大的金毛制定养护计划",
            "steps": [
                {
                    "step_id": "query_knowledge",
                    "title": "查询幼犬知识",
                    "assigned_agent": assigned_agent,
                    "depends_on": ["load_profile"],
                    "expected_output": "幼犬养护知识",
                    "status": "pending",
                },
                {
                    "step_id": "load_profile",
                    "title": "读取狗狗资料",
                    "assigned_agent": "memory_agent",
                    "depends_on": [],
                    "expected_output": "狗狗年龄和犬种",
                    "status": "pending",
                },
            ],
            "status": "planned",
            "reason": "先读取资料，再查询对应知识。",
            "requires_user_input": False,
            "clarification_prompt": "",
        },
        ensure_ascii=False,
    )


def test_planner_should_generate_and_order_valid_plan() -> None:
    """
    检查 PlannerAgent 是否生成合法计划并按依赖关系重新排序。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakePlannerLLMProvider([build_plan_json()])
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取长期记忆和用户资料。",
            "dog_knowledge_agent": "查询狗狗知识库。",
        },
    )

    plan = asyncio.run(
        planner.create_plan(
            "为三个月大的金毛制定养护计划",
            plan_id="plan_001",
            context={"user_id": "user_a"},
        )
    )

    assert [step.step_id for step in plan.steps] == [
        "load_profile",
        "query_knowledge",
    ]
    assert len(provider.prompts) == 1
    assert '"memory_agent"' in provider.prompts[0]
    assert '"user_id": "user_a"' in provider.prompts[0]
    assert "不要创建“汇总全部步骤”" in provider.prompts[0]
    assert "ResultAggregator 统一生成最终回答" in provider.prompts[0]


def test_planner_should_repair_invalid_first_output() -> None:
    """
    检查第一次输出不是 JSON 时是否发起一次结构修复。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakePlannerLLMProvider(
        [
            "我建议先读取资料，然后查询知识。",
            build_plan_json(),
        ]
    )
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取资料。",
            "dog_knowledge_agent": "查询知识。",
        },
    )

    plan = asyncio.run(
        planner.create_plan(
            "为三个月大的金毛制定养护计划",
            plan_id="plan_001",
        )
    )

    assert plan.status == "planned"
    assert len(provider.prompts) == 2
    assert "上一次输出没有通过程序校验" in provider.prompts[1]


def test_planner_should_reject_unregistered_agent() -> None:
    """
    检查 LLM 使用未注册 Agent 时是否最终生成失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    invalid_plan = build_plan_json(assigned_agent="invented_agent")
    provider = FakePlannerLLMProvider([invalid_plan, invalid_plan])
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取资料。",
            "dog_knowledge_agent": "查询知识。",
        },
    )

    with pytest.raises(
        PlannerGenerationError,
        match="未注册 Agent",
    ):
        asyncio.run(
            planner.create_plan(
                "为三个月大的金毛制定养护计划",
                plan_id="plan_001",
            )
        )


def test_planner_should_reject_changed_plan_id() -> None:
    """
    检查 LLM 修改程序计划编号时是否拒绝该输出。

    参数含义：
        无。

    返回值含义：
        None。
    """

    wrong_id_plan = build_plan_json(plan_id="llm_changed_id")
    provider = FakePlannerLLMProvider([wrong_id_plan, wrong_id_plan])
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取资料。",
            "dog_knowledge_agent": "查询知识。",
        },
    )

    with pytest.raises(
        PlannerGenerationError,
        match="plan_id",
    ):
        asyncio.run(
            planner.create_plan(
                "为三个月大的金毛制定养护计划",
                plan_id="plan_001",
            )
        )


def test_planner_should_accept_plan_waiting_for_user_input() -> None:
    """
    检查缺少关键信息时是否允许生成等待用户澄清的计划。

    参数含义：
        无。

    返回值含义：
        None。
    """

    plan_data = json.loads(build_plan_json())
    plan_data["status"] = "awaiting_input"
    plan_data["requires_user_input"] = True
    plan_data["clarification_prompt"] = "请问狗狗目前是否存在关节问题？"
    provider = FakePlannerLLMProvider(
        [json.dumps(plan_data, ensure_ascii=False)]
    )
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取资料。",
            "dog_knowledge_agent": "查询知识。",
        },
    )

    plan = asyncio.run(
        planner.create_plan(
            "为三个月大的金毛制定养护计划",
            plan_id="plan_001",
        )
    )

    assert plan.status == "awaiting_input"
    assert plan.requires_user_input is True
    assert "关节问题" in plan.clarification_prompt


def test_planner_should_use_injected_planning_llm() -> None:
    """
    检查显式传入的规划模型是否优先于 Provider 默认主模型。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = FakePlannerLLMProvider([build_plan_json()])
    custom_planning_llm = object()
    planner = PlannerAgent(
        llm_provider=provider,
        available_agents={
            "memory_agent": "读取资料。",
            "dog_knowledge_agent": "查询知识。",
        },
        planning_llm=custom_planning_llm,
    )

    asyncio.run(
        planner.create_plan(
            "为三个月大的金毛制定养护计划",
            plan_id="plan_001",
        )
    )

    assert provider.received_llms == [custom_planning_llm]


def test_planner_should_require_available_agents() -> None:
    """
    检查没有注册 Worker Agent 时是否禁止创建 PlannerAgent。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="至少注册一个"):
        PlannerAgent(
            llm_provider=FakePlannerLLMProvider([]),
            available_agents={},
        )
