"""
Graph Agent Worker Adapter 测试。

功能：
    验证步骤输入、前置结果、普通回答和等待用户状态能在现有 Agent state
    与多 Agent Worker 结果之间正确转换。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskResult,
    AgentTaskStep,
    GraphAgentWorkerAdapter,
    build_default_agent_state,
)


def test_graph_worker_adapter_should_convert_completed_state() -> None:
    """
    检查普通 Agent 最终 state 是否转换成 completed Worker 结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录输入 state 并返回健康知识回答。"""

        received_states.append(state)
        return {
            **state,
            "final_answer": "幼犬应按计划免疫。",
            "evidence_ids": ["health_chunk_001"],
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="health_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_health",
        title="查询健康知识",
        description="根据幼犬资料查询健康知识",
        assigned_agent="health_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "completed"
    assert result.summary == "幼犬应按计划免疫。"
    assert result.output["final_answer"] == "幼犬应按计划免疫。"
    assert result.evidence_ids == ["health_chunk_001"]
    assert received_states[0]["question"] == "根据幼犬资料查询健康知识"


def test_graph_worker_adapter_should_include_dependency_results() -> None:
    """
    检查前置 Worker 结果是否同时进入结构化 state 和当前问题文本。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录同步调用输入并返回训练建议。"""

        received_states.append(state)
        return {
            "final_answer": "每天进行短时正向训练。",
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="training_agent",
        runner=runner,
    )
    dependency_result = AgentTaskResult(
        step_id="load_profile",
        assigned_agent="profile_agent",
        status="completed",
        summary="已读取三个月大金毛资料。",
        output={"age_months": 3},
    )
    step = AgentTaskStep(
        step_id="query_training",
        title="查询训练知识",
        assigned_agent="training_agent",
        depends_on=["load_profile"],
    )

    asyncio.run(
        adapter(
            step,
            {"load_profile": dependency_result},
        )
    )

    input_state = received_states[0]
    assert input_state["multi_agent_dependency_results"][
        "load_profile"
    ]["output"] == {"age_months": 3}
    assert "已读取三个月大金毛资料" in input_state["question"]


def test_graph_worker_adapter_should_convert_waiting_state() -> None:
    """
    检查 ToolAgent 等待确认状态是否转换成 awaiting_input 结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回等待用户确认的 ToolAgent state。"""

        return {
            **state,
            "waiting_user_input": True,
            "tool_confirmation_prompt": "是否允许查询健康数据库？",
            "tool_agent_response": {
                "status": "awaiting_confirmation",
            },
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_database",
        title="查询健康数据库",
        assigned_agent="tool_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "awaiting_input"
    assert result.requires_user_input is True
    assert result.clarification_prompt == "是否允许查询健康数据库？"


def test_graph_worker_adapter_should_reject_wrong_agent() -> None:
    """
    检查步骤指定的 Agent 与适配器不一致时是否停止执行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回不会在本测试中真正使用的 state。"""

        return state

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_health",
        title="查询健康知识",
        assigned_agent="health_agent",
    )

    with pytest.raises(ValueError, match="只负责 tool_agent"):
        asyncio.run(adapter(step, {}))


def test_graph_worker_adapter_should_require_waiting_prompt() -> None:
    """
    检查 Agent 声明等待用户却没有提示时是否拒绝生成无效暂停结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回缺少用户提示的等待状态。"""

        return {
            **state,
            "waiting_user_input": True,
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_database",
        title="查询数据库",
        assigned_agent="tool_agent",
    )

    with pytest.raises(ValueError, match="没有提供等待提示"):
        asyncio.run(adapter(step, {}))


def test_default_state_builder_should_include_resume_context() -> None:
    """
    检查恢复步骤是否把上次输出和用户回答追加到 Agent 问题中。

    参数含义：
        无。

    返回值含义：
        None。
    """

    step = AgentTaskStep(
        step_id="confirm_profile",
        title="确认读取资料",
        assigned_agent="profile_agent",
        input_data={
            "multi_agent_is_resuming": True,
            "multi_agent_resume_input": "允许读取",
            "multi_agent_previous_worker_output": {
                "pending_action": "读取宠物档案"
            },
        },
    )

    state = build_default_agent_state(step, {})

    assert "当前步骤正在从等待用户输入的状态恢复" in state["question"]
    assert "读取宠物档案" in state["question"]
    assert "允许读取" in state["question"]
