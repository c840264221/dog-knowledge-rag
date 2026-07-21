"""
多 Agent 跨轮恢复 Checkpoint 集成测试。

功能：
    使用真实 LangGraph InMemorySaver 验证多 Agent 暂停结果可以按 thread_id
    跨两轮保存和恢复，并由真实语义路由适配器识别为 resume 动作。

测试边界：
    使用真实 semantic_router_node、RootAgent 和多 Agent 恢复输入适配器；
    使用 fake 多 Agent 节点避免调用真实 LLM、RAG 和外部服务。
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
    build_multi_agent_state_update,
)
from src.graph.nodes.router_node import semantic_router_node
from src.graph.routes.route_after_semantic import route_after_semantic
from src.graph.states.dog_state import DogState


def build_paused_multi_agent_result() -> MultiAgentTaskResult:
    """
    构建第一轮等待用户确认的多 Agent 任务结果。

    功能：
        模拟“读取资料”Worker 已经暂停并询问用户是否允许继续。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskResult:
            可以写入主图状态和 Checkpoint 的暂停任务结果。
    """

    step = AgentTaskStep(
        step_id="read_profile",
        title="读取资料",
        assigned_agent="dog_knowledge_agent",
        status="awaiting_input",
    )
    plan = AgentTaskPlan(
        plan_id="checkpoint_resume_plan",
        objective="生成健康和训练综合方案",
        steps=[step],
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="是否允许读取资料？",
    )
    return MultiAgentTaskResult(
        collaboration_id="checkpoint_resume_task",
        plan=plan,
        status="awaiting_input",
        task_results=[
            AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="是否允许读取资料？",
            )
        ],
    )


def build_completed_multi_agent_result() -> MultiAgentTaskResult:
    """
    构建第二轮恢复执行后的已完成任务结果。

    功能：
        模拟等待步骤收到用户授权后成功执行，并已经完成最终结果聚合。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskResult:
            包含最终回答的已完成任务结果。
    """

    step = AgentTaskStep(
        step_id="read_profile",
        title="读取资料",
        assigned_agent="dog_knowledge_agent",
        status="completed",
    )
    plan = AgentTaskPlan(
        plan_id="checkpoint_resume_plan",
        objective="生成健康和训练综合方案",
        steps=[step],
        status="completed",
    )
    return MultiAgentTaskResult(
        collaboration_id="checkpoint_resume_task",
        plan=plan,
        status="completed",
        task_results=[
            AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
                summary="已读取资料。",
            )
        ],
        final_answer="健康和训练综合方案已生成。",
    )


def build_checkpoint_test_graph():
    """
    构建带真实路由和内存 Checkpointer 的最小主图。

    功能：
        保留真实 semantic_router 路由与多 Agent 恢复适配逻辑，只替换最终
        Agent 节点，使测试集中观察跨轮状态和路由行为。

    参数含义：
        无。

    返回值含义：
        CompiledStateGraph:
            注入 InMemorySaver 后可以跨轮执行的 LangGraph 测试图。
    """

    graph = StateGraph(DogState)
    graph.add_node("semantic_router", semantic_router_node)
    graph.add_node("multi_agent", fake_multi_agent_node)
    graph.add_node("dog_knowledge_agent", fake_other_agent_node)
    graph.add_node("general_agent", fake_other_agent_node)
    graph.add_node("tool_agent", fake_other_agent_node)
    graph.set_entry_point("semantic_router")
    graph.add_conditional_edges(
        "semantic_router",
        route_after_semantic,
        {
            "multi_agent": "multi_agent",
            "dog_knowledge_agent": "dog_knowledge_agent",
            "general_agent": "general_agent",
            "tool_agent": "tool_agent",
            "FINISH": END,
        },
    )
    for node_name in (
        "multi_agent",
        "dog_knowledge_agent",
        "general_agent",
        "tool_agent",
    ):
        graph.add_edge(node_name, END)
    return graph.compile(checkpointer=InMemorySaver())


async def fake_multi_agent_node(
    state: DogState,
) -> dict[str, Any]:
    """
    模拟第一轮暂停和第二轮恢复完成的多 Agent 节点。

    功能：
        resume 动作出现前返回暂停结果；恢复适配器写入 resume 动作后，
        验证用户回答并返回完成结果。

    参数含义：
        state:
            真实语义路由节点处理后的当前主图状态。

    返回值含义：
        dict[str, Any]:
            可以合并进 DogState 并自动写入 Checkpoint 的局部状态。
    """

    if state.get("multi_agent_resume_action") == "resume":
        assert state.get("multi_agent_resume_inputs") == {
            "read_profile": "允许读取资料"
        }
        return build_multi_agent_state_update(
            build_completed_multi_agent_result()
        )
    return build_multi_agent_state_update(
        build_paused_multi_agent_result()
    )


async def fake_other_agent_node(
    state: DogState,
) -> dict[str, Any]:
    """
    标记测试请求错误进入了非多 Agent 节点。

    功能：
        如果真实路由没有进入 multi_agent，返回可被断言识别的错误文本。

    参数含义：
        state:
            当前主图状态，本测试不读取其中字段。

    返回值含义：
        dict[str, Any]:
            包含错误路由标记的局部状态。
    """

    return {
        "current_agent": "unexpected_agent",
        "final_answer": "测试请求进入了错误节点。",
    }


@pytest.mark.asyncio
async def test_same_thread_should_resume_paused_multi_agent_task() -> None:
    """
    测试相同 thread_id 能保存暂停任务并在下一轮恢复执行。

    功能：
        第一轮由 RootAgent 路由到 multi_agent 并保存 awaiting_input 结果；
        第二轮只提交用户回答，验证恢复适配器生成 resume 动作并完成任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    graph = build_checkpoint_test_graph()
    config = {
        "configurable": {
            "thread_id": "multi-agent-resume-thread",
        }
    }

    first_result = await graph.ainvoke(
        {
            "question": "请多 Agent 协作生成健康和训练综合方案",
        },
        config,
    )

    assert first_result["waiting_user_input"] is True
    assert first_result["final_answer"] == "是否允许读取资料？"
    assert first_result["multi_agent_task_result"]["status"] == (
        "awaiting_input"
    )

    checkpoint = await graph.aget_state(config)
    assert checkpoint.values["multi_agent_pending_prompt"] == (
        "是否允许读取资料？"
    )

    second_result = await graph.ainvoke(
        {
            "question": "允许读取资料",
        },
        config,
    )

    assert second_result["current_agent"] == "multi_agent"
    assert second_result["waiting_user_input"] is False
    assert second_result["final_answer"] == (
        "健康和训练综合方案已生成。"
    )
    assert second_result["multi_agent_task_result"]["status"] == (
        "completed"
    )
    assert second_result["multi_agent_resume_action"] == "none"
    assert second_result["multi_agent_resume_ready"] is False
