"""
ToolAgent 多轮澄清 Checkpoint 集成测试。

功能：
    使用真实 LangGraph InMemorySaver 验证澄清字段可以按 thread_id
    跨两次图运行保存、恢复和隔离。

测试边界：
    使用真实 semantic_router_node、RootAgent 和澄清恢复适配器；
    使用 fake ToolAgent 节点避免调用真实 LLM、数据库和外部工具。
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.graph.nodes.router_node import semantic_router_node
from src.graph.routes.route_after_semantic import route_after_semantic
from src.graph.states.dog_state import DogState


def build_checkpoint_test_graph():
    """
    构建带内存 Checkpointer 的最小主图。

    功能：
        保留真实语义路由和澄清恢复逻辑，使用 fake Agent 节点收敛外部依赖，
        使测试可以准确观察同一 thread_id 下的 state 恢复行为。

    参数：
        无。

    返回值：
        CompiledStateGraph:
            注入 InMemorySaver 后的可执行 LangGraph 测试图。
    """

    graph = StateGraph(
        DogState
    )
    graph.add_node(
        "semantic_router",
        semantic_router_node,
    )
    graph.add_node(
        "tool_agent",
        fake_tool_agent_node,
    )
    graph.add_node(
        "general_agent",
        fake_general_agent_node,
    )
    graph.add_node(
        "dog_knowledge_agent",
        fake_dog_knowledge_agent_node,
    )

    graph.set_entry_point(
        "semantic_router"
    )
    graph.add_conditional_edges(
        "semantic_router",
        route_after_semantic,
        {
            "tool_agent": "tool_agent",
            "general_agent": "general_agent",
            "dog_knowledge_agent": "dog_knowledge_agent",
            "FINISH": END,
        },
    )
    graph.add_edge(
        "tool_agent",
        END,
    )
    graph.add_edge(
        "general_agent",
        END,
    )
    graph.add_edge(
        "dog_knowledge_agent",
        END,
    )

    return graph.compile(
        checkpointer=InMemorySaver(),
    )


async def fake_tool_agent_node(
    state: DogState,
) -> dict[str, Any]:
    """
    模拟第一轮生成澄清请求和第二轮完成工具执行。

    功能：
        如果 state 已存在恢复后的 database_name，则模拟工具执行成功；
        否则生成缺少 database_name 的结构化澄清状态。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            第一轮返回澄清字段，第二轮返回模拟执行结果。
    """

    tool_calls = state.get(
        "tool_calls",
        [],
    )
    if tool_calls:
        args = tool_calls[0].get(
            "args",
            {},
        )
        database_name = args.get(
            "database_name"
        )
        if database_name:
            return {
                "tool_executed": True,
                "final_answer": f"已查询 {database_name} 数据库。",
                "tool_agent_clarification_request": None,
                "tool_agent_pending_tool_call": None,
                "tool_agent_clarification_resume_ready": False,
            }

    return {
        "need_tool": False,
        "final_answer": "请选择数据库：memory 或 rag。",
        "waiting_user_input": True,
        "tool_agent_clarification_request": {
            "status": "pending",
            "tool_name": "sqlite_list_tables",
            "missing_fields": [
                "database_name",
            ],
            "question": "请选择数据库：memory 或 rag。",
            "options": {
                "database_name": [
                    "memory",
                    "rag",
                ],
            },
            "reason": "工具调用缺少必填参数。",
        },
        "tool_agent_pending_tool_call": {
            "name": "sqlite_list_tables",
            "args": {},
        },
        "tool_agent_pending_original_question": str(
            state.get(
                "question",
                "",
            )
        ),
        "tool_agent_pending_created_at": "2026-07-11T00:00:00+00:00",
    }


async def fake_general_agent_node(
    state: DogState,
) -> dict[str, Any]:
    """
    模拟普通问题 Agent。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            标记请求进入普通 Agent 的状态更新。
    """

    return {
        "final_answer": "普通问题",
        "current_agent": "general_agent",
    }


async def fake_dog_knowledge_agent_node(
    state: DogState,
) -> dict[str, Any]:
    """
    模拟狗狗知识 Agent。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            标记请求进入狗狗知识 Agent 的状态更新。
    """

    return {
        "final_answer": "狗狗知识问题",
        "current_agent": "dog_knowledge_agent",
    }


@pytest.mark.asyncio
async def test_same_thread_should_restore_clarification_and_resume_tool_call() -> None:
    """
    测试相同 thread_id 可以恢复澄清字段并补全工具参数。

    功能：
        第一轮生成澄清请求并检查真实 Checkpoint；
        第二轮只输入 memory，验证历史 state 自动恢复并完成模拟工具执行。

    参数：
        无。

    返回值：
        None。
    """

    graph = build_checkpoint_test_graph()
    config = {
        "configurable": {
            "thread_id": "conversation-same-thread",
        }
    }

    first_result = await graph.ainvoke(
        {
            "question": "帮我查一下数据库有哪些表",
        },
        config,
    )

    assert first_result["tool_agent_clarification_request"]["status"] == "pending"
    assert first_result["tool_agent_pending_tool_call"]["args"] == {}

    first_checkpoint = await graph.aget_state(
        config
    )
    assert first_checkpoint.values[
        "tool_agent_clarification_request"
    ]["missing_fields"] == [
        "database_name",
    ]
    assert first_checkpoint.values[
        "tool_agent_pending_original_question"
    ] == "帮我查一下数据库有哪些表"

    second_result = await graph.ainvoke(
        {
            "question": "memory",
        },
        config,
    )

    assert second_result["tool_executed"] is True
    assert second_result["final_answer"] == "已查询 memory 数据库。"
    assert second_result["tool_calls"][0]["args"] == {
        "database_name": "memory",
    }
    assert second_result["tool_agent_clarification_request"] is None
    assert second_result["tool_agent_pending_tool_call"] is None
    assert second_result["tool_agent_clarification_resume_ready"] is False


@pytest.mark.asyncio
async def test_different_thread_should_not_restore_other_conversation_state() -> None:
    """
    测试不同 thread_id 之间不会共享澄清状态。

    功能：
        第一条线程保存澄清请求后，在另一条全新线程输入 memory，
        验证它不会取得第一条线程的待处理工具调用。

    参数：
        无。

    返回值：
        None。
    """

    graph = build_checkpoint_test_graph()
    first_thread_config = {
        "configurable": {
            "thread_id": "conversation-first",
        }
    }
    other_thread_config = {
        "configurable": {
            "thread_id": "conversation-other",
        }
    }

    await graph.ainvoke(
        {
            "question": "帮我查一下数据库有哪些表",
        },
        first_thread_config,
    )
    other_result = await graph.ainvoke(
        {
            "question": "memory",
        },
        other_thread_config,
    )

    assert other_result["current_agent"] == "general_agent"
    assert other_result["final_answer"] == "普通问题"
    assert "tool_agent_clarification_request" not in other_result
    assert "tool_agent_pending_tool_call" not in other_result
    assert other_result.get(
        "tool_executed",
        False,
    ) is False
