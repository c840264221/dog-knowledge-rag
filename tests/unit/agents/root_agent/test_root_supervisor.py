"""
RootAgent Supervisor 单元测试。

功能：
    测试 V1.7.1 RootAgent 的根调度逻辑是否正确。

测试目标：
    1. 狗狗推荐问题应该路由到 dog_knowledge_agent。
    2. 狗狗知识问题应该路由到 dog_knowledge_agent。
    3. 工具类问题应该路由到 tool_agent。
    4. 普通聊天应该路由到 general_agent。
    5. 结束类问题应该路由到 FINISH。

注意：
    当前 V1.7.1 阶段 tool_agent 只是逻辑路由结果。
    主图真实节点映射中，tool_agent 暂时映射到 general_agent。
    后续独立 ToolAgent / MCP 阶段再改为真实 tool_agent 节点。
"""

import pytest

from src.agents.root_agent.supervisor import (
    root_supervisor_node,
)


@pytest.fixture
def base_state() -> dict:
    """
    构建基础测试状态。

    功能：
        提供 RootAgent 测试所需的最小 DogState 字段。

    参数：
        无。

    返回值：
        dict:
            测试用 state，包含 user_id、session_id、trace_id。
    """

    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question, expected_route, expected_query_type",
    [
        (
            "推荐几种适合公寓养的狗",
            "dog_knowledge_agent",
            "dog_recommendation",
        ),
        (
            "金毛寿命多久？",
            "dog_knowledge_agent",
            "dog_knowledge",
        ),
        (
            "金毛和拉布拉多有什么区别？",
            "dog_knowledge_agent",
            "dog_knowledge",
        ),
        (
            "现在几点？",
            "tool_agent",
            "tool_request",
        ),
        (
            "今天纽约天气怎么样？",
            "tool_agent",
            "tool_request",
        ),
        (
            "你好，你是谁？",
            "general_agent",
            "general_chat",
        ),
        (
            "先这样，结束",
            "FINISH",
            "finish",
        ),
    ],
)
async def test_root_supervisor_route_decision(
        base_state: dict,
        question: str,
        expected_route: str,
        expected_query_type: str,
) -> None:
    """
    测试 Root Supervisor 的路由决策。

    功能：
        调用 root_supervisor_node，验证 route_decision 是否符合预期。

    参数：
        base_state:
            pytest fixture，基础测试 state。

        question:
            用户问题。

        expected_route:
            预期路由目标。

        expected_query_type:
            预期问题类型。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        **base_state,
        "question": question,
    }

    result = await root_supervisor_node(
        state,
    )

    route_decision = result.get(
        "route_decision",
        {},
    )

    assert route_decision.get(
        "route",
    ) == expected_route

    assert route_decision.get(
        "query_type",
    ) == expected_query_type

    assert result.get(
        "next_agent",
    ) == expected_route

    assert result.get(
        "current_agent",
    ) == "root_agent"


@pytest.mark.asyncio
async def test_root_supervisor_does_not_generate_rag_filters(
        base_state: dict,
) -> None:
    """
    测试 RootAgent 不生成 RAG 细节字段。

    功能：
        RootAgent 只负责粗路由，不应该在 Root 层生成 filters、tags、
        features、dog_name 等旧 query_parse 字段。

    参数：
        base_state:
            pytest fixture，基础测试 state。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        **base_state,
        "question": "推荐几种适合新手养、掉毛少的狗",
    }

    result = await root_supervisor_node(
        state,
    )

    assert "filters" not in result
    assert "tags" not in result
    assert "features" not in result
    assert "dog_name" not in result
    assert "rag_query" not in result

    route_decision = result.get(
        "route_decision",
        {},
    )

    assert route_decision.get(
        "route",
    ) == "dog_knowledge_agent"

    assert route_decision.get(
        "requires_rag",
    ) is True