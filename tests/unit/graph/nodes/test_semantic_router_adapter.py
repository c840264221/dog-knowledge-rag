"""
semantic_router Adapter 单元测试。

功能：
    测试旧主图节点 semantic_router_node 是否已经正确转调新版 RootAgent。

背景：
    V1.7.1 阶段采用 Adapter 过渡方案：
        1. 主图节点名 semantic_router 暂时不变。
        2. 真实路由逻辑迁移到 src.agents.root_agent.supervisor。
        3. router_node.py 只作为兼容入口。

测试目标：
    确保 semantic_router_node 的输出和 RootAgent 标准输出一致。
"""

import pytest

from src.graph.nodes.router_node import (
    semantic_router_node,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question, expected_route",
    [
        (
            "推荐几种适合公寓养的狗",
            "dog_knowledge_agent",
        ),
        (
            "金毛寿命多久？",
            "dog_knowledge_agent",
        ),
        (
            "现在几点？",
            "tool_agent",
        ),
        (
            "你好，你是谁？",
            "general_agent",
        ),
    ],
)
async def test_semantic_router_adapter_calls_root_supervisor(
        question: str,
        expected_route: str,
) -> None:
    """
    测试 semantic_router_node 适配新版 RootAgent。

    功能：
        调用旧入口 semantic_router_node，
        验证它返回新版 route_decision 结构。

    参数：
        question:
            用户问题。

        expected_route:
            预期 route。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        "question": question,
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = await semantic_router_node(
        state,
    )

    route_decision = result.get(
        "route_decision",
        {},
    )

    assert route_decision.get(
        "route",
    ) == expected_route

    assert result.get(
        "next_agent",
    ) == expected_route

    assert result.get(
        "current_agent",
    ) == "root_agent"


@pytest.mark.asyncio
async def test_semantic_router_adapter_does_not_return_legacy_query_parse_fields() -> None:
    """
    测试 semantic_router_node 不再返回旧 query_parse 字段。

    功能：
        验证旧入口不会再输出 intent、filters、tags、features、dog_name
        这些早期 QueryParseResult 字段。

    参数：
        无。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        "question": "推荐适合新手养的狗",
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = await semantic_router_node(
        state,
    )

    assert "intent" not in result
    assert "filters" not in result
    assert "tags" not in result
    assert "features" not in result
    assert "dog_name" not in result