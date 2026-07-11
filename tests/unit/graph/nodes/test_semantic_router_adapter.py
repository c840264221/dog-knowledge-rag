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


@pytest.mark.asyncio
async def test_semantic_router_adapter_should_restore_pending_tool_argument() -> None:
    """
    测试语义路由入口恢复上一轮缺失的工具参数。

    功能：
        Checkpoint state 中存在待补全 database_name 时，输入 memory 应补回参数，
        清理澄清请求并路由到 ToolAgent。

    参数：
        无。

    返回值：
        None:
            pytest 根据断言判断测试结果。
    """

    result = await semantic_router_node(
        {
            "question": "memory",
            "user_id": "test_user",
            "session_id": "test_session",
            "trace_id": "test_trace",
            "tool_agent_clarification_request": {
                "status": "pending",
                "missing_fields": ["database_name"],
                "options": {
                    "database_name": ["memory", "rag"],
                },
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_list_tables",
                "args": {},
            },
        }
    )

    assert result["next_agent"] == "tool_agent"
    assert result["tool_calls"][0]["args"]["database_name"] == "memory"
    assert result["tool_agent_clarification_request"] is None
    assert result["tool_agent_clarification_resume_ready"] is True


@pytest.mark.asyncio
async def test_semantic_router_should_keep_partial_clarification_in_tool_agent() -> None:
    """测试只补完一个字段时仍留在 ToolAgent 继续询问剩余字段。"""

    result = await semantic_router_node(
        {
            "question": "memory",
            "user_id": "test_user",
            "session_id": "test_session",
            "trace_id": "test_trace",
            "tool_agent_clarification_request": {
                "status": "pending",
                "tool_name": "sqlite_describe_table",
                "missing_fields": [
                    "database_name",
                    "table_name",
                ],
                "options": {
                    "database_name": ["memory", "rag"],
                    "table_name": [],
                },
                "question": "请补充数据库别名和表名。",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_describe_table",
                "args": {},
            },
        }
    )

    assert result["next_agent"] == "tool_agent"
    assert result["tool_agent_clarification_resolution"]["action"] == "partial"
    assert result["tool_agent_pending_tool_call"]["args"] == {
        "database_name": "memory",
    }
    assert result["tool_agent_clarification_request"]["missing_fields"] == [
        "table_name",
    ]
