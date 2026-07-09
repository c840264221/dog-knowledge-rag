from src.graph.routes.main_route_alias import (
    DOG_KNOWLEDGE_AGENT_NODE,
    DOG_KNOWLEDGE_AGENT_ROUTE,
    EXACT_AGENT_ROUTE,
    EXACT_SEARCH_AGENT_ROUTE,
    FINISH_ROUTE,
    GENERAL_AGENT_NODE,
    GENERAL_AGENT_ROUTE,
    RECOMMENDATION_AGENT_ROUTE,
    TOOL_AGENT_NODE,
    TOOL_AGENT_ROUTE,
    build_main_route_alias_map,
)


def test_recommendation_agent_route_alias_to_dog_knowledge_agent():
    """
    测试 recommendation_agent 路由会映射到 dog_knowledge_agent。

    功能：
        防止后续主图迁移过程中，
        recommendation_agent 被误接回旧 recommendation_agent 子图。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert (
        route_map[RECOMMENDATION_AGENT_ROUTE]
        == DOG_KNOWLEDGE_AGENT_NODE
    )


def test_exact_agent_route_alias_to_dog_knowledge_agent():
    """
    测试 exact_agent 路由会映射到 dog_knowledge_agent。

    功能：
        防止后续主图迁移过程中，
        exact_agent 被误接回旧 exact_search_agent 子图。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert (
        route_map[EXACT_AGENT_ROUTE]
        == DOG_KNOWLEDGE_AGENT_NODE
    )


def test_general_agent_route_keeps_general_agent():
    """
    测试 general_agent 仍然映射到 general 节点。

    功能：
        确认普通问答不会错误进入 dog_knowledge_agent。
        general_agent 仍然应该由 general_qa_agent 处理。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert (
        route_map[GENERAL_AGENT_ROUTE]
        == GENERAL_AGENT_NODE
    )


def test_finish_route_maps_to_end_node():
    """
    测试 FINISH 路由会映射到 END 节点。

    功能：
        确认主图可以正常结束。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert route_map[FINISH_ROUTE] == "__END__"


def test_tool_agent_route_maps_to_tool_agent_node():
    """
    测试 tool_agent 路由会映射到新版 ToolAgent 节点。

    功能：
        V1.8 起，工具类请求不再临时回流 general 节点，
        而是进入独立 tool_agent 子图。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert (
        route_map[TOOL_AGENT_ROUTE]
        == TOOL_AGENT_NODE
    )


def test_main_route_alias_map_contains_only_expected_routes():
    """
    测试主图路由别名表只包含预期 route。

    功能：
        防止后续意外加入未设计的 route key，
        导致主图路由行为不清晰。

    参数：
        无。

    返回值：
        无。
    """

    route_map = build_main_route_alias_map(
        end_node="__END__",
    )

    assert set(route_map.keys()) == {
        DOG_KNOWLEDGE_AGENT_ROUTE,
        RECOMMENDATION_AGENT_ROUTE,
        EXACT_AGENT_ROUTE,
        EXACT_SEARCH_AGENT_ROUTE,
        GENERAL_AGENT_ROUTE,
        TOOL_AGENT_ROUTE,
        FINISH_ROUTE,
    }
