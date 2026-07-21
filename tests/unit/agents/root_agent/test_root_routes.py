"""
RootAgent Routes 单元测试。

功能：
    测试 RootAgent 路由归一化逻辑和 route_after_root_supervisor。

测试目标：
    1. 新版 route key 可以正常返回。
    2. 旧版 route key 可以兼容映射。
    3. 非法 route key 会兜底到 general_agent。
    4. tool_agent 当前会归一化到新版工具智能体路由。
"""

import pytest

from src.agents.root_agent.routes import (
    normalize_root_route,
    route_after_root_supervisor,
)


@pytest.mark.parametrize(
    "raw_route, expected_route",
    [
        (
            "dog_knowledge_agent",
            "dog_knowledge_agent",
        ),
        (
            "general_agent",
            "general_agent",
        ),
        (
            "tool_agent",
            "tool_agent",
        ),
        (
            "multi_agent",
            "multi_agent",
        ),
        (
            "FINISH",
            "FINISH",
        ),
        (
            "recommendation_agent",
            "dog_knowledge_agent",
        ),
        (
            "exact_agent",
            "dog_knowledge_agent",
        ),
        (
            "exact_search_agent",
            "dog_knowledge_agent",
        ),
        (
            "general",
            "general_agent",
        ),
        (
            "unknown_agent",
            "general_agent",
        ),
    ],
)
def test_normalize_root_route(
        raw_route: str,
        expected_route: str,
) -> None:
    """
    测试 Root 路由 key 归一化。

    功能：
        验证新旧路由 key 都能被转换成 V1.7 标准 route。

    参数：
        raw_route:
            原始路由字符串。

        expected_route:
            预期归一化后的路由字符串。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    actual_route = normalize_root_route(
        raw_route,
    )

    assert actual_route == expected_route


@pytest.mark.parametrize(
    "route, expected_route",
    [
        (
            "dog_knowledge_agent",
            "dog_knowledge_agent",
        ),
        (
            "general_agent",
            "general_agent",
        ),
        (
            "tool_agent",
            "tool_agent",
        ),
        (
            "multi_agent",
            "multi_agent",
        ),
        (
            "FINISH",
            "FINISH",
        ),
    ],
)
def test_route_after_root_supervisor_reads_route_decision(
        route: str,
        expected_route: str,
) -> None:
    """
    测试 route_after_root_supervisor 读取 route_decision。

    功能：
        验证条件边路由函数可以从 state["route_decision"]["route"]
        中读取路由目标。

    参数：
        route:
            写入 state 的 route。

        expected_route:
            预期返回 route。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        "route_decision": {
            "route": route,
        }
    }

    actual_route = route_after_root_supervisor(
        state,
    )

    assert actual_route == expected_route


def test_route_after_root_supervisor_fallback_to_general_agent() -> None:
    """
    测试 route_after_root_supervisor 的兜底行为。

    功能：
        当 state 中没有 route_decision 时，应该兜底到 general_agent。

    参数：
        无。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {}

    actual_route = route_after_root_supervisor(
        state,
    )

    assert actual_route == "general_agent"
