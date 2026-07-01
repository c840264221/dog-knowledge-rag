from __future__ import annotations

from typing import (
    Any,
)

from src.agents.root_agent.routes import (
    build_root_route_alias_map,
)


RECOMMENDATION_AGENT_ROUTE = "recommendation_agent"

EXACT_AGENT_ROUTE = "exact_agent"

EXACT_SEARCH_AGENT_ROUTE = "exact_search_agent"

DOG_KNOWLEDGE_AGENT_ROUTE = "dog_knowledge_agent"

GENERAL_AGENT_ROUTE = "general_agent"

TOOL_AGENT_ROUTE = "tool_agent"

FINISH_ROUTE = "FINISH"

DOG_KNOWLEDGE_AGENT_NODE = "dog_knowledge_agent"

GENERAL_AGENT_NODE = "general"


def build_main_route_alias_map(
        end_node: Any,
) -> dict[str, Any]:
    """
    构建主图路由别名映射表。

    功能：
        V1.7 兼容适配层。
        主图仍然调用 build_main_route_alias_map，
        但新版标准路由映射由 root_agent.routes.build_root_route_alias_map 管理。

        当前支持两类 route key：
        1. 新版 route key：
           - dog_knowledge_agent
           - general_agent
           - tool_agent
           - FINISH

        2. 旧版 route key：
           - recommendation_agent -> dog_knowledge_agent
           - exact_agent -> dog_knowledge_agent
           - exact_search_agent -> dog_knowledge_agent

    参数：
        end_node:
            LangGraph END 节点。

    返回值：
        dict[str, Any]:
            主图 conditional_edges 使用的路由映射表。
    """

    route_map = build_root_route_alias_map(
        end_node=end_node,
    )

    route_map.update(
        {
            RECOMMENDATION_AGENT_ROUTE: DOG_KNOWLEDGE_AGENT_NODE,
            EXACT_AGENT_ROUTE: DOG_KNOWLEDGE_AGENT_NODE,
            EXACT_SEARCH_AGENT_ROUTE: DOG_KNOWLEDGE_AGENT_NODE,
        }
    )

    return route_map
