from typing import Any


RECOMMENDATION_AGENT_ROUTE = "recommendation_agent"

EXACT_AGENT_ROUTE = "exact_agent"

GENERAL_AGENT_ROUTE = "general_agent"

FINISH_ROUTE = "FINISH"

DOG_KNOWLEDGE_AGENT_NODE = "dog_knowledge_agent"

GENERAL_AGENT_NODE = "general"


def build_main_route_alias_map(
        end_node: Any,
) -> dict[str, Any]:
    """
    构建主图路由别名映射表。

    功能：
        将 semantic_router_node 输出的旧 route key 映射到当前主图中真实执行的节点。

        v1.5 当前迁移策略：
        1. semantic_router_node 继续输出 recommendation_agent。
        2. semantic_router_node 继续输出 exact_agent。
        3. 主图不再构建旧 recommendation_agent / exact_search_agent。
        4. recommendation_agent 和 exact_agent 都映射到 dog_knowledge_agent。
        5. general_agent 仍然映射到 general。
        6. FINISH 映射到 LangGraph END。

    参数：
        end_node:
            LangGraph 的 END 节点。
            这里通过参数传入，而不是在本文件直接 import END，
            可以避免 route 工具文件依赖 LangGraph 运行时对象，
            同时方便单元测试传入假 END。

    返回值：
        dict[str, Any]:
            主图 conditional_edges 使用的路由映射表。

    输出格式：
        {
            "recommendation_agent": "dog_knowledge_agent",
            "exact_agent": "dog_knowledge_agent",
            "general_agent": "general",
            "FINISH": END
        }

    专业名词：
        Route Alias：
            路由别名。旧路由 key 不变，但实际执行新节点。

        Main Graph：
            主图。Dog Agent Framework 最外层 LangGraph。

        END：
            LangGraph 的结束节点，表示图执行完成。
    """

    return {
        RECOMMENDATION_AGENT_ROUTE: DOG_KNOWLEDGE_AGENT_NODE,
        EXACT_AGENT_ROUTE: DOG_KNOWLEDGE_AGENT_NODE,
        GENERAL_AGENT_ROUTE: GENERAL_AGENT_NODE,
        FINISH_ROUTE: end_node,
    }