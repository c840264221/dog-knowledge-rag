from __future__ import annotations

from src.agents.root_agent.routes import (
    route_after_root_supervisor,
)
from src.agents.root_agent.schemas import (
    RootRoute,
)
from src.graph.states.dog_state import (
    DogState,
)


SemanticRoute = RootRoute


def route_after_semantic(
        state: DogState,
) -> SemanticRoute:
    """
    semantic_router 后置路由函数。

    功能：
        V1.7 兼容适配层。
        主图中仍然保留旧函数名 route_after_semantic，
        但真实路由逻辑已经迁移到 root_agent.routes.route_after_root_supervisor。

        这样可以做到：
        1. 不大改 GraphRuntimeService。
        2. 不立刻重命名主图节点 semantic_router。
        3. 新版路由 key 统一由 RootAgent 管理。
        4. 兼容旧 route key，例如 recommendation_agent / exact_agent。

    参数：
        state:
            DogState，LangGraph 当前状态。

    返回值：
        SemanticRoute:
            主图 conditional_edges 允许的路由 key。
    """

    return route_after_root_supervisor(
        state=state,
    )
