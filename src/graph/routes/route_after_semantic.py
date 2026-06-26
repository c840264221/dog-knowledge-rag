from typing import Literal, Any

from src.graph.states.state import (
    DogState
)

from src.logger import logger

from src.runtime.context import (
    runtime_ctx
)


SemanticRoute = Literal[
    "recommendation_agent",
    "exact_agent",
    "general_agent",
    "FINISH",
]


def get_route_from_state(
        state: DogState
) -> str:
    """
    从 state 中读取主图路由。

    功能：
    - 优先读取新版 route_decision.route
    - 如果 route_decision 不存在或格式异常，则回退读取旧版 next_agent
    - 如果两个字段都不存在，则兜底到 general_agent

    参数：
    - state: DogState
      LangGraph 当前状态。
      中文释义：Graph 节点之间传递的数据结构。

    返回值：
    - str
      主图路由 key。
      例如 recommendation_agent、exact_agent、general_agent、FINISH。
    """

    route_decision = state.get(
        "route_decision"
    )

    if isinstance(
            route_decision,
            dict
    ):

        route = route_decision.get(
            "route"
        )

        if route:

            return str(
                route
            ).strip()

    next_agent = state.get(
        "next_agent"
    )

    if next_agent:

        return str(
            next_agent
        ).strip()

    return "general_agent"


def route_after_semantic(
        state: DogState
) -> SemanticRoute:
    """
    根据 semantic_router_node 的结果选择下一个 Agent。

    功能：
    - 优先从 route_decision.route 读取新版结构化路由结果
    - 如果 route_decision 不存在，则兼容旧版 next_agent
    - 校验路由 key 是否属于主图 conditional_edges 允许范围
    - 如果路由为空或非法，则兜底到 general_agent
    - 写入 Runtime Context 当前 agent
    - 避免 LangGraph 因非法路由 key 抛出 KeyError

    参数：
    - state: DogState
      LangGraph 当前状态。
      中文释义：Graph 节点之间传递的数据结构。

    返回值：
    - SemanticRoute
      主图允许的路由 key。
      只能是 recommendation_agent、exact_agent、general_agent、FINISH。
    """

    allowed_routes = {
        "recommendation_agent",
        "exact_agent",
        "general_agent",
        "FINISH",
    }

    next_agent = get_route_from_state(
        state
    )

    if next_agent not in allowed_routes:

        logger.warning(
            f"非法主图路由，已兜底到 general_agent: {next_agent!r}"
        )

        next_agent = "general_agent"

    try:
        runtime_context = runtime_ctx.get()

        runtime_context.state().set_agent(
            next_agent
        )

        runtime_context.timeline().add_event(
            event_type="route",
            name="route_after_semantic"
        )

    except Exception as e:
        logger.debug(
            f"route_after_semantic 写入 runtime context 失败: {e}"
        )

    return next_agent  # type: ignore[return-value]