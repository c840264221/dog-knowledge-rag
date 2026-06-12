from typing import Literal

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


def route_after_semantic(
        state: DogState
) -> SemanticRoute:
    """
    根据 semantic_router_node 的结果选择下一个 Agent。

    功能：
    - 从 state 中读取 next_agent
    - 校验 next_agent 是否属于主图 conditional_edges 允许的路由 key
    - 如果 next_agent 是空字符串、None 或非法值，则兜底到 general_agent
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

    next_agent = str(
        state.get(
            "next_agent"
        )
        or "general_agent"
    ).strip()

    if next_agent not in allowed_routes:

        logger.warning(
            f"非法 next_agent，已兜底到 general_agent: {next_agent!r}"
        )

        next_agent = "general_agent"

    try:
        runtime_ctx.get().state().set_agent(
            next_agent
        )

        runtime_ctx.get().timeline().add_event(
            event_type="route",
            name="route_after_semantic"
        )

    except Exception as e:
        logger.debug(
            f"route_after_semantic 写入 runtime context 失败: {e}"
        )

    return next_agent  # type: ignore[return-value]