from __future__ import annotations

from typing import (
    Any,
)

from src.agents.root_agent.schemas import (
    RootRoute,
)
from src.graph.states.dog_state import (
    DogState,
)
from src.logger import logger
from src.runtime.context import (
    runtime_ctx,
)


ROUTE_ALIASES: dict[str, RootRoute] = {
    "dog_knowledge_agent": "dog_knowledge_agent",
    "dog_knowledge": "dog_knowledge_agent",
    "recommendation_agent": "dog_knowledge_agent",
    "exact_agent": "dog_knowledge_agent",
    "exact_search_agent": "dog_knowledge_agent",
    "general_agent": "general_agent",
    "general": "general_agent",
    # "tool_agent": "tool_agent",
    # "tool": "tool_agent",
    # todo: 当前tool_agent还未抽取出来 现融进了general_agent 所以暂时路由到general_agent
    # todo：后续单独抽出后再改回上面的路由
    "tool_agent": "general_agent",
    "tool": "general_agent",
    "FINISH": "FINISH",
    "finish": "FINISH",
}


def get_root_route_from_state(
        state: DogState,
) -> str:
    """
    从 state 中读取 Root 路由。

    功能：
        优先读取新版 route_decision.route。
        如果没有，则兼容读取旧版 next_agent。
        如果仍然没有，则兜底到 general_agent。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            原始路由字符串，后续还需要 normalize_root_route 归一化。
    """

    route_decision = state.get(
        "route_decision",
    )

    if isinstance(
            route_decision,
            dict,
    ):
        route = route_decision.get(
            "route",
        )

        if route:
            return str(
                route,
            ).strip()

    next_agent = state.get(
        "next_agent",
    )

    if next_agent:
        return str(
            next_agent,
        ).strip()

    return "general_agent"


def normalize_root_route(
        route: Any,
) -> RootRoute:
    """
    将新旧路由 key 归一化为 V1.7 RootRoute。

    功能：
        支持新版 route key：
        1. dog_knowledge_agent
        2. general_agent
        3. tool_agent
        4. FINISH

        同时兼容旧 route key：
        1. recommendation_agent -> dog_knowledge_agent
        2. exact_agent -> dog_knowledge_agent
        3. exact_search_agent -> dog_knowledge_agent

    参数：
        route:
            原始路由 key。

    返回值：
        RootRoute:
            归一化后的新版主图路由。
    """

    route_text = str(
        route or "",
    ).strip()

    normalized_route = ROUTE_ALIASES.get(
        route_text,
    )

    if normalized_route:
        return normalized_route

    logger.warning(
        f"非法 Root 路由，已兜底到 general_agent: {route_text!r}"
    )

    return "general_agent"


def route_after_root_supervisor(
        state: DogState,
) -> RootRoute:
    """
    Root Supervisor 后置路由函数。

    功能：
        根据 root_supervisor_node 写入的 route_decision 决定主图下一步。

        注意：
            V1.7.1 中主图节点名仍然保留 semantic_router，
            但真实路由语义已经由 root_agent.routes 管理。

    参数：
        state:
            当前 DogState。

    返回值：
        RootRoute:
            主图 conditional_edges 允许的路由 key。
    """

    raw_route = get_root_route_from_state(
        state=state,
    )

    next_route = normalize_root_route(
        route=raw_route,
    )

    try:
        runtime_context = runtime_ctx.get()

        if runtime_context is not None:
            runtime_context.state().set_agent(
                next_route,
            )

            runtime_context.timeline().add_event(
                event_type="route",
                name="route_after_root_supervisor",
            )

    except Exception as exc:
        logger.debug(
            f"route_after_root_supervisor 写入 runtime context 失败: {exc}"
        )

    return next_route


def build_root_route_alias_map(
        end_node: Any,
) -> dict[str, Any]:
    """
    构建 RootAgent 主图路由映射表。

    功能：
        将 RootRoute 映射到主图真实节点。

        当前 V1.7.1 说明：
        1. dog_knowledge_agent -> dog_knowledge_agent。
        2. general_agent -> general。
        3. tool_agent -> general。
           因为工具链路当前仍在 general_qa_agent 内部，后续再独立拆 tool_agent。
        4. FINISH -> END。

    参数：
        end_node:
            LangGraph END 节点。

    返回值：
        dict[str, Any]:
            StateGraph.add_conditional_edges 使用的映射表。
    """

    return {
        "dog_knowledge_agent": "dog_knowledge_agent",
        "general_agent": "general",
        "tool_agent": "general",
        "FINISH": end_node,
    }
