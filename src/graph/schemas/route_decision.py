"""
Deprecated legacy route decision schema.

兼容性说明：
    该模块属于旧版 semantic_router（语义路由）兼容模型，仅为历史代码保留。
    V1.7.1 之后，新的主图路由标准模型是 RootRouteDecision。

新代码要求：
    新的 RootAgent / semantic_router adapter 主链路应优先使用
    src.agents.root_agent.schemas.RootRouteDecision。

该模块属于旧版 semantic_router 路由模型。
V1.7 起，新版主路由决策请使用：

    src.agents.root_agent.schemas.RootRouteDecision

当前文件仅为旧代码兼容保留。
新代码不要继续 import RouteDecision。
"""


from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


MainRoute = Literal[
    "dog_knowledge_agent",
    "general_agent",
    "tool_agent",
    "recommendation_agent",
    "exact_agent",
    "exact_search_agent",
    "FINISH",
]


class RouteDecision(BaseModel):
    """
    主图路由决策结果。

    功能：
        表示 Main Graph（主图）对用户问题做出的结构化路由判断。

        V1.7 说明：
        1. 新版主路由标准优先使用 dog_knowledge_agent / general_agent / tool_agent / FINISH。
        2. recommendation_agent / exact_agent / exact_search_agent 仅作为旧链路兼容 key。
        3. 新代码不应该再主动输出 recommendation_agent 或 exact_agent。
        4. RootAgent 的正式路由模型位于 src.agents.root_agent.schemas.RootRouteDecision。

    字段说明：
        route:
            主图路由 key。

        confidence:
            路由置信度，范围 0 到 1。

        reason:
            路由原因，用于 Debug（调试）、Trace（链路追踪）、Evaluation（评估）。

        hints:
            路由辅助信息。
            注意：hints 不是正式 RAG filters，不能直接用于 Chroma metadata filter。

    返回值：
        RouteDecision 实例。
        该类主要用于旧模块兼容，新版 RootAgent 优先使用 RootRouteDecision。
    """

    route: MainRoute = Field(
        ...,
        description="主图路由 key",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="路由置信度，范围 0 到 1",
    )

    reason: str = Field(
        default="",
        description="路由原因",
    )

    hints: dict[str, Any] = Field(
        default_factory=dict,
        description="路由阶段解析出的辅助信息，不作为正式 RAG 检索条件",
    )
