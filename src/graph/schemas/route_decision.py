from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


MainRoute = Literal[
    "recommendation_agent",
    "exact_agent",
    "general_agent",
    "FINISH",
]


class RouteDecision(BaseModel):
    """
    主图路由决策结果。

    功能：
        表示 Main Supervisor（主监督者）对用户问题做出的结构化路由判断。

        v1.5 设计目标：
        1. route 表示主图下一步进入哪个 Agent。
        2. confidence 表示当前路由选择的置信度。
        3. reason 表示为什么选择这个路由。
        4. hints 保存语义路由阶段解析出来的辅助信息。

    重要说明：
        hints 只是路由阶段的辅助信息，不是正式 RAG 检索条件。

        正式 RAG 检索条件应该来自：
            state["rag_query"]["filters"]

        不应该直接使用：
            route_decision["hints"]["filters"]

    技术名词：
        Route:
            路由。表示主图下一步进入哪个 Agent。

        Confidence:
            置信度。表示当前路由判断的可信程度，范围 0 到 1。

        Reason:
            原因说明。用于 Debug（调试）、Trace（链路追踪）、Evaluation（评估）。

        Hints:
            提示信息。表示语义路由阶段解析出来的辅助信息，
            例如 intent、filters、tags、features、dog_name。
            它只用于观察、调试、后续增强，不作为正式检索条件。

    参数：
        route:
            主图路由 key。
            当前必须匹配 Graph.add_conditional_edges 中注册的 key。
            例如 recommendation_agent、exact_agent、general_agent、FINISH。

        confidence:
            路由置信度，范围是 0 到 1。

        reason:
            路由原因。
            用于 Debug（调试）、Trace（链路追踪）、Evaluation（评估）。

        hints:
            路由提示信息。
            保存 semantic_router_node 解析出来的辅助字段。

    返回值：
        RouteDecision 实例。
        用于表达主图下一步应该进入哪个 Agent。
    """

    route: MainRoute = Field(
        ...,
        description="主图路由 key"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="路由置信度，范围 0 到 1"
    )

    reason: str = Field(
        default="",
        description="路由原因"
    )

    hints: dict[str, Any] = Field(
        default_factory=dict,
        description="路由阶段解析出的辅助信息，不作为正式 RAG 检索条件"
    )