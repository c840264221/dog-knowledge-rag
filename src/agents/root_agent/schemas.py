from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


RootRoute = Literal[
    "dog_knowledge_agent",
    "general_agent",
    "tool_agent",
    "multi_agent",
    "FINISH",
]


RootQueryType = Literal[
    "dog_knowledge",
    "dog_recommendation",
    "tool_request",
    "multi_agent_task",
    "general_chat",
    "finish",
]


class RootRouteDecision(BaseModel):
    """
    RootAgent 路由决策模型。

    功能：
        表示 Root Supervisor（根调度器）对用户问题做出的主图路由判断。

        V1.7 起，RootAgent 只负责粗路由（Coarse Routing）：
        1. 判断问题是否应该进入 dog_knowledge_agent。
        2. 判断问题是否应该进入 general_agent。
        3. 判断问题是否属于 tool_agent 请求。
        4. 判断是否可以直接 FINISH。

        RootAgent 不负责解析 RAG filters、tags、features、dog_name，
        也不负责创建 RagQuery。
        狗狗知识类问题的细粒度解析由 dog_knowledge_agent 内部 extractor 负责。

    字段说明：
        route:
            主图路由目标。

        query_type:
            用户问题的大类。

        confidence:
            当前路由判断的置信度，范围 0 到 1。

        reason:
            路由原因，用于 Debug（调试）和 Observability（可观测）。

        requires_rag:
            是否需要 RAG（Retrieval-Augmented Generation，检索增强生成）。

        requires_tool:
            是否需要工具调用。

        requires_memory:
            是否建议使用 Memory（记忆）上下文。

        source:
            路由决策来源。
            当前第一版是 rule-based（规则版）。

        hints:
            路由阶段的辅助信息。
            注意：hints 不是正式 RAG filters，不能直接用于 Chroma 检索过滤。

    返回值：
        RootRouteDecision 实例。
        该模型不直接执行 LangGraph 路由，只保存结构化路由结果。
    """

    route: RootRoute = Field(
        ...,
        description="主图路由目标",
    )

    query_type: RootQueryType = Field(
        default="general_chat",
        description="用户问题大类",
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

    requires_rag: bool = Field(
        default=False,
        description="是否需要 RAG 检索增强生成",
    )

    requires_tool: bool = Field(
        default=False,
        description="是否需要工具调用",
    )

    requires_memory: bool = Field(
        default=True,
        description="是否建议使用 Memory 记忆上下文",
    )

    source: str = Field(
        default="root_supervisor_rule_v1",
        description="路由决策来源",
    )

    hints: dict[str, Any] = Field(
        default_factory=dict,
        description="路由辅助信息，不作为正式 RAG filters 使用",
    )
