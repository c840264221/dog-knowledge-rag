from langgraph.graph import (
    StateGraph,
    END,
)

from src.graph.states.dog_state import (
    DogState,
)

from src.agents.recommendation_agent.router import (
    route_after_evaluate,
)

from src.graph.nodes.retrieve_node import (
    build_retrieve_node,
)

from src.graph.nodes.evaluate_node import (
    evaluate_retrieval_node,
)

from src.graph.nodes.rerank_node import (
    build_rerank_node,
)

from src.graph.nodes.generate_node import (
    build_generate_node,
)

from src.graph.nodes.ask_user_node import (
    ask_user_node,
)

from src.graph.nodes.retrieval_retry_node import (
    retrieval_retry_node,
)

from src.graph.nodes.modify_filter_node import (
    build_modify_filter_node,
)

from src.graph.routes.route_after_ask_user import (
    route_after_ask_user,
)

from src.logger import (
    logger,
)


def build_recommendation_agent(
        llm_provider=None,
        memory_provider=None,
        checkpoint_provider=None,
        retriever_provider=None,
        reranker_provider=None,
):
    """
    构建 recommendation_agent 推荐智能体子图。

    功能：
        构建 recommendation_agent 的 LangGraph 子图，
        用于处理“根据用户条件推荐犬种”的问题。

        v1.5 当前设计：
        1. 使用全局 DogState，替代旧版 RecommendationState。
        2. 使用 build_retrieve_node 注入 retriever_provider。
        3. 使用 build_rerank_node 注入 reranker_provider。
        4. 使用 build_generate_node 注入 llm_provider、memory_provider、checkpoint_provider。
        5. 移除旧版 tag_filter_node，避免旧 tags 把新版 RAG 召回结果过滤为空。
        6. 移除旧版 filter_node，让推荐链路的正式 filters 统一由 RagQuery.filters 生成。

    当前主链路：
        retrieve -> evaluate -> rerank -> generate

    异常处理链路：
        evaluate -> retry -> retrieve
        evaluate -> ask_user
        ask_user -> retry -> retrieve
        ask_user -> modify_filter -> retrieve
        ask_user -> generate

    技术名词：
        Recommendation Agent：
            推荐智能体。根据用户需求推荐合适犬种。

        DogState：
            Dog Agent Framework 的全局状态对象。
            用来在主图和子图之间传递 question、rag_query、rag_context 等字段。

        RetrieverProvider：
            检索器提供者。
            负责统一提供 DogQueryFilterParser 和 MetadataFilterRetriever。

        RerankerProvider：
            重排器提供者。
            负责统一提供 reranker 模型。

        RAG：
            Retrieval-Augmented Generation，检索增强生成。
            先检索相关资料，再让 LLM 根据资料生成答案。

        RagQuery.filters：
            新版 RAG 的正式 metadata filter 字段。
            recommendation_agent 中正式检索条件应该统一从这里读取。

        filter_node：
            旧版过滤节点。
            当前 recommendation_agent 新版 RAG 主链路中不再使用。

    参数：
        llm_provider:
            LLMProvider 实例。
            中文释义：用于提供主模型、备用模型、安全调用等能力。

        memory_provider:
            MemoryProvider 实例。
            中文释义：用于提供用户长期记忆召回能力。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint。

        retriever_provider:
            RetrieverProvider 实例。
            中文释义：用于提供新版 RAG parser 和 retriever。

        reranker_provider:
            RerankerProvider 实例。
            中文释义：用于提供 reranker 模型。

    返回值：
        CompiledStateGraph:
            编译后的 recommendation_agent 子图。
            该子图可以作为主图中的一个 node 使用。
    """

    logger.info(
        "构建 recommendation_agent 中..."
    )

    if retriever_provider is None:
        raise RuntimeError(
            "build_recommendation_agent 缺少 retriever_provider。"
            "v1.5 recommendation_agent 新版 RAG 改造后，"
            "必须从 GraphRuntimeService 注入 container.get('retriever')。"
        )

    if reranker_provider is None:
        raise RuntimeError(
            "build_recommendation_agent 缺少 reranker_provider。"
            "v1.5 recommendation_agent 新版 RAG rerank 改造后，"
            "必须从 GraphRuntimeService 注入 container.get('reranker')。"
        )

    builder = StateGraph(
        DogState
    )

    retrieve_node = build_retrieve_node(
        retriever_provider=retriever_provider,
        checkpoint_provider=checkpoint_provider,
    )

    rerank_node = build_rerank_node(
        reranker_provider=reranker_provider,
        checkpoint_provider=checkpoint_provider,
    )

    generate_node = build_generate_node(
        llm_provider=llm_provider,
        memory_provider=memory_provider,
        checkpoint_provider=checkpoint_provider,
    )

    builder.add_node(
        "retrieve",
        retrieve_node,
    )

    builder.add_node(
        "evaluate",
        evaluate_retrieval_node,
    )

    builder.add_node(
        "retry",
        retrieval_retry_node,
    )

    builder.add_node(
        "rerank",
        rerank_node,
    )

    builder.add_node(
        "generate",
        generate_node,
    )

    builder.add_node(
        "ask_user",
        ask_user_node,
    )

    modify_filter_node = build_modify_filter_node(
        checkpoint_provider=checkpoint_provider,
    )
    builder.add_node(
        "modify_filter",
        modify_filter_node,
    )

    # v1.5 关键变化：
    # recommendation_agent 不再从 filter_node 开始，
    # 而是直接从 retrieve_node 开始。
    # 正式 filters 由 build_rag_query_from_state 构建到 RagQuery.filters。
    builder.set_entry_point(
        "retrieve",
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "good": "rerank",
            "retry": "retry",
            "give_up": "generate",
            "ask_user": "ask_user",
        },
    )

    builder.add_edge(
        "retrieve",
        "evaluate",
    )

    builder.add_edge(
        "retry",
        "retrieve",
    )

    builder.add_edge(
        "rerank",
        "generate",
    )

    builder.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "retry": "retry",
            "modify_filter": "modify_filter",
            "generate": "generate",
        },
    )

    builder.add_edge(
        "modify_filter",
        "retrieve",
    )

    builder.add_edge(
        "generate",
        END,
    )

    logger.info(
        "✅ recommendation_agent 构建完成"
    )

    return builder.compile()