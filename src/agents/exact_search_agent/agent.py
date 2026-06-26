from langgraph.graph import (
    StateGraph,
    END,
)

from src.graph.states.dog_state import (
    DogState,
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

from src.graph.nodes.retrieval_retry_node import (
    build_retrieval_retry_node,
)

from src.agents.exact_search_agent.supervisor import (
    build_exact_search_supervisor_node,
)

from src.agents.exact_search_agent.routes import (
    route_exact_worker,
)

from src.logger import (
    logger,
)


VALID_WORKERS = [
    "retrieve",
    "evaluate",
    "retry",
    "rerank",
    "generate",
]


def build_exact_search_agent(
        llm_provider=None,
        memory_provider=None,
        checkpoint_provider=None,
        retriever_provider=None,
        reranker_provider=None,
):
    """
    构建 exact_search_agent 子图。

    功能：
        exact_search_agent 用于处理犬种精确信息查询类问题。

        v1.5 当前设计：
        1. 使用全局 DogState。
        2. 使用 build_retrieve_node 接入新版 RAG 检索。
        3. 使用 build_rerank_node 接入新版 RAG 重排。
        4. 不再注册旧版 filter_node。
        5. 正式检索条件统一由 RagQuery.filters 生成。
        6. 主链路统一为：
           retrieve -> evaluate -> rerank -> generate。

    子图流程：
        supervisor
          ↓
        retrieve / evaluate / retry / rerank / generate
          ↓
        supervisor
          ↓
        finish

    技术名词：
        Exact Search Agent：
            精确查询智能体。
            用于回答某个具体犬种的信息问题，例如寿命、体型、性格等。

        Rerank：
            重排。
            对 retrieve_node 召回的 chunks 进行二次排序，让最相关的内容排在前面。

        RerankerProvider：
            重排器提供者。
            用于统一提供 reranker 模型，避免节点内部直接加载模型。

        RAG：
            Retrieval-Augmented Generation，检索增强生成。
            先检索知识库，再让大模型基于上下文生成答案。

    参数：
        llm_provider:
            LLMProvider 实例。
            中文释义：用于 generate_node 和 supervisor_node 调用大模型。

        memory_provider:
            MemoryProvider 实例。
            中文释义：用于 generate_node 召回用户长期记忆。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于节点执行后保存 checkpoint。

        retriever_provider:
            RetrieverProvider 实例。
            中文释义：用于 retrieve_node 获取 DogQueryFilterParser 和 MetadataFilterRetriever。

        reranker_provider:
            RerankerProvider 实例。
            中文释义：用于 rerank_node 获取 reranker 模型。

    返回值：
        compiled graph:
            编译后的 LangGraph 子图。
    """

    logger.info(
        "构建 exact_search_agent 中..."
    )

    if llm_provider is None:
        raise RuntimeError(
            "build_exact_search_agent 缺少 llm_provider"
        )

    if retriever_provider is None:
        raise RuntimeError(
            "build_exact_search_agent 缺少 retriever_provider。"
            "v1.5 exact_agent 新版 RAG 改造后，"
            "必须从 GraphRuntimeService 注入 container.get('retriever')。"
        )

    if reranker_provider is None:
        raise RuntimeError(
            "build_exact_search_agent 缺少 reranker_provider。"
            "v1.5 exact_agent 接入 rerank 后，"
            "必须从 GraphRuntimeService 注入 container.get('reranker')。"
        )

    supervisor_node = build_exact_search_supervisor_node(
        llm_provider=llm_provider,
    )

    retrieve_node = build_retrieve_node(
        retriever_provider=retriever_provider,
        checkpoint_provider=checkpoint_provider,
    )

    retry_node = build_retrieval_retry_node(
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

    graph = StateGraph(
        DogState
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "evaluate",
        evaluate_retrieval_node,
    )

    graph.add_node(
        "retry",
        retry_node,
    )

    graph.add_node(
        "rerank",
        rerank_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    graph.set_entry_point(
        "supervisor"
    )

    for worker in VALID_WORKERS:
        graph.add_edge(
            worker,
            "supervisor",
        )

    graph.add_conditional_edges(
        "supervisor",
        route_exact_worker,
        {
            # v1.5 兼容旧 supervisor：
            # 如果某些旧逻辑仍然输出 filter，
            # 暂时重定向到 retrieve。
            "filter": "retrieve",

            "retrieve": "retrieve",
            "evaluate": "evaluate",
            "retry": "retry",
            "rerank": "rerank",
            "generate": "generate",
            "finish": END,
        },
    )

    logger.info(
        "✅ exact_search_agent 构建完成"
    )

    return graph.compile()