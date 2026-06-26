from langgraph.graph import (
    StateGraph,
    END,
)

from src.agents.dog_knowledge_agent.models import (
    extract_model_node,
    recommendation_model_node,
)
from src.agents.dog_knowledge_agent.routes import (
    route_after_dog_knowledge_evaluate,
    route_dog_knowledge_model,
)
from src.graph.nodes.ask_user_node import (
    ask_user_node,
)
from src.graph.nodes.evaluate_node import (
    evaluate_retrieval_node,
)
from src.graph.nodes.generate_node import (
    build_generate_node,
)
from src.graph.nodes.modify_filter_node import (
    build_modify_filter_node,
)
from src.graph.nodes.rerank_node import (
    build_rerank_node,
)
from src.graph.nodes.retrieval_retry_node import (
    build_retrieval_retry_node,
)
from src.graph.nodes.retrieve_node import (
    build_retrieve_node,
)
from src.graph.routes.route_after_ask_user import (
    route_after_ask_user,
)
from src.graph.states.dog_state import (
    DogState,
)
from src.logger import (
    logger,
)


def build_dog_knowledge_agent(
        llm_provider=None,
        memory_provider=None,
        checkpoint_provider=None,
        retriever_provider=None,
        reranker_provider=None,
):
    """
    构建 dog_knowledge_agent 犬种知识领域智能体。

    功能：
        dog_knowledge_agent 是 v1.5 阶段用于统一承载犬种知识类任务的领域 Agent。

        它将原来分散的：
        1. extract_agent / exact_search_agent
        2. recommendation_agent

        收敛为一个统一的领域 Agent，并在内部拆分为：
        1. extract_model
        2. recommendation_model

        两个内部 model 分支共享同一套 RAG 执行链路：
        retrieve -> evaluate -> retry / ask_user / rerank -> generate

    当前主流程：
        dog_knowledge_router
            ↓
        extract_model / recommendation_model
            ↓
        retrieve
            ↓
        evaluate
            ↓
        rerank
            ↓
        generate
            ↓
        END

    当前异常流程：
        evaluate -> retry -> retrieve
        evaluate -> ask_user
        ask_user -> retry -> retrieve
        ask_user -> modify_filter -> retrieve
        ask_user -> generate

    参数：
        llm_provider:
            LLMProvider 实例。
            中文释义：大语言模型提供者，用于 generate_node 调用主模型和备用模型。

        memory_provider:
            MemoryProvider 实例。
            中文释义：记忆提供者，用于 generate_node 召回用户长期记忆。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：检查点提供者，用于保存图运行状态。

        retriever_provider:
            RetrieverProvider 实例。
            中文释义：检索器提供者，用于 retrieve_node 解析 RagQuery 并召回 RAG 上下文。

        reranker_provider:
            RerankerProvider 实例。
            中文释义：重排序提供者，用于 rerank_node 对检索结果二次排序。

    返回值：
        CompiledStateGraph:
            编译后的 LangGraph 子图。
            后续可以作为主图中的 dog_knowledge_agent 节点使用。

    输出格式：
        最终输出仍然是 DogState。
        常见最终字段：
        - answer
        - final_answer
        - memory_context
        - answer_strategy
        - rag_query
        - rag_context
        - route_decision
        - messages

    专业名词：
        Domain Agent：
            领域智能体。面向某个业务领域的统一 Agent。
            dog_knowledge_agent 就是面向犬种知识领域的 Agent。

        Internal Model：
            内部模型 / 内部分支。这里不是 LLM，而是 Agent 内部的业务分支。

        RAG：
            Retrieval-Augmented Generation，检索增强生成。
            先检索知识库，再让 LLM 基于检索上下文生成答案。

        Retriever：
            检索器。负责从知识库中召回相关内容。

        Reranker：
            重排序器。负责对召回内容重新排序。

        Checkpoint：
            检查点。用于保存图运行状态，支持恢复和多轮执行。
    """

    logger.info(
        "构建 dog_knowledge_agent 中..."
    )

    if llm_provider is None:
        raise RuntimeError(
            "build_dog_knowledge_agent 缺少 llm_provider。"
            "dog_knowledge_agent 需要通过 generate_node 调用 LLM。"
        )

    if retriever_provider is None:
        raise RuntimeError(
            "build_dog_knowledge_agent 缺少 retriever_provider。"
            "dog_knowledge_agent 需要通过 retrieve_node 执行 RAG 检索。"
        )

    if reranker_provider is None:
        raise RuntimeError(
            "build_dog_knowledge_agent 缺少 reranker_provider。"
            "dog_knowledge_agent 需要通过 rerank_node 执行检索结果重排序。"
        )

    builder = StateGraph(
        DogState
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

    modify_filter_node = build_modify_filter_node(
        checkpoint_provider=checkpoint_provider,
    )

    # =========================
    # 1. 内部 model 分支
    # =========================

    builder.add_node(
        "extract_model",
        extract_model_node,
    )

    builder.add_node(
        "recommendation_model",
        recommendation_model_node,
    )

    # =========================
    # 2. 共享 RAG 节点
    # =========================

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
        retry_node,
    )

    builder.add_node(
        "rerank",
        rerank_node,
    )

    builder.add_node(
        "generate",
        generate_node,
    )

    # =========================
    # 3. 人机交互 / 条件修正节点
    # =========================

    builder.add_node(
        "ask_user",
        ask_user_node,
    )

    builder.add_node(
        "modify_filter",
        modify_filter_node,
    )

    # =========================
    # 4. 入口路由
    # =========================

    builder.set_conditional_entry_point(
        route_dog_knowledge_model,
        {
            "extract_model": "extract_model",
            "recommendation_model": "recommendation_model",
        },
    )

    # =========================
    # 5. model -> retrieve
    # =========================

    builder.add_edge(
        "extract_model",
        "retrieve",
    )

    builder.add_edge(
        "recommendation_model",
        "retrieve",
    )

    # =========================
    # 6. RAG 主链路
    # =========================

    builder.add_edge(
        "retrieve",
        "evaluate",
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_dog_knowledge_evaluate,
        {
            "rerank": "rerank",
            "retry": "retry",
            "ask_user": "ask_user",
            "generate": "generate",
        },
    )

    builder.add_edge(
        "retry",
        "retrieve",
    )

    builder.add_edge(
        "rerank",
        "generate",
    )

    # =========================
    # 7. ask_user 后续链路
    # =========================

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

    # =========================
    # 8. 结束
    # =========================

    builder.add_edge(
        "generate",
        END,
    )

    logger.info(
        "✅ dog_knowledge_agent 构建完成"
    )

    return builder.compile()