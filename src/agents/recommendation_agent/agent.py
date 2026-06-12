from langgraph.graph import (
    StateGraph,
    END
)

from src.agents.recommendation_agent.state import (
    RecommendationState
)

from src.agents.recommendation_agent.router import (
    route_after_evaluate
)

from src.graph.nodes.filter_node import (
    filter_node
)

from src.graph.nodes.retrieve_node import (
    retrieve_node
)

from src.graph.nodes.evaluate_node import (
    evaluate_retrieval_node
)

from src.graph.nodes.rerank_node import (
    rerank_node
)

from src.graph.nodes.generate_node import (
    build_generate_node
)

from src.graph.nodes.ask_user_node import (
    ask_user_node
)

from src.graph.nodes.retrieval_retry_node import (
    retrieval_retry_node
)

from src.graph.nodes.modify_filter_node import (
    modify_filter_node
)

from src.graph.nodes.tag_filter_node import (
    tag_filter_node
)
from src.graph.routes.route_after_ask_user import (
    route_after_ask_user
)


from src.logger import (
    logger
)

def build_recommendation_agent(
        llm_provider=None,
        memory_provider=None,
        checkpoint_provider=None
):
    """
    构建 recommendation_agent。

    功能：
    - 构建推荐 Agent 的 LangGraph 图
    - 注册 filter、retrieve、evaluate、rerank、generate 等节点
    - 将 LLMProvider、MemoryProvider、CheckpointProvider 注入 generate_node
    - 避免 generate_node 内部直接 import container

    技术名词：
    - Recommendation：推荐，指根据用户需求筛选合适狗狗
    - Provider：提供者，负责统一管理和提供服务对象
    - Node：节点，LangGraph 中的一个执行步骤
    - Rerank：重排序，对召回结果再次精排

    参数：
    - llm_provider:
      LLMProvider 实例。
      中文释义：用于提供主模型和安全调用能力。

    - memory_provider:
      MemoryProvider 实例。
      中文释义：用于召回用户长期记忆。

    - checkpoint_provider:
      CheckpointProvider 实例。
      中文释义：用于保存运行时检查点。

    返回值：
    - compiled graph
      编译后的 recommendation_agent 图对象。
    """

    logger.info(
        "构建 recommendation_agent 中..."
    )

    builder = StateGraph(
        RecommendationState
    )

    generate_node = build_generate_node(
        llm_provider=llm_provider,
        memory_provider=memory_provider,
        checkpoint_provider=checkpoint_provider
    )

    builder.add_node(
        "filter",
        filter_node
    )

    builder.add_node(
        "retrieve",
        retrieve_node
    )

    builder.add_node(
        "evaluate",
        evaluate_retrieval_node
    )

    builder.add_node(
        "retry",
        retrieval_retry_node
    )

    builder.add_node(
        "rerank",
        rerank_node
    )

    builder.add_node(
        "generate",
        generate_node
    )

    builder.add_node(
        "ask_user",
        ask_user_node
    )

    builder.add_node(
        "modify_filter",
        modify_filter_node
    )

    builder.add_node(
        "tag_filter",
        tag_filter_node
    )

    builder.set_entry_point(
        "filter"
    )

    builder.add_edge(
        "filter",
        "retrieve"
    )

    builder.add_edge(
        "retrieve",
        "tag_filter"
    )

    builder.add_edge(
        "tag_filter",
        "evaluate"
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "good": "rerank",
            "retry": "retry",
            "give_up": "generate",
            "ask_user": "ask_user"
        }
    )

    builder.add_edge(
        "retry",
        "retrieve"
    )

    builder.add_edge(
        "rerank",
        "generate"
    )

    builder.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "retry": "retry",
            "modify_filter": "modify_filter",
            "generate": "generate"
        }
    )

    builder.add_edge(
        "modify_filter",
        "retrieve"
    )

    builder.add_edge(
        "generate",
        END
    )

    logger.info(
        "✅ recommendation_agent 构建完成"
    )

    return builder.compile()
