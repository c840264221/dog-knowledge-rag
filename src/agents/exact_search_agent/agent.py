from langgraph.graph import (
    StateGraph,
    END
)

from src.graph.states.state import DogState

from src.graph.nodes.filter_node import (
    filter_node
)

from src.graph.nodes.retrieve_node import (
    retrieve_node
)

from src.graph.nodes.evaluate_node import (
    evaluate_retrieval_node
)

from src.graph.nodes.generate_node import (
    build_generate_node
)

from src.graph.nodes.retrieval_retry_node import (
    retrieval_retry_node
)

from src.agents.exact_search_agent.supervisor import (
    exact_search_supervisor_node
)

from src.agents.exact_search_agent.routes import (
    route_exact_worker
)

from src.logger import logger

def build_exact_search_agent(
        llm_provider=None,
        memory_provider=None,
        checkpoint_provider=None
):
    """
    构建 exact_search_agent。

    功能：
    - 构建精准搜索 Agent 的 LangGraph 图
    - 注册 filter、retrieve、evaluate、retry、generate、supervisor 等节点
    - 将 LLMProvider、MemoryProvider、CheckpointProvider 注入 generate_node
    - 避免 generate_node 内部直接 import container

    技术名词：
    - Agent：智能体，负责完成某一类任务的子流程
    - Provider：提供者，负责统一管理和提供服务对象
    - Node：节点，LangGraph 中的一个执行步骤
    - Container：容器，统一管理项目依赖对象的地方

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
      编译后的 exact_search_agent 图对象。
    """

    logger.info(
        "构建 exact_search_agent 中..."
    )

    graph = StateGraph(
        DogState
    )

    generate_node = build_generate_node(
        llm_provider=llm_provider,
        memory_provider=memory_provider,
        checkpoint_provider=checkpoint_provider
    )

    graph.add_node(
        "filter",
        filter_node
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "evaluate",
        evaluate_retrieval_node
    )

    graph.add_node(
        "retry",
        retrieval_retry_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "supervisor",
        exact_search_supervisor_node
    )

    graph.set_entry_point(
        "supervisor"
    )

    from src.agents.exact_search_agent.valid_workers import VALID_WORKERS

    for worker in VALID_WORKERS:

        graph.add_edge(
            worker,
            "supervisor"
        )

    graph.add_conditional_edges(
        "supervisor",
        route_exact_worker,
        {
            "filter": "filter",
            "retrieve": "retrieve",
            "evaluate": "evaluate",
            "retry": "retry",
            "generate": "generate",
            "finish": END
        }
    )

    app = graph.compile()

    logger.info(
        "✅ exact_search_agent 构建完成"
    )

    return app