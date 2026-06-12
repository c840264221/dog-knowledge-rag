from langgraph.graph import (
    StateGraph,
    END
)

from src.graph.states.state import DogState

from src.graph.nodes.tool_parse_node import (
    tool_parse_node
)

from src.graph.nodes.ask_confirm_tool_node import (
    ask_confirm_tool_node
)

from src.agents.general_qa_agent.routes import route_after_executing_tool_worker

from src.graph.routes.route_afer_confirm import route_after_confirm

from src.graph.nodes.execute_tool_node import (
    execute_tool_node
)

from src.graph.nodes.answer_gen_node import (
    answer_gen_node
)

from src.agents.general_qa_agent.supervisor import (
    general_qa_supervisor_node
)

from src.agents.general_qa_agent.routes import (
    route_general_qa_worker
)

from src.graph.nodes.memory_retrieve_node import (
    build_memory_retrieve_node
)


from src.logger import logger


def build_general_qa_agent(
    llm_provider=None,
    memory_provider=None,
    checkpoint_provider = None
):
    """
       构建 general_qa_agent。

       功能：
       - 构建通用问答 Agent 的 LangGraph 图
       - 注册 tool_parse、ask_confirm、execute_tool、answer_gen、supervisor 等节点
       - 如果传入 memory_provider，则额外注册 memory_retrieve 节点
       - memory_retrieve 节点会在 supervisor 之前执行，用于召回用户长期记忆
       - 不在 node 内部直接 import container，避免循环导入

       参数：
       - memory_provider:
         MemoryProvider 实例。
         如果传入，则启用 Memory 语义召回。
         如果不传入，则保持旧流程，直接从 supervisor 开始。

       返回值：
       - compiled graph
         编译后的 LangGraph 图对象。
       """

    logger.info(
        "构建 general_qa_agent中..."
    )

    graph = StateGraph(DogState)

    # ========= Workers =========

    # ========= Workers =========

    if memory_provider is not None and checkpoint_provider is not None:
        graph.add_node(
            "memory_retrieve",
            build_memory_retrieve_node(
                semantic_recall=(
                    memory_provider.semantic_recall
                ),
                checkpoint_manager=checkpoint_provider.manager
            )
        )

    graph.add_node(
        "tool_parse",
        tool_parse_node
    )

    graph.add_node(
        "ask_confirm",
        ask_confirm_tool_node
    )

    graph.add_node(
        "execute_tool",
        execute_tool_node
    )

    graph.add_node(
        "answer_gen",
        answer_gen_node
    )

    # ========= Supervisor =========

    graph.add_node(
        "supervisor",
        general_qa_supervisor_node
    )

    # ========= Entry =========

    if memory_provider is not None and checkpoint_provider is not None:

        graph.set_entry_point(
            "memory_retrieve"
        )

        graph.add_edge(
            "memory_retrieve",
            "supervisor"
        )

    else:

        graph.set_entry_point(
            "supervisor"
        )

    # ========= Workers返回Supervisor =========
    from src.agents.general_qa_agent.valid_workers import VALID_WORKERS

    for worker in VALID_WORKERS:
        if worker not in  ["ask_confirm","execute_tool"]:
            graph.add_edge(
                worker,
                "supervisor"
            )

    # ========= Dynamic Routing =========

    graph.add_conditional_edges(
        "ask_confirm",
        route_after_confirm,
        {
            "call_tool": "execute_tool",
            "no_call_tool": "answer_gen"
        }
    )

    graph.add_conditional_edges(
        "execute_tool",
        route_after_executing_tool_worker,
        {
            "ask_confirm": "ask_confirm",
            "answer_gen": "answer_gen"
        }
    )

    graph.add_conditional_edges(

        "supervisor",

        route_general_qa_worker,

        {

            "tool_parse": "tool_parse",

            "ask_confirm": "ask_confirm",

            "execute_tool": "execute_tool",

            "answer_gen": "answer_gen",

            "finish": END
        }
    )

    logger.info(
        "✅ general_qa_agent 构建完成"
    )

    return graph.compile()