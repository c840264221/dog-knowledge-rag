from langgraph.graph import (
    StateGraph,
    END
)

from src.graph.states.state import DogState

from src.graph.nodes.tool_parse_node import (
    build_tool_parse_node
)

from src.graph.nodes.ask_confirm_tool_node import (
    build_ask_confirm_tool_node
)

from src.agents.general_qa_agent.routes import route_after_executing_tool_worker

from src.graph.routes.route_afer_confirm import route_after_confirm

from src.graph.nodes.execute_tool_node import (
    build_execute_tool_node
)

from src.graph.nodes.answer_gen_node import (
    build_answer_gen_node
)

from src.agents.general_qa_agent.supervisor import (
    build_general_qa_supervisor_node
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
    checkpoint_provider=None
):
    """
    构建 general_qa_agent。

    功能：
        构建通用问答 Agent 的 LangGraph 图。
        注册 memory_retrieve、tool_parse、ask_confirm、execute_tool、answer_gen、supervisor 等节点。
        通过依赖注入方式向 node 传入 llm_provider、memory_provider、checkpoint_provider。
        避免 node 内部直接 import container。

    参数：
        llm_provider：
            LLM Provider（大语言模型提供者）。
            tool_parse_node 需要使用它调用 backup_llm。

        memory_provider：
            MemoryProvider（记忆提供者）。
            如果传入，则启用 Memory 语义召回。

        checkpoint_provider：
            CheckpointProvider（检查点提供者）。
            如果传入，则支持 memory_retrieve 和 tool_parse 保存 checkpoint。

    返回值：
        compiled graph：
            编译后的 LangGraph 图对象。
    """

    if llm_provider is None:
        raise ValueError(
            "build_general_qa_agent 缺少 llm_provider，"
            "请从 container.get('llm') 获取后传入。"
        )

    logger.info(
        "构建 general_qa_agent中..."
    )

    graph = StateGraph(
        DogState
    )

    if memory_provider is not None and checkpoint_provider is not None:
        graph.add_node(
            "memory_retrieve",
            build_memory_retrieve_node(
                semantic_recall=(
                    memory_provider.semantic_recall
                ),
                checkpoint_manager=(
                    checkpoint_provider.manager
                )
            )
        )

    graph.add_node(
        "tool_parse",
        build_tool_parse_node(
            llm_provider=llm_provider,
            checkpoint_manager=(
                checkpoint_provider.manager
                if checkpoint_provider is not None
                else None
            )
        )
    )

    graph.add_node(
        "ask_confirm",
        build_ask_confirm_tool_node(
            checkpoint_manager=(
                checkpoint_provider.manager
                if checkpoint_provider is not None
                else None
            )
        )
    )

    graph.add_node(
        "execute_tool",
        build_execute_tool_node(
            checkpoint_manager=(
                checkpoint_provider.manager
                if checkpoint_provider is not None
                else None
            )
        )
    )

    graph.add_node(
        "answer_gen",
        build_answer_gen_node(
            llm_provider=llm_provider,
            checkpoint_manager=(
                checkpoint_provider.manager
                if checkpoint_provider is not None
                else None
            )
        )
    )

    graph.add_node(
        "supervisor",
        build_general_qa_supervisor_node(
            llm_provider=llm_provider,
            checkpoint_manager=(
                checkpoint_provider.manager
                if checkpoint_provider is not None
                else None
            )
        )
    )


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

    from src.agents.general_qa_agent.valid_workers import (
        VALID_WORKERS
    )

    for worker in VALID_WORKERS:
        if worker not in [
            "ask_confirm",
            "execute_tool"
        ]:
            graph.add_edge(
                worker,
                "supervisor"
            )

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