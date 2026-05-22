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


def build_general_qa_agent():

    graph = StateGraph(DogState)

    # ========= Workers =========

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

    return graph.compile()