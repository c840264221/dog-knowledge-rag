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
    generate_node
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


def build_exact_search_agent():

    graph = StateGraph(DogState)

    # ========= Workers =========

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

    # ========= Supervisor =========

    graph.add_node(
        "supervisor",
        exact_search_supervisor_node
    )

    # ========= Entry =========

    graph.set_entry_point(
        "supervisor"
    )

    # ========= Worker返回Supervisor =========

    from src.agents.exact_search_agent.valid_workers import VALID_WORKERS

    for worker in VALID_WORKERS:

        graph.add_edge(
            worker,
            "supervisor"
        )

    # ========= Dynamic Routing =========

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
    return app