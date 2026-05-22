from langgraph.graph import StateGraph

from src.graph.states.state import DogState

from src.graph.nodes.filter_node import filter_node
from src.graph.nodes.retrieve_node import retrieve_node
from src.graph.nodes.evaluate_node import evaluate_retrieval_node
from src.graph.nodes.retrieval_retry_node import retrieval_retry_node
from src.graph.nodes.generate_node import generate_node


def route_after_evaluate(state):

    if state["retrieval_ok"]:
        return "good"

    if state.get("retry_count", 0) >= 2:
        return "give_up"

    return "retry"


def build_exact_search_graph():

    graph = StateGraph(DogState)

    graph.add_node("filter", filter_node)
    graph.add_node("retrieve", retrieve_node)

    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("retry", retrieval_retry_node)

    graph.add_node("generate", generate_node)

    graph.set_entry_point("filter")

    graph.add_edge("filter", "retrieve")

    graph.add_edge("retrieve", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "good": "generate",
            "retry": "retry",
            "give_up": "generate"
        }
    )

    graph.add_edge("retry", "retrieve")

    return graph.compile()