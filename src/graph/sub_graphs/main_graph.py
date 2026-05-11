from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from src.graph.state import DogState

from src.graph.nodes.parse_node import parse_node
from src.graph.nodes.router_node import strategy_router_node

from src.graph.sub_graphs.recommend_subgraph import (
    build_recommendation_subgraph, build_recommendation_subgraph_with_human
)

from src.graph.sub_graphs.exact_search_subgraph import (
    build_exact_search_graph
)

from src.graph.sub_graphs.qa_subgraph import (
    build_general_qa_graph, build_general_qa_subgraph
)
from src.graph.routes.route_by_strategy import route_by_strategy
import sqlite3



def build_main_graph():

    # recommendation_graph = build_recommendation_subgraph()
    recommendation_graph = build_recommendation_subgraph_with_human()
    exact_graph = build_exact_search_graph()
    # general_graph = build_general_qa_graph()
    general_graph = build_general_qa_subgraph()

    def recommendation_node(state):
        return recommendation_graph.invoke(state)

    def exact_node(state):
        return exact_graph.invoke(state)

    def general_node(state):
        return general_graph.invoke(state)

    graph = StateGraph(DogState)

    graph.add_node("parse", parse_node)

    graph.add_node("router", strategy_router_node)

    graph.add_node("recommendation", recommendation_node)
    graph.add_node("exact", exact_node)
    graph.add_node("general", general_node)

    graph.set_entry_point("parse")

    graph.add_edge("parse", "router")

    graph.add_conditional_edges(
        "router",
        route_by_strategy,
        {
            "filtered": "recommendation",
            "exact": "exact",
            "semantic": "general",
            "direct": "general"
        }
    )
    from src.config import CHECKPOINTS_DB_PATH
    conn = sqlite3.connect(CHECKPOINTS_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    # from langgraph.checkpoint.memory import MemorySaver
    # return graph.compile(checkpointer=MemorySaver())
    return graph.compile(checkpointer=checkpointer)