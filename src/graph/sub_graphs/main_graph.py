from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END


from src.graph.states.state import DogState

from src.graph.nodes.parse_node import parse_node
from src.graph.nodes.router_node import strategy_router_node

from src.graph.sub_graphs.recommend_subgraph import (
    build_recommendation_subgraph_with_human
)

from src.graph.sub_graphs.exact_search_subgraph import (
    build_exact_search_graph
)

from src.graph.sub_graphs.qa_subgraph import (
    build_general_qa_subgraph
)

from src.agents.general_qa_agent.agent import build_general_qa_agent

from src.agents.recommendation_agent.agent import build_recommendation_agent

from src.agents.exact_search_agent.agent import build_exact_search_agent

from src.agents.supervisor.supervisor_node import supervisor_node

from src.graph.nodes.router_node import semantic_router_node

from src.graph.nodes.memory_extract_node import memory_extract_node

from src.graph.routes.route_by_strategy import route_by_strategy
import sqlite3

from src.logger import logger


def build_main_graph():
    logger.info(f"主图构建中......")
    # recommendation_graph = build_recommendation_subgraph()
    # recommendation_agent = build_recommendation_subgraph_with_human()
    recommendation_agent = build_recommendation_agent()

    exact_graph = build_exact_search_graph()
    exact_agent = build_exact_search_agent()

    # general_graph = build_general_qa_graph()
    # general_graph = build_general_qa_subgraph()
    general_qa_agent = build_general_qa_agent()

    def recommendation_node(state):
        return recommendation_graph.invoke(state)

    def exact_node(state):
        return exact_graph.invoke(state)

    def general_node(state):
        return general_graph.invoke(state)

    graph = StateGraph(DogState)

    # graph.add_node("supervisor", supervisor_node)

    graph.add_node(
        "memory_extract",
        memory_extract_node
    )

    graph.add_node(
        "semantic_router",
        semantic_router_node
    )

    # graph.add_node("parse", parse_node)

    # graph.add_node("router", strategy_router_node)

    # graph.add_node("recommendation", recommendation_node)
    graph.add_node("recommendation_agent", recommendation_agent)

    # graph.add_node("exact", exact_node)
    graph.add_node("exact", exact_agent)
    graph.add_node("general", general_qa_agent)

    # graph.set_entry_point("parse")
    # graph.set_entry_point("semantic_router")
    graph.set_entry_point("memory_extract")

    graph.add_edge("memory_extract", "semantic_router")

    # graph.add_edge("parse", "router")

    graph.add_conditional_edges(

        "semantic_router",

        lambda s: s["next_agent"],

        {

            "recommendation_agent":
                "recommendation_agent",

            "exact_agent":
                "exact",

            "general_agent":
                "general",

            "FINISH":
                END
        }
    )

    graph.add_edge(

        "recommendation_agent",

        END
    )

    graph.add_edge(
        "exact",
        END
    )

    graph.add_edge(
        "general",
        END
    )

    # graph.add_conditional_edges(
    #     "router",
    #     route_by_strategy,
    #     {
    #         # "filtered": "recommendation",
    #         "filtered": "recommendation_agent",
    #         "exact": "exact",
    #         "semantic": "general",
    #         "direct": "general"
    #     }
    # )
    logger.info(f"主图构建完成...")

    from src.config import CHECKPOINTS_DB_PATH

    conn = sqlite3.connect(CHECKPOINTS_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    logger.info(f"checkpointer已就绪")

    # from langgraph.checkpoint.memory import MemorySaver
    # return graph.compile(checkpointer=MemorySaver())
    return graph.compile(checkpointer=checkpointer)