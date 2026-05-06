from langgraph.graph import StateGraph
from src.graph.state import DogState

from src.graph.nodes.parse_node import parse_node
from src.graph.nodes.filter_node import filter_node
from src.graph.nodes.retrieve_node import RetrieveNode
from src.graph.nodes.tag_filter_node import tag_filter_node
from src.graph.nodes.rerank_node import rerank_node
from src.graph.nodes.generate_node import generate_node


def build_graph(db):
    graph = StateGraph(DogState)

    graph.add_node("parse", parse_node)
    graph.add_node("filter", filter_node)
    graph.add_node("retrieve", RetrieveNode(db))
    graph.add_node("tag_filter", tag_filter_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("parse")

    graph.add_edge("parse", "filter")
    graph.add_edge("filter", "retrieve")
    graph.add_edge("retrieve", "tag_filter")
    graph.add_edge("tag_filter", "rerank")
    graph.add_edge("rerank", "generate")

    graph.set_finish_point("generate")

    return graph.compile()