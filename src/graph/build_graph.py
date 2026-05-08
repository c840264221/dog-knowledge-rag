from langgraph.graph import StateGraph
from src.graph.state import DogState

from src.graph.nodes.parse_node import parse_node
from src.graph.nodes.router_node import strategy_router_node
from src.graph.nodes.filter_node import filter_node
from src.graph.nodes.retrieve_node import retrieve_node
from src.graph.nodes.tag_filter_node import tag_filter_node
from src.graph.nodes.rerank_node import rerank_node
from src.graph.nodes.generate_node import generate_node
from src.graph.routes.route_by_strategy import route_by_strategy
from src.graph.nodes.evaluate_node import evaluate_retrieval_node
from src.graph.routes.route_after_retrieval import route_after_retrieval
from src.graph.nodes.retrieval_retry_node import retrieval_retry_node


def build_graph(db):
    graph = StateGraph(DogState)

    graph.add_node("parse", parse_node)

    # 注册路由
    graph.add_node("strategy_router", strategy_router_node)

    # 推荐流
    graph.add_node("filter", filter_node)
    graph.add_node("retrieve", retrieve_node(db))
    graph.add_node("tag_filter", tag_filter_node)
    graph.add_node("rerank", rerank_node)

    # QA流
    graph.add_node("retrieve_qa", retrieve_node(db))

    # 注册评估检索节点
    graph.add_node("evaluate_retrieval", evaluate_retrieval_node)

    # 注册重试检索节点
    graph.add_node("retrieval_retry", retrieval_retry_node)


    # 生成答案
    graph.add_node("generate", generate_node)

    graph.set_entry_point("parse")

    # graph.add_conditional_edges(
    #     "parse",
    #     route_after_parse,
    #     {
    #         "recommend": "filter",
    #         "qa_with_name": "filter",
    #         "qa_general": "retrieve_qa",
    #     }
    # )
    graph.add_conditional_edges(
        "strategy_router",
        route_by_strategy,
        {
            "exact": "filter",
            "filtered": "filter",
            "semantic": "retrieve_qa",
            "direct": "generate"
        }
    )
    graph.add_conditional_edges(
        "evaluate_retrieval",
        route_after_retrieval,
        {
            "good": "rerank",
            "retry": "retrieval_retry",
        }
    )

    graph.add_edge("parse", "strategy_router")
    graph.add_edge("filter", "retrieve")
    graph.add_edge("retrieve", "tag_filter")
    # graph.add_edge("tag_filter", "rerank")
    graph.add_edge("tag_filter", "evaluate_retrieval")
    graph.add_edge("rerank", "generate")
    graph.add_edge("retrieve_qa", "generate")

    # 形成循环
    graph.add_edge("retrieval_retry", "retrieve")

    graph.set_finish_point("generate")

    return graph.compile()