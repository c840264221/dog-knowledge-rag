from langgraph.graph import StateGraph

from src.graph.state import DogState

from src.graph.nodes.retrieve_node import retrieve_node
from src.graph.nodes.evaluate_node import evaluate_retrieval_node
from src.graph.nodes.retrieval_retry_node import retrieval_retry_node
from src.graph.nodes.generate_node import generate_node

from src.graph.nodes.tool_parse_node import tool_parse_node
from src.graph.nodes.execute_tool_node import execute_tool_node
from src.graph.nodes.answer_gen_node import answer_gen_node
from src.graph.nodes.ask_confirm_tool_node import ask_confirm_tool_node
from src.graph.routes.route_afer_confirm import route_after_confirm


def route_after_evaluate(state):

    if state["retrieval_ok"]:
        return "good"

    if state.get("retry_count", 0) >= 2:
        return "give_up"

    return "retry"


def build_general_qa_graph():

    graph = StateGraph(DogState)

    graph.add_node("retrieve", retrieve_node)

    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("retry", retrieval_retry_node)

    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

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


def build_general_qa_subgraph():
    graph = StateGraph(DogState)

    graph.add_node("tool_parse", tool_parse_node)
    graph.add_node("ask_confirm", ask_confirm_tool_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("answer_gen", answer_gen_node)

    graph.set_entry_point("tool_parse")

    # 条件路由：需要工具就去执行，否则直接生成
    graph.add_conditional_edges(
        "tool_parse",
        lambda s: "ask_confirm" if s.get("need_tool", False) else "answer_gen",
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
        lambda s: "tool_parse" if s.get("need_tool") else "answer_gen",
    )
    graph.add_edge("execute_tool", "answer_gen")

    return graph.compile()