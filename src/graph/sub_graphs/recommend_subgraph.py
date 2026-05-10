from langgraph.graph import StateGraph
from src.graph.state import DogState
from src.graph.nodes.filter_node import filter_node
from src.graph.nodes.retrieve_node import retrieve_node
from src.graph.nodes.tag_filter_node import tag_filter_node
from src.graph.nodes.rerank_node import rerank_node
from src.graph.nodes.generate_node import generate_node
from src.graph.nodes.evaluate_node import evaluate_retrieval_node
from src.graph.nodes.retrieval_retry_node import retrieval_retry_node
from src.graph.nodes.ask_user_node import ask_user_node
from src.graph.routes.route_after_ask_user import route_after_ask_user
from src.graph.nodes.modify_filter_node import modify_filter_node
from langgraph.checkpoint.memory import MemorySaver


def route_after_evaluate(state):

    if state["retrieval_ok"]:
        return "good"
    retry_cnt = state.get("retry_count", 0)

    if retry_cnt >= 3:
        return "give_up"
    # 如果已经重试过至少一次，且还不是彻底放弃，就先去询问用户
    # 避免无限循环问用户，所以只问一次（可根据需求调整）
    print("retry_cnt:", retry_cnt)
    print("has_asked_user:", state.get("has_asked_user", False))
    if retry_cnt >= 1 and state.get("has_asked_user", False) is False:
        return "ask_user"

    return "retry"

def build_recommendation_subgraph():
    graph = StateGraph(DogState)

    graph.add_node("filter", filter_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tag_filter", tag_filter_node)

    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("retry", retrieval_retry_node)

    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("filter")

    graph.add_edge("filter", "retrieve")
    graph.add_edge("retrieve", "tag_filter")
    graph.add_edge("tag_filter", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "good": "rerank",
            "retry": "retry",
            "give_up": "generate"
        }
    )

    graph.add_edge("retry", "retrieve")

    graph.add_edge("rerank", "generate")

    return graph.compile()

# 构建带人机交互的recommendation_subgraph  是build_recommendation_subgraph的上位替代
def build_recommendation_subgraph_with_human():
    graph = StateGraph(DogState)

    # 原有的节点
    graph.add_node("filter", filter_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tag_filter", tag_filter_node)
    graph.add_node("evaluate", evaluate_retrieval_node)
    graph.add_node("retry", retrieval_retry_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    # 新增的人机交互节点
    graph.add_node("ask_user", ask_user_node)
    # 新增一个“修改过滤条件”的节点（示例）
    graph.add_node("modify_filter", modify_filter_node)

    # 设置入口
    graph.set_entry_point("filter")

    # 原有边
    graph.add_edge("filter", "retrieve")
    graph.add_edge("retrieve", "tag_filter")
    graph.add_edge("tag_filter", "evaluate")

    # 条件边：从 evaluate 出发，现在可以走向 "ask_user"
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "good": "rerank",
            "retry": "retry",
            "give_up": "generate",
            "ask_user": "ask_user"  # 新增分支
        }
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("rerank", "generate")

    # 从 ask_user 出发的条件边
    graph.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "retry": "retry",
            "modify_filter": "modify_filter",
            "generate": "generate"
        }
    )

    # modify_filter 节点完成后回到 retrieve（重新检索）
    graph.add_edge("modify_filter", "retrieve")

    # 关键：编译时传入 checkpoint才支持断点和恢复
    # MemorySaver 用于测试 生产可用 SqliteSaver
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)