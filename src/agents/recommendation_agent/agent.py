from langgraph.graph import (
    StateGraph,
    END
)

from src.agents.recommendation_agent.state import (
    RecommendationState
)

from src.agents.recommendation_agent.router import (
    route_after_evaluate
)

from src.graph.nodes.filter_node import (
    filter_node
)

from src.graph.nodes.retrieve_node import (
    retrieve_node
)

from src.graph.nodes.evaluate_node import (
    evaluate_retrieval_node
)

from src.graph.nodes.rerank_node import (
    rerank_node
)

from src.graph.nodes.generate_node import (
    generate_node
)

from src.graph.nodes.ask_user_node import (
    ask_user_node
)

from src.graph.nodes.retrieval_retry_node import (
    retrieval_retry_node
)

from src.graph.nodes.modify_filter_node import (
    modify_filter_node
)

from src.graph.nodes.tag_filter_node import (
    tag_filter_node
)
from src.graph.routes.route_after_ask_user import (
    route_after_ask_user
)


def build_recommendation_agent():
    builder = StateGraph(RecommendationState)

    builder.add_node(
        "filter",
        filter_node
    )

    builder.add_node(
        "retrieve",
        retrieve_node
    )

    builder.add_node(
        "evaluate",
        evaluate_retrieval_node
    )

    builder.add_node(
        "retry",
        retrieval_retry_node
    )

    builder.add_node(
        "rerank",
        rerank_node
    )

    builder.add_node(
        "generate",
        generate_node
    )

    builder.add_node(
        "ask_user",
        ask_user_node
    )

    builder.add_node(
        "modify_filter",
        modify_filter_node
    )

    builder.add_node(
        "tag_filter",
        tag_filter_node
    )

    builder.set_entry_point(
        "filter"
    )

    builder.add_edge(
        "filter",
        "retrieve"
    )

    builder.add_edge(
        "retrieve",
        "tag_filter"
    )

    builder.add_edge(
        "tag_filter",
        "evaluate"
    )

    builder.add_conditional_edges(

        "evaluate",

        route_after_evaluate,

        {
            "good": "rerank",
            "retry": "retry",
            "give_up": "generate",
            "ask_user": "ask_user"  # 新增分支
        }
    )


    builder.add_edge(
        "retry",
        "retrieve"
    )

    builder.add_edge(
        "rerank",
        "generate"
    )

    builder.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "retry": "retry",
            "modify_filter": "modify_filter",
            "generate": "generate"
        }
    )

    builder.add_edge(
        "modify_filter",
        "retrieve"
    )

    builder.add_edge(
        "generate",
        END
    )

    return builder.compile()
