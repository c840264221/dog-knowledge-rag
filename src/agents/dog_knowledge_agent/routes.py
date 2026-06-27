from typing import Literal

from src.graph.nodes.generate_strategy_resolver_node import (
    resolve_answer_strategy,
)
from src.graph.states.dog_state import DogState
from src.logger import logger


DogKnowledgeModelRoute = Literal[
    "extract_model",
    "recommendation_model",
]


def route_dog_knowledge_model(
        state: DogState,
) -> DogKnowledgeModelRoute:
    """
    路由到 dog_knowledge_agent 内部模型分支。

    功能：
        根据当前 DogState 判断问题应该进入：
        1. recommendation_model：犬种推荐分支。
        2. extract_model：犬种知识提取 / 精确信息 / 对比 / 护理建议分支。

        当前直接复用 generate_node 已经使用的 resolve_answer_strategy，
        保证“入口分支判断”和“最终 Prompt 策略判断”尽量一致。

    参数：
        state:
            DogState，当前图状态。
            可能包含：
            - question
            - intent
            - rag_query
            - route_decision
            - current_agent
            - next_agent
            - filters

    返回值：
        DogKnowledgeModelRoute:
            "recommendation_model":
                进入推荐模型分支。

            "extract_model":
                进入知识提取模型分支。

    输出格式：
        字符串，例如：
        "recommendation_model"

    专业名词：
        Route：
            路由。根据状态选择下一步执行路径。

        Domain Agent：
            领域智能体。dog_knowledge_agent 就是犬种知识领域的统一 Agent。

        Strategy Reuse：
            策略复用。入口路由复用已有的回答策略解析器，避免重复写判断规则。
    """

    answer_strategy = resolve_answer_strategy(
        state=state,
    )

    logger.info(
        f"[route_dog_knowledge_model] task_type={answer_strategy.task_type}\n "
        f"answer_style={answer_strategy.answer_style}\n "
        f"reason={answer_strategy.reason}",
    )

    if answer_strategy.task_type == "recommendation":
        return "recommendation_model"

    return "extract_model"


def route_after_dog_knowledge_evaluate(
        state: DogState,
) -> str:
    """
    dog_knowledge_agent 中 evaluate 后的路由函数。

    功能：
        根据 evaluate_retrieval_node 写入的状态，决定后续走向。

        当前兼容两类字段：
        1. route_decision 字典。
        2. retrieval_ok / retrieval_failure_type / retry_count 旧字段。

        这样可以同时兼容 exact_search_agent 和 recommendation_agent 的迁移阶段。

    参数：
        state:
            DogState，当前图状态。
            重点读取：
            - route_decision
            - retrieval_ok
            - retrieval_failure_type
            - retry_count
            - has_asked_user

    返回值：
        str:
            下一个节点名称：
            - "rerank"
            - "retry"
            - "ask_user"
            - "generate"

    输出格式：
        字符串节点名。

    专业名词：
        Evaluation：
            评估。判断检索结果是否足够回答问题。

        Retry：
            重试。检索结果不好时，调整条件后重新检索。

        Ask User：
            询问用户。当条件不足或歧义较大时，向用户澄清。

        Rerank：
            重排序。对召回结果重新排序，提高上下文质量。
    """

    route_decision = state.get(
        "route_decision",
        {},
    )

    if isinstance(
            route_decision,
            dict,
    ):
        route = route_decision.get(
            "route"
        )

        reason = route_decision.get(
            "reason",
            "",
        )

        logger.info(
            f"[route_after_dog_knowledge_evaluate] route_decision.route={route} "
            f"reason={reason}",
        )

        if route in {
            "rerank",
            "good",
        }:
            return "rerank"

        if route in {
            "retry",
            "retrieve",
        }:
            return "retry"

        if route in {
            "ask_user",
            "clarify",
        }:
            return "ask_user"

        if route in {
            "generate",
            "finish",
            "give_up",
        }:
            return "generate"

    retrieval_ok = state.get(
        "retrieval_ok",
    )

    retrieval_failure_type = state.get(
        "retrieval_failure_type",
        "",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    has_asked_user = state.get(
        "has_asked_user",
        False,
    )

    logger.info(
        f"[route_after_dog_knowledge_evaluate] "
        f"retrieval_ok={retrieval_ok} "
        f"failure_type={retrieval_failure_type} "
        f"retry_count={retry_count} "
        f"has_asked_user={has_asked_user}",
    )

    if retrieval_ok is True:
        return "rerank"

    if retrieval_failure_type in {
        "ambiguous_query",
        "need_user_clarification",
    } and not has_asked_user:
        return "ask_user"

    if retry_count < 2:
        return "retry"

    return "generate"