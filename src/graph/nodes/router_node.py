def strategy_router_node(state):
    intent = state.get("intent")
    dog_name = state.get("dog_name")
    filters = state.get("filters", {})
    top_k = state.get("top_k", 5)
    # 记忆性问题的关键词
    memory_keywords = ["喜欢什么", "上次说的", "之前推荐", "还记得", "我喜欢的", "我曾经", "最喜欢的"]
    # 精确查询（指定狗）
    if dog_name:
        strategy = "exact"

    # 推荐类（有筛选条件）
    elif intent == "recommend" or filters:
        strategy = "filtered"
        top_k = 10

    # 泛问
    elif intent == "ask_info":
        strategy = "semantic"

    else:
        strategy = "direct"
    strategy = (
        "semantic"
        if any(keyword in state["question"] for keyword in memory_keywords)
        else strategy
    )
    print("路由分流完成，stratege:", strategy)

    return {"strategy": strategy,
            "top_k": top_k
            }

from langchain_core.messages import HumanMessage

from src.graph.routes.agent_type import AgentType

from src.parser.schema import (
    QueryParseResult,
    Intent
)

from src.parser.query_parser import (
    parse_query_with_llm
)

from src.logger import logger

from src.common.decorators.safe_node import (
    safe_node
)

from src.common.decorators.validation_input import (
    validate_question
)

from src.common.decorators.state_validation import (
    validate_state
)

from src.common.decorators.validate_llm_output import (
    validate_llm_output, validate_query_parse_result, default_parse_result
)

@safe_node(
    fallback=lambda state, e: {
        "next_agent": AgentType.GENERAL_QA.value
    }
)
@validate_question
@validate_state(["question"])
@validate_llm_output(
    validator=validate_query_parse_result,
    fallback_factory=default_parse_result
)
def semantic_router_node(state):
    print("semantic_router_node", state)
    question = state["question"]

    logger.info(
        f"进入语义路由节点: {question}"
    )

    messages = state.get(
        "messages",
        []
    )

    messages.append(
        HumanMessage(content=question)
    )

    # 默认结果
    result = QueryParseResult(
        intent=Intent.GENERAL.value,
        filters={},
        tags=["general"],
        features=["general"],
        dog_name=None
    )

    try:

        result = parse_query_with_llm(
            question
        )

    except Exception as e:

        logger.exception(
            f"LLM解析失败: {e}"
        )

    parsed = result.model_dump()
    logger.debug(f"语义节点解析完毕，parsed为：{parsed}")

    intent = parsed["intent"]

    # ========= 核心升级 =========
    # semantic routing
    # ==========================

    if intent == Intent.RECOMMEND.value:

        next_agent = (
            AgentType.RECOMMENDATION.value
        )

    elif intent == Intent.ASK_INFO.value:

        next_agent = (
            AgentType.EXACT_SEARCH.value
        )

    else:

        next_agent = (
            AgentType.GENERAL_QA.value
        )

    logger.info(
        f"路由到Agent: {next_agent}"
    )

    return {

        "intent": parsed["intent"],

        "filters": parsed["filters"],

        "tags": parsed["tags"],

        "features": parsed["features"],

        "dog_name": parsed["dog_name"],

        "messages": messages,

        "next_agent": next_agent,

        "current_agent": "semantic_router"
    }