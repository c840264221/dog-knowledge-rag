from typing import Any

from langchain_core.messages import HumanMessage

from src.common.decorators.safe_node import (
    safe_node
)

from src.common.decorators.validation_input import (
    validate_question
)

from src.common.decorators.state_validation import (
    validate_state
)

from src.graph.states.state import (
    DogState
)

from src.logger import logger

from src.parser.schema import (
    QueryParseResult,
    Intent
)

from src.parser.query_parser import (
    parse_query_with_llm
)

from src.runtime.context import (
    runtime_ctx
)


RECOMMENDATION_AGENT = "recommendation_agent"

EXACT_AGENT = "exact_agent"

GENERAL_AGENT = "general_agent"

FINISH = "FINISH"


def normalize_query_parse_result(
        parsed: dict[str, Any]
) -> dict[str, Any]:
    """
    归一化 LLM 解析结果。

    功能：
    - 修复 LLM 返回的空 intent
    - 修复非法 intent
    - 修复 filters、tags、features、dog_name 的默认值
    - 避免非法字段继续进入 LangGraph 状态
    - 保证 semantic_router_node 后续一定能生成合法 next_agent

    参数：
    - parsed: dict[str, Any]
      LLM 解析用户问题后的原始结果。
      中文释义：大模型把自然语言问题解析成结构化字段后的字典。

    返回值：
    - dict[str, Any]
      清洗后的解析结果。
      至少包含 intent、filters、tags、features、dog_name。
    """

    valid_intents = {
        Intent.RECOMMEND.value,
        Intent.ASK_INFO.value,
        Intent.GENERAL.value,
    }

    intent = str(
        parsed.get("intent")
        or Intent.GENERAL.value
    ).strip()

    if intent not in valid_intents:
        logger.warning(
            f"LLM 返回非法 intent，已兜底为 general: {intent!r}"
        )

        intent = Intent.GENERAL.value

    filters = parsed.get(
        "filters"
    )

    if not isinstance(
            filters,
            dict
    ):
        filters = {}

    tags = parsed.get(
        "tags"
    )

    if not isinstance(
            tags,
            list
    ):
        tags = [
            "general"
        ]

    if not tags:
        tags = [
            "general"
        ]

    features = parsed.get(
        "features"
    )

    if not isinstance(
            features,
            list
    ):
        features = [
            "general"
        ]

    if not features:
        features = [
            "general"
        ]

    dog_name = parsed.get(
        "dog_name"
    )

    if dog_name == "":
        dog_name = None

    return {
        **parsed,
        "intent": intent,
        "filters": filters,
        "tags": tags,
        "features": features,
        "dog_name": dog_name,
    }


def route_intent_to_agent(
        intent: str
) -> str:
    """
    根据 intent 选择下一个 Agent。

    功能：
    - 将语义意图 intent 转换成主图中的 Agent 路由 key
    - recommend 路由到 recommendation_agent
    - ask_info 路由到 exact_agent
    - general 或非法值兜底到 general_agent

    参数：
    - intent: str
      用户问题意图。
      中文释义：LLM 对用户问题类型的判断，例如 recommend、ask_info、general。

    返回值：
    - str
      下一个 Agent 的路由 key。
      必须匹配主图 conditional_edges 中注册的 key。
    """

    if intent == Intent.RECOMMEND.value:

        return RECOMMENDATION_AGENT

    if intent == Intent.ASK_INFO.value:

        return EXACT_AGENT

    return GENERAL_AGENT


def create_default_query_parse_result() -> QueryParseResult:
    """
    创建默认 QueryParseResult。

    功能：
    - 当 LLM 解析异常时使用
    - 返回一个合法的 general 意图解析结果
    - 避免 semantic_router_node 因 LLM 异常中断主流程

    参数：
    - 无

    返回值：
    - QueryParseResult
      默认的问题解析结果。
    """

    return QueryParseResult(
        intent=Intent.GENERAL.value,
        filters={},
        tags=[
            "general"
        ],
        features=[
            "general"
        ],
        dog_name=None
    )


def save_semantic_router_checkpoint() -> None:
    """
    保存 semantic_router_node 检查点。

    功能：
    - 从 RuntimeContainer 中获取 checkpoint manager
    - 保存当前运行检查点
    - 如果保存失败，只记录 warning，不中断主流程

    参数：
    - 无

    返回值：
    - None
      只执行保存逻辑。
    """

    try:
        from src.runtime.container.init import (
            container
        )

        container.get(
            "checkpoint"
        ).manager.save_checkpoint()

    except Exception as checkpoint_error:
        logger.warning(
            f"semantic_router_node 保存 checkpoint 失败: {checkpoint_error}"
        )


@safe_node(
    fallback=lambda state, e: {
        "intent": Intent.GENERAL.value,
        "filters": {},
        "tags": [
            "general"
        ],
        "features": [
            "general"
        ],
        "dog_name": None,
        "next_agent": GENERAL_AGENT,
        "current_agent": "semantic_router",
    }
)
@validate_question
@validate_state(
    [
        "question"
    ]
)
async def semantic_router_node(
        state: DogState
) -> dict[str, Any]:
    """
    语义路由节点。

    功能：
    - 从 state 中读取用户问题 question
    - 将用户问题追加到 messages 历史消息中
    - 调用 LLM 解析用户问题
    - 对 LLM 解析结果进行内部清洗和兜底
    - 根据 intent 选择下一个 Agent
    - 写入 Runtime Context、Timeline、Checkpoint、Logger
    - 返回 LangGraph 需要合并进 state 的字段

    参数：
    - state: DogState
      LangGraph 当前状态。
      中文释义：Graph 节点之间传递的数据结构。

    返回值：
    - dict[str, Any]
      返回语义解析结果和路由结果。
      包含 intent、filters、tags、features、dog_name、messages、next_agent、current_agent。
    """

    node_name = "semantic_router_node"

    runtime_context = runtime_ctx.get()

    runtime_context.state().set_node(
        node_name
    )

    runtime_context.timeline().add_event(
        event_type="node",
        name=node_name
    )

    question = str(
        state.get(
            "question",
            ""
        )
    )

    logger.info(
        f"进入语义路由节点: {question}"
    )

    messages = list(
        state.get(
            "messages",
            []
        )
    )

    messages.append(
        HumanMessage(
            content=question
        )
    )

    result = create_default_query_parse_result()

    try:
        result = await parse_query_with_llm(
            question
        )

    except Exception as e:
        logger.exception(
            f"LLM解析失败，已使用默认 general 结果: {e}"
        )

    parsed = result.model_dump()

    parsed = normalize_query_parse_result(
        parsed
    )

    logger.debug(
        f"语义节点解析完毕，parsed为：{parsed}"
    )

    intent = str(
        parsed.get(
            "intent",
            Intent.GENERAL.value
        )
    )

    next_agent = route_intent_to_agent(
        intent
    )

    logger.info(
        f"路由到Agent: {next_agent}"
    )

    runtime_context.timeline().add_event(
        event_type="agent",
        name=next_agent
    )

    runtime_context.state().set_agent(
        next_agent
    )

    save_semantic_router_checkpoint()

    return {
        "intent": parsed["intent"],
        "filters": parsed["filters"],
        "tags": parsed["tags"],
        "features": parsed["features"],
        "dog_name": parsed["dog_name"],
        "messages": messages,
        "next_agent": next_agent,
        "current_agent": "semantic_router",
    }