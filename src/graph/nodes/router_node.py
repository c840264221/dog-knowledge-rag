from typing import Any

import json

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

from src.graph.states.dog_state import (
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

from src.graph.schemas.route_decision import RouteDecision


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
        tags = []

    if not tags:
        tags = []

    features = parsed.get(
        "features"
    )

    if not isinstance(
            features,
            list
    ):
        features = []

    if not features:
        features = []

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


def build_route_hints(
        parsed: dict[str, Any],
) -> dict[str, Any]:
    """
    构建路由提示信息。

    功能：
        将 semantic_router_node 从 LLM 中解析出来的辅助字段，
        统一放入 route_decision.hints。

        注意：
            hints 不是正式 RAG 检索条件。
            它只用于 Debug（调试）、Trace（链路追踪）、Evaluation（评估）、
            以及后续可能的 Query Rewrite（查询改写）。

            正式检索条件仍然应该由 DogQueryFilterParser 构建到：
                RagQuery.filters

    技术名词：
        Hints：
            提示信息。表示路由阶段得到的辅助线索。

        Query Rewrite：
            查询改写。根据用户问题和上下文重写成更适合检索的查询。

    参数：
        parsed:
            LLM 解析用户问题后的清洗结果。

    返回值：
        dict[str, Any]:
            route_decision.hints 字典。
    """

    return {
        "intent": parsed.get(
            "intent",
            Intent.GENERAL.value,
        ),
        "filters": parsed.get(
            "filters",
            {},
        ) or {},
        "tags": parsed.get(
            "tags",
            [],
        ) or [],
        "features": parsed.get(
            "features",
            [],
        ) or [],
        "dog_name": parsed.get(
            "dog_name",
        ),
        "parser": "semantic_router_node",
        "usage": (
            "route hints only; not official rag filters"
        ),
    }


def clamp_confidence(
        value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
) -> float:
    """
    限制置信度范围。

    功能：
        将 confidence 限制在 0 到 1 之间。

    参数：
        value:
            原始置信度。

        min_value:
            最小值。

        max_value:
            最大值。

    返回值：
        float:
            合法置信度。
    """

    if value < min_value:
        return min_value

    if value > max_value:
        return max_value

    return value


def resolve_route_confidence(
        parsed: dict[str, Any],
        route: str,
        fallback_used: bool = False,
) -> float:
    """
    计算路由置信度。

    功能：
        根据 LLM 解析结果和路由目标计算一个相对合理的 confidence。

        当前说明：
            这不是模型真实概率。
            因为当前 QueryParseResult 里没有 confidence 字段，
            所以这里采用规则评分。

        后续增强方向：
            可以在 QueryParseResult 中增加 confidence 字段，
            并修改 parse_query_with_llm 的 prompt，
            让 LLM 同时输出路由置信度。

    技术名词：
        Confidence Heuristic：
            置信度启发式规则。
            在没有模型概率输出时，根据解析信号估算置信度。

        Fallback：
            兜底。当 LLM 解析失败时使用默认结果。

    参数：
        parsed:
            清洗后的 LLM 解析结果。

        route:
            最终路由到的 Agent。

        fallback_used:
            是否使用了兜底解析结果。

    返回值：
        float:
            0 到 1 之间的置信度。
    """

    if fallback_used:
        return 0.35

    raw_confidence = parsed.get(
        "confidence"
    )

    if isinstance(
            raw_confidence,
            (
                    int,
                    float,
            )
    ):
        return clamp_confidence(
            float(
                raw_confidence
            )
        )

    intent = str(
        parsed.get(
            "intent",
            Intent.GENERAL.value
        )
    )

    filters = parsed.get(
        "filters",
        {}
    ) or {}

    tags = parsed.get(
        "tags",
        []
    ) or []

    features = parsed.get(
        "features",
        []
    ) or []

    dog_name = parsed.get(
        "dog_name"
    )

    if route == RECOMMENDATION_AGENT:
        confidence = 0.82

        if intent == Intent.RECOMMEND.value:
            confidence += 0.05

        if filters:
            confidence += 0.03

        if tags:
            confidence += 0.02

        if features:
            confidence += 0.02

        return clamp_confidence(
            confidence
        )

    if route == EXACT_AGENT:
        confidence = 0.82

        if intent == Intent.ASK_INFO.value:
            confidence += 0.05

        if dog_name:
            confidence += 0.05

        if filters:
            confidence += 0.02

        return clamp_confidence(
            confidence
        )

    if route == GENERAL_AGENT:
        confidence = 0.70

        if intent == Intent.GENERAL.value:
            confidence += 0.08

        return clamp_confidence(
            confidence
        )

    return 0.5


def create_route_decision_from_parsed(
        parsed: dict[str, Any],
        fallback_used: bool = False,
) -> RouteDecision:
    """
    根据 parsed 创建结构化路由决策。

    功能：
        将 semantic_router_node 的解析结果转换成 RouteDecision。

        和旧版 create_route_decision_from_intent 的区别：
        1. 不只依赖 intent。
        2. 会把 filters、tags、features、dog_name 放入 hints。
        3. confidence 不再完全写死，而是通过 resolve_route_confidence 计算。
        4. 返回的 RouteDecision 更适合 Debug、Trace、Evaluation。

    参数：
        parsed:
            清洗后的 LLM 解析结果。
            中文释义：包含 intent、filters、tags、features、dog_name 等字段。

        fallback_used:
            是否使用了默认兜底解析结果。

    返回值：
        RouteDecision:
            主图路由决策对象。
    """

    intent = str(
        parsed.get(
            "intent",
            Intent.GENERAL.value,
        )
    )

    route = route_intent_to_agent(
        intent=intent
    )

    confidence = resolve_route_confidence(
        parsed=parsed,
        route=route,
        fallback_used=fallback_used,
    )

    hints = build_route_hints(
        parsed=parsed
    )

    if route == RECOMMENDATION_AGENT:
        reason = (
            "用户问题被解析为推荐意图，"
            "需要进入 recommendation_agent 进行犬种推荐。"
        )

    elif route == EXACT_AGENT:
        reason = (
            "用户问题被解析为犬种信息查询意图，"
            "需要进入 exact_agent 进行知识库检索问答。"
        )

    else:
        reason = (
            "用户问题未命中推荐或精确查询意图，"
            "兜底进入 general_agent 进行通用问答。"
        )

    if fallback_used:
        reason = (
            "semantic_router_node 使用默认解析结果，"
            "因此低置信度兜底进入 general_agent。"
        )

    return RouteDecision(
        route=route,
        confidence=confidence,
        reason=reason,
        hints=hints,
    )

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
        "next_agent": GENERAL_AGENT,
        "route_decision": RouteDecision(
            route=GENERAL_AGENT,
            confidence=0.0,
            reason=(
                "semantic_router_node 执行失败，"
                "safe_node fallback 兜底到 general_agent。"
            ),
            hints={
                "intent": Intent.GENERAL.value,
                "filters": {},
                "tags": [],
                "features": [],
                "dog_name": None,
                "parser": "semantic_router_node",
                "usage": "fallback route hints only; not official rag filters",
                "error": str(e),
            },
        ).model_dump(),
        "current_agent": "semantic_router",
        "error": str(e),
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

    fallback_used = False

    result = create_default_query_parse_result()

    try:
        result = await parse_query_with_llm(
            question
        )

    except Exception as e:
        fallback_used = True

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

    route_decision = create_route_decision_from_parsed(
        parsed=parsed,
        fallback_used=fallback_used,
    )

    next_agent = route_decision.route

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

    json_normalize_route_decision = json.dumps(route_decision.model_dump(), indent=4, ensure_ascii=False)
    logger.debug(f"语义路由解析完后的route_decision内容：{json_normalize_route_decision}")

    return {
        "intent": parsed.get(
            "intent",
            Intent.GENERAL.value,
        ),
        "next_agent": next_agent,
        "current_agent": "semantic_router",
        "route_decision": route_decision.model_dump(),
        "messages": [
            HumanMessage(
                content=question
            )
        ],
    }