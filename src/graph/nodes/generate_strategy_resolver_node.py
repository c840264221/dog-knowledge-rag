from __future__ import annotations

from typing import Any
from typing import Mapping

from src.graph.schemas.answer_strategy import (
    AnswerStrategy,
)


RECOMMENDATION_KEYWORDS = [
    "推荐",
    "帮我选",
    "哪种狗",
    "什么狗",
    "适合养什么",
    "新手推荐",
    "公寓推荐",
    "recommend",
    "suggest",
    "recommendation",
]


COMPARISON_KEYWORDS = [
    "比较",
    "对比",
    "区别",
    "差别",
    "哪个更",
    "哪个比较",
    "vs",
    "versus",
    "compare",
    "comparison",
    "difference",
]


CARE_ADVICE_KEYWORDS = [
    "怎么养",
    "怎么护理",
    "怎么训练",
    "如何训练",
    "如何护理",
    "护理",
    "喂养",
    "训练",
    "洗澡",
    "美容",
    "掉毛怎么办",
    "care",
    "grooming",
    "training",
    "feed",
    "feeding",
]


EXACT_INFO_KEYWORDS = [
    "寿命",
    "多高",
    "多重",
    "体重",
    "身高",
    "性格",
    "掉毛",
    "爱叫",
    "容易训练吗",
    "适合新手吗",
    "适合公寓吗",
    "lifespan",
    "height",
    "weight",
    "temperament",
    "shedding",
    "barking",
    "trainability",
]


def resolve_answer_strategy(
        state: Mapping[str, Any],
) -> AnswerStrategy:
    """
    解析 generate_node 的回答策略。

    功能：
        根据当前 DogState 判断 generate_node 应该使用哪种回答策略。

    当前判断依据：
        1. route_decision.route
        2. next_agent / current_agent
        3. intent
        4. rag_query.intent
        5. question 关键词
        6. rag_query.filters 是否包含 dog_name

    设计原则：
        1. 对比类问题优先级最高。
        2. 护理 / 训练 / 喂养类问题优先于普通事实问答。
        3. recommendation_agent 或 recommend intent 优先使用推荐策略。
        4. exact_agent 或 ask_info / dog_info intent 优先使用精确信息策略。
        5. 如果 RAG filter 中有 dog_name，通常说明是具体犬种问答。
        6. 最后才根据关键词兜底判断。

    技术名词：
        Strategy Resolver：
            策略解析器。根据 state 判断后续生成答案应该采用什么策略。

        Route Decision：
            路由决策。semantic_router_node 生成的主图路由结果。

        Intent：
            意图。表示用户问题的目标，例如推荐、查询信息等。

        Dog Name Filter：
            犬种名过滤条件。表示 RagQuery.filters 中已经指定了某个 dog_name。

    参数：
        state:
            当前 DogState。

    返回值：
        AnswerStrategy:
            回答策略对象。
    """

    existing_answer_strategy = load_answer_strategy_from_state(
        state=state,
    )

    if existing_answer_strategy is not None:
        return existing_answer_strategy

    question = normalize_text(
        state.get(
            "question",
            "",
        )
    )

    route = resolve_route(
        state=state,
    )

    intent = resolve_intent(
        state=state,
    )

    rag_intent = resolve_rag_intent(
        state=state,
    )

    has_dog_name_filter = detect_dog_name_filter_from_state(
        state=state,
    )

    if contains_any(
            text=question,
            keywords=COMPARISON_KEYWORDS,
    ):
        return AnswerStrategy(
            task_type="comparison",
            answer_style="comparison",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=False,
            reason="用户问题包含对比关键词，使用 comparison 回答策略。",
        )

    if contains_any(
            text=question,
            keywords=CARE_ADVICE_KEYWORDS,
    ):
        return AnswerStrategy(
            task_type="care_advice",
            answer_style="step_by_step_advice",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=False,
            reason="用户问题包含护理、训练或喂养关键词，使用 care_advice 回答策略。",
        )

    if (
            route == "recommendation_agent"
            or intent == "recommend"
            or rag_intent == "recommend"
    ):
        return AnswerStrategy(
            task_type="recommendation",
            answer_style="ranked_recommendation",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=True,
            reason="当前路由或 intent 表示推荐任务，使用 recommendation 回答策略。",
        )

    if (
            route in {
                "exact_agent",
                "exact_search_agent",
            }
            or intent in {
                "ask_info",
                "dog_info",
            }
            or rag_intent in {
                "ask_info",
                "dog_info",
            }
            or has_dog_name_filter
    ):
        return AnswerStrategy(
            task_type="exact_info",
            answer_style="direct_fact",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=False,
            reason="当前问题被识别为具体犬种信息查询，使用 exact_info 回答策略。",
        )

    if contains_any(
            text=question,
            keywords=RECOMMENDATION_KEYWORDS,
    ):
        return AnswerStrategy(
            task_type="recommendation",
            answer_style="ranked_recommendation",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=True,
            reason="用户问题包含推荐关键词，使用 recommendation 回答策略。",
        )

    if contains_any(
            text=question,
            keywords=EXACT_INFO_KEYWORDS,
    ):
        return AnswerStrategy(
            task_type="exact_info",
            answer_style="direct_fact",
            must_use_context=True,
            include_sources=True,
            include_cautions=True,
            include_recommendation_reason=False,
            reason="用户问题包含犬种事实查询关键词，使用 exact_info 回答策略。",
        )

    return AnswerStrategy(
        task_type="general_dog_qa",
        answer_style="general_explanation",
        must_use_context=True,
        include_sources=True,
        include_cautions=True,
        include_recommendation_reason=False,
        reason="用户问题未命中特定策略，使用 general_dog_qa 兜底回答策略。",
    )


def resolve_route(
        state: Mapping[str, Any],
) -> str:
    """
    解析当前问题的主路由。

    功能：
        优先从 route_decision.route 读取。
        如果没有，则从 next_agent / current_agent 读取。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            路由名称。
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

        if route:
            return normalize_text(
                route
            )

    next_agent = state.get(
        "next_agent"
    )

    if next_agent:
        return normalize_text(
            next_agent
        )

    current_agent = state.get(
        "current_agent"
    )

    if current_agent:
        return normalize_text(
            current_agent
        )

    return ""


def resolve_intent(
        state: Mapping[str, Any],
) -> str:
    """
    解析用户意图。

    功能：
        从 state["intent"] 中读取 intent。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            intent 字符串。
    """

    return normalize_text(
        state.get(
            "intent",
            "",
        )
    )


def resolve_rag_intent(
        state: Mapping[str, Any],
) -> str:
    """
    解析 RagQuery 中的 intent。

    功能：
        从 state["rag_query"]["intent"] 中读取新版 RAG intent。

        同时兼容：
        1. dict 格式的 rag_query。
        2. Pydantic 对象格式的 rag_query。
        3. 普通对象属性格式的 rag_query。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            rag intent 字符串。
    """

    rag_query = state.get(
        "rag_query"
    )

    if rag_query is None:
        return ""

    if isinstance(
            rag_query,
            dict,
    ):
        return normalize_text(
            rag_query.get(
                "intent",
                "",
            )
        )

    intent = getattr(
        rag_query,
        "intent",
        "",
    )

    return normalize_text(
        intent
    )

def load_answer_strategy_from_state(
        state: Mapping[str, Any],
) -> AnswerStrategy | None:
    """
    从 DogState 中加载已有 answer_strategy。

    功能：
        如果上游节点已经写入 answer_strategy，则将其恢复成 AnswerStrategy 对象。
        这样 generate_node 可以优先复用上游策略，而不是每次重新解析。

    技术名词：
        Deserialize：
            反序列化。把 dict 数据重新转换成 Pydantic 对象。

        State Reuse：
            状态复用。表示后续节点优先使用前面节点已经做出的决策。

    参数：
        state:
            当前 DogState。
            其中 answer_strategy 可能是 dict，也可能暂时不存在。

    返回值：
        AnswerStrategy | None:
            如果 state 中有合法 answer_strategy，则返回 AnswerStrategy 对象。
            如果没有或解析失败，则返回 None。
    """

    raw_answer_strategy = state.get(
        "answer_strategy"
    )

    if not raw_answer_strategy:
        return None

    if isinstance(
            raw_answer_strategy,
            AnswerStrategy,
    ):
        return raw_answer_strategy

    if isinstance(
            raw_answer_strategy,
            dict,
    ):
        try:
            return AnswerStrategy.model_validate(
                raw_answer_strategy
            )

        except Exception:
            return None

    return None

def detect_dog_name_filter_from_state(
        state: Mapping[str, Any],
) -> bool:
    """
    判断 state 中的 RagQuery.filters 是否包含 dog_name。

    功能：
        如果 filters 中包含 dog_name，通常说明当前问题是具体犬种查询，
        可以优先使用 exact_info 回答策略。

    支持结构：
        1. {"dog_name": {"$eq": "Shih Tzu"}}
        2. {"$and": [{"dog_name": {"$eq": "Shih Tzu"}}]}

    参数：
        state:
            当前 DogState。

    返回值：
        bool:
            True 表示包含 dog_name filter。
            False 表示不包含。
    """

    rag_query = state.get(
        "rag_query"
    )

    filters = None

    if isinstance(
            rag_query,
            dict,
    ):
        filters = rag_query.get(
            "filters"
        )
    elif rag_query is not None:
        filters = getattr(
            rag_query,
            "filters",
            None,
        )

    if filters is None:
        filters = state.get(
            "filters"
        )

    return contains_filter_field(
        filters=filters,
        field_name="dog_name",
    )


def contains_filter_field(
        filters: Any,
        field_name: str,
) -> bool:
    """
    判断 filters 中是否包含某个字段。

    功能：
        支持普通 flat filter 和 Chroma $and filter。

    参数：
        filters:
            Chroma metadata filter。

        field_name:
            目标字段名。

    返回值：
        bool:
            True 表示包含该字段。
            False 表示不包含。
    """

    if not isinstance(
            filters,
            dict,
    ):
        return False

    if field_name in filters:
        return True

    and_conditions = filters.get(
        "$and"
    )

    if isinstance(
            and_conditions,
            list,
    ):
        for condition in and_conditions:
            if contains_filter_field(
                    filters=condition,
                    field_name=field_name,
            ):
                return True

    return False


def contains_any(
        text: str,
        keywords: list[str],
) -> bool:
    """
    判断文本是否包含任意关键词。

    功能：
        遍历 keywords，只要命中任意一个就返回 True。

    参数：
        text:
            待检测文本。

        keywords:
            关键词列表。

    返回值：
        bool:
            True 表示命中。
            False 表示未命中。
    """

    for keyword in keywords:

        normalized_keyword = normalize_text(
            keyword
        )

        if normalized_keyword and normalized_keyword in text:
            return True

    return False


def normalize_text(
        value: Any,
) -> str:
    """
    规范化文本。

    功能：
        将任意值转换成小写字符串，并去除首尾空格。

    参数：
        value:
            原始值。

    返回值：
        str:
            规范化后的文本。
    """

    return str(
        value
        or ""
    ).strip().lower()