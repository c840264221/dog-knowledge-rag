from collections.abc import Mapping
from typing import Any

from src.rag.schemas import RagQuery


MetadataFilter = dict[str, Any]


SUPPORTED_METADATA_FIELDS = {
    "dog_name",
    "size",
    "energy_level",
    "barking_level",
    "trainability_level",
    "shedding_level",
    "good_for_apartment",
    "good_for_beginner",
    "good_with_young_children_level",
    "good_with_other_dogs_level",
    "drooling_level",
    "coat_grooming_frequency_level",
}


LEGACY_FIELD_MAPPING = {
    "name": "dog_name",
}


CHROMA_OPERATOR_KEYS = {
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
}


def build_rag_query_from_state(
        state: Mapping[str, Any],
        parser: Any | None = None,
) -> RagQuery:
    """
    从 Graph State 构建 RagQuery。

    功能：
        将 LangGraph state 中的 question、user_id、top_k、intent、filters、
        dog_name 等字段整理成新版 RAG 使用的 RagQuery。

        v1.5 当前职责：
        1. 从 state 中读取 question。
        2. 从 state 中读取 user_id。
        3. 从 state 中读取 top_k。
        4. 从 state 中读取 intent。
        5. 使用 DogQueryFilterParser 从自然语言问题中解析 parser_filter。
        6. 从旧版 state["filters"] 和 state["dog_name"] 构建 state_filter。
        7. 合并 parser_filter 和 state_filter。
        8. 返回标准 RagQuery。

    设计说明：
        RagQuery.filters 是新版 RAG 的主过滤字段。
        state["filters"] 是旧版兼容字段，后续会逐步弃用。

    技术名词：
        State：
            状态。LangGraph 中节点之间传递的数据字典。

        RagQuery：
            RAG 查询对象，包含 question、user_id、top_k、filters、intent。

        Query Parser：
            查询解析器，负责把自然语言问题解析成结构化 filters。

        Metadata Filter：
            元数据过滤条件，用于限制向量检索范围。

        Chroma Where Filter：
            Chroma 向量数据库使用的过滤语法。
            例如：
            {"size": {"$eq": "small"}}

    参数：
        state:
            当前 Graph State。
            中文释义：包含 question、user_id、top_k、filters、dog_name 等字段。

        parser:
            DogQueryFilterParser 实例。
            中文释义：狗狗查询过滤解析器，用于从用户问题中解析 RagQuery。
            如果为 None，则只使用 state 中已有 filters。

    返回值：
        RagQuery:
            最终传给 MetadataFilterRetriever 的查询对象。
    """

    question = resolve_question_from_state(
        state=state
    )

    user_id = resolve_user_id_from_state(
        state=state
    )

    top_k = resolve_top_k(
        value=state.get(
            "top_k",
            5
        )
    )

    intent = resolve_intent_from_state(
        state=state
    )

    parser_filter = build_parser_metadata_filter(
        parser=parser,
        question=question,
        user_id=user_id,
        top_k=top_k,
        intent=intent,
    )

    state_filter = build_state_metadata_filter(
        state=state
    )

    merged_filter = merge_metadata_filters(
        parser_filter,
        state_filter,
    )

    return RagQuery(
        question=question,
        user_id=user_id,
        top_k=top_k,
        filters=merged_filter or {},
        intent=intent,
    )


def resolve_question_from_state(
        state: Mapping[str, Any],
) -> str:
    """
    从 state 中读取 question。

    功能：
        读取用户原始问题，并做基础清洗。
        如果 question 为空，则抛出异常。

    参数：
        state:
            当前 Graph State。

    返回值：
        str:
            清洗后的用户问题。
    """

    question = str(
        state.get(
            "question",
            ""
        )
        or ""
    ).strip()

    if not question:
        raise ValueError(
            "构建 RagQuery 失败：state 中缺少 question"
        )

    return question


def resolve_user_id_from_state(
        state: Mapping[str, Any],
) -> str:
    """
    从 state 中读取 user_id。

    功能：
        用于多用户隔离、个性化推荐和后续记忆召回。
        如果 state 中没有 user_id，则使用 default。

    参数：
        state:
            当前 Graph State。

    返回值：
        str:
            用户 ID。
    """

    return str(
        state.get(
            "user_id",
            "default"
        )
        or "default"
    )


def resolve_intent_from_state(
        state: Mapping[str, Any],
) -> str:
    """
    从 state 中读取 intent。

    功能：
        读取用户意图。
        如果上游没有写入 intent，则默认使用 dog_recommendation。

    说明：
        recommendation_agent 的核心场景是推荐犬种，
        所以默认值使用 dog_recommendation。
        如果 exact_search_agent 或其他 agent 传入自己的 intent，
        这里会优先保留上游传入值。

    参数：
        state:
            当前 Graph State。

    返回值：
        str:
            用户意图。
    """

    return str(
        state.get(
            "intent",
            "dog_recommendation"
        )
        or "dog_recommendation"
    )


def resolve_top_k(
        value: Any,
        default: int = 5,
        min_value: int = 1,
        max_value: int = 50,
) -> int:
    """
    解析并校正 top_k。

    功能：
        将 state 中可能是字符串、None、非法数字的 top_k，
        转换成合法整数。

    技术名词：
        top_k：
            检索数量。表示最多召回多少个 chunk。

        Clamp：
            夹取范围。把数值限制在最小值和最大值之间。

    参数：
        value:
            原始 top_k 值。

        default:
            默认 top_k。

        min_value:
            最小允许值。

        max_value:
            最大允许值。

    返回值：
        int:
            合法 top_k。
    """

    try:
        top_k = int(
            value
        )
    except (
            TypeError,
            ValueError,
    ):
        top_k = default

    if top_k < min_value:
        return min_value

    if top_k > max_value:
        return max_value

    return top_k


def build_parser_metadata_filter(
        parser: Any | None,
        question: str,
        user_id: str,
        top_k: int,
        intent: str,
) -> MetadataFilter:
    """
    使用 parser 从用户问题中构建 metadata filter。

    功能：
        调用 DogQueryFilterParser.parse，
        从返回的 RagQuery 中读取 filters。

        如果 parser 为 None，则返回空 dict。
        这样方便单元测试，也避免强依赖 parser。

    技术名词：
        Parser Filter：
            解析器从自然语言问题中生成的过滤条件。

    参数：
        parser:
            DogQueryFilterParser 实例。

        question:
            用户问题。

        user_id:
            用户 ID。

        top_k:
            检索返回数量。

        intent:
            用户意图。

    返回值：
        MetadataFilter:
            Parser 解析出的 Chroma metadata filter。
            如果没有解析出条件，则返回空 dict。
    """

    if parser is None:
        return {}

    parsed_rag_query = parser.parse(
        question=question,
        user_id=user_id,
        top_k=top_k,
        intent=intent,
    )

    parsed_filters = get_filters_from_rag_query_like(
        rag_query_like=parsed_rag_query
    )

    return normalize_metadata_filter(
        metadata_filter=parsed_filters
    ) or {}


def build_state_metadata_filter(
        state: Mapping[str, Any],
) -> MetadataFilter:
    """
    从 state 构建 metadata filter。

    功能：
        兼容旧版 semantic_router_node / filter_node / modify_filter_node
        产生的 filters。

        当前处理：
        1. state["filters"] 是 flat dict：
           {"size": "small"}

        2. state["filters"] 是 Chroma where filter：
           {"size": {"$eq": "small"}}

        3. state["filters"] 是 Chroma $and filter：
           {
               "$and": [
                   {"size": {"$eq": "small"}},
                   {"barking_level": {"$lte": 3}}
               ]
           }

        4. state["dog_name"]：
           自动补齐为 {"dog_name": {"$eq": dog_name}}

    技术名词：
        Legacy Filter：
            旧版过滤字段。这里主要指 state["filters"] 和 state["dog_name"]。

        Chroma Filter：
            Chroma 向量数据库使用的 metadata filter 格式。

    参数：
        state:
            当前 Graph State。

    返回值：
        MetadataFilter:
            Chroma metadata filter。
            没有条件时返回空 dict。
    """

    raw_filters = state.get(
        "filters",
        {}
    ) or {}

    state_filter = normalize_metadata_filter(
        metadata_filter=raw_filters
    ) or {}

    dog_name = state.get(
        "dog_name"
    )

    dog_name_filter = {}

    if dog_name:
        dog_name_filter = {
            "dog_name": {
                "$eq": dog_name
            }
        }

    return merge_metadata_filters(
        state_filter,
        dog_name_filter,
    )


def get_filters_from_rag_query_like(
        rag_query_like: Any,
) -> MetadataFilter:
    """
    从 RagQuery 或 dict 中读取 filters。

    功能：
        兼容两种结构：
        1. RagQuery Pydantic 对象
        2. dict 字典对象

        这是为了兼容 LangGraph checkpoint。
        因为 checkpoint 保存后，Pydantic 对象可能会变成 dict。

    参数：
        rag_query_like:
            RagQuery 对象或 dict。

    返回值：
        MetadataFilter:
            filters 字段。
            如果不存在，则返回空 dict。
    """

    if rag_query_like is None:
        return {}

    if isinstance(
            rag_query_like,
            Mapping
    ):
        filters = rag_query_like.get(
            "filters",
            {}
        )

        if isinstance(
                filters,
                dict
        ):
            return filters

        return {}

    filters = getattr(
        rag_query_like,
        "filters",
        {}
    )

    if isinstance(
            filters,
            dict
    ):
        return filters

    return {}


def normalize_metadata_filter(
        metadata_filter: Any,
) -> MetadataFilter | None:
    """
    归一化 metadata filter。

    功能：
        将不同形态的 filters 统一成 Chroma 可读格式。

        支持输入：
        1. None
        2. {}
        3. {"size": "small"}
        4. {"size": {"$eq": "small"}}
        5. {"$and": [{"size": {"$eq": "small"}}, ...]}

    参数：
        metadata_filter:
            原始 metadata filter。

    返回值：
        MetadataFilter | None:
            归一化后的 metadata filter。
            如果没有有效条件，则返回 None。
    """

    if not metadata_filter:
        return None

    if not isinstance(
            metadata_filter,
            Mapping
    ):
        return None

    conditions = split_filter_conditions(
        metadata_filter=dict(
            metadata_filter
        )
    )

    return conditions_to_filter(
        conditions=conditions
    )


def split_filter_conditions(
        metadata_filter: MetadataFilter,
) -> list[MetadataFilter]:
    """
    拆分 metadata filter。

    功能：
        将复杂 filter 拆成单条件列表，方便后续合并、去重和冲突覆盖。

        支持：
        1. {"$and": [A, B]}
        2. {"dog_name": {"$eq": "Golden Retriever"}}
        3. {"dog_name": "Golden Retriever"}
        4. {"size": "small", "good_for_beginner": True}

    参数：
        metadata_filter:
            原始 metadata filter。

    返回值：
        list[MetadataFilter]:
            单条件列表。
    """

    if not metadata_filter:
        return []

    if "$and" in metadata_filter:
        return split_and_filter_conditions(
            metadata_filter=metadata_filter
        )

    conditions: list[MetadataFilter] = []

    for field_name, value in metadata_filter.items():

        field = str(
            field_name
        )

        if field.startswith(
                "$"
        ):
            continue

        normalized_field = normalize_metadata_field_name(
            field_name=field
        )

        if not normalized_field:
            continue

        condition = build_metadata_condition(
            field_name=normalized_field,
            value=value,
        )

        if condition:
            conditions.append(
                condition
            )

    return conditions


def split_and_filter_conditions(
        metadata_filter: MetadataFilter,
) -> list[MetadataFilter]:
    """
    拆分 $and filter。

    功能：
        将 Chroma 的 $and 条件展开成单条件列表。

    示例：
        输入：
            {
                "$and": [
                    {"size": {"$eq": "small"}},
                    {"barking_level": {"$lte": 3}}
                ]
            }

        输出：
            [
                {"size": {"$eq": "small"}},
                {"barking_level": {"$lte": 3}}
            ]

    参数：
        metadata_filter:
            包含 $and 的 Chroma filter。

    返回值：
        list[MetadataFilter]:
            展开后的单条件列表。
    """

    raw_conditions = metadata_filter.get(
        "$and",
        []
    )

    if not isinstance(
            raw_conditions,
            list
    ):
        return []

    conditions: list[MetadataFilter] = []

    for condition in raw_conditions:

        if not isinstance(
                condition,
                Mapping
        ):
            continue

        conditions.extend(
            split_filter_conditions(
                metadata_filter=dict(
                    condition
                )
            )
        )

    return conditions


def normalize_metadata_field_name(
        field_name: Any,
) -> str | None:
    """
    归一化 metadata 字段名。

    功能：
        将旧版字段名转换成新版 RAG metadata 字段名。

        当前主要兼容：
        - name -> dog_name

        如果字段已经是新版支持字段，则原样返回。
        如果字段不支持，则返回 None。

    参数：
        field_name:
            原始字段名。

    返回值：
        str | None:
            新版字段名。
            不支持时返回 None。
    """

    field = str(
        field_name
    ).strip()

    if not field:
        return None

    mapped_field = LEGACY_FIELD_MAPPING.get(
        field,
        field
    )

    if mapped_field not in SUPPORTED_METADATA_FIELDS:
        return None

    return mapped_field


def build_metadata_condition(
        field_name: str,
        value: Any,
) -> MetadataFilter | None:
    """
    构建单个 Chroma metadata filter 条件。

    功能：
        将普通字段和值转换成 Chroma 可用的 filter 条件。

        支持两种输入：
        1. 普通值：
           dog_name = "Golden Retriever"

           转换为：
           {"dog_name": {"$eq": "Golden Retriever"}}

        2. 已经是操作符格式：
           dog_name = {"$eq": "Golden Retriever"}

           保持为：
           {"dog_name": {"$eq": "Golden Retriever"}}

    技术名词：
        Operator：
            操作符。例如 $eq 表示等于，$lte 表示小于等于。

    参数：
        field_name:
            metadata 字段名。

        value:
            metadata 字段值。

    返回值：
        MetadataFilter | None:
            单个 filter 条件。
    """

    if is_empty_filter_value(
            value=value
    ):
        return None

    if isinstance(
            value,
            Mapping
    ):

        operator_payload = dict(
            value
        )

        if not operator_payload:
            return None

        if has_chroma_operator(
                operator_payload=operator_payload
        ):
            return {
                field_name: operator_payload
            }

        return {
            field_name: {
                "$eq": operator_payload
            }
        }

    return {
        field_name: {
            "$eq": value
        }
    }


def has_chroma_operator(
        operator_payload: Mapping[str, Any],
) -> bool:
    """
    判断 dict 是否包含 Chroma 操作符。

    功能：
        判断字段值是否已经是 Chroma filter 操作符格式。

    示例：
        {"$eq": "small"} -> True
        {"value": "small"} -> False

    参数：
        operator_payload:
            字段值对应的 dict。

    返回值：
        bool:
            True 表示包含 Chroma 操作符。
            False 表示普通 dict。
    """

    return any(
        key in CHROMA_OPERATOR_KEYS
        for key in operator_payload.keys()
    )


def merge_metadata_filters(
        *filters: MetadataFilter | None,
) -> MetadataFilter:
    """
    合并多个 metadata filter。

    功能：
        将 Parser 生成的 filter 和 state 中已有的 filter 合并。

        合并策略：
        1. 空 filter 会被忽略。
        2. 所有 filter 先拆成单条件。
        3. 同字段条件发生冲突时，后面的 filter 优先级更高。
        4. 如果最终只有一个条件，直接返回该条件。
        5. 如果最终有多个条件，使用 Chroma 的 $and 组合。

    重要说明：
        当前调用顺序是：
            merge_metadata_filters(parser_filter, state_filter)

        所以 state_filter 优先级高于 parser_filter。
        这样 modify_filter_node 或 retry 过程中的人工修改可以覆盖 parser 结果。

    技术名词：
        Merge：
            合并。把多个 filter 条件组合成一个 filter。

        Conflict Override：
            冲突覆盖。同一个字段出现多个条件时，后者覆盖前者。

        $and：
            Chroma filter 操作符，表示多个条件必须同时满足。

    参数：
        *filters:
            一个或多个 metadata filter。

    返回值：
        MetadataFilter:
            合并后的 metadata filter。
            没有条件时返回空 dict。
    """

    field_order: list[str] = []

    field_to_condition: dict[str, MetadataFilter] = {}

    for metadata_filter in filters:

        normalized_filter = normalize_metadata_filter(
            metadata_filter=metadata_filter
        )

        if not normalized_filter:
            continue

        conditions = split_filter_conditions(
            metadata_filter=normalized_filter
        )

        for condition in conditions:

            field_name = get_single_condition_field_name(
                condition=condition
            )

            if not field_name:
                continue

            if field_name not in field_order:
                field_order.append(
                    field_name
                )

            field_to_condition[
                field_name
            ] = condition

    merged_conditions = [
        field_to_condition[
            field_name
        ]
        for field_name in field_order
        if field_name in field_to_condition
    ]

    return conditions_to_filter(
        conditions=merged_conditions
    ) or {}


def get_single_condition_field_name(
        condition: MetadataFilter,
) -> str | None:
    """
    获取单条件中的字段名。

    功能：
        从单个 Chroma metadata condition 中读取字段名。

    示例：
        输入：
            {"size": {"$eq": "small"}}

        输出：
            "size"

    参数：
        condition:
            单个 metadata condition。

    返回值：
        str | None:
            字段名。
            如果无法读取，则返回 None。
    """

    if not condition:
        return None

    if not isinstance(
            condition,
            Mapping
    ):
        return None

    for field_name in condition.keys():

        field = str(
            field_name
        )

        if field.startswith(
                "$"
        ):
            continue

        return field

    return None


def conditions_to_filter(
        conditions: list[MetadataFilter],
) -> MetadataFilter | None:
    """
    将单条件列表转换成 Chroma filter。

    功能：
        根据条件数量生成最终 filter：
        1. 没有条件：返回 None。
        2. 只有一个条件：直接返回该条件。
        3. 多个条件：使用 $and 组合。

    参数：
        conditions:
            单条件列表。

    返回值：
        MetadataFilter | None:
            Chroma metadata filter。
    """

    cleaned_conditions = [
        condition
        for condition in conditions
        if condition
    ]

    if not cleaned_conditions:
        return None

    if len(
            cleaned_conditions
    ) == 1:
        return cleaned_conditions[0]

    return {
        "$and": cleaned_conditions
    }


def is_empty_filter_value(
        value: Any,
) -> bool:
    """
    判断 filter 值是否为空。

    功能：
        防止 None、空字符串、空列表、空字典等无效值进入 filter。

    参数：
        value:
            原始 filter 值。

    返回值：
        bool:
            True 表示为空，需要丢弃。
            False 表示有效，可以保留。
    """

    if value is None:
        return True

    if isinstance(
            value,
            str
    ) and not value.strip():
        return True

    if isinstance(
            value,
            (
                    list,
                    tuple,
                    set,
                    dict,
            )
    ) and len(
        value
    ) == 0:
        return True

    return False