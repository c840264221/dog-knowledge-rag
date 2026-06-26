from typing import Any

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


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


def filter_node(
        state: DogState,
) -> dict[str, Any]:
    """
    整理 exact_search_agent 的检索过滤条件。

    功能：
        根据当前 DogState 中的 dog_name 和 filters 字段，
        构建新版 RAG metadata filter 使用的基础过滤字段。

        v1.5 改造重点：
        1. 将旧版字段 name 迁移为新版字段 dog_name。
        2. 保留对旧版 filters["name"] 的读取兼容。
        3. 输出时优先使用新版字段 dog_name。
        4. 不在这里构建复杂 Chroma $and / $eq 结构，
           只负责整理基础 filters。
        5. 复杂 Chroma filter 结构仍然交给 retrieve_node / Retriever 处理。

    技术名词：
        Filter：
            过滤条件，用于限制 RAG 检索范围。

        Metadata：
            元数据，描述文档或 chunk 的结构化字段。
            例如 dog_name、size、energy_level、barking_level。

        Legacy Field：
            旧字段。这里主要指旧版 name 字段。

        Normalization：
            归一化，把旧字段、空值、不规范字段整理成统一格式。

    参数：
        state:
            当前 LangGraph 状态。
            中文释义：包含 question、filters、dog_name、tags、features 等字段。

    返回值：
        dict[str, Any]:
            返回需要合并进 DogState 的字段。
            当前主要返回：
            - filters：新版 RAG metadata 字段格式的过滤条件。
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "filter_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="filter_node"
    )

    logger.info(
        "进入 filter_node 节点，"
        f"question={state.get('question')}, "
        f"intent={state.get('intent')}, "
        f"filters={state.get('filters')}, "
        f"tags={state.get('tags')}, "
        f"dog_name={state.get('dog_name')}"
    )

    raw_filters = state.get(
        "filters",
        {}
    ) or {}

    dog_name = state.get(
        "dog_name"
    )

    normalized_filters = normalize_metadata_filters(
        raw_filters=raw_filters
    )

    if dog_name:
        normalized_filters[
            "dog_name"
        ] = dog_name

    logger.debug(
        "filter_node 节点执行完成，"
        f"normalized_filters={normalized_filters}"
    )

    return {
        "filters": normalized_filters
    }


def normalize_metadata_filters(
        raw_filters: dict[str, Any],
) -> dict[str, Any]:
    """
    归一化 metadata filters。

    功能：
        将旧版或不规范的 filters 整理成新版 RAG metadata 字段。

        当前支持两类输入：
        1. 旧版 flat filters：
           {"name": "Golden Retriever"}

        2. Chroma where filters：
           {
               "$and": [
                   {"size": {"$eq": "small"}},
                   {"barking_level": {"$lte": 3}}
               ]
           }

    技术名词：
        Metadata Filter：
            元数据过滤条件，用于在向量搜索前限定候选范围。

        Chroma Where Filter：
            Chroma 向量数据库使用的过滤条件格式。

        Field Mapping：
            字段映射，把旧字段名转换成新字段名。
            例如 name -> dog_name。

    参数：
        raw_filters:
            原始 filters。
            可能来自 semantic_router_node、旧 filter_node、用户规则解析等。

    返回值：
        dict[str, Any]:
            归一化后的 filters。
            字段名会尽量统一成新版 RAG metadata 字段。
    """

    if not isinstance(
            raw_filters,
            dict,
    ):
        return {}

    if "$and" in raw_filters:
        return normalize_chroma_where_filter(
            raw_filters=raw_filters,
        )

    normalized: dict[str, Any] = {}

    for field_name, value in raw_filters.items():

        normalized_field_name = normalize_filter_field_name(
            field_name=field_name,
        )

        if not normalized_field_name:
            continue

        if is_empty_filter_value(
                value=value,
        ):
            continue

        normalized[
            normalized_field_name
        ] = value

    return normalized


def normalize_chroma_where_filter(
        raw_filters: dict[str, Any],
) -> dict[str, Any]:
    """
    归一化 Chroma where filter。

    功能：
        保留 Chroma 的 $and / $eq / $lte / $gte 结构，
        同时对里面的字段名做旧字段兼容处理。

        例如：
            {"name": {"$eq": "Golden Retriever"}}

        会转换成：
            {"dog_name": {"$eq": "Golden Retriever"}}

    参数：
        raw_filters:
            Chroma where filter。

    返回值：
        dict[str, Any]:
            归一化后的 Chroma where filter。
    """

    raw_conditions = raw_filters.get(
        "$and",
        [],
    )

    if not isinstance(
            raw_conditions,
            list,
    ):
        return {}

    normalized_conditions: list[dict[str, Any]] = []

    for condition in raw_conditions:

        if not isinstance(
                condition,
                dict,
        ):
            continue

        normalized_condition = normalize_chroma_condition(
            condition=condition,
        )

        if normalized_condition:
            normalized_conditions.append(
                normalized_condition,
            )

    if not normalized_conditions:
        return {}

    if len(
            normalized_conditions,
    ) == 1:
        return normalized_conditions[0]

    return {
        "$and": normalized_conditions,
    }


def normalize_chroma_condition(
        condition: dict[str, Any],
) -> dict[str, Any]:
    """
    归一化单个 Chroma condition。

    功能：
        处理单个 Chroma metadata 条件。
        主要做字段名转换和空值过滤。

    参数：
        condition:
            单个 Chroma 条件。
            例如 {"name": {"$eq": "Golden Retriever"}}

    返回值：
        dict[str, Any]:
            归一化后的单条件。
    """

    normalized: dict[str, Any] = {}

    for field_name, value in condition.items():

        normalized_field_name = normalize_filter_field_name(
            field_name=field_name,
        )

        if not normalized_field_name:
            continue

        if is_empty_filter_value(
                value=value,
        ):
            continue

        normalized[
            normalized_field_name
        ] = value

    return normalized


def normalize_filter_field_name(
        field_name: Any,
) -> str | None:
    """
    归一化 filter 字段名。

    功能：
        将旧字段名转换成新版 RAG metadata 字段名。

        当前字段迁移：
        - name -> dog_name

        如果字段已经是新版支持字段，则原样返回。
        如果字段不支持，则返回 None。

    参数：
        field_name:
            原始字段名。

    返回值：
        str | None:
            新版字段名。
            如果字段不支持，则返回 None。
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
        logger.debug(
            f"filter_node 忽略暂不支持的 filter 字段: {field}"
        )

        return None

    return mapped_field


def is_empty_filter_value(
        value: Any,
) -> bool:
    """
    判断 filter 值是否为空。

    功能：
        避免把 None、空字符串、空列表、空字典等无效值写入 filters。

    参数：
        value:
            原始 filter 值。

    返回值：
        bool:
            True 表示该值为空，应该丢弃。
            False 表示该值有效，可以保留。
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