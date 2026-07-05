from collections.abc import Mapping
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogQueryLayerOutput,
)

LOGICAL_FILTER_OPERATORS = {
    "$and",
    "$or",
    "$nor",
}

DOG_NAME_FILTER_FIELDS = {
    "dog_name",
    "breed_name",
}

TARGET_FIELD_VALUE_KEYS = {
    "field",
    "section",
}


def build_dog_knowledge_query_layer_output_node():
    """
    构建 DogKnowledgeAgent 查询理解层输出节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点从当前 state 中读取 rag_query、answer_strategy、filters、intent 等字段，
        并生成 V1.7.4 标准查询理解层产物 dog_query_result。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回包含 dog_query_result 的 state update。
    """

    def dog_knowledge_query_layer_output_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 查询理解层输出节点。

        功能：
            将当前 DogState 中的查询相关旧字段转换成 dog_query_result。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update，包含 dog_query_result。
        """

        query_result = build_dog_query_layer_output_from_state(state)

        return {
            "dog_query_result": query_result.model_dump(mode="python"),
        }

    return dog_knowledge_query_layer_output_node


def build_dog_query_layer_output_from_state(
    state: Mapping[str, Any] | Any,
) -> DogQueryLayerOutput:
    """
    从当前 state 构建查询理解层标准输出。

    功能：
        读取 rag_query、answer_strategy、filters、intent、strategy 等字段，
        归一化生成 DogQueryLayerOutput。

    参数：
        state:
            LangGraph state、普通 dict 或带属性对象。

    返回值：
        DogQueryLayerOutput:
            查询理解层标准中间产物。
    """

    state_data = _to_dict(state)
    rag_query = _to_dict(state_data.get("rag_query"))
    answer_strategy = _to_dict(state_data.get("answer_strategy"))
    filters = _to_dict(
        rag_query.get("filters")
        or state_data.get("filters")
    )
    question = _first_non_empty_str(
        rag_query.get("question"),
        state_data.get("question"),
        state_data.get("user_question"),
    )

    return DogQueryLayerOutput(
        question=question,
        query_type=_resolve_query_type(
            state_data=state_data,
            rag_query=rag_query,
            answer_strategy=answer_strategy,
        ),
        task_intent=_first_non_empty_str(
            answer_strategy.get("task_type"),
            rag_query.get("intent"),
            state_data.get("intent"),
        ) or None,
        dog_names=_extract_dog_names(filters),
        target_fields=_extract_target_fields(filters),
        filters=filters,
        confidence=_resolve_query_confidence(answer_strategy),
        reason=_first_non_empty_str(
            answer_strategy.get("reason"),
            "从当前 DogState 查询相关字段生成查询理解层输出。",
        ),
        metadata={
            "source": "query_layer_output_node",
        },
    )


def _resolve_query_type(
    state_data: dict[str, Any],
    rag_query: dict[str, Any],
    answer_strategy: dict[str, Any],
) -> str:
    """
    从 state 中解析标准 query_type。

    功能：
        按优先级读取 answer_strategy、rag_query、intent、strategy，
        并映射成 DogKnowledgeAgent 标准问题类型。

    参数：
        state_data:
            当前状态字典。

        rag_query:
            RAG 查询字典。

        answer_strategy:
            回答策略字典。

    返回值：
        str:
            标准 query_type。
    """

    if state_data.get("error") or state_data.get("fallback_reason"):
        return "fallback"

    candidates = [
        answer_strategy.get("task_type"),
        rag_query.get("intent"),
        state_data.get("intent"),
        state_data.get("strategy"),
    ]

    for candidate in candidates:
        normalized = _normalize_query_type(candidate)

        if normalized:
            return normalized

    return "general_qa"


def _normalize_query_type(
    value: Any,
) -> str | None:
    """
    标准化问题类型。

    参数：
        value:
            原始问题类型。

    返回值：
        str | None:
            标准 query_type；无法识别时返回 None。
    """

    normalized = str(value or "").strip().lower()

    aliases = {
        "exact": "exact_lookup",
        "exact_info": "exact_lookup",
        "exact_search": "exact_lookup",
        "ask_info": "exact_lookup",
        "dog_info": "exact_lookup",
        "recommend": "recommendation",
        "recommendation": "recommendation",
        "comparison": "comparison",
        "compare": "comparison",
        "general": "general_qa",
        "general_dog_qa": "general_qa",
        "general_qa": "general_qa",
        "care_advice": "general_qa",
        "fallback": "fallback",
    }

    return aliases.get(normalized)


def _resolve_query_confidence(
    answer_strategy: dict[str, Any],
) -> float:
    """
    解析查询理解层置信度。

    参数：
        answer_strategy:
            回答策略字典。

    返回值：
        float:
            查询理解层置信度。
    """

    if answer_strategy:
        return 0.7

    return 0.0


def _extract_dog_names(
    filters: dict[str, Any],
) -> list[str]:
    """
    从 filters 中提取犬种名。

    参数：
        filters:
            检索过滤条件。

    返回值：
        list[str]:
            犬种名列表。
    """

    dog_names = []

    for field_name, raw_value in _iter_filter_conditions(filters):
        if field_name not in DOG_NAME_FILTER_FIELDS:
            continue

        dog_names.extend(
            _extract_filter_values(raw_value)
        )

    return _unique_strings(dog_names)


def _extract_target_fields(
    filters: dict[str, Any],
) -> list[str]:
    """
    从 filters 中提取目标字段。

    参数：
        filters:
            检索过滤条件。

    返回值：
        list[str]:
            目标字段列表。
    """

    target_fields = []

    for field_name, raw_value in _iter_filter_conditions(filters):
        if field_name in DOG_NAME_FILTER_FIELDS:
            continue

        if field_name in TARGET_FIELD_VALUE_KEYS:
            target_fields.extend(
                _extract_filter_values(raw_value)
            )
            continue

        target_fields.append(field_name)

    return _unique_strings(target_fields)


def _iter_filter_conditions(
    value: Any,
) -> list[tuple[str, Any]]:
    """
    遍历 metadata filter 中的业务条件。

    功能：
        支持扁平 filters，也支持 $and / $or / $nor 嵌套 filters。
        返回的每一项都是业务字段名和该字段对应的过滤条件。

    参数：
        value:
            原始 filters 或 filters 内部的任意子结构。

    返回值：
        list[tuple[str, Any]]:
            业务字段条件列表。
            每个元素格式为 (字段名, 字段过滤值)。
    """

    conditions = []

    if isinstance(value, Mapping):
        for field_name, raw_value in value.items():
            if field_name in LOGICAL_FILTER_OPERATORS:
                for nested_value in _as_list(raw_value):
                    conditions.extend(
                        _iter_filter_conditions(nested_value)
                    )
                continue

            if str(field_name).startswith("$"):
                continue

            conditions.append(
                (
                    str(field_name),
                    raw_value,
                )
            )

        return conditions

    for nested_value in _as_list(value):
        conditions.extend(
            _iter_filter_conditions(nested_value)
        )

    return conditions


def _extract_filter_values(
    raw_value: Any,
) -> list[str]:
    """
    从单个字段过滤条件中提取字符串值。

    功能：
        支持 {"$eq": "xxx"}、{"$in": ["a", "b"]} 和直接字符串值。

    参数：
        raw_value:
            单个字段的过滤条件。

    返回值：
        list[str]:
            提取出的字符串值列表。
    """

    if isinstance(raw_value, Mapping):
        if "$eq" in raw_value:
            return _as_str_list(raw_value.get("$eq"))

        if "$in" in raw_value:
            return _as_str_list(raw_value.get("$in"))

        return []

    return _as_str_list(raw_value)


def _as_list(
    value: Any,
) -> list[Any]:
    """
    将任意值转换成列表。

    参数：
        value:
            原始值。

    返回值：
        list[Any]:
            列表形式的值。
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [
        value,
    ]


def _as_str_list(
    value: Any,
) -> list[str]:
    """
    将任意值转换成字符串列表。

    参数：
        value:
            原始值。

    返回值：
        list[str]:
            字符串列表。
    """

    result = []

    for item in _as_list(value):
        text = str(item or "").strip()

        if text:
            result.append(text)

    return result


def _unique_strings(
    values: list[str],
) -> list[str]:
    """
    对字符串列表去重并保持原顺序。

    参数：
        values:
            原始字符串列表。

    返回值：
        list[str]:
            去重后的字符串列表。
    """

    result = []
    seen = set()

    for value in values:
        text = str(value or "").strip()

        if not text or text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result


def _first_non_empty_str(
    *values: Any,
) -> str:
    """
    返回第一个非空字符串。

    参数：
        values:
            候选值列表。

    返回值：
        str:
            第一个非空字符串；如果没有找到则返回空字符串。
    """

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    将任意对象转换成 dict。

    参数：
        value:
            原始对象，可以是 dict、Pydantic Model 或普通对象。

    返回值：
        dict[str, Any]:
            转换后的字典；无法转换时返回空字典。
    """

    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(value, "dict"):
        dumped = value.dict()

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(value, "__dict__"):
        return dict(value.__dict__)

    return {}


dog_knowledge_query_layer_output_node = (
    build_dog_knowledge_query_layer_output_node()
)
