from __future__ import annotations

from typing import Any


FILTER_FIELD_ALIASES: dict[str, str] = {
    "trainability_level": "trainability",
    "energy_level": "energy",
    "barking_level": "barking",
    "shedding_level": "shedding",
}


LEVEL_SEMANTIC_FIELDS: set[str] = {
    "trainability",
    "energy",
    "barking",
    "shedding",
}


def flatten_filter_mapping(
    filters: dict[str, Any],
) -> dict[str, Any]:
    """
    扁平化 filter 结构。

    功能：
        将普通 key-value filter 和 Chroma where filter
        统一转换成更容易比较和展示的 dict。

    支持格式：
        1. 普通 key-value：
            {"size": "small"}

        2. Chroma $eq：
            {"dog_name": {"$eq": "Poodle"}}

        3. Chroma $and：
            {
                "$and": [
                    {"size": {"$eq": "small"}},
                    {"trainability_level": {"$gte": 4}}
                ]
            }

    参数含义：
        filters:
            原始 filters。
            可以来自 expected_filters，也可以来自 parsed_filters。

    返回值含义：
        dict[str, Any]:
            扁平化后的 filters。

    专业名词：
        Flatten：
            扁平化。把嵌套结构展开成更简单的一层结构。

        Chroma where：
            Chroma 向量库使用的 metadata 过滤条件格式。
    """

    if not isinstance(filters, dict):
        return {}

    flattened: dict[str, Any] = {}

    for key, value in filters.items():
        if key == "$and":
            merge_and_filter_items(
                target=flattened,
                value=value,
            )

            continue

        if key == "$or":
            # MVP 阶段先不处理 $or。
            # $or 表示“满足任意一个条件”，不能简单合并成普通 key-value。
            continue

        flattened[key] = extract_filter_value(
            value=value,
        )

    return flattened


def merge_and_filter_items(
    target: dict[str, Any],
    value: Any,
) -> None:
    """
    合并 Chroma $and 条件。

    功能：
        将 $and 列表中的多个条件展开后合并到 target 中。

    参数含义：
        target:
            保存合并结果的目标 dict。

        value:
            $and 对应的原始值，通常是 list[dict]。

    返回值含义：
        None。
    """

    if not isinstance(value, list):
        return

    for item in value:
        if not isinstance(item, dict):
            continue

        nested_flattened = flatten_filter_mapping(
            filters=item,
        )

        target.update(
            nested_flattened,
        )


def extract_filter_value(
    value: Any,
) -> Any:
    """
    从 filter value 中提取可比较值。

    功能：
        处理 Chroma 操作符格式，例如 $eq、$gte、$lte。

    规则：
        - $eq 直接提取真实值。
        - $gte / $lte / $gt / $lt 保留操作符 dict。
          因为它们不是简单等于关系，后续需要语义归一化处理。

    参数含义：
        value:
            原始 filter value。

    返回值含义：
        Any:
            提取后的值。
    """

    if not isinstance(value, dict):
        return value

    for operator in (
        "$eq",
        "eq",
    ):
        if operator in value:
            return value[operator]

    for operator in (
        "$gte",
        "gte",
        "$lte",
        "lte",
        "$gt",
        "gt",
        "$lt",
        "lt",
        "$in",
        "in",
    ):
        if operator in value:
            return {
                operator: value[operator]
            }

    return value


def normalize_semantic_filter_mapping(
    filters: dict[str, Any],
) -> dict[str, Any]:
    """
    将 filter 归一化成业务语义格式。

    功能：
        将不同字段名、不同后端查询格式，转换成统一的业务语义格式。

    示例一：
        输入：
            {"trainability": "high"}

        输出：
            {"trainability": "high"}

    示例二：
        输入：
            {"trainability_level": {"$gte": 4}}

        输出：
            {"trainability": "high"}

    示例三：
        输入：
            {"energy_level": {"$lte": 3}}

        输出：
            {"energy": "low"}

    参数含义：
        filters:
            原始 filters。

    返回值含义：
        dict[str, Any]:
            语义归一化后的 filters。

    专业名词：
        Semantic Normalization：
            语义归一化。不是只比较字段和值是否完全相同，
            而是把不同写法转换成同一个业务含义后再比较。

        Canonical Format：
            标准格式。这里指后端无关、业务可读的 filter 格式。
    """

    flattened_filters = flatten_filter_mapping(
        filters=filters,
    )

    normalized_filters: dict[str, Any] = {}

    for raw_key, raw_value in flattened_filters.items():
        semantic_key = normalize_filter_key(
            key=raw_key,
        )

        semantic_value = normalize_filter_value(
            semantic_key=semantic_key,
            raw_value=raw_value,
        )

        normalized_filters[semantic_key] = semantic_value

    return normalized_filters


def normalize_filter_key(
    key: str,
) -> str:
    """
    归一化 filter 字段名。

    功能：
        将后端 metadata 字段名转换成业务语义字段名。

    示例：
        trainability_level -> trainability
        energy_level -> energy
        barking_level -> barking
        shedding_level -> shedding

    参数含义：
        key:
            原始字段名。

    返回值含义：
        str:
            归一化后的字段名。
    """

    return FILTER_FIELD_ALIASES.get(
        key,
        key,
    )


def normalize_filter_value(
    semantic_key: str,
    raw_value: Any,
) -> Any:
    """
    归一化 filter 值。

    功能：
        根据字段语义处理不同值格式。

    参数含义：
        semantic_key:
            归一化后的字段名，例如 trainability、energy。

        raw_value:
            原始值，例如 "high"、{"$gte": 4}、{"$lte": 3}。

    返回值含义：
        Any:
            归一化后的值。
    """

    if semantic_key in LEVEL_SEMANTIC_FIELDS:
        return normalize_level_value(
            value=raw_value,
        )

    return normalize_compare_value(
        value=raw_value,
    )


def normalize_level_value(
    value: Any,
) -> Any:
    """
    归一化等级类字段值。

    功能：
        将 1-5 分等级、Chroma 范围条件、字符串等级，
        统一转换成 low / medium / high。

    当前 MVP 规则：
        - $gte >= 4 视为 high。
        - $lte <= 3 视为 low。
        - 数字 >= 4 视为 high。
        - 数字 <= 2 视为 low。
        - 数字 3 视为 medium。
        - 字符串直接小写归一化。

    参数含义：
        value:
            原始等级值。

    返回值含义：
        Any:
            归一化后的等级语义值。
    """

    if isinstance(value, dict):
        return normalize_level_operator_value(
            value=value,
        )

    if isinstance(value, (int, float)):
        return normalize_numeric_level(
            value=float(value),
        )

    return normalize_compare_value(
        value=value,
    )


def normalize_level_operator_value(
    value: dict[str, Any],
) -> Any:
    """
    归一化带操作符的等级值。

    功能：
        处理 {"$gte": 4}、{"$lte": 3} 这类 Chroma 范围条件。

    参数含义：
        value:
            带操作符的 dict。

    返回值含义：
        Any:
            归一化后的等级语义。
    """

    greater_equal_value = get_first_existing_operator_value(
        value=value,
        operators=[
            "$gte",
            "gte",
        ],
    )

    if greater_equal_value is not None:
        try:
            numeric_value = float(greater_equal_value)
        except (
            TypeError,
            ValueError,
        ):
            return value

        if numeric_value >= 4:
            return "high"

        return {
            "$gte": numeric_value
        }

    less_equal_value = get_first_existing_operator_value(
        value=value,
        operators=[
            "$lte",
            "lte",
        ],
    )

    if less_equal_value is not None:
        try:
            numeric_value = float(less_equal_value)
        except (
            TypeError,
            ValueError,
        ):
            return value

        if numeric_value <= 3:
            return "low"

        return {
            "$lte": numeric_value
        }

    return value


def get_first_existing_operator_value(
    value: dict[str, Any],
    operators: list[str],
) -> Any:
    """
    获取第一个存在的操作符值。

    参数含义：
        value:
            操作符 dict。

        operators:
            候选操作符列表。

    返回值含义：
        Any:
            第一个命中的操作符值。
            如果没有命中，返回 None。
    """

    for operator in operators:
        if operator in value:
            return value[operator]

    return None


def normalize_numeric_level(
    value: float,
) -> str:
    """
    将数字等级归一化成 low / medium / high。

    参数含义：
        value:
            数字等级，通常是 1 到 5。

    返回值含义：
        str:
            low / medium / high。
    """

    if value >= 4:
        return "high"

    if value <= 2:
        return "low"

    return "medium"


def normalize_compare_value(
    value: Any,
) -> Any:
    """
    归一化普通比较值。

    功能：
        处理字符串大小写、首尾空格、布尔字符串、列表等。

    参数含义：
        value:
            任意待比较值。

    返回值含义：
        Any:
            归一化后的值。
    """

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        if normalized_value == "true":
            return True

        if normalized_value == "false":
            return False

        return normalized_value

    if isinstance(value, list):
        return tuple(
            sorted(
                [
                    normalize_compare_value(item)
                    for item in value
                ],
                key=str,
            )
        )

    if isinstance(value, tuple):
        return tuple(
            sorted(
                [
                    normalize_compare_value(item)
                    for item in value
                ],
                key=str,
            )
        )

    return value


def is_semantic_filter_subset_matched(
    expected_filters: dict[str, Any],
    parsed_filters: dict[str, Any],
) -> bool:
    """
    判断 parsed_filters 是否语义覆盖 expected_filters。

    功能：
        使用语义归一化后的 filters 做子集匹配。

    匹配规则：
        1. expected_filters 为空，则默认匹配。
        2. expected_filters 中的每一个字段，都必须出现在 parsed_filters 中。
        3. expected_filters 中的每一个值，都必须和 parsed_filters 归一化后的值相等。
        4. parsed_filters 可以有额外字段。

    参数含义：
        expected_filters:
            评估用例中的期望 filters。
            推荐使用业务可读格式，例如 {"trainability": "high"}。

        parsed_filters:
            Parser 实际输出的 filters。
            可以是 Chroma where 格式，例如 {"trainability_level": {"$gte": 4}}。

    返回值含义：
        bool:
            True 表示语义匹配。
            False 表示语义不匹配。
    """

    if not expected_filters:
        return True

    normalized_expected = normalize_semantic_filter_mapping(
        filters=expected_filters,
    )

    normalized_actual = normalize_semantic_filter_mapping(
        filters=parsed_filters,
    )

    for key, expected_value in normalized_expected.items():
        if key not in normalized_actual:
            return False

        actual_value = normalized_actual[key]

        if actual_value != expected_value:
            return False

    return True