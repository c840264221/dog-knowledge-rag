from src.rag.evaluation import (
    flatten_filter_mapping,
    is_semantic_filter_subset_matched,
    normalize_semantic_filter_mapping,
)


def test_flatten_simple_key_value_filter() -> None:
    """
    测试简单 key-value filter 可以正常扁平化。

    参数含义：
        无。

    返回值含义：
        None。
    """

    filters = {
        "size": "small",
        "trainability": "high",
    }

    flattened = flatten_filter_mapping(
        filters=filters,
    )

    assert flattened == {
        "size": "small",
        "trainability": "high",
    }


def test_flatten_chroma_eq_filter() -> None:
    """
    测试 Chroma $eq filter 可以正常扁平化。

    参数含义：
        无。

    返回值含义：
        None。
    """

    filters = {
        "dog_name": {
            "$eq": "Golden Retriever",
        }
    }

    flattened = flatten_filter_mapping(
        filters=filters,
    )

    assert flattened == {
        "dog_name": "Golden Retriever",
    }


def test_flatten_chroma_and_filter() -> None:
    """
    测试 Chroma $and filter 可以正常扁平化。

    参数含义：
        无。

    返回值含义：
        None。
    """

    filters = {
        "$and": [
            {
                "size": {
                    "$eq": "small",
                }
            },
            {
                "trainability_level": {
                    "$gte": 4,
                }
            },
            {
                "good_for_beginner": {
                    "$eq": True,
                }
            },
        ]
    }

    flattened = flatten_filter_mapping(
        filters=filters,
    )

    assert flattened == {
        "size": "small",
        "trainability_level": {
            "$gte": 4,
        },
        "good_for_beginner": True,
    }


def test_semantic_normalize_trainability_high() -> None:
    """
    测试 trainability_level >= 4 可以归一化成 trainability=high。

    参数含义：
        无。

    返回值含义：
        None。
    """

    filters = {
        "trainability_level": {
            "$gte": 4,
        }
    }

    normalized = normalize_semantic_filter_mapping(
        filters=filters,
    )

    assert normalized == {
        "trainability": "high",
    }


def test_semantic_normalize_energy_low() -> None:
    """
    测试 energy_level <= 3 可以归一化成 energy=low。

    参数含义：
        无。

    返回值含义：
        None。
    """

    filters = {
        "energy_level": {
            "$lte": 3,
        }
    }

    normalized = normalize_semantic_filter_mapping(
        filters=filters,
    )

    assert normalized == {
        "energy": "low",
    }


def test_semantic_filter_subset_matched_with_extra_fields() -> None:
    """
    测试 parsed_filters 多出额外字段时，只要覆盖 expected_filters 就算匹配。

    参数含义：
        无。

    返回值含义：
        None。
    """

    expected_filters = {
        "size": "small",
        "trainability": "high",
    }

    parsed_filters = {
        "$and": [
            {
                "size": {
                    "$eq": "small",
                }
            },
            {
                "trainability_level": {
                    "$gte": 4,
                }
            },
            {
                "good_for_beginner": {
                    "$eq": True,
                }
            },
        ]
    }

    matched = is_semantic_filter_subset_matched(
        expected_filters=expected_filters,
        parsed_filters=parsed_filters,
    )

    assert matched is True


def test_semantic_filter_subset_not_matched_when_key_missing() -> None:
    """
    测试 parsed_filters 缺少 expected_filters 字段时不匹配。

    参数含义：
        无。

    返回值含义：
        None。
    """

    expected_filters = {
        "size": "small",
        "trainability": "high",
    }

    parsed_filters = {
        "$and": [
            {
                "size": {
                    "$eq": "small",
                }
            }
        ]
    }

    matched = is_semantic_filter_subset_matched(
        expected_filters=expected_filters,
        parsed_filters=parsed_filters,
    )

    assert matched is False


def test_semantic_filter_subset_not_matched_when_value_different() -> None:
    """
    测试字段存在但语义值不一致时不匹配。

    参数含义：
        无。

    返回值含义：
        None。
    """

    expected_filters = {
        "energy": "low",
    }

    parsed_filters = {
        "energy_level": {
            "$gte": 4,
        }
    }

    matched = is_semantic_filter_subset_matched(
        expected_filters=expected_filters,
        parsed_filters=parsed_filters,
    )

    assert matched is False