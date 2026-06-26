"""
DogQueryFilterParser 单元测试。

本测试只验证规则解析逻辑：
1. 不连接 Chroma
2. 不调用 LLM
3. 不依赖 RuntimeContainer
4. 不读取真实 Markdown
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.query_parsers import (
    DogQueryFilterParser,
)

from src.rag.schemas import (
    RagQuery,
)


@pytest.fixture
def parser() -> DogQueryFilterParser:
    """
    构建 DogQueryFilterParser。

    参数：
        无。

    返回值：
        DogQueryFilterParser：
            狗狗查询过滤解析器。
    """

    return DogQueryFilterParser()


def assert_condition_exists(
        where_filter: dict[str, Any],
        expected_condition: dict[str, Any],
) -> None:
    """
    断言 Chroma where filter 中存在某个条件。

    功能：
        兼容单条件 filter 和 $and 多条件 filter。

    参数：
        where_filter: dict[str, Any]
            实际生成的 metadata filter。

        expected_condition: dict[str, Any]
            期望存在的单个条件。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    conditions = extract_conditions(
        where_filter=where_filter,
    )

    assert expected_condition in conditions


def extract_conditions(
        where_filter: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    从 Chroma where filter 中提取条件列表。

    功能：
        如果是 $and 结构，返回 $and 内部列表；
        如果是单条件结构，包装成 list 返回。

    参数：
        where_filter: dict[str, Any]
            Chroma metadata filter。

    返回值：
        list[dict[str, Any]]：
            单条件列表。
    """

    if "$and" in where_filter:
        return list(
            where_filter[
                "$and"
            ]
        )

    return [
        where_filter,
    ]


def test_parse_filters_should_parse_small_apartment_quiet_dog(
        parser,
):
    """
    测试解析：适合公寓、不太爱叫的小型犬。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="推荐适合公寓、不太爱叫的小型犬",
    )

    assert result is not None

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "size": {
                "$eq": "small",
            }
        },
    )

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "barking_level": {
                "$lte": 3,
            }
        },
    )

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "good_for_apartment": {
                "$eq": True,
            }
        },
    )


def test_parse_filters_should_parse_chinese_breed_alias(
        parser,
):
    """
    测试中文犬种别名是否能映射成英文 dog_name。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="金毛适合新手吗",
    )

    assert result is not None

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "dog_name": {
                "$eq": "Golden Retriever",
            }
        },
    )

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "good_for_beginner": {
                "$eq": True,
            }
        },
    )


def test_parse_filters_should_parse_english_breed_name(
        parser,
):
    """
    测试英文犬种名是否能识别。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="Is Golden Retriever easy to train?",
    )

    assert result is not None

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "dog_name": {
                "$eq": "Golden Retriever",
            }
        },
    )

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "trainability_level": {
                "$gte": 4,
            }
        },
    )


def test_parse_filters_should_parse_easy_training(
        parser,
):
    """
    测试容易训练关键词是否能解析成 trainability_level。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="我想要容易训练、听话的狗",
    )

    assert result is not None

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "trainability_level": {
                "$gte": 4,
            }
        },
    )


def test_parse_filters_should_parse_low_shedding_and_low_drooling(
        parser,
):
    """
    测试低掉毛和低流口水关键词。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="推荐掉毛少、不流口水的狗",
    )

    assert result is not None

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "shedding_level": {
                "$lte": 2,
            }
        },
    )

    assert_condition_exists(
        where_filter=result,
        expected_condition={
            "drooling_level": {
                "$lte": 2,
            }
        },
    )


def test_parse_filters_should_return_none_when_no_rule_matched(
        parser,
):
    """
    测试没有命中规则时是否返回 None。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse_filters(
        question="介绍一下狗狗",
    )

    assert result is None


def test_parse_should_return_rag_query(
        parser,
):
    """
    测试 parse 是否返回 RagQuery。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = parser.parse(
        question="推荐适合公寓、不太爱叫的小型犬",
        user_id="user-001",
        top_k=8,
    )

    assert isinstance(
        result,
        RagQuery,
    )

    assert result.question == "推荐适合公寓、不太爱叫的小型犬"

    assert result.user_id == "user-001"

    assert result.top_k == 8

    assert result.intent == "dog_info"

    assert result.filters != {}


def test_parse_should_raise_error_when_question_empty(
        parser,
):
    """
    测试 question 为空时是否抛出异常。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    with pytest.raises(
            ValueError,
    ):
        parser.parse(
            question="   ",
        )


def test_parse_should_raise_error_when_top_k_invalid(
        parser,
):
    """
    测试 top_k 非法时是否抛出异常。

    参数：
        parser:
            pytest fixture 注入的 DogQueryFilterParser。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    with pytest.raises(
            ValueError,
    ):
        parser.parse(
            question="推荐小型犬",
            top_k=0,
        )