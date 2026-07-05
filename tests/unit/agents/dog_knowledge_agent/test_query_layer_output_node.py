from src.agents.dog_knowledge_agent.nodes.query_layer_output_node import (
    build_dog_knowledge_query_layer_output_node,
    build_dog_query_layer_output_from_state,
)


def test_query_layer_output_should_build_exact_lookup_result() -> None:
    """
    测试查询层节点可以生成精确查询契约。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_query_layer_output_from_state(
        {
            "question": "金毛寿命多久？",
            "rag_query": {
                "question": "金毛寿命多久？",
                "intent": "dog_info",
                "filters": {
                    "dog_name": "golden_retriever",
                    "field": "lifespan",
                },
            },
            "answer_strategy": {
                "task_type": "exact_info",
                "reason": "具体犬种字段查询。",
            },
        }
    )

    assert result.question == "金毛寿命多久？"
    assert result.query_type == "exact_lookup"
    assert result.task_intent == "exact_info"
    assert result.dog_names == [
        "golden_retriever",
    ]
    assert result.target_fields == [
        "lifespan",
    ]
    assert result.confidence == 0.7
    assert result.metadata["source"] == "query_layer_output_node"


def test_query_layer_output_should_build_recommendation_result() -> None:
    """
    测试查询层节点可以生成推荐查询契约。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_query_layer_output_node()

    update = node(
        {
            "question": "新手适合养什么狗？",
            "rag_query": {
                "question": "新手适合养什么狗？",
                "intent": "recommend",
            },
        }
    )

    assert update["dog_query_result"]["question"] == "新手适合养什么狗？"
    assert update["dog_query_result"]["query_type"] == "recommendation"
    assert update["dog_query_result"]["task_intent"] == "recommend"


def test_query_layer_output_should_build_fallback_result_when_error_exists() -> None:
    """
    测试错误状态可以生成 fallback 查询契约。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_query_layer_output_from_state(
        {
            "question": "哪种狗适合在火星生活？",
            "error": "问题超出当前知识库边界。",
        }
    )

    assert result.query_type == "fallback"
    assert result.question == "哪种狗适合在火星生活？"


def test_query_layer_output_should_extract_nested_and_filter_fields() -> None:
    """
    测试查询层节点可以解析 $and 嵌套 metadata filters。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_query_layer_output_from_state(
        {
            "question": "小型犬有哪些？",
            "rag_query": {
                "question": "小型犬有哪些？",
                "intent": "recommend",
                "filters": {
                    "$and": [
                        {
                            "size": {
                                "$eq": "small",
                            },
                        },
                    ],
                },
            },
        }
    )

    assert result.query_type == "recommendation"
    assert result.dog_names == []
    assert result.target_fields == [
        "size",
    ]


def test_query_layer_output_should_extract_nested_dog_name_filter() -> None:
    """
    测试查询层节点可以从嵌套 filters 中提取犬种名。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_query_layer_output_from_state(
        {
            "question": "金毛寿命多久？",
            "rag_query": {
                "question": "金毛寿命多久？",
                "intent": "dog_info",
                "filters": {
                    "$and": [
                        {
                            "dog_name": {
                                "$eq": "golden_retriever",
                            },
                        },
                        {
                            "section": {
                                "$eq": "lifespan",
                            },
                        },
                    ],
                },
            },
        }
    )

    assert result.dog_names == [
        "golden_retriever",
    ]
    assert result.target_fields == [
        "lifespan",
    ]


def test_query_layer_output_should_extract_or_filter_fields_without_duplicates() -> None:
    """
    测试查询层节点可以解析 $or filters 并去重目标字段。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_query_layer_output_from_state(
        {
            "question": "小型犬或中型犬推荐？",
            "rag_query": {
                "question": "小型犬或中型犬推荐？",
                "intent": "recommend",
                "filters": {
                    "$or": [
                        {
                            "size": {
                                "$eq": "small",
                            },
                        },
                        {
                            "size": {
                                "$eq": "medium",
                            },
                        },
                        {
                            "breed_group": {
                                "$in": [
                                    "Toy Group",
                                    "Sporting Group",
                                ],
                            },
                        },
                    ],
                },
            },
        }
    )

    assert result.target_fields == [
        "size",
        "breed_group",
    ]
