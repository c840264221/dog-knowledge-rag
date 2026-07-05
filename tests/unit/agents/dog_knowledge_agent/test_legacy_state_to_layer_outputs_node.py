from src.agents.dog_knowledge_agent.nodes.legacy_state_to_layer_outputs_node import (
    build_layer_outputs_from_legacy_state,
    build_legacy_state_to_dog_knowledge_layer_outputs_node,
)


def test_legacy_state_to_layer_outputs_should_build_query_and_generation_outputs() -> None:
    """
    测试旧 state 可以适配成查询层和生成层输出。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
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
                "reason": "具体犬种信息查询。",
            },
            "final_answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
        }
    )

    assert update["dog_query_result"]["query_type"] == "exact_lookup"
    assert update["dog_query_result"]["dog_names"] == [
        "golden_retriever",
    ]
    assert update["dog_query_result"]["target_fields"] == [
        "lifespan",
    ]
    assert update["dog_generation_result"]["generated_answer"] == (
        "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    )


def test_legacy_state_to_layer_outputs_should_build_retrieval_output_from_rag_context() -> None:
    """
    测试旧 rag_context 可以适配成检索层输出。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "金毛寿命多久？",
            "rag_query": {
                "intent": "dog_info",
            },
            "rag_context": {
                "status": "success",
                "source_count": 1,
                "chunks": [
                    {
                        "chunk": {
                            "chunk_id": "golden-retriever-lifespan-001",
                            "title": "Golden Retriever",
                            "content": "Golden Retrievers usually live around 10 to 12 years.",
                            "metadata": {
                                "dog_name": "golden_retriever",
                            },
                        },
                        "final_score": 0.91,
                    }
                ],
            },
        }
    )

    retrieval_result = update["dog_retrieval_result"]

    assert retrieval_result["retrieved_count"] == 1
    assert retrieval_result["confidence"] == 0.91
    assert retrieval_result["evidences"][0]["evidence_id"] == (
        "golden-retriever-lifespan-001"
    )
    assert retrieval_result["evidences"][0]["metadata"]["dog_name"] == (
        "golden_retriever"
    )


def test_legacy_state_to_layer_outputs_should_build_recommendation_output_when_present() -> None:
    """
    测试旧推荐字段存在时可以适配成推荐层输出。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "新手适合养什么狗？",
            "intent": "recommend",
            "recommendations": [
                {
                    "dog_name": "labrador_retriever",
                    "display_name": "Labrador Retriever / 拉布拉多寻回犬",
                    "reason": "性格友好，训练难度相对较低。",
                    "score": 0.88,
                }
            ],
        }
    )

    recommendation_result = update["dog_recommendation_result"]

    assert update["dog_query_result"]["query_type"] == "recommendation"
    assert recommendation_result["recommended_breeds"][0]["breed_name"] == (
        "labrador_retriever"
    )
    assert recommendation_result["confidence"] == 0.88


def test_legacy_state_to_layer_outputs_should_build_fallback_output_when_error_exists() -> None:
    """
    测试旧 state 存在错误字段时可以适配成兜底层输出。

    参数：
        无。

    返回值：
        None。
    """

    node = build_legacy_state_to_dog_knowledge_layer_outputs_node()

    update = node(
        {
            "question": "哪种狗适合在火星生活？",
            "error": "问题超出当前犬种知识库边界。",
            "final_answer": "我暂时无法基于当前犬种知识库可靠回答这个问题。",
        }
    )

    fallback_result = update["dog_fallback_result"]

    assert update["dog_query_result"]["query_type"] == "fallback"
    assert fallback_result["is_fallback"] is True
    assert fallback_result["fallback_reason"] == "问题超出当前犬种知识库边界。"
    assert fallback_result["generated_answer"] == (
        "我暂时无法基于当前犬种知识库可靠回答这个问题。"
    )


def test_legacy_state_to_layer_outputs_should_not_override_existing_query_result() -> None:
    """
    测试旧适配器不会覆盖已经存在的查询层契约。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "金毛寿命多久？",
            "dog_query_result": {
                "question": "金毛寿命多久？",
                "query_type": "exact_lookup",
                "task_intent": "exact_info",
                "dog_names": [
                    "golden_retriever",
                ],
                "target_fields": [
                    "lifespan",
                ],
                "filters": {
                    "dog_name": "golden_retriever",
                    "field": "lifespan",
                },
                "confidence": 0.9,
                "reason": "由 query_layer_output 节点提前生成。",
                "metadata": {
                    "source": "query_layer_output_node",
                },
            },
            "rag_context": {
                "status": "success",
                "chunks": [],
            },
        }
    )

    assert "dog_query_result" not in update
    assert update["dog_retrieval_result"]["query_type"] == "exact_lookup"


def test_legacy_state_to_layer_outputs_should_not_override_existing_retrieval_result() -> None:
    """
    测试旧适配器不会覆盖已经存在的检索层契约。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "金毛寿命多久？",
            "dog_query_result": {
                "question": "金毛寿命多久？",
                "query_type": "exact_lookup",
                "task_intent": "exact_info",
                "dog_names": [
                    "golden_retriever",
                ],
                "target_fields": [
                    "lifespan",
                ],
                "filters": {},
                "confidence": 0.8,
                "reason": "查询层已生成。",
                "metadata": {},
            },
            "dog_retrieval_result": {
                "query_type": "exact_lookup",
                "evidences": [],
                "retrieved_count": 0,
                "confidence": 0.88,
                "reason": "由 retrieval_layer_output 节点提前生成。",
                "metadata": {
                    "source": "retrieval_layer_output_node",
                },
            },
            "rag_context": {
                "status": "success",
                "chunks": [
                    {
                        "chunk": {
                            "chunk_id": "should-not-be-used",
                            "content": "这条证据不应该覆盖已有检索层契约。",
                        },
                    }
                ],
            },
        }
    )

    assert "dog_query_result" not in update
    assert "dog_retrieval_result" not in update


def test_legacy_state_to_layer_outputs_should_not_override_existing_generation_result() -> None:
    """
    测试旧适配器不会覆盖已经存在的生成层契约。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "金毛寿命多久？",
            "final_answer": "这条旧答案不应该覆盖生成层契约。",
            "dog_generation_result": {
                "generated_answer": "生成层节点已经产出的答案。",
                "confidence": 0.8,
                "reason": "由 generation_layer_output 节点提前生成。",
                "used_evidence_ids": [
                    "evidence-001",
                ],
                "metadata": {
                    "source": "generation_layer_output_node",
                },
            },
        }
    )

    assert "dog_generation_result" not in update


def test_legacy_state_to_layer_outputs_should_not_override_existing_fallback_result() -> None:
    """
    测试旧适配器不会覆盖已经存在的兜底层契约。

    参数：
        无。

    返回值：
        None。
    """

    update = build_layer_outputs_from_legacy_state(
        {
            "question": "哪种狗适合在火星生活？",
            "error": "旧 error 不应该覆盖兜底层契约。",
            "dog_fallback_result": {
                "is_fallback": True,
                "fallback_reason": "fallback_layer_output 已经生成兜底原因。",
                "generated_answer": "fallback_layer_output 已经生成兜底答案。",
                "confidence": 0.1,
                "reason": "fallback_layer_output 已经生成兜底原因。",
                "metadata": {
                    "source": "fallback_layer_output_node",
                },
            },
        }
    )

    assert "dog_fallback_result" not in update
