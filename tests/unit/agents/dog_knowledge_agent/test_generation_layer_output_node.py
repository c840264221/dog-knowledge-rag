from src.agents.dog_knowledge_agent.nodes.generation_layer_output_node import (
    build_dog_generation_layer_output_from_state,
    build_dog_knowledge_generation_layer_output_node,
)


def test_generation_layer_output_should_build_result_from_final_answer() -> None:
    """
    测试生成层节点可以从 final_answer 构建标准输出。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_generation_layer_output_from_state(
        {
            "final_answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
            "dog_retrieval_result": {
                "confidence": 0.86,
                "evidences": [
                    {
                        "evidence_id": "golden-retriever-lifespan-001",
                    },
                    {
                        "evidence_id": "golden-retriever-lifespan-001",
                    },
                    {
                        "evidence_id": "akc-golden-retriever-overview",
                    },
                ],
            },
        }
    )

    assert result.generated_answer == "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    assert result.confidence == 0.86
    assert result.used_evidence_ids == [
        "golden-retriever-lifespan-001",
        "akc-golden-retriever-overview",
    ]
    assert result.metadata["source"] == "generation_layer_output_node"
    assert result.metadata["has_final_answer"] is True
    assert result.metadata["confidence_source"] == (
        "dog_retrieval_result.confidence"
    )


def test_generation_layer_output_node_should_return_state_update() -> None:
    """
    测试生成层节点可以返回 LangGraph state update。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_generation_layer_output_node()

    update = node(
        {
            "answer": "新手可以优先考虑拉布拉多寻回犬。",
            "dog_retrieval_result": {
                "confidence": 0.7,
                "evidences": [],
            },
        }
    )

    assert update["dog_generation_result"]["generated_answer"] == (
        "新手可以优先考虑拉布拉多寻回犬。"
    )
    assert update["dog_generation_result"]["confidence"] == 0.7


def test_generation_layer_output_should_use_default_when_answer_missing() -> None:
    """
    测试缺少答案文本时生成层节点使用默认答案。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_generation_layer_output_from_state({})

    assert result.generated_answer == "当前没有生成可用的狗狗知识答案。"
    assert result.confidence == 0.0
    assert result.used_evidence_ids == []
    assert result.metadata["confidence_source"] == "default"
