from src.agents.dog_knowledge_agent.nodes.retrieval_layer_output_node import (
    build_dog_knowledge_retrieval_layer_output_node,
    build_dog_retrieval_layer_output_from_state,
)


def test_retrieval_layer_output_should_build_evidences_from_rag_context() -> None:
    """
    测试检索层节点可以从 rag_context 构建标准证据。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_retrieval_layer_output_from_state(
        {
            "dog_query_result": {
                "query_type": "exact_lookup",
            },
            "retrieval_ok": True,
            "retrieval_evaluated": True,
            "retrieval_failure_type": "",
            "retrieval_quality": {
                "quality_score": 0.86,
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

    assert result.query_type == "exact_lookup"
    assert result.retrieved_count == 1
    assert result.confidence == 0.86
    assert result.evidences[0].evidence_id == "golden-retriever-lifespan-001"
    assert result.evidences[0].metadata["dog_name"] == "golden_retriever"
    assert result.metadata["source"] == "retrieval_layer_output_node"
    assert result.metadata["retrieval_ok"] is True


def test_retrieval_layer_output_node_should_return_state_update() -> None:
    """
    测试检索层节点可以返回 LangGraph state update。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_retrieval_layer_output_node()

    update = node(
        {
            "dog_query_result": {
                "query_type": "recommendation",
            },
            "retrieval_ok": True,
            "rag_context": {
                "status": "success",
                "chunks": [],
            },
        }
    )

    assert update["dog_retrieval_result"]["query_type"] == "recommendation"
    assert update["dog_retrieval_result"]["retrieved_count"] == 0


def test_retrieval_layer_output_should_use_chunk_score_when_quality_score_missing() -> None:
    """
    测试缺少质量分时使用 chunk 分数作为检索层置信度。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_retrieval_layer_output_from_state(
        {
            "dog_query_result": {
                "query_type": "exact_lookup",
            },
            "rag_context": {
                "status": "success",
                "chunks": [
                    {
                        "chunk": {
                            "chunk_id": "chunk-001",
                            "content": "测试内容。",
                        },
                        "retrieval_score": 0.72,
                    }
                ],
            },
        }
    )

    assert result.confidence == 0.72
