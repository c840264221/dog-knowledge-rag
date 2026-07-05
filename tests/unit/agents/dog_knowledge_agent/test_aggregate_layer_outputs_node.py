from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogFallbackLayerOutput,
    DogGenerationLayerOutput,
    DogQueryLayerOutput,
    DogRecommendationLayerOutput,
    DogRetrievalLayerOutput,
)
from src.agents.dog_knowledge_agent.nodes.aggregate_layer_outputs_node import (
    aggregate_dog_knowledge_layer_outputs,
    build_aggregate_dog_knowledge_layer_outputs_node,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeEvidence,
    DogKnowledgeRecommendationItem,
)


def test_aggregate_layer_outputs_should_build_exact_lookup_pipeline_result() -> None:
    """
    测试聚合节点可以汇总精确查询路径。

    参数：
        无。

    返回值：
        None。
    """

    evidence = DogKnowledgeEvidence(
        evidence_id="golden-retriever-lifespan-001",
        source_kind="rag_chunk",
        title="Golden Retriever",
        content="Golden Retrievers usually live around 10 to 12 years.",
        score=0.91,
    )

    state = {
        "dog_query_result": DogQueryLayerOutput(
            question="金毛寿命多久？",
            query_type="exact_lookup",
            task_intent="dog_attribute_lookup",
            confidence=0.86,
            reason="用户询问金毛寿命，属于犬种属性精确查询。",
        ),
        "dog_retrieval_result": DogRetrievalLayerOutput(
            query_type="exact_lookup",
            evidences=[
                evidence,
            ],
            retrieved_count=1,
            confidence=0.82,
            reason="检索命中了 Golden Retriever 的寿命相关知识片段。",
        ),
        "dog_generation_result": DogGenerationLayerOutput(
            generated_answer="金毛寻回犬的寿命通常在 10 到 12 年左右。",
            confidence=0.84,
            reason="答案基于高相关度寿命证据生成。",
            used_evidence_ids=[
                "golden-retriever-lifespan-001",
            ],
        ),
    }

    update = aggregate_dog_knowledge_layer_outputs(state)

    result = update["dog_knowledge_pipeline_result"]

    assert result["question"] == "金毛寿命多久？"
    assert result["query_type"] == "exact_lookup"
    assert result["status"] == "success"
    assert result["answer"] == "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    assert result["evidences"][0]["evidence_id"] == "golden-retriever-lifespan-001"
    assert result["confidence"] == 0.84


def test_aggregate_layer_outputs_should_build_recommendation_pipeline_result() -> None:
    """
    测试聚合节点可以汇总推荐路径。

    参数：
        无。

    返回值：
        None。
    """

    recommendation = DogKnowledgeRecommendationItem(
        breed_name="labrador_retriever",
        display_name="Labrador Retriever / 拉布拉多寻回犬",
        reason="性格友好，训练难度相对较低。",
        score=0.88,
    )

    state = {
        "dog_query_result": {
            "question": "新手适合养什么狗？",
            "query_type": "recommendation",
            "confidence": 0.8,
        },
        "dog_recommendation_result": DogRecommendationLayerOutput(
            recommended_breeds=[
                recommendation,
            ],
            confidence=0.8,
            reason="根据用户的新手需求，推荐命中新手友好特征的犬种。",
        ),
        "dog_generation_result": DogGenerationLayerOutput(
            generated_answer="新手可以优先考虑拉布拉多寻回犬。",
            confidence=0.78,
            reason="根据推荐项生成自然语言答案。",
        ),
    }

    update = aggregate_dog_knowledge_layer_outputs(state)

    result = update["dog_knowledge_pipeline_result"]

    assert result["query_type"] == "recommendation"
    assert result["recommended_breeds"][0]["breed_name"] == "labrador_retriever"
    assert result["answer"] == "新手可以优先考虑拉布拉多寻回犬。"
    assert result["status"] == "success"


def test_aggregate_layer_outputs_should_build_fallback_pipeline_result() -> None:
    """
    测试聚合节点可以汇总 fallback 路径。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "哪种狗适合在火星生活？",
        "dog_query_result": {
            "question": "哪种狗适合在火星生活？",
            "query_type": "fallback",
        },
        "dog_fallback_result": DogFallbackLayerOutput(
            fallback_reason="问题超出当前犬种知识库边界。",
            generated_answer="我暂时无法基于当前犬种知识库可靠回答这个问题。",
            confidence=0.1,
            reason="问题超出当前犬种知识库边界。",
        ),
    }

    node = build_aggregate_dog_knowledge_layer_outputs_node()
    update = node(state)

    result = update["dog_knowledge_pipeline_result"]

    assert result["query_type"] == "fallback"
    assert result["status"] == "fallback"
    assert result["is_fallback"] is True
    assert result["fallback_reason"] == "问题超出当前犬种知识库边界。"
    assert result["confidence"] == 0.1


def test_aggregate_layer_outputs_should_record_available_layers_in_debug() -> None:
    """
    测试聚合节点会记录可用 layer 信息。

    参数：
        无。

    返回值：
        None。
    """

    update = aggregate_dog_knowledge_layer_outputs(
        {
            "question": "测试问题",
            "dog_query_result": {
                "question": "测试问题",
                "query_type": "general_qa",
            },
            "dog_generation_result": {
                "generated_answer": "测试答案",
            },
        }
    )

    result = update["dog_knowledge_pipeline_result"]

    assert result["debug"]["aggregator"]["name"] == (
        "aggregate_dog_knowledge_layer_outputs_node"
    )
    assert result["debug"]["aggregator"]["available_layers"] == [
        "query",
        "generation",
    ]
