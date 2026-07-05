from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogFallbackLayerOutput,
    DogGenerationLayerOutput,
    DogQueryLayerOutput,
    DogRecommendationLayerOutput,
    DogRetrievalLayerOutput,
)
from src.agents.dog_knowledge_agent.nodes.aggregate_layer_outputs_node import (
    aggregate_dog_knowledge_layer_outputs,
)
from src.agents.dog_knowledge_agent.nodes.finalize_answer_node import (
    build_finalize_dog_knowledge_answer_node,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswer,
    DogKnowledgeEvidence,
    DogKnowledgeRecommendationItem,
)


def test_layer_contract_flow_should_finalize_exact_lookup_answer() -> None:
    """
    测试分层契约聚合结果可以进入最终答案节点。

    功能：
        模拟 exact_lookup 路径：
        layer outputs -> dog_knowledge_pipeline_result -> DogKnowledgeAnswer。

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

    aggregate_update = aggregate_dog_knowledge_layer_outputs(state)
    state.update(aggregate_update)

    finalize_node = build_finalize_dog_knowledge_answer_node()
    final_update = finalize_node(state)

    answer = final_update["dog_knowledge_answer"]

    assert isinstance(answer, DogKnowledgeAnswer)
    assert answer.question == "金毛寿命多久？"
    assert answer.query_type == "exact_lookup"
    assert answer.status == "success"
    assert answer.answer == "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    assert answer.has_evidences() is True
    assert final_update["final_answer"] == answer.answer
    assert final_update["dog_knowledge_answer_public"]["query_type"] == "exact_lookup"


def test_layer_contract_flow_should_finalize_recommendation_answer() -> None:
    """
    测试推荐路径的分层契约聚合结果可以进入最终答案节点。

    功能：
        模拟 recommendation 路径：
        推荐层输出 recommended_breeds，最终 DogKnowledgeAnswer 保留推荐项。

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
        "dog_query_result": DogQueryLayerOutput(
            question="新手适合养什么狗？",
            query_type="recommendation",
            task_intent="beginner_recommendation",
            confidence=0.8,
        ),
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

    aggregate_update = aggregate_dog_knowledge_layer_outputs(state)
    state.update(aggregate_update)

    finalize_node = build_finalize_dog_knowledge_answer_node()
    final_update = finalize_node(state)

    answer = final_update["dog_knowledge_answer"]

    assert answer.query_type == "recommendation"
    assert answer.has_recommendations() is True
    assert answer.recommended_breeds[0].breed_name == "labrador_retriever"
    assert final_update["dog_knowledge_answer_public"]["recommended_breeds"][0][
        "breed_name"
    ] == "labrador_retriever"


def test_layer_contract_flow_should_finalize_fallback_answer() -> None:
    """
    测试 fallback 路径的分层契约聚合结果可以进入最终答案节点。

    功能：
        模拟 fallback 路径：
        兜底层输出 fallback_reason，最终 DogKnowledgeAnswer 保留兜底状态。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "哪种狗适合在火星生活？",
        "dog_query_result": DogQueryLayerOutput(
            question="哪种狗适合在火星生活？",
            query_type="fallback",
            confidence=0.3,
        ),
        "dog_fallback_result": DogFallbackLayerOutput(
            fallback_reason="问题超出当前犬种知识库边界。",
            generated_answer="我暂时无法基于当前犬种知识库可靠回答这个问题。",
            confidence=0.1,
            reason="问题超出当前犬种知识库边界。",
        ),
    }

    aggregate_update = aggregate_dog_knowledge_layer_outputs(state)
    state.update(aggregate_update)

    finalize_node = build_finalize_dog_knowledge_answer_node()
    final_update = finalize_node(state)

    answer = final_update["dog_knowledge_answer"]

    assert answer.query_type == "fallback"
    assert answer.status == "fallback"
    assert answer.is_fallback is True
    assert answer.fallback_reason == "问题超出当前犬种知识库边界。"
    assert final_update["final_answer"] == "我暂时无法基于当前犬种知识库可靠回答这个问题。"
