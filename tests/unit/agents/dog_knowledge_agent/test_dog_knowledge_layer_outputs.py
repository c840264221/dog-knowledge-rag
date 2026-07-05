import pytest
from pydantic import ValidationError

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogFallbackLayerOutput,
    DogGenerationLayerOutput,
    DogKnowledgePipelineResult,
    DogQueryLayerOutput,
    DogRecommendationLayerOutput,
    DogRetrievalLayerOutput,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeEvidence,
    DogKnowledgeRecommendationItem,
)


def test_query_layer_output_should_describe_user_question() -> None:
    """
    测试查询理解层可以输出标准中间产物。

    参数：
        无。

    返回值：
        None。
    """

    result = DogQueryLayerOutput(
        question="金毛寿命多久？",
        query_type="exact_lookup",
        task_intent="dog_attribute_lookup",
        dog_names=[
            "golden_retriever",
        ],
        target_fields=[
            "lifespan",
        ],
        filters={
            "dog_name": "golden_retriever",
        },
        confidence=0.86,
        reason="用户询问金毛寿命，属于犬种属性精确查询。",
    )

    assert result.query_type == "exact_lookup"
    assert result.task_intent == "dog_attribute_lookup"
    assert result.filters["dog_name"] == "golden_retriever"


def test_retrieval_layer_output_should_reuse_answer_evidence_schema() -> None:
    """
    测试检索层复用 DogKnowledgeEvidence 作为标准证据结构。

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
        metadata={
            "dog_name": "golden_retriever",
        },
    )

    result = DogRetrievalLayerOutput(
        query_type="exact_lookup",
        evidences=[
            evidence,
        ],
        retrieved_count=1,
        confidence=0.82,
        reason="检索命中了 Golden Retriever 的寿命相关知识片段。",
    )

    assert result.evidences[0] == evidence
    assert result.retrieved_count == 1
    assert result.confidence == 0.82


def test_recommendation_layer_output_should_reuse_recommendation_item_schema() -> None:
    """
    测试推荐层复用 DogKnowledgeRecommendationItem 作为标准推荐项结构。

    参数：
        无。

    返回值：
        None。
    """

    recommendation = DogKnowledgeRecommendationItem(
        breed_name="labrador_retriever",
        display_name="Labrador Retriever / 拉布拉多寻回犬",
        reason="性格友好，训练难度相对较低，通常适合新手家庭。",
        matched_traits=[
            "新手友好",
            "容易训练",
        ],
        warnings=[
            "运动量较高，需要规律遛狗。",
        ],
        evidence_ids=[
            "labrador-retriever-001",
        ],
        score=0.88,
    )

    result = DogRecommendationLayerOutput(
        recommended_breeds=[
            recommendation,
        ],
        confidence=0.8,
        reason="根据用户的新手需求，推荐命中新手友好特征的犬种。",
    )

    assert result.recommended_breeds[0] == recommendation
    assert result.recommended_breeds[0].breed_name == "labrador_retriever"


def test_generation_layer_output_should_hold_generated_answer() -> None:
    """
    测试生成层可以输出自然语言答案和使用过的证据 ID。

    参数：
        无。

    返回值：
        None。
    """

    result = DogGenerationLayerOutput(
        generated_answer="金毛寻回犬的寿命通常在 10 到 12 年左右。",
        confidence=0.84,
        reason="答案基于高相关度寿命证据生成。",
        used_evidence_ids=[
            "golden-retriever-lifespan-001",
        ],
    )

    assert "10 到 12 年" in result.generated_answer
    assert result.used_evidence_ids == [
        "golden-retriever-lifespan-001",
    ]


def test_fallback_layer_output_should_require_fallback_reason() -> None:
    """
    测试兜底层必须输出 fallback_reason。

    参数：
        无。

    返回值：
        None。
    """

    result = DogFallbackLayerOutput(
        fallback_reason="没有找到足够可靠的犬种知识证据。",
        generated_answer="我暂时无法基于当前犬种知识库可靠回答这个问题。",
        confidence=0.1,
        reason="检索结果为空或问题超出当前犬种知识库边界。",
    )

    assert result.is_fallback is True
    assert result.fallback_reason == "没有找到足够可靠的犬种知识证据。"


def test_pipeline_result_should_match_final_response_contract_shape() -> None:
    """
    测试聚合层输出字段形状贴近最终 DogKnowledgeAnswer。

    参数：
        无。

    返回值：
        None。
    """

    evidence = DogKnowledgeEvidence(
        evidence_id="golden-retriever-lifespan-001",
        source_kind="rag_chunk",
        content="Golden Retrievers usually live around 10 to 12 years.",
    )

    result = DogKnowledgePipelineResult(
        question="金毛寿命多久？",
        query_type="exact_lookup",
        status="success",
        answer="金毛寻回犬的寿命通常在 10 到 12 年左右。",
        evidences=[
            evidence,
        ],
        confidence=0.9,
        reason="聚合了查询理解层、检索层和生成层结果。",
    )

    assert result.question == "金毛寿命多久？"
    assert result.status == "success"
    assert result.answer == "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    assert result.evidences[0].evidence_id == "golden-retriever-lifespan-001"


def test_layer_outputs_should_forbid_extra_fields() -> None:
    """
    测试分层输出契约不允许传入未定义字段。

    参数：
        无。

    返回值：
        None。
    """

    with pytest.raises(ValidationError):
        DogQueryLayerOutput(
            question="金毛寿命多久？",
            query_type="exact_lookup",
            unexpected_field="不允许的字段",
        )


def test_layer_outputs_should_validate_confidence_range() -> None:
    """
    测试分层输出契约会校验 confidence 范围。

    参数：
        无。

    返回值：
        None。
    """

    with pytest.raises(ValidationError):
        DogGenerationLayerOutput(
            generated_answer="测试答案",
            confidence=1.5,
        )
