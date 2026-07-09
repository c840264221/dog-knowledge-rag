import pytest
from pydantic import ValidationError

from src.agents.dog_knowledge_agent.contracts.schemas import (
    DogKnowledgeAnswer,
    DogKnowledgeEvidence,
    DogKnowledgeRecommendationItem,
)


def test_create_exact_lookup_answer_success():
    """
    测试可以创建精确查询类 DogKnowledgeAnswer。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    evidence = DogKnowledgeEvidence(
        evidence_id="chunk-golden-retriever-001",
        source_kind="rag_chunk",
        title="Golden Retriever",
        content="Golden Retrievers usually live around 10 to 12 years.",
        score=0.92,
        metadata={
            "dog_name": "golden_retriever",
            "chunk_index": 1,
        },
    )

    answer = DogKnowledgeAnswer(
        question="金毛寿命多久？",
        query_type="exact_lookup",
        status="success",
        answer="金毛寻回犬的寿命通常在 10 到 12 年左右。",
        evidences=[evidence],
        confidence=0.9,
        reason="命中了 Golden Retriever 的寿命信息。",
    )

    assert answer.question == "金毛寿命多久？"
    assert answer.query_type == "exact_lookup"
    assert answer.status == "success"
    assert answer.confidence == 0.9
    assert answer.has_evidences() is True
    assert answer.has_recommendations() is False
    assert answer.evidences[0].evidence_id == "chunk-golden-retriever-001"


def test_create_recommendation_answer_success():
    """
    测试可以创建推荐类 DogKnowledgeAnswer。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    recommendation = DogKnowledgeRecommendationItem(
        breed_name="labrador_retriever",
        display_name="Labrador Retriever / 拉布拉多寻回犬",
        reason="性格友好、训练难度相对较低，通常比较适合新手家庭。",
        matched_traits=["新手友好", "容易训练", "家庭友好"],
        warnings=["运动量较高，需要规律遛狗"],
        evidence_ids=["chunk-labrador-001"],
        score=0.88,
        metadata={
            "energy": "high",
            "trainability": "high",
        },
    )

    answer = DogKnowledgeAnswer(
        question="新手适合养什么狗？",
        query_type="recommendation",
        status="success",
        answer="新手可以优先考虑拉布拉多寻回犬，但要注意它运动量较高。",
        recommended_breeds=[recommendation],
        confidence=0.86,
        reason="用户需求是新手友好，拉布拉多命中了训练难度和家庭友好特征。",
    )

    assert answer.query_type == "recommendation"
    assert answer.has_recommendations() is True
    assert answer.recommended_breeds[0].breed_name == "labrador_retriever"
    assert answer.recommended_breeds[0].score == 0.88


def test_build_fallback_answer_success():
    """
    测试可以通过 build_fallback 快速创建 fallback 答案。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    answer = DogKnowledgeAnswer.build_fallback(
        question="哪种狗适合在火星生活？",
        answer="这个问题超出了当前犬种知识库的可靠范围，我暂时不能给出确定推荐。",
        fallback_reason="问题超出当前 DogKnowledgeAgent 知识边界。",
        confidence=0.1,
        debug={
            "retrieved_chunks": 0,
        },
    )

    assert answer.query_type == "fallback"
    assert answer.status == "fallback"
    assert answer.is_fallback is True
    assert answer.fallback_reason == "问题超出当前 DogKnowledgeAgent 知识边界。"
    assert answer.confidence == 0.1


def test_to_public_dict_hide_debug_by_default():
    """
    测试 to_public_dict 默认会隐藏 debug 字段。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    answer = DogKnowledgeAnswer.build_fallback(
        question="未知问题",
        answer="暂时无法回答。",
        fallback_reason="测试 fallback。",
        debug={
            "internal_node": "dog_knowledge_answer_formatter",
        },
    )

    public_data = answer.to_public_dict()

    assert "debug" not in public_data
    assert public_data["question"] == "未知问题"
    assert public_data["status"] == "fallback"


def test_to_public_dict_include_debug_when_enabled():
    """
    测试 to_public_dict 在 include_debug=True 时会保留 debug 字段。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    answer = DogKnowledgeAnswer.build_fallback(
        question="未知问题",
        answer="暂时无法回答。",
        fallback_reason="测试 fallback。",
        debug={
            "internal_node": "dog_knowledge_answer_formatter",
        },
    )

    public_data = answer.to_public_dict(include_debug=True)

    assert "debug" in public_data
    assert public_data["debug"]["internal_node"] == "dog_knowledge_answer_formatter"


def test_confidence_must_be_between_zero_and_one():
    """
    测试 confidence 必须在 0 到 1 之间。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    with pytest.raises(ValidationError):
        DogKnowledgeAnswer(
            question="测试问题",
            query_type="exact_lookup",
            status="success",
            answer="测试答案",
            confidence=1.5,
        )


def test_extra_fields_are_forbidden():
    """
    测试 Schema 不允许传入未定义字段。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    with pytest.raises(ValidationError):
        DogKnowledgeAnswer(
            question="测试问题",
            query_type="exact_lookup",
            status="success",
            answer="测试答案",
            confidence=0.8,
            unknown_field="不允许的字段",
        )
