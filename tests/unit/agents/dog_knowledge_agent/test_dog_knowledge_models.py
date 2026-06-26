from src.agents.dog_knowledge_agent.models.extract_model_node import (
    extract_model_node,
)
from src.agents.dog_knowledge_agent.models.recommendation_model_node import (
    recommendation_model_node,
)


def test_extract_model_node_returns_partial_state():
    """
    测试 extract_model_node 返回正确的局部状态。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "question": "金毛寿命多长？",
        "intent": "ask_info",
    }

    result = extract_model_node(
        state=state,
    )

    assert result["current_agent"] == "dog_knowledge_agent"
    assert result["strategy"] == "extract_model"
    assert result["next_worker"] == "retrieve"
    assert "answer_strategy" not in result


def test_recommendation_model_node_returns_partial_state():
    """
    测试 recommendation_model_node 返回正确的局部状态。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "question": "新手适合养什么狗？",
    }

    result = recommendation_model_node(
        state=state,
    )

    assert result["current_agent"] == "recommendation_agent"
    assert result["intent"] == "recommend"
    assert result["strategy"] == "recommendation_model"
    assert result["next_worker"] == "retrieve"
    assert "answer_strategy" not in result