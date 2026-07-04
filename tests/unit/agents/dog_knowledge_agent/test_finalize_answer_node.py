from src.agents.dog_knowledge_agent.nodes.finalize_answer_node import (
    build_finalize_dog_knowledge_answer_node,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswer,
)


def test_finalize_answer_node_returns_state_update():
    """
    测试 finalize answer node 可以返回 LangGraph state update。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    node = build_finalize_dog_knowledge_answer_node()

    state = {
        "question": "金毛寿命多久？",
        "pipeline_result": {
            "query_type": "exact_lookup",
            "answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
            "confidence": 0.9,
        },
    }

    update = node(state)

    assert "dog_knowledge_answer" in update
    assert "dog_knowledge_answer_public" in update
    assert "final_answer" in update

    assert isinstance(update["dog_knowledge_answer"], DogKnowledgeAnswer)
    assert update["dog_knowledge_answer"].question == "金毛寿命多久？"
    assert update["dog_knowledge_answer"].query_type == "exact_lookup"
    assert update["final_answer"] == "金毛寻回犬的寿命通常在 10 到 12 年左右。"


def test_finalize_answer_node_hides_debug_by_default():
    """
    测试 finalize answer node 默认隐藏 public dict 中的 debug。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    node = build_finalize_dog_knowledge_answer_node()

    state = {
        "question": "测试问题",
        "pipeline_result": {
            "answer": "测试答案",
            "debug": {
                "internal_node": "test_node",
            },
        },
    }

    update = node(state)

    assert "debug" not in update["dog_knowledge_answer_public"]


def test_finalize_answer_node_can_include_debug():
    """
    测试 finalize answer node 可以通过 include_debug=True 保留 debug。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    node = build_finalize_dog_knowledge_answer_node(
        include_debug=True,
    )

    state = {
        "question": "测试问题",
        "pipeline_result": {
            "answer": "测试答案",
            "debug": {
                "internal_node": "test_node",
            },
        },
    }

    update = node(state)

    assert "debug" in update["dog_knowledge_answer_public"]
    assert update["dog_knowledge_answer_public"]["debug"]["internal_node"] == "test_node"


def test_finalize_answer_node_formats_recommendation_result():
    """
    测试 finalize answer node 可以格式化推荐结果。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    node = build_finalize_dog_knowledge_answer_node()

    state = {
        "question": "新手适合养什么狗？",
        "recommendation_result": {
            "intent": "recommend",
            "recommendations": [
                {
                    "dog_name": "labrador_retriever",
                    "display_name": "Labrador Retriever / 拉布拉多寻回犬",
                    "reason": "性格友好，训练难度相对较低。",
                    "score": 0.88,
                }
            ],
        },
    }

    update = node(state)

    answer = update["dog_knowledge_answer"]

    assert answer.query_type == "recommendation"
    assert answer.has_recommendations() is True
    assert answer.recommended_breeds[0].breed_name == "labrador_retriever"