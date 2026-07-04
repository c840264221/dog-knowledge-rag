from src.agents.dog_knowledge_agent.response_adapter import (
    DogKnowledgeAgentResponseAdapter,
    finalize_dog_knowledge_response,
    finalize_dog_knowledge_state,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswer,
)


def test_response_adapter_finalize_returns_answer_object():
    """
    测试 response adapter 可以返回 DogKnowledgeAnswer 对象。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    pipeline_result = {
        "question": "金毛寿命多久？",
        "query_type": "exact_lookup",
        "answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
        "confidence": 0.9,
    }

    answer = adapter.finalize(
        pipeline_result=pipeline_result,
        as_public_dict=False,
    )

    assert isinstance(answer, DogKnowledgeAnswer)
    assert answer.question == "金毛寿命多久？"
    assert answer.query_type == "exact_lookup"
    assert answer.status == "success"
    assert answer.confidence == 0.9


def test_response_adapter_finalize_returns_public_dict():
    """
    测试 response adapter 可以返回 public dict。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    pipeline_result = {
        "question": "哈士奇适合新手吗？",
        "query_type": "general_qa",
        "answer": "哈士奇精力旺盛，通常不太适合完全没有经验的新手。",
        "debug": {
            "node": "test_node",
        },
    }

    public_answer = adapter.finalize(
        pipeline_result=pipeline_result,
        as_public_dict=True,
    )

    assert isinstance(public_answer, dict)
    assert public_answer["question"] == "哈士奇适合新手吗？"
    assert public_answer["status"] == "success"
    assert "debug" not in public_answer


def test_response_adapter_public_dict_can_include_debug():
    """
    测试 public dict 在 include_debug=True 时可以包含 debug。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    pipeline_result = {
        "question": "哈士奇适合新手吗？",
        "answer": "哈士奇精力旺盛。",
        "debug": {
            "node": "test_node",
        },
    }

    public_answer = adapter.finalize(
        pipeline_result=pipeline_result,
        include_debug=True,
        as_public_dict=True,
    )

    assert "debug" in public_answer
    assert public_answer["debug"]["node"] == "test_node"


def test_response_adapter_question_argument_has_higher_priority():
    """
    测试显式传入 question 的优先级高于 pipeline_result 内部 question。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    pipeline_result = {
        "question": "旧问题",
        "answer": "测试答案",
    }

    answer = adapter.finalize(
        pipeline_result=pipeline_result,
        question="新问题",
    )

    assert isinstance(answer, DogKnowledgeAnswer)
    assert answer.question == "新问题"


def test_response_adapter_accepts_existing_dog_knowledge_answer():
    """
    测试 response adapter 可以接收已经格式化好的 DogKnowledgeAnswer。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    original_answer = DogKnowledgeAnswer(
        question="金毛寿命多久？",
        query_type="exact_lookup",
        status="success",
        answer="金毛寻回犬的寿命通常在 10 到 12 年左右。",
        confidence=0.9,
    )

    answer = adapter.finalize(original_answer)

    assert answer is original_answer


def test_response_adapter_finalize_state_from_pipeline_result():
    """
    测试 response adapter 可以从 state.pipeline_result 中生成最终输出。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    state = {
        "question": "新手适合养什么狗？",
        "pipeline_result": {
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

    update = adapter.finalize_state(state)

    assert "dog_knowledge_answer" in update
    assert "dog_knowledge_answer_public" in update
    assert "final_answer" in update

    answer = update["dog_knowledge_answer"]

    assert isinstance(answer, DogKnowledgeAnswer)
    assert answer.question == "新手适合养什么狗？"
    assert answer.query_type == "recommendation"
    assert answer.has_recommendations() is True
    assert "拉布拉多" in update["final_answer"]


def test_response_adapter_finalize_state_fallback_to_whole_state():
    """
    测试 state 中没有 pipeline_result 时，可以直接从整个 state 格式化。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    state = {
        "question": "边境牧羊犬聪明吗？",
        "query_type": "exact_lookup",
        "answer": "边境牧羊犬通常非常聪明，也比较容易训练。",
        "retrieved_chunks": [
            {
                "chunk_id": "border-collie-001",
                "content": "Border Collies are highly intelligent.",
                "score": 0.9,
            }
        ],
    }

    update = adapter.finalize_state(state)

    answer = update["dog_knowledge_answer"]

    assert answer.query_type == "exact_lookup"
    assert answer.has_evidences() is True
    assert update["final_answer"] == "边境牧羊犬通常非常聪明，也比较容易训练。"


def test_finalize_dog_knowledge_response_convenience_function():
    """
    测试 finalize_dog_knowledge_response 便捷函数。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    answer = finalize_dog_knowledge_response(
        pipeline_result={
            "question": "柯基掉毛吗？",
            "answer": "柯基通常会掉毛，换毛季会更明显。",
        }
    )

    assert isinstance(answer, DogKnowledgeAnswer)
    assert answer.question == "柯基掉毛吗？"


def test_finalize_dog_knowledge_state_convenience_function():
    """
    测试 finalize_dog_knowledge_state 便捷函数。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    update = finalize_dog_knowledge_state(
        state={
            "question": "贵宾犬适合公寓吗？",
            "answer": "贵宾犬体型选择多，部分小型贵宾犬比较适合公寓生活。",
        }
    )

    assert "dog_knowledge_answer" in update
    assert update["dog_knowledge_answer"].question == "贵宾犬适合公寓吗？"