from src.agents.dog_knowledge_agent.formatters.answer_formatter import (
    DogKnowledgeAnswerFormatter,
    format_dog_knowledge_answer,
)


def test_format_exact_lookup_answer_from_pipeline_dict():
    """
    测试 formatter 可以格式化精确查询类 pipeline dict。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "金毛寿命多久？",
        "query_type": "exact_lookup",
        "answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
        "retrieved_chunks": [
            {
                "chunk_id": "golden-retriever-001",
                "source_kind": "rag_chunk",
                "title": "Golden Retriever",
                "content": "Golden Retrievers usually live around 10 to 12 years.",
                "score": 0.93,
                "metadata": {
                    "dog_name": "golden_retriever",
                    "chunk_index": 1,
                },
            }
        ],
        "confidence": 0.9,
        "reason": "命中了 Golden Retriever 的寿命信息。",
    }

    answer = formatter.format(pipeline_result)

    assert answer.question == "金毛寿命多久？"
    assert answer.query_type == "exact_lookup"
    assert answer.status == "success"
    assert answer.answer == "金毛寻回犬的寿命通常在 10 到 12 年左右。"
    assert answer.confidence == 0.9
    assert answer.has_evidences() is True
    assert answer.evidences[0].evidence_id == "golden-retriever-001"
    assert answer.evidences[0].source_kind == "rag_chunk"


def test_format_recommendation_answer_from_pipeline_dict():
    """
    测试 formatter 可以格式化推荐类 pipeline dict。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "新手适合养什么狗？",
        "intent": "recommend",
        "recommendations": [
            {
                "dog_name": "labrador_retriever",
                "display_name": "Labrador Retriever / 拉布拉多寻回犬",
                "reason": "性格友好，训练难度相对较低，通常适合新手家庭。",
                "matched_traits": [
                    "新手友好",
                    "容易训练",
                    "家庭友好",
                ],
                "warnings": [
                    "运动量较高，需要规律遛狗",
                ],
                "evidence_ids": [
                    "chunk-labrador-001",
                ],
                "score": 88,
                "metadata": {
                    "energy": "high",
                    "trainability": "high",
                },
            }
        ],
    }

    answer = formatter.format(pipeline_result)

    assert answer.query_type == "recommendation"
    assert answer.status == "success"
    assert answer.has_recommendations() is True
    assert answer.recommended_breeds[0].breed_name == "labrador_retriever"
    assert answer.recommended_breeds[0].score == 0.88
    assert "拉布拉多" in answer.answer


def test_format_fallback_answer_from_pipeline_dict():
    """
    测试 formatter 可以格式化 fallback 类 pipeline dict。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "哪种狗适合在火星生活？",
        "is_fallback": True,
        "fallback_reason": "问题超出当前犬种知识库边界。",
        "debug": {
            "retrieved_chunks": 0,
        },
    }

    answer = formatter.format(pipeline_result)

    assert answer.query_type == "fallback"
    assert answer.status == "fallback"
    assert answer.is_fallback is True
    assert answer.fallback_reason == "问题超出当前犬种知识库边界。"
    assert answer.confidence == 0.1
    assert "无法基于当前犬种知识库可靠回答" in answer.answer


def test_format_source_documents_as_evidences():
    """
    测试 formatter 可以把 source_documents 转换成 evidences。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "边境牧羊犬容易训练吗？",
        "query_type": "rag",
        "answer": "边境牧羊犬通常非常聪明，训练能力较强。",
        "source_documents": [
            {
                "id": "border-collie-doc-001",
                "page_content": "Border Collies are highly intelligent and very trainable.",
                "metadata": {
                    "dog_name": "border_collie",
                },
                "similarity_score": 0.87,
            }
        ],
    }

    answer = formatter.format(pipeline_result)

    assert answer.query_type == "exact_lookup"
    assert answer.has_evidences() is True
    assert answer.evidences[0].evidence_id == "border-collie-doc-001"
    assert answer.evidences[0].content == "Border Collies are highly intelligent and very trainable."
    assert answer.evidences[0].score == 0.87


def test_format_exact_lookup_from_graph_answer_strategy():
    """
    测试 formatter 可以从真实 graph state 的 answer_strategy 中识别精确查询。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "金毛寿命多久？",
        "answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
        "answer_strategy": {
            "task_type": "exact_info",
            "answer_style": "direct_fact",
            "reason": "用户问题被识别为具体犬种信息查询。",
        },
    }

    answer = formatter.format(pipeline_result)

    assert answer.query_type == "exact_lookup"
    assert answer.status == "success"
    assert answer.answer == "金毛寻回犬的寿命通常在 10 到 12 年左右。"


def test_format_recommendation_from_graph_rag_context_chunks():
    """
    测试 formatter 可以从真实 graph state 的 rag_context.chunks 合成推荐项。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "新手适合养什么狗？",
        "answer": "可以优先考虑拉布拉多寻回犬，它通常比较友好，也比较容易训练。",
        "answer_strategy": {
            "task_type": "recommendation",
            "answer_style": "ranked_recommendation",
            "reason": "用户问题包含推荐需求。",
        },
        "rag_context": {
            "chunks": [
                {
                    "chunk": {
                        "id": "labrador-retriever-001",
                        "title": "Labrador Retriever",
                        "content": "Labrador Retrievers are friendly and trainable family dogs.",
                        "metadata": {
                            "dog_name": "labrador_retriever",
                            "energy": "high",
                            "trainability": "high",
                        },
                    },
                    "final_score": 0.86,
                }
            ],
        },
    }

    answer = formatter.format(pipeline_result)

    assert answer.query_type == "recommendation"
    assert answer.has_recommendations() is True
    assert answer.recommended_breeds[0].breed_name == "labrador_retriever"
    assert answer.recommended_breeds[0].evidence_ids == ["labrador-retriever-001"]
    assert answer.has_evidences() is True


def test_format_question_argument_has_higher_priority():
    """
    测试外部显式传入的 question 优先级高于 pipeline_result 中的 question。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    formatter = DogKnowledgeAnswerFormatter()

    pipeline_result = {
        "question": "旧问题",
        "answer": "测试答案",
    }

    answer = formatter.format(
        pipeline_result=pipeline_result,
        question="新问题",
    )

    assert answer.question == "新问题"


def test_format_convenience_function():
    """
    测试 format_dog_knowledge_answer 便捷函数可用。

    参数：
        无。

    返回值：
        无。pytest 根据断言判断测试是否通过。
    """

    answer = format_dog_knowledge_answer(
        pipeline_result={
            "question": "哈士奇适合新手吗？",
            "answer": "哈士奇精力旺盛，通常不太适合完全没有经验的新手。",
            "query_type": "general_qa",
        }
    )

    assert answer.question == "哈士奇适合新手吗？"
    assert answer.query_type == "general_qa"
    assert answer.status == "success"

