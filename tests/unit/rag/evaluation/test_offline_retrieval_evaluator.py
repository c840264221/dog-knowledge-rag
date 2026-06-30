from typing import Any

from src.rag.evaluation import (
    OfflineRetrievalEvaluator,
    RagEvalCase,
)


def fake_parse_query_func(
    eval_case: RagEvalCase,
) -> dict[str, Any]:
    """
    测试用 query parser。

    参数含义：
        eval_case:
            RAG 评估用例。

    返回值含义：
        dict[str, Any]:
            模拟解析后的 query。
    """

    return {
        "question": eval_case.question,
        "filters": eval_case.expected_filters,
        "top_k": eval_case.top_k,
    }


def fake_retrieve_context_func(
    parsed_query: dict[str, Any],
) -> dict[str, Any]:
    """
    测试用 retriever。

    参数含义：
        parsed_query:
            模拟解析后的 query。

    返回值含义：
        dict[str, Any]:
            模拟 RagContext。
    """

    return {
        "status": "success",
        "context_text": (
            "Poodle 是一种聪明、友好、可训练性较高的犬种。"
            "Poodle 适合很多家庭场景，也经常被认为比较容易训练。"
            "这个上下文用于测试离线评估器是否能正确提取召回结果。"
        ),
        "chunks": [
            {
                "retrieval_score": 0.1,
                "chunk": {
                    "chunk_id": "poodle_chunk_001",
                    "content": "Poodle is intelligent and highly trainable.",
                    "metadata": {
                        "dog_name": "Poodle",
                        "section_title": "训练",
                        "source": "data/dog_markdown/poodle.md",
                    },
                },
            }
        ],
    }


def test_evaluate_case_success() -> None:
    """
    测试单条评估用例可以成功转换成 RagEvalResult。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = RagEvalCase(
        case_id="dog_eval_001",
        question="Poodle 的训练难度怎么样？",
        expected_dog_names=["Poodle"],
        expected_filters={
            "dog_name": "Poodle",
        },
        top_k=5,
    )

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=fake_parse_query_func,
        retrieve_context_func=fake_retrieve_context_func,
        require_quality_usable=False,
    )

    result = evaluator.evaluate_case(
        eval_case=eval_case,
    )

    assert result.case_id == "dog_eval_001"
    assert result.hit is True
    assert result.hit_rank == 1
    assert result.top1_hit is True
    assert result.filter_matched is True
    assert result.empty_retrieval is False
    assert result.passed is True
    assert result.retrieved_dog_names == ["Poodle"]
    assert result.retrieved_items[0].dog_name == "Poodle"


def test_evaluate_case_filter_mismatch() -> None:
    """
    测试 parsed_filters 和 expected_filters 不一致时，评估结果不通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def fake_parse_wrong_filter_func(
        eval_case: RagEvalCase,
    ) -> dict[str, Any]:
        """
        测试用错误 query parser。

        参数含义：
            eval_case:
                RAG 评估用例。

        返回值含义：
            dict[str, Any]:
                故意返回错误 dog_name filter。
        """

        return {
            "question": eval_case.question,
            "filters": {
                "dog_name": "Poodle",
            },
            "top_k": eval_case.top_k,
        }

    eval_case = RagEvalCase(
        case_id="dog_eval_002",
        question="Golden Retriever 的训练难度怎么样？",
        expected_dog_names=["Golden Retriever"],
        expected_filters={
            "dog_name": "Golden Retriever",
        },
        top_k=5,
    )

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=fake_parse_wrong_filter_func,
        retrieve_context_func=fake_retrieve_context_func,
        require_quality_usable=False,
    )

    result = evaluator.evaluate_case(
        eval_case=eval_case,
    )

    assert result.filter_matched is False
    assert result.passed is False


def test_evaluate_case_empty_retrieval() -> None:
    """
    测试空召回时，评估结果不通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def empty_retrieve_context_func(
        parsed_query: dict[str, Any],
    ) -> dict[str, Any]:
        """
        测试用空召回 retriever。

        参数含义：
            parsed_query:
                模拟 query。

        返回值含义：
            dict[str, Any]:
                空 RagContext。
        """

        return {
            "status": "empty",
            "context_text": "",
            "chunks": [],
        }

    eval_case = RagEvalCase(
        case_id="dog_eval_003",
        question="Chihuahua 的寿命是多少？",
        expected_dog_names=["Chihuahua"],
        expected_filters={
            "dog_name": "Chihuahua",
        },
        top_k=5,
    )

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=fake_parse_query_func,
        retrieve_context_func=empty_retrieve_context_func,
        require_quality_usable=False,
    )

    result = evaluator.evaluate_case(
        eval_case=eval_case,
    )

    assert result.hit is False
    assert result.hit_rank is None
    assert result.top1_hit is False
    assert result.empty_retrieval is True
    assert result.passed is False


def test_evaluate_many() -> None:
    """
    测试批量执行评估用例。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = [
        RagEvalCase(
            case_id="dog_eval_001",
            question="Poodle 的训练难度怎么样？",
            expected_dog_names=["Poodle"],
            expected_filters={
                "dog_name": "Poodle",
            },
            top_k=5,
        ),
        RagEvalCase(
            case_id="dog_eval_002",
            question="Poodle 适合新手吗？",
            expected_dog_names=["Poodle"],
            expected_filters={
                "dog_name": "Poodle",
            },
            top_k=5,
        ),
    ]

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=fake_parse_query_func,
        retrieve_context_func=fake_retrieve_context_func,
        require_quality_usable=False,
    )

    results = evaluator.evaluate_many(
        eval_cases=eval_cases,
    )

    assert len(results) == 2
    assert results[0].case_id == "dog_eval_001"
    assert results[1].case_id == "dog_eval_002"



def test_evaluate_case_writes_flattened_and_semantic_filters() -> None:
    """
    测试 OfflineRetrievalEvaluator 会把扁平化和语义归一化后的 filters 写入 extra。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def fake_parse_query_func(
        eval_case: RagEvalCase,
    ) -> dict[str, Any]:
        """
        测试用 query parser。

        参数含义：
            eval_case:
                RAG 评估用例。

        返回值含义：
            dict[str, Any]:
                模拟解析后的 query。
        """

        return {
            "question": eval_case.question,
            "filters": {
                "$and": [
                    {
                        "size": {
                            "$eq": "small",
                        }
                    },
                    {
                        "trainability_level": {
                            "$gte": 4,
                        }
                    },
                    {
                        "good_for_beginner": {
                            "$eq": True,
                        }
                    },
                ]
            },
            "top_k": eval_case.top_k,
        }

    def fake_retrieve_context_func(
        parsed_query: dict[str, Any],
    ) -> dict[str, Any]:
        """
        测试用 retriever。

        参数含义：
            parsed_query:
                模拟解析后的 query。

        返回值含义：
            dict[str, Any]:
                模拟 RagContext。
        """

        return {
            "status": "success",
            "context_text": (
                "Poodle is intelligent and highly trainable. "
                "This context is long enough for evaluation."
            ),
            "chunks": [
                {
                    "retrieval_score": 0.1,
                    "chunk": {
                        "chunk_id": "poodle_chunk_001",
                        "content": "Poodle is intelligent and highly trainable.",
                        "metadata": {
                            "dog_name": "Poodle",
                            "section_title": "Training",
                            "source": "data/dog_markdown/poodle.md",
                        },
                    },
                }
            ],
        }

    eval_case = RagEvalCase(
        case_id="dog_eval_filter_001",
        question="适合新手养的、比较容易训练的小型犬有哪些？",
        expected_dog_names=["Poodle"],
        expected_filters={
            "size": "small",
            "trainability": "high",
        },
        top_k=5,
    )

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=fake_parse_query_func,
        retrieve_context_func=fake_retrieve_context_func,
        require_quality_usable=False,
    )

    result = evaluator.evaluate_case(
        eval_case=eval_case,
    )

    assert result.filter_matched is True

    assert result.extra["expected_filters_flattened"] == {
        "size": "small",
        "trainability": "high",
    }

    assert result.extra["parsed_filters_flattened"] == {
        "size": "small",
        "trainability_level": {
            "$gte": 4,
        },
        "good_for_beginner": True,
    }

    assert result.extra["expected_filters_semantic"] == {
        "size": "small",
        "trainability": "high",
    }

    assert result.extra["parsed_filters_semantic"] == {
        "size": "small",
        "trainability": "high",
        "good_for_beginner": True,
    }