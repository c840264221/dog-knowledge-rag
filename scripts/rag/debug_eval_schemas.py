from src.rag.evaluation import (
    RagEvalCase,
    RagEvalResult,
    RagEvalRetrievedItem,
)


def main() -> None:
    """
    测试 RAG Evaluation Schema 是否可以正常创建。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = RagEvalCase(
        case_id="dog_eval_001",
        question="适合新手养的小型犬有哪些？",
        expected_dog_names=["Poodle"],
        expected_filters={
            "size": "small",
            "trainability": "high",
        },
        top_k=5,
    )

    retrieved_item = RagEvalRetrievedItem(
        rank=1,
        chunk_id="poodle_chunk_001",
        dog_name="Poodle",
        score=0.92,
        source="data/dog_markdown/poodle.md",
        section_title="基本信息",
        content_preview="Poodle is intelligent and highly trainable.",
    )

    result = RagEvalResult(
        case_id=eval_case.case_id,
        question=eval_case.question,
        expected_dog_names=eval_case.expected_dog_names,
        expected_filters=eval_case.expected_filters,
        parsed_filters={
            "size": "small",
            "trainability": "high",
        },
        retrieved_items=[retrieved_item],
        retrieved_dog_names=["Poodle"],
        hit=True,
        hit_rank=1,
        top1_hit=True,
        filter_matched=True,
        empty_retrieval=False,
        passed=True,
    )

    print(eval_case)
    print(result)
    print("是否成功:", result.is_successful())


if __name__ == "__main__":
    main()