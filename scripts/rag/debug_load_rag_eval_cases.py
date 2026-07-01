from pathlib import Path

from src.rag.evaluation import load_rag_eval_cases


def main() -> None:
    """
    调试加载 RAG Evaluation 数据集。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dataset_path = Path("data/rag_eval/dog_rag_eval_cases.json")

    cases = load_rag_eval_cases(dataset_path)

    print(f"成功加载 RAG 评估用例数量: {len(cases)}")

    for eval_case in cases:
        print("=" * 80)
        print(f"case_id: {eval_case.case_id}")
        print(f"question: {eval_case.question}")
        print(f"expected_dog_names: {eval_case.expected_dog_names}")
        print(f"expected_filters: {eval_case.expected_filters}")
        print(f"top_k: {eval_case.top_k}")
        print(f"tags: {eval_case.tags}")
        print(f"note: {eval_case.note}")


if __name__ == "__main__":
    main()