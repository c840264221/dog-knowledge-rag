from __future__ import annotations

import asyncio
from pathlib import Path

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import RagRetrievalBehaviorEvaluator


DEFAULT_DATASET_PATH = Path(
    "evaluation/datasets/rag_retrieval_behavior_cases.json"
)


async def run_rag_retrieval_behavior_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> int:
    """
    执行 V1.12.8 真实 RAG 检索行为评估。

    功能：
        加载统一 RAG 黄金集，通过真实 Parser、Retriever 和本地 Chroma
        执行检索，并在终端输出逐条失败检查项。

    参数含义：
        dataset_path:
            真实 RAG 检索黄金评估集路径。

    返回值含义：
        int:
            全部用例通过返回 0；存在失败或执行异常返回 1。
    """

    eval_cases = load_agent_evaluation_cases(dataset_path)
    evaluator = RagRetrievalBehaviorEvaluator()
    results = await evaluator.evaluate_many(eval_cases)
    passed_results = [result for result in results if result.passed]
    failed_results = [result for result in results if not result.passed]

    print("=" * 80)
    print("V1.12.8 Real RAG Retrieval Behavior Evaluation Report")
    print("=" * 80)
    print(f"dataset: {dataset_path.as_posix()}")
    print(f"total_cases: {len(results)}")
    print(f"passed_cases: {len(passed_results)}")
    print(f"failed_cases: {len(failed_results)}")
    print(
        "pass_rate: "
        f"{(len(passed_results) / len(results)) if results else 0.0:.2%}"
    )

    if failed_results:
        print("-" * 80)
        print("失败用例:")
        for result in failed_results:
            print(f"- {result.case_id}")
            if result.error_message:
                print(f"  error: {result.error_message}")
            for check in result.failed_checks():
                print(
                    f"  check={check.check_name}, "
                    f"expected={check.expected!r}, "
                    f"actual={check.actual!r}, "
                    f"message={check.message}"
                )
            if any(
                check.check_name == "quality_is_usable"
                for check in result.failed_checks()
            ):
                print(
                    "  quality_score="
                    f"{result.output.get('quality_score')!r}, "
                    "failure_type="
                    f"{result.output.get('quality_failure_type')!r}"
                )
                for reason in result.output.get("quality_reasons", []):
                    print(f"  quality_reason={reason}")

    print("=" * 80)
    return 1 if failed_results else 0


def main() -> None:
    """
    运行真实 RAG 检索行为评估命令行入口。

    参数含义：
        无。

    返回值含义：
        None；通过进程退出码表示真实 RAG 评估是否全部通过。
    """

    raise SystemExit(
        asyncio.run(run_rag_retrieval_behavior_evaluation())
    )


if __name__ == "__main__":
    main()
