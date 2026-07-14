from __future__ import annotations

import asyncio
from pathlib import Path

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import DogKnowledgeBehaviorEvaluator


DEFAULT_DATASET_PATH = Path(
    "evaluation/datasets/dog_knowledge_behavior_cases.json"
)


async def run_dog_knowledge_behavior_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> int:
    """
    执行 V1.12.4 DogKnowledgeAgent 行为与响应契约评估。

    参数含义：
        dataset_path:
            DogKnowledgeAgent 黄金行为评估集路径。

    返回值含义：
        int:
            全部用例通过返回 0；存在失败用例返回 1。
    """

    eval_cases = load_agent_evaluation_cases(dataset_path)
    evaluator = DogKnowledgeBehaviorEvaluator()
    results = await evaluator.evaluate_many(eval_cases)
    passed_results = [result for result in results if result.passed]
    failed_results = [result for result in results if not result.passed]

    print("=" * 80)
    print("V1.12.4 DogKnowledgeAgent Behavior Evaluation Report")
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

    print("=" * 80)
    return 1 if failed_results else 0


def main() -> None:
    """
    运行 DogKnowledgeAgent 行为评估命令行入口。

    参数含义：
        无。

    返回值含义：
        None；通过进程退出码表示评估是否全部通过。
    """

    exit_code = asyncio.run(
        run_dog_knowledge_behavior_evaluation()
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
