from __future__ import annotations

import asyncio
from pathlib import Path

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import MultiAgentOrchestrationEvaluator


DEFAULT_DATASET_PATH = Path(
    "evaluation/datasets/multi_agent_orchestration_cases.json"
)


async def run_multi_agent_orchestration_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> int:
    """
    执行 V1.17 多 Agent 总编排集成评估。

    参数含义：
        dataset_path:
            总编排行为黄金数据集路径。

    返回值含义：
        int:
            全部用例通过返回 0；存在失败用例返回 1。
    """

    eval_cases = load_agent_evaluation_cases(dataset_path)
    results = await MultiAgentOrchestrationEvaluator().evaluate_many(
        eval_cases
    )
    failed_results = [result for result in results if not result.passed]

    print("=" * 80)
    print("V1.17 Multi-Agent Orchestration Evaluation Report")
    print("=" * 80)
    print(f"dataset: {dataset_path.as_posix()}")
    print(f"total_cases: {len(results)}")
    print(f"passed_cases: {len(results) - len(failed_results)}")
    print(f"failed_cases: {len(failed_results)}")
    if failed_results:
        print("-" * 80)
        for result in failed_results:
            print(f"- {result.case_id}")
            if result.error_message:
                print(f"  error: {result.error_message}")
            for check in result.failed_checks():
                print(
                    f"  check={check.check_name}, "
                    f"expected={check.expected!r}, "
                    f"actual={check.actual!r}"
                )
    print("=" * 80)
    return 1 if failed_results else 0


def main() -> None:
    """
    运行总编排评估命令行入口。

    参数含义：
        无。

    返回值含义：
        None:
            使用进程退出码表示黄金集是否全部通过。
    """

    raise SystemExit(
        asyncio.run(run_multi_agent_orchestration_evaluation())
    )


if __name__ == "__main__":
    main()
