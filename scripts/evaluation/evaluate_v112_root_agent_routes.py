from __future__ import annotations

import asyncio
from pathlib import Path

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import RootRouteEvaluator


DEFAULT_DATASET_PATH = Path(
    "evaluation/datasets/root_agent_route_cases.json"
)


async def run_root_route_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> int:
    """
    执行 V1.12.2 RootAgent 路由黄金集评估。

    参数含义：
        dataset_path:
            RootAgent 路由黄金评估集路径。

    返回值含义：
        int:
            全部用例通过返回 0；存在失败用例返回 1。
    """

    # 加载外部黄金评估集，使业务期望不与 pytest 测试代码绑定。
    eval_cases = load_agent_evaluation_cases(dataset_path)
    evaluator = RootRouteEvaluator()

    # 默认评估主图真实 semantic_router 入口及其条件边路由结果。
    results = await evaluator.evaluate_many(eval_cases)
    passed_results = [result for result in results if result.passed]
    failed_results = [result for result in results if not result.passed]

    print("=" * 80)
    print("V1.12.2 RootAgent Route Evaluation Report")
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
    运行 RootAgent 路由评估命令行入口。

    参数含义：
        无。

    返回值含义：
        None；通过进程退出码向命令行返回评估是否成功。
    """

    exit_code = asyncio.run(
        run_root_route_evaluation()
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
