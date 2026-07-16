from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.baseline import (
    compare_evaluation_report_to_baseline,
    load_evaluation_baseline,
    load_evaluation_suite_report,
    write_evaluation_regression_report,
)


DEFAULT_CURRENT_REPORT_PATH = Path(
    "evaluation/reports/v112_agent_evaluation_report.json"
)
DEFAULT_BASELINE_PATH = Path(
    "evaluation/baselines/v112_full_evaluation_baseline.json"
)
DEFAULT_JSON_OUTPUT_PATH = Path(
    "evaluation/reports/v113_evaluation_regression_report.json"
)
DEFAULT_MARKDOWN_OUTPUT_PATH = Path(
    "evaluation/reports/v113_evaluation_regression_report.md"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    规定基线比较命令可以接收哪些参数。

    功能：
        这个命令一共支持四个路径：当前报告、历史基线、JSON 输出和
        Markdown 输出。用户没有传入路径时，就使用项目中的默认位置。

    参数含义：
        无。

    返回值含义：
        argparse.ArgumentParser:
            一个参数读取器，后面可以从中拿到用户传入的四个文件路径。
    """

    parser = argparse.ArgumentParser(
        description="比较当前统一评估报告与 V1.12 历史成绩基线。",
    )
    parser.add_argument(
        "--current-report",
        type=Path,
        default=DEFAULT_CURRENT_REPORT_PATH,
        help="当前统一评估 JSON 报告路径。",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="需要使用的历史评估基线 JSON 路径。",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="回归比较 JSON 报告输出路径。",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT_PATH,
        help="回归比较 Markdown 报告输出路径。",
    )
    return parser


def run_baseline_comparison(
    *,
    current_report_path: str | Path,
    baseline_path: str | Path,
    json_output_path: str | Path,
    markdown_output_path: str | Path,
) -> int:
    """
    完成一次“当前成绩和 V1.12 历史成绩”的完整比较。

    功能：
        先读取当前报告和历史基线，再逐项比较整体、类别和 RAG 指标，
        然后生成 JSON、Markdown，并在终端打印最终是否发生退步。

    参数含义：
        current_report_path:
            当前版本评估成绩所在的 JSON 文件。
        baseline_path:
            V1.12 历史成绩所在的 JSON 文件。
        json_output_path、markdown_output_path:
            新旧成绩比较完成后，两个结果文件分别保存到哪里。

    返回值含义：
        int:
            没有成绩退步返回 0；任一成绩退步或缺失返回 1。CI 会根据这个
            数字判断当前步骤成功还是失败。
    """

    current_report = load_evaluation_suite_report(current_report_path)
    baseline = load_evaluation_baseline(baseline_path)
    regression_report = compare_evaluation_report_to_baseline(
        report=current_report,
        baseline=baseline,
    )
    write_evaluation_regression_report(
        report=regression_report,
        json_path=json_output_path,
        markdown_path=markdown_output_path,
    )

    print("=" * 80)
    print("V1.13 Evaluation Baseline Regression Report")
    print("=" * 80)
    print(f"baseline: {regression_report.baseline_name}")
    print(f"current_suite: {regression_report.current_suite_name}")
    print(f"current_version: {regression_report.current_version}")
    print(f"passed: {regression_report.passed}")
    print(f"json_report: {Path(json_output_path).as_posix()}")
    print(f"markdown_report: {Path(markdown_output_path).as_posix()}")

    if regression_report.failed_checks():
        print("-" * 80)
        print("发现以下质量回退:")
        for check in regression_report.failed_checks():
            print(f"- {check.message}")

    print("=" * 80)
    return 0 if regression_report.passed else 1


def main() -> None:
    """
    让这个文件可以通过 python -m 命令直接运行。

    功能：
        读取用户在命令行中传入的四个路径，调用
        run_baseline_comparison 完成比较，最后把 0 或 1 返回给终端和 CI。

    参数含义：
        无。

    返回值含义：
        None。函数本身不返回数据，而是通过 SystemExit 结束程序并设置
        退出码：0 表示没有退步，1 表示发生退步。
    """

    args = build_argument_parser().parse_args()
    raise SystemExit(
        run_baseline_comparison(
            current_report_path=args.current_report,
            baseline_path=args.baseline,
            json_output_path=args.json_output,
            markdown_output_path=args.markdown_output,
        )
    )


if __name__ == "__main__":
    main()
