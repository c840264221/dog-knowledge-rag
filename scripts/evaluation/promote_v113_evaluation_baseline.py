from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.baseline import (
    build_evaluation_baseline_snapshot,
    load_evaluation_baseline,
    load_evaluation_suite_report,
    write_evaluation_baseline_snapshot,
)


DEFAULT_CURRENT_REPORT_PATH = Path(
    "evaluation/reports/v112_agent_evaluation_report.json"
)
DEFAULT_CURRENT_BASELINE_PATH = Path(
    "evaluation/baselines/v112_full_evaluation_baseline.json"
)
DEFAULT_CANDIDATE_OUTPUT_PATH = Path(
    "evaluation/baselines/candidates/v113_full_evaluation_baseline.json"
)
DEFAULT_BASELINE_NAME = "v113_full_evaluation_candidate"


def build_argument_parser() -> argparse.ArgumentParser:
    """
    规定生成候选基线命令可以接收哪些参数。

    功能：
        支持指定当前报告、正在使用的正式基线、候选基线名称、输出位置和
        是否覆盖已有文件。用户不传参数时使用项目约定的默认值。

    参数含义：
        无。

    返回值含义：
        argparse.ArgumentParser:
            一个参数读取器，后面可以从中取得用户传入的设置。
    """

    parser = argparse.ArgumentParser(
        description="从通过质量门禁的完整报告中生成候选评估基线。",
    )
    parser.add_argument(
        "--current-report",
        type=Path,
        default=DEFAULT_CURRENT_REPORT_PATH,
        help="准备晋升为基线的当前评估 JSON 报告。",
    )
    parser.add_argument(
        "--current-baseline",
        type=Path,
        default=DEFAULT_CURRENT_BASELINE_PATH,
        help="团队目前正在使用的正式基线 JSON。",
    )
    parser.add_argument(
        "--baseline-name",
        default=DEFAULT_BASELINE_NAME,
        help="候选基线名称。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT_PATH,
        help="候选基线 JSON 输出路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确允许覆盖已经存在的候选基线文件。",
    )
    return parser


def run_baseline_promotion(
    *,
    current_report_path: str | Path,
    current_baseline_path: str | Path,
    baseline_name: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> int:
    """
    从当前合格成绩中生成一份等待人工审查的候选基线。

    功能：
        读取当前报告和正式基线，确认所有旧成绩没有下降。至少一项成绩提高
        时写入候选 JSON；全部成绩相同时不写文件。这个函数不会修改正式
        基线，也不会执行 Git 操作。

    参数含义：
        current_report_path:
            准备作为新标准的当前评估报告位置。
        current_baseline_path:
            团队目前使用的正式基线位置。
        baseline_name:
            候选基线名称。
        output_path:
            生成的候选基线需要保存到哪里。
        overwrite:
            是否允许覆盖已有候选文件，默认不允许。

    返回值含义：
        int:
            成功生成候选基线，或者成绩相同无需更新时返回 0；安全检查失败
            时会抛出异常并停止。
    """

    report = load_evaluation_suite_report(current_report_path)
    current_baseline = load_evaluation_baseline(current_baseline_path)
    baseline = build_evaluation_baseline_snapshot(
        report=report,
        current_baseline=current_baseline,
        baseline_name=baseline_name,
        metadata={
            "candidate": True,
            "promotion_requires_review": True,
        },
    )
    if baseline is None:
        print("=" * 80)
        print("V1.13 Evaluation Baseline Candidate")
        print("=" * 80)
        print(f"current_baseline: {current_baseline.baseline_name}")
        print("status: no_update_required")
        print("reason: 当前成绩与正式基线完全相同，未生成候选文件。")
        print("=" * 80)
        return 0

    resolved_output = write_evaluation_baseline_snapshot(
        baseline=baseline,
        output_path=output_path,
        overwrite=overwrite,
    )

    print("=" * 80)
    print("V1.13 Evaluation Baseline Candidate")
    print("=" * 80)
    print(f"baseline_name: {baseline.baseline_name}")
    print(f"source_suite: {baseline.source_suite_name}")
    print(f"source_version: {baseline.source_version}")
    print(f"overall_pass_rate: {baseline.overall_pass_rate:.2%}")
    print(f"category_count: {len(baseline.category_pass_rates)}")
    print(f"metric_count: {len(baseline.metrics)}")
    print(f"candidate_file: {resolved_output.as_posix()}")
    print("status: candidate_requires_human_review")
    print("=" * 80)
    return 0


def main() -> None:
    """
    让候选基线生成脚本可以通过 python -m 命令直接运行。

    功能：
        读取命令行参数，调用 run_baseline_promotion 生成候选文件，并把
        执行结果交给终端。这里只生成候选文件，不会自动提交或替换正式基线。

    参数含义：
        无。

    返回值含义：
        None。成功时通过 SystemExit 返回退出码 0；异常时程序直接失败。
    """

    args = build_argument_parser().parse_args()
    raise SystemExit(
        run_baseline_promotion(
            current_report_path=args.current_report,
            current_baseline_path=args.current_baseline,
            baseline_name=args.baseline_name,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
