from __future__ import annotations

import argparse
import asyncio
import platform
from pathlib import Path
from typing import Literal

from src.evaluation.report_builder import (
    build_evaluation_suite_report,
    write_evaluation_suite_report,
)
from src.evaluation.runner import (
    CORE_EVALUATION_TARGETS,
    EvaluationSuiteRunner,
    EvaluationTarget,
    FULL_EVALUATION_TARGETS,
)
from src.evaluation.schemas import EvaluationSuiteReport


DEFAULT_JSON_REPORT_PATH = Path(
    "evaluation/reports/v112_agent_evaluation_report.json"
)
DEFAULT_MARKDOWN_REPORT_PATH = Path(
    "evaluation/reports/v112_agent_evaluation_report.md"
)
CORE_JSON_REPORT_PATH = Path(
    "evaluation/reports/v112_core_evaluation_report.json"
)
CORE_MARKDOWN_REPORT_PATH = Path(
    "evaluation/reports/v112_core_evaluation_report.md"
)

EvaluationProfile = Literal["core", "full"]


def resolve_evaluation_targets(
    profile: EvaluationProfile,
) -> tuple[EvaluationTarget, ...]:
    """
    根据运行档位选择统一评估目标。

    参数含义：
        profile:
            core 表示确定性快速门禁；full 表示包含真实 RAG 的完整门禁。

    返回值含义：
        tuple[EvaluationTarget, ...]:
            当前运行档位需要顺序执行的评估目标。
    """

    if profile == "core":
        return CORE_EVALUATION_TARGETS
    return FULL_EVALUATION_TARGETS


def resolve_report_paths(
    profile: EvaluationProfile,
) -> tuple[Path, Path]:
    """
    根据运行档位选择默认 JSON 和 Markdown 报告路径。

    参数含义：
        profile:
            当前评估运行档位。

    返回值含义：
        tuple[Path, Path]:
            第一项是 JSON 路径，第二项是 Markdown 路径。
    """

    if profile == "core":
        return CORE_JSON_REPORT_PATH, CORE_MARKDOWN_REPORT_PATH
    return DEFAULT_JSON_REPORT_PATH, DEFAULT_MARKDOWN_REPORT_PATH


def print_category_results(report: EvaluationSuiteReport) -> None:
    """
    在终端打印分类成绩和专业总体指标。

    参数含义：
        report:
            已构建完成的统一评估报告。

    返回值含义：
        None。
    """

    print("-" * 80)
    print("分类成绩:")
    for summary in report.category_summaries:
        print(
            f"- {summary.category}: "
            f"{summary.passed_cases}/{summary.total_cases}, "
            f"pass_rate={summary.pass_rate:.2%}"
        )
        for metric_name, metric_value in summary.metrics.items():
            value_text = (
                f"{metric_value:.2%}"
                if isinstance(metric_value, float)
                else str(metric_value)
            )
            print(f"  {metric_name}: {value_text}")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建 V1.12 统一评估命令行参数解析器。

    参数含义：
        无。

    返回值含义：
        argparse.ArgumentParser:
            支持选择 core 或 full 运行档位的参数解析器。
    """

    parser = argparse.ArgumentParser(
        description="执行 V1.12 统一 Agent 与 RAG 质量门禁。",
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="full",
        help="core 为快速确定性门禁，full 额外执行真实 RAG。",
    )
    return parser


async def run_all_v112_evaluations(
    profile: EvaluationProfile = "full",
    json_report_path: Path | None = None,
    markdown_report_path: Path | None = None,
) -> int:
    """
    执行 V1.12 全部 Agent 评估并生成统一质量报告。

    参数含义：
        profile:
            core 为快速确定性门禁；full 为包含真实 RAG 的完整门禁。
        json_report_path:
            可选机器可读 JSON 报告路径；不传时按 profile 选择默认路径。
        markdown_report_path:
            可选人工可读 Markdown 报告路径；不传时按 profile 选择默认路径。

    返回值含义：
        int:
            质量门禁通过返回 0；不通过返回 1。
    """

    default_json_path, default_markdown_path = resolve_report_paths(profile)
    resolved_json_path = json_report_path or default_json_path
    resolved_markdown_path = markdown_report_path or default_markdown_path
    suite_run = await EvaluationSuiteRunner(
        targets=resolve_evaluation_targets(profile),
    ).run()
    report = build_evaluation_suite_report(
        suite_run=suite_run,
        metadata={
            "evaluation_profile": profile,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    write_evaluation_suite_report(
        report=report,
        json_path=resolved_json_path,
        markdown_path=resolved_markdown_path,
    )

    print("=" * 80)
    print("V1.12.9 Unified Agent and RAG Evaluation Report")
    print("=" * 80)
    print(f"profile: {profile}")
    print(f"total_cases: {report.total_cases}")
    print(f"passed_cases: {report.passed_cases}")
    print(f"failed_cases: {report.failed_cases}")
    print(f"error_cases: {report.error_cases}")
    print(f"pass_rate: {report.pass_rate:.2%}")
    print(f"quality_gate_passed: {report.quality_gate.passed}")
    print(f"json_report: {resolved_json_path.as_posix()}")
    print(f"markdown_report: {resolved_markdown_path.as_posix()}")
    print_category_results(report)

    if report.quality_gate.violations:
        print("-" * 80)
        print("质量门禁未通过原因:")
        for violation in report.quality_gate.violations:
            print(f"- {violation}")

    print("=" * 80)
    return 0 if report.quality_gate.passed else 1


def main() -> None:
    """
    运行 V1.12 统一评估命令行入口。

    参数含义：
        无。

    返回值含义：
        None；通过进程退出码表达质量门禁结果。
    """

    args = build_argument_parser().parse_args()
    raise SystemExit(
        asyncio.run(
            run_all_v112_evaluations(profile=args.profile)
        )
    )


if __name__ == "__main__":
    main()
