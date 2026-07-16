from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.evaluation.compare_v113_evaluation_baseline import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_CURRENT_REPORT_PATH,
    build_argument_parser,
    run_baseline_comparison,
)
from src.evaluation.schemas import (
    AgentEvaluationResult,
    EvaluationBaselineSnapshot,
    EvaluationCategorySummary,
    EvaluationCheckResult,
    EvaluationQualityGate,
    EvaluationSuiteReport,
)


def write_comparison_inputs(
    tmp_path: Path,
    *,
    current_pass_rate: float,
) -> tuple[Path, Path]:
    """
    为命令行测试准备两份最小 JSON 文件。

    功能：
        在临时目录中生成一份当前报告和一份历史基线。两份文件只包含
        root_route 类别，这样测试不需要依赖本机已有的真实报告。

    参数含义：
        tmp_path:
            两份测试文件需要保存到的临时目录。
        current_pass_rate:
            想让当前报告模拟多少通过率，例如 1.0 表示 100%。

    返回值含义：
        tuple[Path, Path]:
            两个文件的位置：第一项是当前报告，第二项是历史基线。
    """

    current_report = EvaluationSuiteReport(
        suite_name="v113_cli_test",
        version="V1.13.0",
        generated_at=datetime.now(timezone.utc),
        total_cases=1,
        passed_cases=round(current_pass_rate),
        failed_cases=1 - round(current_pass_rate),
        pass_rate=current_pass_rate,
        category_summaries=[
            EvaluationCategorySummary(
                category="root_route",
                dataset_path="evaluation/root.json",
                total_cases=1,
                passed_cases=round(current_pass_rate),
                failed_cases=1 - round(current_pass_rate),
                pass_rate=current_pass_rate,
            )
        ],
        quality_gate=EvaluationQualityGate(
            policy_name="cli_test",
            passed=True,
        ),
        results=[
            AgentEvaluationResult(
                case_id="root_cli_001",
                category="root_route",
                checks=[
                    EvaluationCheckResult(
                        check_name="route",
                        passed=current_pass_rate == 1.0,
                        expected="dog_knowledge_agent",
                        actual=(
                            "dog_knowledge_agent"
                            if current_pass_rate == 1.0
                            else "general_agent"
                        ),
                    )
                ],
            )
        ],
    )
    baseline = EvaluationBaselineSnapshot(
        baseline_name="v112_cli_test",
        source_suite_name="v112_cli_source",
        source_version="V1.12.9",
        overall_pass_rate=1.0,
        category_pass_rates={"root_route": 1.0},
    )
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    current_path.write_text(
        current_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    baseline_path.write_text(
        baseline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return current_path, baseline_path


def test_argument_parser_should_use_project_default_paths() -> None:
    """
    检查用户不传路径时，命令是否会使用项目默认路径。

    参数含义：
        无。

    返回值含义：
        None。
    """

    args = build_argument_parser().parse_args([])

    assert args.current_report == DEFAULT_CURRENT_REPORT_PATH
    assert args.baseline == DEFAULT_BASELINE_PATH


def test_matching_report_should_write_reports_and_return_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    检查成绩没有退步时，命令是否生成报告并返回 0。

    功能：
        准备一份 100% 的当前报告和一份 100% 的历史基线，执行比较后
        检查两个输出文件存在、退出码为 0、终端显示 passed: True。

    参数含义：
        tmp_path:
            测试输入和输出文件使用的临时目录。
        capsys:
            pytest 用来读取 print 输出的工具。

    返回值含义：
        None。
    """

    current_path, baseline_path = write_comparison_inputs(
        tmp_path,
        current_pass_rate=1.0,
    )
    json_output = tmp_path / "output" / "regression.json"
    markdown_output = tmp_path / "output" / "regression.md"

    exit_code = run_baseline_comparison(
        current_report_path=current_path,
        baseline_path=baseline_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert exit_code == 0
    assert json_output.is_file()
    assert markdown_output.is_file()
    assert "passed: True" in capsys.readouterr().out


def test_regressed_report_should_return_one_and_explain_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    检查当前成绩下降时，命令是否返回 1 并说明原因。

    功能：
        准备一份 0% 的当前报告和一份 100% 的历史基线，执行比较后
        检查退出码为 1，并确认终端打印“发现以下质量回退”。

    参数含义：
        tmp_path:
            测试输入和输出文件使用的临时目录。
        capsys:
            pytest 用来读取 print 输出的工具。

    返回值含义：
        None。
    """

    current_path, baseline_path = write_comparison_inputs(
        tmp_path,
        current_pass_rate=0.0,
    )

    exit_code = run_baseline_comparison(
        current_report_path=current_path,
        baseline_path=baseline_path,
        json_output_path=tmp_path / "regression.json",
        markdown_output_path=tmp_path / "regression.md",
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "passed: False" in output
    assert "发现以下质量回退" in output
