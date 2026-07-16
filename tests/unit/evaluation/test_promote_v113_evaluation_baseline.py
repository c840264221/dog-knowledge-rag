from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.evaluation.promote_v113_evaluation_baseline import (
    DEFAULT_BASELINE_NAME,
    DEFAULT_CANDIDATE_OUTPUT_PATH,
    DEFAULT_CURRENT_BASELINE_PATH,
    build_argument_parser,
    run_baseline_promotion,
)
from src.evaluation.baseline import (
    build_evaluation_baseline_snapshot,
    load_evaluation_baseline,
    write_evaluation_baseline_snapshot,
)
from src.evaluation.schemas import (
    EvaluationBaselineMetric,
    EvaluationBaselineSnapshot,
    EvaluationCategorySummary,
    EvaluationQualityGate,
    EvaluationSuiteReport,
)


def build_source_report(
    *,
    gate_passed: bool = True,
    rag_metrics: dict[str, float] | None = None,
) -> EvaluationSuiteReport:
    """
    为候选基线测试准备一份最小完整评估报告。

    功能：
        生成 Root 和 RAG 两个类别的成绩。调用测试可以控制质量门禁是否
        通过，也可以换入未知指标，模拟允许晋升和必须阻止的场景。

    参数含义：
        gate_passed:
            想让这份报告模拟门禁通过还是失败。
        rag_metrics:
            想放进 RAG 类别的指标；不传时使用四个正式 RAG 指标。

    返回值含义：
        EvaluationSuiteReport:
            可以用于生成候选基线的测试成绩单。
    """

    resolved_rag_metrics = (
        {
            "hit_at_k": 1.0,
            "top1_accuracy": 0.9,
            "filter_match_rate": 1.0,
            "empty_retrieval_rate": 0.0,
        }
        if rag_metrics is None
        else rag_metrics
    )
    return EvaluationSuiteReport(
        suite_name="v113_promotion_test",
        version="V1.13.0",
        generated_at=datetime.now(timezone.utc),
        total_cases=2,
        passed_cases=2,
        pass_rate=1.0,
        category_summaries=[
            EvaluationCategorySummary(
                category="root_route",
                dataset_path="evaluation/root.json",
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            EvaluationCategorySummary(
                category="rag_retrieval_behavior",
                dataset_path="evaluation/rag.json",
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
                metrics=resolved_rag_metrics,
            ),
        ],
        quality_gate=EvaluationQualityGate(
            policy_name="promotion_test",
            passed=gate_passed,
        ),
    )


def build_current_baseline() -> EvaluationBaselineSnapshot:
    """
    为候选晋升测试准备团队当前正在使用的正式基线。

    功能：
        规定 Root 和 RAG 是候选报告必须包含的两个类别，并记录四个 RAG
        指标的历史成绩。测试会用它判断当前报告是提高、持平还是下降。

    参数含义：
        无。

    返回值含义：
        EvaluationBaselineSnapshot:
            包含必需类别和历史指标值的测试基线。
    """

    return EvaluationBaselineSnapshot(
        baseline_name="v112_test_baseline",
        source_suite_name="v112_test_suite",
        source_version="V1.12.9",
        overall_pass_rate=1.0,
        category_pass_rates={
            "root_route": 1.0,
            "rag_retrieval_behavior": 1.0,
        },
        metrics=[
            EvaluationBaselineMetric(
                category="rag_retrieval_behavior",
                metric_name="hit_at_k",
                baseline_value=1.0,
                direction="higher_is_better",
            ),
            EvaluationBaselineMetric(
                category="rag_retrieval_behavior",
                metric_name="top1_accuracy",
                baseline_value=0.9,
                direction="higher_is_better",
            ),
            EvaluationBaselineMetric(
                category="rag_retrieval_behavior",
                metric_name="filter_match_rate",
                baseline_value=1.0,
                direction="higher_is_better",
            ),
            EvaluationBaselineMetric(
                category="rag_retrieval_behavior",
                metric_name="empty_retrieval_rate",
                baseline_value=0.0,
                direction="lower_is_better",
            ),
        ],
    )


def test_passing_report_should_become_baseline_candidate() -> None:
    """
    检查通过门禁的报告是否能正确生成候选基线。

    功能：
        从测试报告生成候选对象，然后确认整体成绩、两个类别和四个 RAG
        指标都被保存，空召回率还必须标记为越低越好。

    参数含义：
        无。

    返回值含义：
        None。
    """

    baseline = build_evaluation_baseline_snapshot(
        report=build_source_report(
            rag_metrics={
                "hit_at_k": 1.0,
                "top1_accuracy": 0.95,
                "filter_match_rate": 1.0,
                "empty_retrieval_rate": 0.0,
            }
        ),
        current_baseline=build_current_baseline(),
        baseline_name="v113_candidate",
    )

    assert baseline is not None
    assert baseline.overall_pass_rate == 1.0
    assert baseline.category_pass_rates == {
        "root_route": 1.0,
        "rag_retrieval_behavior": 1.0,
    }
    assert len(baseline.metrics) == 4
    top1_metric = next(
        metric
        for metric in baseline.metrics
        if metric.metric_name == "top1_accuracy"
    )
    assert top1_metric.baseline_value == 0.95
    empty_metric = next(
        metric
        for metric in baseline.metrics
        if metric.metric_name == "empty_retrieval_rate"
    )
    assert empty_metric.direction == "lower_is_better"


def test_failed_quality_gate_should_not_create_candidate() -> None:
    """
    检查没有通过质量门禁的报告是否会被拒绝。

    功能：
        准备一份门禁结果为 False 的报告，确认生成函数直接报错，避免把
        已知不合格的成绩登记成下一版本标准。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="没有通过质量门禁"):
        build_evaluation_baseline_snapshot(
            report=build_source_report(gate_passed=False),
            current_baseline=build_current_baseline(),
            baseline_name="invalid_candidate",
        )


def test_unknown_metric_direction_should_not_be_guessed() -> None:
    """
    检查新指标没有配置好坏方向时是否会停止生成。

    功能：
        放入一个名为 new_metric 的未知指标，确认程序不会猜它越高越好或
        越低越好，而是要求开发者先明确规则。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="不知道指标"):
        build_evaluation_baseline_snapshot(
            report=build_source_report(rag_metrics={"new_metric": 0.5}),
            current_baseline=build_current_baseline(),
            baseline_name="unknown_metric_candidate",
        )


def test_equal_report_should_not_create_candidate() -> None:
    """
    检查当前成绩与正式基线完全相同时是否跳过更新。

    功能：
        使用一份所有数值都与正式基线相同的报告，确认函数返回 None，避免
        生成内容没有变化的候选基线。

    参数含义：
        无。

    返回值含义：
        None。
    """

    baseline = build_evaluation_baseline_snapshot(
        report=build_source_report(),
        current_baseline=build_current_baseline(),
        baseline_name="unchanged_candidate",
    )

    assert baseline is None


def test_missing_required_category_should_block_candidate() -> None:
    """
    检查候选报告缺少正式基线要求的类别时是否被拒绝。

    功能：
        从当前报告中移除 Root 类别，确认即使剩余类别全部通过，也不能生成
        候选基线。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_source_report()
    report.category_summaries = [
        summary
        for summary in report.category_summaries
        if summary.category != "root_route"
    ]

    with pytest.raises(ValueError, match="缺少正式基线要求"):
        build_evaluation_baseline_snapshot(
            report=report,
            current_baseline=build_current_baseline(),
            baseline_name="missing_category_candidate",
        )


def test_regressed_metric_should_block_candidate() -> None:
    """
    检查任一旧指标下降时是否禁止生成候选基线。

    功能：
        模拟 Top1 从正式基线的 90% 降到 85%。即使报告自己的质量门禁为
        通过，也必须拒绝更新历史基线。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_source_report(
        rag_metrics={
            "hit_at_k": 1.0,
            "top1_accuracy": 0.85,
            "filter_match_rate": 1.0,
            "empty_retrieval_rate": 0.0,
        }
    )

    with pytest.raises(ValueError, match="成绩发生下降"):
        build_evaluation_baseline_snapshot(
            report=report,
            current_baseline=build_current_baseline(),
            baseline_name="regressed_candidate",
        )


def test_existing_baseline_file_should_not_be_overwritten_by_default(
    tmp_path: Path,
) -> None:
    """
    检查默认设置是否会保护已经存在的基线文件。

    功能：
        先写入一次候选基线，再向同一路径写第二次，确认第二次抛出
        FileExistsError，并且第一次写入的文件仍然可以正常读取。

    参数含义：
        tmp_path:
            pytest 创建的临时目录，用来保存测试候选文件。

    返回值含义：
        None。
    """

    baseline = build_evaluation_baseline_snapshot(
        report=build_source_report(
            rag_metrics={
                "hit_at_k": 1.0,
                "top1_accuracy": 0.95,
                "filter_match_rate": 1.0,
                "empty_retrieval_rate": 0.0,
            }
        ),
        current_baseline=build_current_baseline(),
        baseline_name="protected_candidate",
    )
    assert baseline is not None
    output_path = tmp_path / "candidate.json"
    write_evaluation_baseline_snapshot(baseline, output_path)

    with pytest.raises(FileExistsError, match="未执行覆盖"):
        write_evaluation_baseline_snapshot(baseline, output_path)

    assert load_evaluation_baseline(output_path).baseline_name == (
        "protected_candidate"
    )


def test_promotion_command_should_write_reviewable_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    检查命令是否会生成候选文件并提醒人工审查。

    功能：
        把合格报告写入临时目录，再执行完整命令流程，确认退出码为 0、
        候选文件存在，并且终端明确显示 candidate_requires_human_review。

    参数含义：
        tmp_path:
            当前报告和候选基线使用的临时目录。
        capsys:
            pytest 用来读取 print 输出的工具。

    返回值含义：
        None。
    """

    report_path = tmp_path / "report.json"
    current_baseline_path = tmp_path / "current_baseline.json"
    output_path = tmp_path / "candidate" / "baseline.json"
    report_path.write_text(
        build_source_report(
            rag_metrics={
                "hit_at_k": 1.0,
                "top1_accuracy": 0.95,
                "filter_match_rate": 1.0,
                "empty_retrieval_rate": 0.0,
            }
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    current_baseline_path.write_text(
        build_current_baseline().model_dump_json(indent=2),
        encoding="utf-8",
    )

    exit_code = run_baseline_promotion(
        current_report_path=report_path,
        current_baseline_path=current_baseline_path,
        baseline_name="v113_command_candidate",
        output_path=output_path,
    )
    output = capsys.readouterr().out
    candidate = load_evaluation_baseline(output_path)

    assert exit_code == 0
    assert output_path.is_file()
    assert candidate.metadata["promotion_requires_review"] is True
    assert "candidate_requires_human_review" in output


def test_promotion_command_should_skip_equal_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    检查命令遇到持平成绩时是否不创建候选文件。

    功能：
        把相同成绩的报告和正式基线写入临时目录，确认命令成功结束但不生成
        文件，并在终端显示 no_update_required。

    参数含义：
        tmp_path:
            当前报告、正式基线和候选文件使用的临时目录。
        capsys:
            pytest 用来读取 print 输出的工具。

    返回值含义：
        None。
    """

    report_path = tmp_path / "report.json"
    current_baseline_path = tmp_path / "current_baseline.json"
    output_path = tmp_path / "candidate.json"
    report_path.write_text(
        build_source_report().model_dump_json(indent=2),
        encoding="utf-8",
    )
    current_baseline_path.write_text(
        build_current_baseline().model_dump_json(indent=2),
        encoding="utf-8",
    )

    exit_code = run_baseline_promotion(
        current_report_path=report_path,
        current_baseline_path=current_baseline_path,
        baseline_name="unchanged_command_candidate",
        output_path=output_path,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert not output_path.exists()
    assert "no_update_required" in output


def test_argument_parser_should_use_safe_candidate_defaults() -> None:
    """
    检查默认命令是否写入 candidates 目录并禁止覆盖。

    参数含义：
        无。

    返回值含义：
        None。
    """

    args = build_argument_parser().parse_args([])

    assert args.baseline_name == DEFAULT_BASELINE_NAME
    assert args.current_baseline == DEFAULT_CURRENT_BASELINE_PATH
    assert args.output == DEFAULT_CANDIDATE_OUTPUT_PATH
    assert args.overwrite is False
