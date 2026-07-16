from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.baseline import (
    compare_evaluation_report_to_baseline,
    load_evaluation_baseline,
)
from src.evaluation.schemas import (
    EvaluationBaselineMetric,
    EvaluationBaselineSnapshot,
    EvaluationCategorySummary,
    EvaluationQualityGate,
    EvaluationSuiteReport,
)


def build_baseline() -> EvaluationBaselineSnapshot:
    """
    构建回归比较测试使用的最小历史基线。

    功能：
        模拟 V1.12 已验证版本的整体、类别和 RAG 专业指标历史成绩。

    参数含义：
        无。

    返回值含义：
        EvaluationBaselineSnapshot:
            可直接交给回归比较器的测试基线。
    """

    return EvaluationBaselineSnapshot(
        baseline_name="v112_test_baseline",
        source_suite_name="dog_agent_v112_evaluation",
        source_version="V1.12.9",
        overall_pass_rate=1.0,
        category_pass_rates={
            "root_route": 1.0,
            "rag_retrieval_behavior": 1.0,
        },
        metrics=[
            EvaluationBaselineMetric(
                category="rag_retrieval_behavior",
                metric_name="top1_accuracy",
                baseline_value=0.9,
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


def build_current_report(
    *,
    overall_pass_rate: float = 1.0,
    root_pass_rate: float = 1.0,
    rag_pass_rate: float = 1.0,
    top1_accuracy: float | None = 0.9,
    empty_retrieval_rate: float | None = 0.0,
) -> EvaluationSuiteReport:
    """
    构建具有可调成绩的当前统一评估报告。

    功能：
        分别模拟成绩不变、类别退步、指标退步和指标缺失场景。

    参数含义：
        overall_pass_rate:
            当前整套评估通过率。
        root_pass_rate、rag_pass_rate:
            当前 Root 和 RAG 类别通过率。
        top1_accuracy、empty_retrieval_rate:
            当前 RAG 专业指标；None 表示报告中缺少该指标。

    返回值含义：
        EvaluationSuiteReport:
            可交给回归比较器的当前成绩单。
    """

    rag_metrics = {}
    if top1_accuracy is not None:
        rag_metrics["top1_accuracy"] = top1_accuracy
    if empty_retrieval_rate is not None:
        rag_metrics["empty_retrieval_rate"] = empty_retrieval_rate

    return EvaluationSuiteReport(
        suite_name="dog_agent_current_evaluation",
        version="V1.13.0",
        generated_at=datetime.now(timezone.utc),
        total_cases=2,
        passed_cases=round(overall_pass_rate * 2),
        failed_cases=2 - round(overall_pass_rate * 2),
        pass_rate=overall_pass_rate,
        category_summaries=[
            EvaluationCategorySummary(
                category="root_route",
                dataset_path="evaluation/root.json",
                total_cases=1,
                passed_cases=round(root_pass_rate),
                failed_cases=1 - round(root_pass_rate),
                pass_rate=root_pass_rate,
            ),
            EvaluationCategorySummary(
                category="rag_retrieval_behavior",
                dataset_path="evaluation/rag.json",
                total_cases=1,
                passed_cases=round(rag_pass_rate),
                failed_cases=1 - round(rag_pass_rate),
                pass_rate=rag_pass_rate,
                metrics=rag_metrics,
            ),
        ],
        quality_gate=EvaluationQualityGate(
            policy_name="test_policy",
            passed=True,
        ),
    )


def test_v112_baseline_file_should_load_with_expected_metrics() -> None:
    """
    测试仓库中的 V1.12 完整评估基线可以被 Schema 正确加载。

    参数含义：
        无。

    返回值含义：
        None。
    """

    baseline = load_evaluation_baseline(
        Path("evaluation/baselines/v112_full_evaluation_baseline.json")
    )

    assert baseline.overall_pass_rate == 1.0
    assert baseline.category_pass_rates["rag_retrieval_behavior"] == 1.0
    assert {
        metric.metric_name
        for metric in baseline.metrics
    } == {
        "hit_at_k",
        "top1_accuracy",
        "filter_match_rate",
        "empty_retrieval_rate",
    }


def test_current_report_matching_baseline_should_pass() -> None:
    """
    测试当前成绩等于历史基线时整份回归报告通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    regression_report = compare_evaluation_report_to_baseline(
        report=build_current_report(),
        baseline=build_baseline(),
    )

    assert regression_report.passed is True
    assert regression_report.failed_checks() == []
    assert len(regression_report.checks) == 5


def test_category_or_metric_regression_should_fail() -> None:
    """
    测试类别通过率或越高越好的指标下降时回归检测失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    regression_report = compare_evaluation_report_to_baseline(
        report=build_current_report(
            overall_pass_rate=0.5,
            root_pass_rate=0.0,
            top1_accuracy=0.8,
        ),
        baseline=build_baseline(),
    )

    assert regression_report.passed is False
    failed_scopes_and_names = {
        (check.scope, check.metric_name)
        for check in regression_report.failed_checks()
    }
    assert failed_scopes_and_names == {
        ("overall", "overall_pass_rate"),
        ("category", "pass_rate"),
        ("metric", "top1_accuracy"),
    }


def test_lower_is_better_metric_increase_should_fail() -> None:
    """
    测试越低越好的空召回率上升时回归检测失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    regression_report = compare_evaluation_report_to_baseline(
        report=build_current_report(empty_retrieval_rate=0.1),
        baseline=build_baseline(),
    )

    empty_check = next(
        check
        for check in regression_report.failed_checks()
        if check.metric_name == "empty_retrieval_rate"
    )
    assert empty_check.direction == "lower_is_better"
    assert empty_check.delta == 0.1


def test_missing_baseline_metric_should_fail_closed() -> None:
    """
    测试当前报告缺少基线要求的指标时按失败处理。

    功能：
        验证 Fail Closed（故障关闭）原则，防止漏算指标被误判为没有退步。

    参数含义：
        无。

    返回值含义：
        None。
    """

    regression_report = compare_evaluation_report_to_baseline(
        report=build_current_report(top1_accuracy=None),
        baseline=build_baseline(),
    )

    missing_check = next(
        check
        for check in regression_report.failed_checks()
        if check.metric_name == "top1_accuracy"
    )
    assert missing_check.current_value is None
    assert "缺失" in missing_check.message
