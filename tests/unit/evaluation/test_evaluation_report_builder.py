from pathlib import Path

from src.evaluation import (
    AgentEvaluationResult,
    EvaluationCategorySummary,
    EvaluationCheckResult,
    EvaluationQualityGate,
    EvaluationSuiteReport,
)
from src.evaluation.report_builder import (
    build_evaluation_suite_report,
    render_evaluation_report_markdown,
    write_evaluation_suite_report,
)
from src.evaluation.runner import (
    EvaluationCategoryRun,
    EvaluationSuiteRun,
    EvaluationTarget,
)


def build_result(
    case_id: str,
    passed: bool,
    with_checks: bool = True,
) -> AgentEvaluationResult:
    """
    构建报告测试使用的单条评估结果。

    参数含义：
        case_id:
            测试结果唯一编号。
        passed:
            期望构造通过还是失败的检查项。
        with_checks:
            是否生成结构化检查项。

    返回值含义：
        AgentEvaluationResult:
            可用于类别和套件汇总的测试成绩。
    """

    checks = (
        [
            EvaluationCheckResult(
                check_name="route",
                passed=passed,
                expected="tool_agent",
                actual=("tool_agent" if passed else "general_agent"),
            )
        ]
        if with_checks
        else []
    )
    return AgentEvaluationResult(
        case_id=case_id,
        category="root_route",
        checks=checks,
        latency_ms=10.0,
    )


def build_suite_run(
    results: list[AgentEvaluationResult],
    error_message: str | None = None,
) -> EvaluationSuiteRun:
    """
    构建报告测试使用的整套原始运行记录。

    参数含义：
        results:
            当前类别包含的单条评估结果。
        error_message:
            可选的类别级执行异常。

    返回值含义：
        EvaluationSuiteRun:
            包含一个 RootAgent 类别的原始套件记录。
    """

    target = EvaluationTarget(
        category="root_route",
        dataset_path=Path("evaluation/root.json"),
        evaluator_factory=lambda: None,
    )
    return EvaluationSuiteRun(
        category_runs=[
            EvaluationCategoryRun(
                target=target,
                results=results,
                duration_ms=20.0,
                error_message=error_message,
            )
        ],
        duration_ms=20.0,
    )


def test_evaluation_report_schemas_should_form_complete_scorecard() -> None:
    """
    测试三个报告 Schema 可以组成完整成绩单关系。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_evaluation_suite_report(
        build_suite_run([build_result("root_001", passed=True)])
    )

    assert isinstance(
        report.category_summaries[0],
        EvaluationCategorySummary,
    )
    assert isinstance(report.quality_gate, EvaluationQualityGate)
    assert isinstance(report, EvaluationSuiteReport)
    assert report.category_summaries[0].pass_rate == 1.0
    assert report.quality_gate.passed is True
    assert report.quality_gate.metric_results == []
    assert report.pass_rate == 1.0


def test_quality_gate_should_fail_when_result_has_no_checks() -> None:
    """
    测试单条结果没有 checks 时严格质量门禁不通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_evaluation_suite_report(
        build_suite_run(
            [
                build_result(
                    "root_without_checks_001",
                    passed=False,
                    with_checks=False,
                )
            ]
        )
    )

    assert report.quality_gate.passed is False
    assert any(
        "没有产生结构化 checks" in violation
        for violation in report.quality_gate.violations
    )


def test_report_builder_should_render_and_write_json_and_markdown(
    tmp_path: Path,
) -> None:
    """
    测试完整成绩单可以渲染并写入两种报告格式。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    report = build_evaluation_suite_report(
        build_suite_run([build_result("root_001", passed=True)])
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_evaluation_suite_report(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert '"quality_gate"' in json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Category Summaries" in markdown
    assert "root_route" in markdown
    assert render_evaluation_report_markdown(report) == markdown


def test_rag_category_should_keep_specialized_retrieval_metrics() -> None:
    """
    测试统一类别汇总会保留旧 RAG 报告的专业检索指标。

    参数含义：
        无。

    返回值含义：
        None。
    """

    rag_result = AgentEvaluationResult(
        case_id="rag_001",
        category="rag_retrieval_behavior",
        checks=[
            EvaluationCheckResult(
                check_name="expected_dog_names",
                passed=True,
                expected=["Golden Retriever"],
                actual=["Golden Retriever"],
            )
        ],
        latency_ms=20.0,
        output={
            "hit_at_k": True,
            "top1_hit": True,
            "filter_matched": True,
            "empty_retrieval": False,
        },
    )
    target = EvaluationTarget(
        category="rag_retrieval_behavior",
        dataset_path=Path("evaluation/rag.json"),
        evaluator_factory=lambda: None,
    )
    report = build_evaluation_suite_report(
        EvaluationSuiteRun(
            category_runs=[
                EvaluationCategoryRun(
                    target=target,
                    results=[rag_result],
                )
            ]
        )
    )

    metrics = report.category_summaries[0].metrics
    assert metrics == {
        "hit_at_k": 1.0,
        "top1_accuracy": 1.0,
        "filter_match_rate": 1.0,
        "empty_retrieval_rate": 0.0,
    }
    markdown = render_evaluation_report_markdown(report)
    assert "## Category Metrics" in markdown
    assert "- hit_at_k: 100.00%" in markdown


def test_rag_metric_threshold_should_fail_quality_gate() -> None:
    """
    测试 RAG 单条用例通过但总体 Top1 指标低于阈值时门禁失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    rag_result = AgentEvaluationResult(
        case_id="rag_recommendation_001",
        category="rag_retrieval_behavior",
        checks=[
            EvaluationCheckResult(
                check_name="expected_dog_names",
                passed=True,
                expected=["Golden Retriever"],
                actual=["Golden Retriever"],
            )
        ],
        output={
            "hit_at_k": True,
            "top1_hit": False,
            "filter_matched": True,
            "empty_retrieval": False,
        },
    )
    target = EvaluationTarget(
        category="rag_retrieval_behavior",
        dataset_path=Path("evaluation/rag.json"),
        evaluator_factory=lambda: None,
    )
    report = build_evaluation_suite_report(
        EvaluationSuiteRun(
            category_runs=[
                EvaluationCategoryRun(
                    target=target,
                    results=[rag_result],
                )
            ]
        )
    )

    assert rag_result.passed is True
    assert report.pass_rate == 1.0
    assert report.quality_gate.passed is False
    top1_gate = next(
        metric_result
        for metric_result in report.quality_gate.metric_results
        if metric_result.metric_name == "top1_accuracy"
    )
    assert top1_gate.actual == 0.0
    assert top1_gate.threshold == 0.8
    assert top1_gate.passed is False
    assert report.quality_gate.failed_categories == [
        "rag_retrieval_behavior"
    ]
    assert any(
        "top1_accuracy" in violation
        for violation in report.quality_gate.violations
    )
