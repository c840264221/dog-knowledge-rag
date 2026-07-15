from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.evaluation.runner import (
    EvaluationCategoryRun,
    EvaluationSuiteRun,
)
from src.evaluation.schemas import (
    AgentEvaluationResult,
    EvaluationCategorySummary,
    EvaluationMetricGateResult,
    EvaluationQualityGate,
    EvaluationSuiteReport,
)


@dataclass(frozen=True)
class EvaluationMetricThreshold:
    """
    定义一项类别总体指标的质量门禁阈值。

    参数含义：
        category:
            指标所属评估类别。
        metric_name:
            需要判断的类别专属指标名称。
        operator:
            gte 表示最低要求；lte 表示最高允许值。
        threshold:
            当前指标的门禁阈值。

    返回值含义：
        EvaluationMetricThreshold:
            报告构建器使用的只读总体指标规则。
    """

    category: str
    metric_name: str
    operator: Literal["gte", "lte"]
    threshold: float


DEFAULT_RAG_METRIC_THRESHOLDS = (
    EvaluationMetricThreshold(
        category="rag_retrieval_behavior",
        metric_name="hit_at_k",
        operator="gte",
        threshold=1.0,
    ),
    EvaluationMetricThreshold(
        category="rag_retrieval_behavior",
        metric_name="top1_accuracy",
        operator="gte",
        threshold=0.8,
    ),
    EvaluationMetricThreshold(
        category="rag_retrieval_behavior",
        metric_name="filter_match_rate",
        operator="gte",
        threshold=1.0,
    ),
    EvaluationMetricThreshold(
        category="rag_retrieval_behavior",
        metric_name="empty_retrieval_rate",
        operator="lte",
        threshold=0.0,
    ),
)


@dataclass(frozen=True)
class EvaluationQualityGatePolicy:
    """
    定义统一评估使用的 Quality Gate Policy（质量门禁策略）。

    参数含义：
        policy_name:
            门禁策略名称。
        required_overall_pass_rate:
            最低整体通过率。
        require_all_categories_passed:
            是否要求每个类别全部通过。
        maximum_error_cases:
            最多允许的执行异常数量。
        require_checks:
            是否要求每条结果至少包含一个检查项。
        metric_thresholds:
            需要参与门禁的类别专业总体指标规则。

    返回值含义：
        EvaluationQualityGatePolicy:
            报告构建器执行门禁判断时使用的只读规则。
    """

    policy_name: str = "v112_strict"
    required_overall_pass_rate: float = 1.0
    require_all_categories_passed: bool = True
    maximum_error_cases: int = 0
    require_checks: bool = True
    metric_thresholds: tuple[EvaluationMetricThreshold, ...] = (
        DEFAULT_RAG_METRIC_THRESHOLDS
    )


DEFAULT_QUALITY_GATE_POLICY = EvaluationQualityGatePolicy()


def _build_category_metrics(
    category: str,
    results: list[AgentEvaluationResult],
) -> dict[str, Any]:
    """
    计算指定评估类别的专属汇总指标。

    功能：
        为通用类别保留空指标；为真实 RAG 检索类别统计 Top K 命中率、
        Top1 准确率、过滤条件匹配率和空召回率，保留旧 V1.6 报告能力。

    参数含义：
        category:
            当前评估类别名称。
        results:
            当前类别包含的所有统一单条评估结果。

    返回值含义：
        dict[str, Any]:
            当前类别专属指标；没有专属指标时返回空字典。
    """

    if category != "rag_retrieval_behavior" or not results:
        return {}

    total_cases = len(results)
    return {
        "hit_at_k": sum(
            1 for result in results if result.output.get("hit_at_k") is True
        ) / total_cases,
        "top1_accuracy": sum(
            1 for result in results if result.output.get("top1_hit") is True
        ) / total_cases,
        "filter_match_rate": sum(
            1
            for result in results
            if result.output.get("filter_matched") is True
        ) / total_cases,
        "empty_retrieval_rate": sum(
            1
            for result in results
            if result.output.get("empty_retrieval") is True
        ) / total_cases,
    }


def build_evaluation_category_summary(
    category_run: EvaluationCategoryRun,
) -> EvaluationCategorySummary:
    """
    根据一门评估科目的原始结果计算类别汇总。

    参数含义：
        category_run:
            Runner（运行器）产生的类别执行记录。

    返回值含义：
        EvaluationCategorySummary:
            当前类别的用例数量、通过率、耗时和失败编号。
    """

    results = list(category_run.results)
    total_cases = len(results)
    passed_results = [result for result in results if result.passed]
    failed_results = [result for result in results if not result.passed]
    error_results = [
        result
        for result in results
        if result.error_message is not None
    ]
    latencies = [
        float(result.latency_ms)
        for result in results
        if result.latency_ms is not None
    ]

    return EvaluationCategorySummary(
        category=category_run.target.category,
        dataset_path=category_run.target.dataset_path.as_posix(),
        total_cases=total_cases,
        passed_cases=len(passed_results),
        failed_cases=len(failed_results),
        error_cases=len(error_results),
        pass_rate=(
            len(passed_results) / total_cases
            if total_cases
            else 0.0
        ),
        average_latency_ms=(
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        failed_case_ids=[result.case_id for result in failed_results],
        error_case_ids=[result.case_id for result in error_results],
        execution_error=category_run.error_message,
        metrics=_build_category_metrics(
            category=category_run.target.category,
            results=results,
        ),
    )


def build_evaluation_quality_gate(
    category_summaries: list[EvaluationCategorySummary],
    results: list[AgentEvaluationResult],
    runner_errors: list[str],
    policy: EvaluationQualityGatePolicy = DEFAULT_QUALITY_GATE_POLICY,
) -> EvaluationQualityGate:
    """
    根据汇总成绩和策略生成质量门禁结论。

    参数含义：
        category_summaries:
            所有评估类别的汇总成绩。
        results:
            所有单条评估结果。
        runner_errors:
            数据集加载或类别执行产生的套件错误。
        policy:
            当前使用的质量门禁策略。

    返回值含义：
        EvaluationQualityGate:
            是否通过、失败类别和所有违规原因。
    """

    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    actual_pass_rate = (
        passed_cases / total_cases
        if total_cases
        else 0.0
    )
    result_error_cases = sum(
        1
        for result in results
        if result.error_message is not None
    )
    actual_error_cases = result_error_cases + len(runner_errors)
    failed_categories = sorted(
        {
            summary.category
            for summary in category_summaries
            if (
                summary.total_cases == 0
                or summary.failed_cases > 0
                or summary.execution_error is not None
            )
        }
    )
    violations: list[str] = []
    summaries_by_category = {
        summary.category: summary
        for summary in category_summaries
    }
    metric_results: list[EvaluationMetricGateResult] = []

    if not total_cases:
        violations.append("整套评估没有产生任何可检查用例。")

    if actual_pass_rate < policy.required_overall_pass_rate:
        violations.append(
            f"整体通过率 {actual_pass_rate:.2%}，低于要求的 "
            f"{policy.required_overall_pass_rate:.2%}。"
        )

    if policy.require_all_categories_passed and failed_categories:
        violations.append(
            "以下评估类别没有全部通过："
            + "、".join(failed_categories)
            + "。"
        )

    if actual_error_cases > policy.maximum_error_cases:
        violations.append(
            f"执行异常数量为 {actual_error_cases}，超过允许值 "
            f"{policy.maximum_error_cases}。"
        )

    if policy.require_checks:
        missing_check_case_ids = [
            result.case_id
            for result in results
            if not result.checks
        ]
        if missing_check_case_ids:
            violations.append(
                "以下用例没有产生结构化 checks（检查项）："
                + "、".join(missing_check_case_ids)
                + "。"
            )

    for threshold in policy.metric_thresholds:
        summary = summaries_by_category.get(threshold.category)
        if summary is None:
            continue

        raw_actual = summary.metrics.get(threshold.metric_name)
        actual = (
            float(raw_actual)
            if isinstance(raw_actual, (int, float))
            and not isinstance(raw_actual, bool)
            else None
        )
        if actual is None:
            passed = False
        elif threshold.operator == "gte":
            passed = actual >= threshold.threshold
        else:
            passed = actual <= threshold.threshold

        operator_text = "大于等于" if threshold.operator == "gte" else "小于等于"
        actual_text = "缺失" if actual is None else f"{actual:.2%}"
        message = (
            f"{threshold.category}.{threshold.metric_name} 实际值 "
            f"{actual_text}，要求{operator_text} {threshold.threshold:.2%}。"
        )
        metric_results.append(
            EvaluationMetricGateResult(
                category=threshold.category,
                metric_name=threshold.metric_name,
                operator=threshold.operator,
                threshold=threshold.threshold,
                actual=actual,
                passed=passed,
                message=message,
            )
        )
        if not passed:
            violations.append(message)
            if threshold.category not in failed_categories:
                failed_categories.append(threshold.category)

    failed_categories.sort()

    violations.extend(runner_errors)

    return EvaluationQualityGate(
        policy_name=policy.policy_name,
        passed=not violations,
        required_overall_pass_rate=policy.required_overall_pass_rate,
        actual_overall_pass_rate=actual_pass_rate,
        require_all_categories_passed=policy.require_all_categories_passed,
        maximum_error_cases=policy.maximum_error_cases,
        actual_error_cases=actual_error_cases,
        require_checks=policy.require_checks,
        failed_categories=failed_categories,
        violations=violations,
        metric_results=metric_results,
    )


def build_evaluation_suite_report(
    suite_run: EvaluationSuiteRun,
    suite_name: str = "dog_agent_v112_evaluation",
    version: str = "V1.12.9",
    policy: EvaluationQualityGatePolicy = DEFAULT_QUALITY_GATE_POLICY,
    metadata: dict[str, Any] | None = None,
) -> EvaluationSuiteReport:
    """
    将统一运行记录转换成完整评估成绩单。

    参数含义：
        suite_run:
            EvaluationSuiteRunner 返回的整套原始运行记录。
        suite_name:
            报告中的评估套件名称。
        version:
            当前评估体系版本。
        policy:
            质量门禁策略。
        metadata:
            执行环境等可选扩展信息。

    返回值含义：
        EvaluationSuiteReport:
            包含总体成绩、分类成绩、门禁结论和单条结果的完整报告。
    """

    category_summaries = [
        build_evaluation_category_summary(category_run)
        for category_run in suite_run.category_runs
    ]
    results = [
        result
        for category_run in suite_run.category_runs
        for result in category_run.results
    ]
    runner_errors = [
        f"{category_run.target.category} 执行失败："
        f"{category_run.error_message}"
        for category_run in suite_run.category_runs
        if category_run.error_message
    ]
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases
    error_cases = sum(
        1
        for result in results
        if result.error_message is not None
    )
    quality_gate = build_evaluation_quality_gate(
        category_summaries=category_summaries,
        results=results,
        runner_errors=runner_errors,
        policy=policy,
    )

    return EvaluationSuiteReport(
        suite_name=suite_name,
        version=version,
        generated_at=datetime.now(timezone.utc),
        duration_ms=suite_run.duration_ms,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        error_cases=error_cases,
        pass_rate=(
            passed_cases / total_cases
            if total_cases
            else 0.0
        ),
        category_summaries=category_summaries,
        quality_gate=quality_gate,
        results=results,
        runner_errors=runner_errors,
        metadata=dict(metadata or {}),
    )


def render_evaluation_report_markdown(
    report: EvaluationSuiteReport,
) -> str:
    """
    将完整评估报告渲染成人工可读的 Markdown 文本。

    参数含义：
        report:
            已构建完成的整套评估报告。

    返回值含义：
        str:
            包含总体成绩、分类成绩、门禁和失败详情的 Markdown 文本。
    """

    lines = [
        f"# {report.version} Agent Evaluation Report",
        "",
        "## Overall Summary",
        "",
        f"- suite_name: {report.suite_name}",
        f"- generated_at: {report.generated_at.isoformat()}",
        f"- duration_ms: {report.duration_ms:.2f}",
        f"- total_cases: {report.total_cases}",
        f"- passed_cases: {report.passed_cases}",
        f"- failed_cases: {report.failed_cases}",
        f"- error_cases: {report.error_cases}",
        f"- pass_rate: {report.pass_rate:.2%}",
        "",
        "## Category Summaries",
        "",
        "| Category | Total | Passed | Failed | Errors | Pass Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in report.category_summaries:
        lines.append(
            f"| {summary.category} | {summary.total_cases} | "
            f"{summary.passed_cases} | {summary.failed_cases} | "
            f"{summary.error_cases} | {summary.pass_rate:.2%} |"
        )

    category_metrics = [
        summary
        for summary in report.category_summaries
        if summary.metrics
    ]
    if category_metrics:
        lines.extend(["", "## Category Metrics", ""])
        for summary in category_metrics:
            lines.append(f"### {summary.category}")
            lines.extend(
                f"- {metric_name}: "
                f"{metric_value:.2%}"
                if isinstance(metric_value, float)
                else f"- {metric_name}: {metric_value}"
                for metric_name, metric_value in summary.metrics.items()
            )

    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            f"- policy: {report.quality_gate.policy_name}",
            f"- passed: {report.quality_gate.passed}",
        ]
    )
    if report.quality_gate.violations:
        lines.append("- violations:")
        lines.extend(
            f"  - {violation}"
            for violation in report.quality_gate.violations
        )
    else:
        lines.append("- violations: none")

    if report.quality_gate.metric_results:
        lines.extend(["", "### Metric Thresholds", ""])
        lines.extend(
            "- "
            + metric_result.message
            + (" 通过" if metric_result.passed else " 未通过")
            for metric_result in report.quality_gate.metric_results
        )

    lines.extend(["", "## Runner Errors", ""])
    if report.runner_errors:
        lines.extend(f"- {error}" for error in report.runner_errors)
    else:
        lines.append("none")

    failed_results = [result for result in report.results if not result.passed]
    lines.extend(["", "## Failed Cases", ""])
    if not failed_results:
        lines.append("全部评估用例通过。")
    else:
        for result in failed_results:
            lines.append(f"### {result.case_id}")
            if result.error_message:
                lines.append(f"- error: {result.error_message}")
            for check in result.failed_checks():
                lines.append(
                    f"- {check.check_name}: expected={check.expected!r}, "
                    f"actual={check.actual!r}, message={check.message}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_evaluation_suite_report(
    report: EvaluationSuiteReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """
    将完整评估报告同时写入 JSON 和 Markdown 文件。

    参数含义：
        report:
            需要落盘的完整评估报告。
        json_path:
            JSON 机器可读报告路径。
        markdown_path:
            Markdown 人工可读报告路径。

    返回值含义：
        None。
    """

    resolved_json_path = Path(json_path)
    resolved_markdown_path = Path(markdown_path)
    resolved_json_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    resolved_markdown_path.write_text(
        render_evaluation_report_markdown(report),
        encoding="utf-8",
    )
