from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from src.evaluation.schemas import (
    EvaluationBaselineMetric,
    EvaluationBaselineSnapshot,
    EvaluationCategorySummary,
    EvaluationRegressionCheck,
    EvaluationRegressionReport,
    EvaluationSuiteReport,
)


MetricDirection = Literal[
    "higher_is_better",
    "lower_is_better",
]

DEFAULT_BASELINE_METRIC_DIRECTIONS: dict[
    tuple[str, str],
    MetricDirection,
] = {
    # 同名指标可能出现在不同类别中，所以必须用“类别 + 指标名”共同定位。
    ("rag_retrieval_behavior", "hit_at_k"): "higher_is_better",
    ("rag_retrieval_behavior", "top1_accuracy"): "higher_is_better",
    ("rag_retrieval_behavior", "filter_match_rate"): "higher_is_better",
    ("rag_retrieval_behavior", "empty_retrieval_rate"): "lower_is_better",
}


def load_evaluation_baseline(
    baseline_path: str | Path,
) -> EvaluationBaselineSnapshot:
    """
    从 JSON 文件加载并校验统一评估基线。

    功能：
        使用 Pydantic Schema（数据模型）校验基线文件，阻止字段拼写错误、
        非法通过率或未知扩展字段静默进入回归判断。

    参数含义：
        baseline_path:
            需要读取的评估基线 JSON 文件路径。

    返回值含义：
        EvaluationBaselineSnapshot:
            校验通过后的结构化评估基线快照。
    """

    resolved_path = Path(baseline_path)
    return EvaluationBaselineSnapshot.model_validate_json(
        resolved_path.read_text(encoding="utf-8")
    )


def load_evaluation_suite_report(
    report_path: str | Path,
) -> EvaluationSuiteReport:
    """
    读取当前版本已经生成的评估报告。

    功能：
        打开指定的 JSON 报告文件，把文件内容恢复成
        EvaluationSuiteReport 对象。后面的比较函数可以直接从这个对象中
        读取整体通过率、各类别通过率和 RAG 指标。

    参数含义：
        report_path:
            当前评估报告放在哪里，例如
            evaluation/reports/v112_agent_evaluation_report.json。

    返回值含义：
        EvaluationSuiteReport:
            当前版本的完整成绩单，可以继续和 V1.12 历史成绩比较。
    """

    resolved_path = Path(report_path)
    raw_report = json.loads(resolved_path.read_text(encoding="utf-8"))
    raw_results = raw_report.get("results", [])
    # passed 会根据 checks 重新计算，所以读取旧报告时先删掉文件中的旧值。
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if isinstance(raw_result, dict):
                raw_result.pop("passed", None)
    return EvaluationSuiteReport.model_validate(raw_report)


def _build_regression_check(
    *,
    scope: Literal["overall", "category", "metric"],
    category: str | None,
    metric_name: str,
    baseline_value: float,
    current_value: float | None,
    direction: Literal["higher_is_better", "lower_is_better"],
    maximum_regression: float,
) -> EvaluationRegressionCheck:
    """
    比较一项当前成绩和基线成绩并生成结构化检查。

    参数含义：
        scope:
            overall、category 或 metric 比较范围。
        category:
            当前指标所属类别；整体指标为 None。
        metric_name:
            当前参与比较的指标名称。
        baseline_value、current_value:
            基线值和当前值；当前报告缺少指标时 current_value 为 None。
        direction:
            higher_is_better 或 lower_is_better 指标质量方向。
        maximum_regression:
            相对基线最多允许的绝对退步幅度。

    返回值含义：
        EvaluationRegressionCheck:
            包含变化量、判断结论和中文说明的回归检查结果。
    """

    if current_value is None:
        return EvaluationRegressionCheck(
            scope=scope,
            category=category,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=None,
            delta=None,
            direction=direction,
            maximum_regression=maximum_regression,
            passed=False,
            message=f"{metric_name} 在当前评估报告中缺失。",
        )

    delta = current_value - baseline_value
    if direction == "higher_is_better":
        passed = current_value >= baseline_value - maximum_regression
    else:
        passed = current_value <= baseline_value + maximum_regression

    category_prefix = f"{category}." if category else ""
    direction_text = (
        "越高越好"
        if direction == "higher_is_better"
        else "越低越好"
    )
    return EvaluationRegressionCheck(
        scope=scope,
        category=category,
        metric_name=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        delta=delta,
        direction=direction,
        maximum_regression=maximum_regression,
        passed=passed,
        message=(
            f"{category_prefix}{metric_name} 当前值 {current_value:.2%}，"
            f"基线值 {baseline_value:.2%}，指标{direction_text}，"
            f"最多允许退步 {maximum_regression:.2%}。"
        ),
    )


def compare_evaluation_report_to_baseline(
    report: EvaluationSuiteReport,
    baseline: EvaluationBaselineSnapshot,
) -> EvaluationRegressionReport:
    """
    比较当前统一评估报告与已验证版本基线。

    功能：
        检查整体通过率、基线要求的各类别通过率以及专业总体指标；当前报告
        缺少基线类别或指标时按回归失败处理，避免漏跑科目被误判为合格。

    参数含义：
        report:
            当前代码执行后生成的完整统一评估报告。
        baseline:
            已发布或已人工确认版本的评估基线快照。

    返回值含义：
        EvaluationRegressionReport:
            包含全部比较项及总通过结论的结构化回归报告。
    """

    summaries_by_category = {
        summary.category: summary
        for summary in report.category_summaries
    }
    checks = [
        _build_regression_check(
            scope="overall",
            category=None,
            metric_name="overall_pass_rate",
            baseline_value=baseline.overall_pass_rate,
            current_value=report.pass_rate,
            direction="higher_is_better",
            maximum_regression=0.0,
        )
    ]

    for category, baseline_pass_rate in baseline.category_pass_rates.items():
        summary = summaries_by_category.get(category)
        checks.append(
            _build_regression_check(
                scope="category",
                category=category,
                metric_name="pass_rate",
                baseline_value=baseline_pass_rate,
                current_value=(summary.pass_rate if summary else None),
                direction="higher_is_better",
                maximum_regression=0.0,
            )
        )

    for metric in baseline.metrics:
        checks.append(
            _compare_category_metric(
                metric=metric,
                summaries_by_category=summaries_by_category,
            )
        )

    return EvaluationRegressionReport(
        baseline_name=baseline.baseline_name,
        current_suite_name=report.suite_name,
        current_version=report.version,
        checks=checks,
    )


def build_evaluation_baseline_snapshot(
    report: EvaluationSuiteReport,
    current_baseline: EvaluationBaselineSnapshot,
    baseline_name: str,
    *,
    maximum_regression: float = 0.0,
    metric_directions: Mapping[
        tuple[str, str],
        MetricDirection,
    ] = DEFAULT_BASELINE_METRIC_DIRECTIONS,
    metadata: dict[str, Any] | None = None,
) -> EvaluationBaselineSnapshot | None:
    """
    只有当前成绩比正式基线更好时，才生成候选基线。

    功能：
        先确认当前报告包含正式基线要求的所有类别和指标，再检查旧成绩没有
        任何下降。例如正式 Top1 是 90%，当前提高到 95% 时生成候选；如果
        仍是 90%，返回 None，表示没有必要更新；如果降到 85%，直接报错。

    参数含义：
        report:
            准备和正式基线比较的当前完整成绩单。
        current_baseline:
            团队目前正在使用的正式历史基线，用来判断当前成绩是否真正提高。
        baseline_name:
            新基线的名称，例如 v113_full_evaluation_candidate。
        maximum_regression:
            后续版本最多允许比这份基线差多少。0.0 表示不允许下降。
        metric_directions:
            说明每个专业指标是越高越好还是越低越好，不能在这里自动猜测。
        metadata:
            需要额外记录的信息，例如 Git commit 或数据版本。

    返回值含义：
        EvaluationBaselineSnapshot:
            当前成绩至少有一项提高时，返回新的候选基线对象。
        None:
            当前所有成绩和正式基线完全相同时返回，表示不需要生成新文件。
    """

    if not baseline_name.strip():
        raise ValueError("候选基线名称不能为空")
    if maximum_regression < 0.0:
        raise ValueError("maximum_regression 不能小于 0")
    if not report.quality_gate.passed:
        raise ValueError("当前评估没有通过质量门禁，不能生成候选基线")
    if report.runner_errors:
        raise ValueError("当前评估包含 Runner 错误，不能生成候选基线")
    if not report.category_summaries:
        raise ValueError("当前评估没有类别成绩，不能生成候选基线")

    report_categories = {
        summary.category
        for summary in report.category_summaries
    }
    required_categories = set(current_baseline.category_pass_rates)
    missing_categories = sorted(required_categories - report_categories)
    if missing_categories:
        raise ValueError(
            "当前报告缺少正式基线要求的评估类别："
            + "、".join(missing_categories)
        )

    category_pass_rates: dict[str, float] = {}
    metrics: list[EvaluationBaselineMetric] = []
    for summary in report.category_summaries:
        if summary.total_cases == 0:
            raise ValueError(
                f"{summary.category} 没有运行任何用例，不能写入候选基线"
            )
        category_pass_rates[summary.category] = summary.pass_rate
        for metric_name, raw_value in summary.metrics.items():
            direction = metric_directions.get(
                (summary.category, metric_name)
            )
            if direction is None:
                raise ValueError(
                    f"不知道指标 {summary.category}.{metric_name} "
                    "是越高越好还是越低越好"
                )
            if (
                not isinstance(raw_value, (int, float))
                or isinstance(raw_value, bool)
            ):
                raise ValueError(
                    f"指标 {summary.category}.{metric_name} 不是数字"
                )
            metrics.append(
                EvaluationBaselineMetric(
                    category=summary.category,
                    metric_name=metric_name,
                    baseline_value=float(raw_value),
                    direction=direction,
                    maximum_regression=maximum_regression,
                )
            )

    strict_baseline = current_baseline.model_copy(deep=True)
    strict_baseline.metrics = [
        metric.model_copy(update={"maximum_regression": 0.0})
        for metric in current_baseline.metrics
    ]
    regression_report = compare_evaluation_report_to_baseline(
        report=report,
        baseline=strict_baseline,
    )
    if not regression_report.passed:
        failure_messages = "；".join(
            check.message
            for check in regression_report.failed_checks()
        )
        raise ValueError(
            "当前报告缺少正式基线要求的类别或指标，或者成绩发生下降："
            f"{failure_messages}"
        )

    if not _report_has_strict_improvement(
        report=report,
        baseline=current_baseline,
    ):
        return None

    baseline_metadata = dict(metadata or {})
    baseline_metadata.update(
        {
            "source_generated_at": report.generated_at.isoformat(),
            "source_total_cases": report.total_cases,
            "source_quality_gate_policy": report.quality_gate.policy_name,
            "source_quality_gate_passed": report.quality_gate.passed,
        }
    )
    return EvaluationBaselineSnapshot(
        baseline_name=baseline_name.strip(),
        source_suite_name=report.suite_name,
        source_version=report.version,
        overall_pass_rate=report.pass_rate,
        category_pass_rates=category_pass_rates,
        metrics=metrics,
        metadata=baseline_metadata,
    )


def _report_has_strict_improvement(
    *,
    report: EvaluationSuiteReport,
    baseline: EvaluationBaselineSnapshot,
) -> bool:
    """
    检查当前报告是否至少有一项成绩真正优于正式基线。

    功能：
        整体通过率和类别通过率变高算提高；越高越好的指标数值变大算提高；
        越低越好的指标数值变小也算提高。所有数值完全相同时返回 False。

    参数含义：
        report:
            当前版本的完整评估成绩单。
        baseline:
            团队当前使用的正式历史基线。

    返回值含义：
        bool:
            至少一项成绩真正变好时返回 True；全部相同时返回 False。
    """

    if report.pass_rate > baseline.overall_pass_rate:
        return True

    summaries_by_category = {
        summary.category: summary
        for summary in report.category_summaries
    }
    for category, baseline_pass_rate in baseline.category_pass_rates.items():
        summary = summaries_by_category.get(category)
        if summary is not None and summary.pass_rate > baseline_pass_rate:
            return True

    for baseline_metric in baseline.metrics:
        summary = summaries_by_category.get(baseline_metric.category)
        if summary is None:
            continue
        raw_current = summary.metrics.get(baseline_metric.metric_name)
        if (
            not isinstance(raw_current, (int, float))
            or isinstance(raw_current, bool)
        ):
            continue
        current_value = float(raw_current)
        if (
            baseline_metric.direction == "higher_is_better"
            and current_value > baseline_metric.baseline_value
        ):
            return True
        if (
            baseline_metric.direction == "lower_is_better"
            and current_value < baseline_metric.baseline_value
        ):
            return True
    return False


def write_evaluation_baseline_snapshot(
    baseline: EvaluationBaselineSnapshot,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    把候选基线保存成 JSON 文件。

    功能：
        把内存中的候选基线写到指定位置。默认情况下，如果目标文件已经
        存在就停止，避免不小心覆盖团队正在使用的正式历史成绩。

    参数含义：
        baseline:
            已经从合格评估报告中生成的候选基线。
        output_path:
            候选基线 JSON 文件需要保存到哪里。
        overwrite:
            是否允许覆盖已有文件。默认 False，只有明确传入 True 才覆盖。

    返回值含义：
        Path:
            实际写入的候选基线文件路径。
    """

    resolved_path = Path(output_path)
    if resolved_path.exists() and not overwrite:
        raise FileExistsError(
            f"候选基线文件已经存在，未执行覆盖: {resolved_path}"
        )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        baseline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return resolved_path


def _compare_category_metric(
    *,
    metric: EvaluationBaselineMetric,
    summaries_by_category: dict[str, EvaluationCategorySummary],
) -> EvaluationRegressionCheck:
    """
    比较一项类别专业指标与其基线值。

    参数含义：
        metric:
            当前需要判断的专业指标基线配置。
        summaries_by_category:
            当前报告按类别建立的汇总索引。

    返回值含义：
        EvaluationRegressionCheck:
            专业指标存在性和退步幅度对应的检查结果。
    """

    summary = summaries_by_category.get(metric.category)
    raw_current = (
        getattr(summary, "metrics", {}).get(metric.metric_name)
        if summary is not None
        else None
    )
    current_value = (
        float(raw_current)
        if isinstance(raw_current, (int, float))
        and not isinstance(raw_current, bool)
        else None
    )
    return _build_regression_check(
        scope="metric",
        category=metric.category,
        metric_name=metric.metric_name,
        baseline_value=metric.baseline_value,
        current_value=current_value,
        direction=metric.direction,
        maximum_regression=metric.maximum_regression,
    )


def render_evaluation_regression_markdown(
    report: EvaluationRegressionReport,
) -> str:
    """
    把当前成绩和历史成绩的比较结果整理成 Markdown 表格。

    功能：
        例如把“V1.12 Top1 是 90%，当前 Top1 是 80%”整理成一行表格，
        同时标记这一项是否通过。发生退步时，还会在表格下面单独列出原因。

    参数含义：
        report:
            已经计算完成的基线比较结果，里面包含每个指标的新旧成绩。

    返回值含义：
        str:
            一段 Markdown 文本，可以保存成文件，也可以显示在 GitHub
            Actions 的 Summary 页面。
    """

    lines = [
        "# V1.13 Evaluation Regression Report",
        "",
        f"- baseline: {report.baseline_name}",
        f"- current_suite: {report.current_suite_name}",
        f"- current_version: {report.current_version}",
        f"- passed: {report.passed}",
        "",
        "## Regression Checks",
        "",
        "| Scope | Category | Metric | Baseline | Current | Delta | "
        "Direction | Allowed Regression | Passed |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]

    for check in report.checks:
        lines.append(
            f"| {check.scope} | {check.category or '-'} | "
            f"{check.metric_name} | {_format_percentage(check.baseline_value)} | "
            f"{_format_percentage(check.current_value)} | "
            f"{_format_percentage(check.delta, signed=True)} | "
            f"{check.direction} | "
            f"{_format_percentage(check.maximum_regression)} | "
            f"{check.passed} |"
        )

    failed_checks = report.failed_checks()
    lines.extend(["", "## Regressions", ""])
    if failed_checks:
        lines.extend(f"- {check.message}" for check in failed_checks)
    else:
        lines.append("没有发现相对历史基线的质量回退。")

    return "\n".join(lines).rstrip() + "\n"


def _format_percentage(
    value: float | None,
    *,
    signed: bool = False,
) -> str:
    """
    把程序使用的小数成绩转换成容易阅读的百分比。

    功能：
        例如把 0.9 转换成 90.00%，把 -0.1 转换成 -10.00%。如果当前
        报告没有这个指标，则把 None 显示成 missing。

    参数含义：
        value:
            需要显示的成绩小数；None 表示当前报告没有这个指标。
        signed:
            是否显示正负号。例如比较成绩变化时，0.1 显示为 +10.00%。

    返回值含义：
        str:
            转换后的百分比文本；指标缺失时返回 missing。
    """

    if value is None:
        return "missing"
    format_spec = "+.2%" if signed else ".2%"
    return format(value, format_spec)


def write_evaluation_regression_report(
    report: EvaluationRegressionReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """
    把基线比较结果同时保存成 JSON 和 Markdown 文件。

    功能：
        JSON 文件保存完整字段，方便程序和 CI 读取；Markdown 文件保存
        成绩表格，方便开发者查看。如果输出目录不存在，这里会自动创建。

    参数含义：
        report:
            当前成绩和历史成绩的完整比较结果。
        json_path:
            JSON 报告需要保存到哪里。
        markdown_path:
            Markdown 报告需要保存到哪里。

    返回值含义：
        None。函数执行完成后，指定位置会出现两个报告文件。
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
        render_evaluation_regression_markdown(report),
        encoding="utf-8",
    )
