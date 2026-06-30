from __future__ import annotations

import json
import re
from pathlib import Path

from src.rag.evaluation.schemas import (
    RagEvalMetrics,
    RagEvalReport,
    RagEvalResult,
    RagEvalRetrievedItem,
)

from src.settings import settings


class RagEvalReportWriter:
    """
    RAG 评估报告写入器。

    功能：
        将 RagEvalReport 渲染成 Markdown 文本，
        并写入本地 logs/report/rag_evaluate 目录。

    参数含义：
        output_dir:
            报告输出目录。
            默认可以使用 logs/report/rag_evaluate。

    返回值含义：
        RagEvalReportWriter 实例。

    专业名词：
        Report Writer：
            报告写入器，负责把结构化评估结果转换成可读文档。

        Markdown：
            一种轻量级文本格式，适合保存评估报告、技术文档和复习材料。

        Render：
            渲染。这里指把 Python 对象转换成 Markdown 字符串。
    """

    def __init__(
        self,
        output_dir: str | Path = "logs/report/rag_evaluate",
    ) -> None:
        """
        初始化 RAG 评估报告写入器。

        参数含义：
            output_dir:
                Markdown 报告输出目录。

        返回值含义：
            None。
        """

        self.output_dir = Path(output_dir)

    def write(
        self,
        report: RagEvalReport,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        写入 RAG 评估 Markdown 报告。

        参数含义：
            report:
                RAG 评估报告对象，包含 metrics、results、metadata 等信息。

            output_path:
                可选的报告输出路径。
                如果不传，则根据 report.run_id 自动生成文件名。

        返回值含义：
            Path:
                最终写入的 Markdown 报告路径。
        """

        final_output_path = self._resolve_output_path(
            report=report,
            output_path=output_path,
        )

        final_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        markdown_text = self.render(
            report=report,
        )

        final_output_path.write_text(
            markdown_text,
            encoding="utf-8",
        )

        report.report_path = str(
            final_output_path.as_posix()
        )

        return final_output_path

    def render(
        self,
        report: RagEvalReport,
    ) -> str:
        """
        渲染 RAG 评估报告为 Markdown 字符串。

        参数含义：
            report:
                RAG 评估报告对象。

        返回值含义：
            str:
                Markdown 格式的报告文本。
        """

        sections = [
            self._render_title(report),
            self._render_summary(report),
            self._render_metadata(report),
            self._render_metrics(report.metrics),
            self._render_failed_results(report.results),
            self._render_all_results(report.results),
        ]

        return "\n\n".join(
            section
            for section in sections
            if section.strip()
        ) + "\n"

    def _resolve_output_path(
        self,
        report: RagEvalReport,
        output_path: str | Path | None,
    ) -> Path:
        """
        解析最终报告输出路径。

        参数含义：
            report:
                RAG 评估报告对象。

            output_path:
                用户指定的输出路径。

        返回值含义：
            Path:
                最终报告路径。
        """

        if output_path is not None:
            return Path(output_path)

        safe_run_id = self._safe_filename(
            value=report.run_id,
        )

        return self.output_dir / f"{safe_run_id}.md"

    def _safe_filename(
        self,
        value: str,
    ) -> str:
        """
        将字符串转换成安全文件名。

        参数含义：
            value:
                原始字符串，例如 run_id。

        返回值含义：
            str:
                可用于文件名的安全字符串。
        """

        normalized_value = value.strip()

        if not normalized_value:
            return "rag_eval_report"

        return re.sub(
            pattern=r"[^a-zA-Z0-9_\-]+",
            repl="_",
            string=normalized_value,
        )

    def _render_title(
        self,
        report: RagEvalReport,
    ) -> str:
        """
        渲染报告标题区域。

        参数含义：
            report:
                RAG 评估报告对象。

        返回值含义：
            str:
                Markdown 标题内容。
        """

        created_at = report.created_at.isoformat()

        return "\n".join(
            [
                "# RAG Evaluation Report",
                "",
                f"- Run ID: `{report.run_id}`",
                f"- Created At: `{created_at}`",
                f"- Dataset Path: `{report.dataset_path or ''}`",
                f"- Report Path: `{report.report_path or ''}`",
            ]
        )

    def _render_summary(
        self,
        report: RagEvalReport,
    ) -> str:
        """
        渲染评估摘要。

        参数含义：
            report:
                RAG 评估报告对象。

        返回值含义：
            str:
                Markdown 摘要内容。
        """

        metrics = report.metrics

        if metrics.total_cases == 0:
            pass_rate = 0.0
        else:
            pass_rate = metrics.passed_cases / metrics.total_cases

        return "\n".join(
            [
                "## Summary",
                "",
                f"- Total Cases: **{metrics.total_cases}**",
                f"- Passed Cases: **{metrics.passed_cases}**",
                f"- Failed Cases: **{metrics.failed_cases}**",
                f"- Pass Rate: **{self._format_rate(pass_rate)}**",
            ]
        )

    def _render_metadata(
        self,
        report: RagEvalReport,
    ) -> str:
        """
        渲染报告元数据。

        参数含义：
            report:
                RAG 评估报告对象。

        返回值含义：
            str:
                Markdown 元数据内容。
        """

        if not report.metadata:
            return ""

        metadata_json = json.dumps(
            report.metadata,
            ensure_ascii=False,
            indent=2,
        )

        return "\n".join(
            [
                "## Metadata",
                "",
                "```json",
                metadata_json,
                "```",
            ]
        )

    def _render_metrics(
        self,
        metrics: RagEvalMetrics,
    ) -> str:
        """
        渲染汇总指标。

        参数含义：
            metrics:
                RAG 评估汇总指标对象。

        返回值含义：
            str:
                Markdown 指标内容。
        """

        return "\n".join(
            [
                "## Metrics",
                "",
                "| Metric | Value | 中文含义 |",
                "| --- | ---: | --- |",
                f"| total_cases | {metrics.total_cases} | 总评估用例数 |",
                f"| passed_cases | {metrics.passed_cases} | 通过用例数 |",
                f"| failed_cases | {metrics.failed_cases} | 失败用例数 |",
                f"| hit_at_k | {self._format_rate(metrics.hit_at_k)} | 前 k 个召回结果命中率 |",
                f"| top1_accuracy | {self._format_rate(metrics.top1_accuracy)} | 第一名召回结果准确率 |",
                f"| filter_match_rate | {self._format_rate(metrics.filter_match_rate)} | Filter 解析匹配率 |",
                f"| empty_retrieval_rate | {self._format_rate(metrics.empty_retrieval_rate)} | 空召回比例 |",
                f"| average_latency_ms | {self._format_latency(metrics.average_latency_ms)} | 平均耗时 |",
            ]
        )

    def _render_failed_results(
        self,
        results: list[RagEvalResult],
    ) -> str:
        """
        渲染失败用例列表。

        参数含义：
            results:
                所有 RAG 评估结果。

        返回值含义：
            str:
                Markdown 失败用例内容。
        """

        failed_results = [
            result
            for result in results
            if not result.is_successful()
        ]

        if not failed_results:
            return "\n".join(
                [
                    "## Failed Cases",
                    "",
                    "No failed cases.",
                ]
            )

        lines = [
            "## Failed Cases",
            "",
            "| Case ID | Hit | Top1 | Filter Matched | Empty Retrieval | Failure Type | Error |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        for result in failed_results:
            failure_type = str(
                result.extra.get(
                    "quality_failure_type",
                    "",
                )
            )

            error_message = result.error_message or ""

            lines.append(
                "| "
                f"`{result.case_id}` | "
                f"{self._format_bool(result.hit)} | "
                f"{self._format_bool(result.top1_hit)} | "
                f"{self._format_bool(result.filter_matched)} | "
                f"{self._format_bool(result.empty_retrieval)} | "
                f"{failure_type} | "
                f"{self._escape_table_text(error_message)} |"
            )

        return "\n".join(lines)

    def _render_all_results(
        self,
        results: list[RagEvalResult],
    ) -> str:
        """
        渲染所有评估用例详情。

        参数含义：
            results:
                所有 RAG 评估结果。

        返回值含义：
            str:
                Markdown 详情内容。
        """

        lines = [
            "## Case Details",
        ]

        for result in results:
            lines.append(
                self._render_single_result(
                    result=result,
                )
            )

        return "\n\n".join(lines)

    def _render_single_result(
        self,
        result: RagEvalResult,
    ) -> str:
        """
        渲染单条评估结果详情。

        参数含义：
            result:
                单条 RAG 评估结果。

        返回值含义：
            str:
                Markdown 单条详情内容。
        """

        lines = [
            f"### `{result.case_id}`",
            "",
            f"- Question: {result.question}",
            f"- Passed: **{self._format_bool(result.is_successful())}**",
            f"- Hit: **{self._format_bool(result.hit)}**",
            f"- Hit Rank: `{result.hit_rank}`",
            f"- Top1 Hit: **{self._format_bool(result.top1_hit)}**",
            f"- Filter Matched: **{self._format_bool(result.filter_matched)}**",
            f"- Empty Retrieval: **{self._format_bool(result.empty_retrieval)}**",
            f"- Latency: `{self._format_latency(result.latency_ms)}`",
        ]

        if result.error_message:
            lines.append(
                f"- Error: `{result.error_message}`"
            )

        lines.extend(
            [
                "",
                "**Expected Dog Names**",
                "",
                "```json",
                json.dumps(
                    result.expected_dog_names,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "**Retrieved Dog Names**",
                "",
                "```json",
                json.dumps(
                    result.retrieved_dog_names,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "**Expected Filters**",
                "",
                "```json",
                json.dumps(
                    result.expected_filters,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "**Parsed Filters**",
                "",
                "```json",
                json.dumps(
                    result.parsed_filters,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "**Flattened Filter Comparison**",
                "",
                "```json",
                json.dumps(
                    {
                        "expected_filters_flattened": result.extra.get(
                            "expected_filters_flattened",
                            {},
                        ),
                        "parsed_filters_flattened": result.extra.get(
                            "parsed_filters_flattened",
                            {},
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "**Semantic Filter Comparison**",
                "",
                "```json",
                json.dumps(
                    {
                        "expected_filters_semantic": result.extra.get(
                            "expected_filters_semantic",
                            {},
                        ),
                        "parsed_filters_semantic": result.extra.get(
                            "parsed_filters_semantic",
                            {},
                        ),
                        "filter_compare_mode": result.extra.get(
                            "filter_compare_mode",
                            "",
                        ),
                        "filter_matched": result.filter_matched,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

        if result.extra:
            lines.extend(
                [
                    "",
                    "**Quality Extra**",
                    "",
                    "```json",
                    json.dumps(
                        result.extra,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                ]
            )

        if result.retrieved_items:
            lines.extend(
                [
                    "",
                    "**Retrieved Items**",
                    "",
                    self._render_retrieved_items(
                        retrieved_items=result.retrieved_items,
                    ),
                ]
            )

        return "\n".join(lines)

    def _render_retrieved_items(
        self,
        retrieved_items: list[RagEvalRetrievedItem],
    ) -> str:
        """
        渲染召回结果列表。

        参数含义：
            retrieved_items:
                召回结果项列表。

        返回值含义：
            str:
                Markdown 表格内容。
        """

        lines = [
            "| Rank | Dog Name | Score | Section | Source | Preview |",
            "| ---: | --- | ---: | --- | --- | --- |",
        ]

        for item in retrieved_items:
            lines.append(
                "| "
                f"{item.rank} | "
                f"{self._escape_table_text(item.dog_name or '')} | "
                f"{self._format_optional_float(item.score)} | "
                f"{self._escape_table_text(item.section_title or '')} | "
                f"{self._escape_table_text(item.source or '')} | "
                f"{self._escape_table_text(item.content_preview or '')} |"
            )

        return "\n".join(lines)

    def _format_rate(
        self,
        value: float,
    ) -> str:
        """
        格式化比例为百分比。

        参数含义：
            value:
                0 到 1 之间的小数。

        返回值含义：
            str:
                百分比字符串。
        """

        return f"{value * 100:.2f}%"

    def _format_latency(
        self,
        value: float | None,
    ) -> str:
        """
        格式化耗时。

        参数含义：
            value:
                毫秒耗时。

        返回值含义：
            str:
                格式化后的耗时字符串。
        """

        if value is None:
            return "N/A"

        return f"{value:.3f} ms"

    def _format_optional_float(
        self,
        value: float | None,
    ) -> str:
        """
        格式化可选浮点数。

        参数含义：
            value:
                可选浮点数。

        返回值含义：
            str:
                格式化后的字符串。
        """

        if value is None:
            return ""

        return f"{value:.4f}"

    def _format_bool(
        self,
        value: bool,
    ) -> str:
        """
        格式化布尔值。

        参数含义：
            value:
                布尔值。

        返回值含义：
            str:
                True 显示为 YES，False 显示为 NO。
        """

        return "YES" if value else "NO"

    def _escape_table_text(
        self,
        value: str,
    ) -> str:
        """
        转义 Markdown 表格文本。

        参数含义：
            value:
                原始文本。

        返回值含义：
            str:
                转义后的表格文本。
        """

        return (
            value.replace("|", "\\|")
            .replace("\n", " ")
            .strip()
        )


def write_rag_eval_report(
    report: RagEvalReport,
    output_dir: str | Path = settings.path.RAG_EVALUATE_REPORT_DIR,
    output_path: str | Path | None = None,
) -> Path:
    """
    快捷写入 RAG Evaluation Markdown 报告。

    参数含义：
        report:
            RAG 评估报告对象。

        output_dir:
            默认报告输出目录。

        output_path:
            可选的完整输出路径。

    返回值含义：
        Path:
            最终写入的报告路径。
    """

    writer = RagEvalReportWriter(
        output_dir=output_dir,
    )

    return writer.write(
        report=report,
        output_path=output_path,
    )