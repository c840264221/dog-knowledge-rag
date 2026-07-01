from __future__ import annotations

from src.rag.evaluation.schemas import (
    RagEvalMetrics,
    RagEvalResult,
)


class RagEvalMetricsCalculator:
    """
    RAG 评估指标计算器。

    用于将多条 RagEvalResult 汇总成 RagEvalMetrics。

    参数含义：
        无。

    返回值含义：
        RagEvalMetricsCalculator 实例。

    设计说明：
        MetricsCalculator 只负责统计指标，不负责执行检索、不负责解析问题、
        不负责生成 Markdown 报告，避免和 Evaluator、ReportWriter 职责混在一起。

    专业名词：
        Metrics：指标，用来量化系统质量。
        Calculator：计算器，这里指专门负责计算评估指标的组件。
    """

    def calculate(
        self,
        results: list[RagEvalResult],
    ) -> RagEvalMetrics:
        """
        计算 RAG 评估汇总指标。

        参数含义：
            results:
                多条 RAG 评估结果。
                每一条 RagEvalResult 代表一个评估 case 的执行结果。

        返回值含义：
            RagEvalMetrics:
                汇总后的评估指标，包括总用例数、通过数、失败数、
                hit@k、top1_accuracy、filter_match_rate、
                empty_retrieval_rate、average_latency_ms。

        计算规则：
            total_cases:
                results 的总数量。

            passed_cases:
                result.is_successful() 为 True 的数量。

            failed_cases:
                total_cases - passed_cases。

            hit_at_k:
                hit=True 的数量 / total_cases。

            top1_accuracy:
                top1_hit=True 的数量 / total_cases。

            filter_match_rate:
                filter_matched=True 的数量 / total_cases。

            empty_retrieval_rate:
                empty_retrieval=True 的数量 / total_cases。

            average_latency_ms:
                所有非 None latency_ms 的平均值。
                如果没有任何 latency_ms，则返回 None。
        """

        total_cases = len(results)

        if total_cases == 0:
            return RagEvalMetrics()

        passed_cases = self._count_passed_cases(results)
        failed_cases = total_cases - passed_cases

        hit_at_k = self._calculate_boolean_rate(
            results=results,
            field_name="hit",
        )

        top1_accuracy = self._calculate_boolean_rate(
            results=results,
            field_name="top1_hit",
        )

        filter_match_rate = self._calculate_boolean_rate(
            results=results,
            field_name="filter_matched",
        )

        empty_retrieval_rate = self._calculate_boolean_rate(
            results=results,
            field_name="empty_retrieval",
        )

        average_latency_ms = self._calculate_average_latency_ms(results)

        return RagEvalMetrics(
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            hit_at_k=hit_at_k,
            top1_accuracy=top1_accuracy,
            filter_match_rate=filter_match_rate,
            empty_retrieval_rate=empty_retrieval_rate,
            average_latency_ms=average_latency_ms,
        )

    def _count_passed_cases(
        self,
        results: list[RagEvalResult],
    ) -> int:
        """
        统计通过的评估用例数量。

        参数含义：
            results:
                多条 RAG 评估结果。

        返回值含义：
            int:
                成功通过的评估结果数量。

        说明：
            这里不直接使用 result.passed，
            而是调用 result.is_successful()。
            因为 is_successful 会同时判断：
                1. passed=True
                2. error_message 为空

            这样可以避免某条 case 虽然 passed=True，
            但是执行过程中有 error_message 的异常情况被误判为通过。
        """

        return sum(
            1
            for result in results
            if result.is_successful()
        )

    def _calculate_boolean_rate(
        self,
        results: list[RagEvalResult],
        field_name: str,
    ) -> float:
        """
        计算布尔字段的命中比例。

        参数含义：
            results:
                多条 RAG 评估结果。

            field_name:
                RagEvalResult 中的布尔字段名称。
                例如 hit、top1_hit、filter_matched、empty_retrieval。

        返回值含义：
            float:
                该布尔字段为 True 的数量 / 总数量。
                结果范围为 0.0 到 1.0。

        异常：
            AttributeError:
                当 field_name 在 RagEvalResult 中不存在时抛出。
        """

        if not results:
            return 0.0

        matched_count = sum(
            1
            for result in results
            if bool(getattr(result, field_name))
        )

        return matched_count / len(results)

    def _calculate_average_latency_ms(
        self,
        results: list[RagEvalResult],
    ) -> float | None:
        """
        计算平均评估耗时。

        参数含义：
            results:
                多条 RAG 评估结果。

        返回值含义：
            float | None:
                如果至少有一条结果包含 latency_ms，则返回平均耗时。
                如果所有 latency_ms 都是 None，则返回 None。

        说明：
            latency_ms 是毫秒单位。
            这里只统计非 None 的耗时，避免未记录耗时的 case 影响平均值。
        """

        latency_values = [
            result.latency_ms
            for result in results
            if result.latency_ms is not None
        ]

        if not latency_values:
            return None

        return sum(latency_values) / len(latency_values)


def calculate_rag_eval_metrics(
    results: list[RagEvalResult],
) -> RagEvalMetrics:
    """
    快捷计算 RAG 评估指标。

    这是一个函数式入口，适合调试脚本、ReportWriter 或 Evaluator 直接调用。

    参数含义：
        results:
            多条 RAG 评估结果。

    返回值含义：
        RagEvalMetrics:
            汇总后的 RAG 评估指标。
    """

    calculator = RagEvalMetricsCalculator()

    return calculator.calculate(results)