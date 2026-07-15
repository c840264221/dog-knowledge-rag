from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.rag.evaluation import OfflineRetrievalEvaluator, RagEvalCase
from src.rag.evaluation.schemas import RagEvalResult


RetrievalEvaluatorBuilder = Callable[[], OfflineRetrievalEvaluator]

SUPPORTED_EXPECTED_FIELDS = {
    "expected_dog_names",
    "expected_filters",
    "top1_hit",
    "empty_retrieval",
    "quality_is_usable",
}


def build_real_rag_retrieval_evaluator() -> OfflineRetrievalEvaluator:
    """
    构建连接真实 Parser 和本地 Chroma 的 RAG 检索评估器。

    功能：
        从 Runtime Container（运行时容器）取得真实向量库，创建真实
        DogQueryFilterParser 和 MetadataFilterRetriever，并注入旧版
        OfflineRetrievalEvaluator 复用既有 RAG 指标判断逻辑。质量可用性
        始终保留在输出中，是否作为硬性检查由每条统一黄金用例决定。

    参数含义：
        无。

    返回值含义：
        OfflineRetrievalEvaluator:
            会执行真实查询解析和真实本地知识库检索的离线评估器。
    """

    from src.rag.query_parsers import DogQueryFilterParser
    from src.rag.retrievers import MetadataFilterRetriever
    from src.runtime.container.init import container

    parser = DogQueryFilterParser()
    vectorstore_provider = container.get("vectorstore")
    retriever = MetadataFilterRetriever(vectorstore_provider.db)

    def parse_query(eval_case: RagEvalCase) -> Any:
        """
        使用真实 DogQueryFilterParser 解析一条旧 RAG 评估用例。

        参数含义：
            eval_case:
                包含用户问题和 top_k 的旧 RAG 评估用例。

        返回值含义：
            Any:
                Parser 生成的标准 RagQuery 查询对象。
        """

        return parser.parse(
            question=eval_case.question,
            top_k=eval_case.top_k,
        )

    return OfflineRetrievalEvaluator(
        parse_query_func=parse_query,
        retrieve_context_func=retriever.retrieve,
        require_quality_usable=False,
    )


class RagRetrievalBehaviorEvaluator:
    """
    将真实 RAG 检索成绩适配成统一 Agent Evaluation 成绩。

    功能：
        把 AgentEvaluationCase 转换成旧 RAG 评估用例，复用真实
        OfflineRetrievalEvaluator 执行 Parser、Retriever 和 Chroma 检索，
        再把 RagEvalResult 转换成统一检查项和单条评估结果。

    参数含义：
        retrieval_evaluator_builder:
            创建底层 RAG 检索评估器的函数；测试时可注入确定性替身。

    返回值含义：
        RagRetrievalBehaviorEvaluator:
            支持统一 Runner 异步调用的真实 RAG 评估适配器。
    """

    def __init__(
        self,
        retrieval_evaluator_builder: RetrievalEvaluatorBuilder = (
            build_real_rag_retrieval_evaluator
        ),
    ) -> None:
        self.retrieval_evaluator = retrieval_evaluator_builder()

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条真实 RAG 检索评估用例。

        功能：
            校验统一用例、转换成 RagEvalCase，并在线程中执行同步的真实
            检索评估，避免阻塞统一 Runner 的异步事件循环。

        参数含义：
            eval_case:
                包含期望犬种、过滤条件和 top_k 的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                包含真实召回摘要和逐项检查结果的统一单条成绩。
        """

        try:
            self._validate_case(eval_case)
            rag_eval_case = self._build_rag_eval_case(eval_case)
            rag_result = await asyncio.to_thread(
                self.retrieval_evaluator.evaluate_case,
                rag_eval_case,
            )
            return self._build_agent_result(eval_case, rag_result)
        except Exception as exc:
            return AgentEvaluationResult(
                case_id=eval_case.case_id,
                category=eval_case.category,
                checks=[],
                error_message=str(exc),
                metadata={
                    "evaluator": type(self).__name__,
                    "external_dependencies": "real_local_rag",
                },
            )

    async def evaluate_many(
        self,
        eval_cases: list[AgentEvaluationCase],
    ) -> list[AgentEvaluationResult]:
        """
        按黄金集顺序批量执行真实 RAG 检索评估。

        参数含义：
            eval_cases:
                待执行的统一 RAG 检索评估用例列表。

        返回值含义：
            list[AgentEvaluationResult]:
                与输入顺序一致的统一评估结果列表。
        """

        results: list[AgentEvaluationResult] = []
        for eval_case in eval_cases:
            results.append(await self.evaluate_case(eval_case))
        return results

    def _validate_case(self, eval_case: AgentEvaluationCase) -> None:
        """
        校验 RAG 评估类别和支持的黄金期望字段。

        参数含义：
            eval_case:
                当前统一评估用例。

        返回值含义：
            None；类别或 expected 字段不合法时抛出 ValueError。
        """

        if eval_case.category != "rag_retrieval_behavior":
            raise ValueError(
                "RagRetrievalBehaviorEvaluator 只接受 "
                "category=rag_retrieval_behavior"
            )

        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "RAG 检索评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

        if "expected_dog_names" not in eval_case.expected:
            raise ValueError("RAG 检索评估必须声明 expected_dog_names")
        if "expected_filters" not in eval_case.expected:
            raise ValueError("RAG 检索评估必须声明 expected_filters")

    def _build_rag_eval_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> RagEvalCase:
        """
        将统一评估用例转换成旧 RAG 评估用例。

        参数含义：
            eval_case:
                已通过类别和字段校验的统一评估用例。

        返回值含义：
            RagEvalCase:
                可交给 OfflineRetrievalEvaluator 执行的旧 RAG 用例。
        """

        return RagEvalCase(
            case_id=eval_case.case_id,
            question=eval_case.question,
            expected_dog_names=list(
                eval_case.expected["expected_dog_names"]
            ),
            expected_filters=dict(eval_case.expected["expected_filters"]),
            top_k=int(eval_case.input_state.get("top_k", 5)),
            tags=list(eval_case.tags),
            note=eval_case.note,
            metadata=dict(eval_case.metadata),
        )

    def _build_agent_result(
        self,
        eval_case: AgentEvaluationCase,
        rag_result: RagEvalResult,
    ) -> AgentEvaluationResult:
        """
        将旧 RAG 单条成绩转换成统一评估结果。

        参数含义：
            eval_case:
                声明黄金期望的统一评估用例。
            rag_result:
                OfflineRetrievalEvaluator 返回的真实检索成绩。

        返回值含义：
            AgentEvaluationResult:
                包含统一 checks、output 和 metadata 的单条成绩。
        """

        output = {
            "parsed_filters": dict(rag_result.parsed_filters),
            "retrieved_dog_names": list(rag_result.retrieved_dog_names),
            "retrieved_items": [
                item.model_dump(mode="python")
                for item in rag_result.retrieved_items
            ],
            "hit_at_k": rag_result.hit,
            "hit_rank": rag_result.hit_rank,
            "top1_hit": rag_result.top1_hit,
            "filter_matched": rag_result.filter_matched,
            "empty_retrieval": rag_result.empty_retrieval,
            "quality_is_usable": bool(
                rag_result.extra.get("quality_is_usable", False)
            ),
            "quality_status": rag_result.extra.get("quality_status", ""),
            "quality_score": rag_result.extra.get("quality_score"),
            "quality_failure_type": rag_result.extra.get(
                "quality_failure_type",
                "",
            ),
            "quality_reasons": list(
                rag_result.extra.get("quality_reasons", [])
            ),
            "quality_metrics": dict(
                rag_result.extra.get("quality_metrics", {})
            ),
        }
        checks = self._build_checks(eval_case.expected, output)
        return AgentEvaluationResult(
            case_id=eval_case.case_id,
            category=eval_case.category,
            checks=checks,
            latency_ms=rag_result.latency_ms,
            output=output,
            error_message=rag_result.error_message,
            metadata={
                "evaluator": type(self).__name__,
                "delegate_evaluator": type(
                    self.retrieval_evaluator
                ).__name__,
                "parser": "DogQueryFilterParser",
                "retriever": "MetadataFilterRetriever",
                "vector_store": "Chroma",
                "external_dependencies": "real_local_rag",
            },
        )

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        根据 RAG 黄金期望生成统一检查项。

        参数含义：
            expected:
                期望犬种、期望过滤条件及可选质量要求。
            output:
                真实 Parser 和 Retriever 产生的标准摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                犬种命中、过滤条件和可选质量要求对应的检查结果。
        """

        checks = [
            self._build_check(
                check_name="expected_dog_names",
                expected=expected["expected_dog_names"],
                actual=output["retrieved_dog_names"],
                passed=bool(output["hit_at_k"]),
            ),
            self._build_check(
                check_name="expected_filters",
                expected=expected["expected_filters"],
                actual=output["parsed_filters"],
                passed=bool(output["filter_matched"]),
            ),
        ]

        for field_name in (
            "top1_hit",
            "empty_retrieval",
            "quality_is_usable",
        ):
            if field_name not in expected:
                continue
            actual = output.get(field_name)
            checks.append(
                self._build_check(
                    check_name=field_name,
                    expected=expected[field_name],
                    actual=actual,
                    passed=actual == expected[field_name],
                )
            )
        return checks

    def _build_check(
        self,
        check_name: str,
        expected: Any,
        actual: Any,
        passed: bool,
    ) -> EvaluationCheckResult:
        """
        创建一个统一 RAG 检索检查结果。

        参数含义：
            check_name:
                当前检查项名称。
            expected:
                黄金用例声明的期望值。
            actual:
                真实检索链路产生的实际值。
            passed:
                当前比较是否通过。

        返回值含义：
            EvaluationCheckResult:
                包含期望、实际值和中文说明的结构化检查结果。
        """

        return EvaluationCheckResult(
            check_name=check_name,
            passed=passed,
            expected=expected,
            actual=actual,
            message=(
                f"{check_name} 符合预期。"
                if passed
                else f"{check_name} 不符合预期。"
            ),
        )
