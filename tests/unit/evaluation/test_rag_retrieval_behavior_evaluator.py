from __future__ import annotations

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators.rag_retrieval_behavior_evaluator import (
    RagRetrievalBehaviorEvaluator,
)
from src.rag.evaluation.schemas import (
    RagEvalCase,
    RagEvalResult,
    RagEvalRetrievedItem,
)


class FakeOfflineRetrievalEvaluator:
    """
    返回固定真实检索形状结果的离线评估器替身。

    参数含义：
        result:
            每次 evaluate_case 固定返回的旧 RAG 评估结果。

    返回值含义：
        FakeOfflineRetrievalEvaluator:
            支持记录输入用例并返回固定结果的测试替身。
    """

    def __init__(self, result: RagEvalResult) -> None:
        self.result = result
        self.cases: list[RagEvalCase] = []

    def evaluate_case(self, eval_case: RagEvalCase) -> RagEvalResult:
        """
        记录转换后的旧 RAG 用例并返回固定成绩。

        参数含义：
            eval_case:
                统一适配器转换生成的旧 RAG 用例。

        返回值含义：
            RagEvalResult:
                构造函数传入的固定 RAG 检索成绩。
        """

        self.cases.append(eval_case)
        return self.result


def build_eval_case() -> AgentEvaluationCase:
    """
    构建 RAG 统一适配器测试使用的黄金用例。

    参数含义：
        无。

    返回值含义：
        AgentEvaluationCase:
            要求金毛命中、过滤条件匹配和质量可用的用例。
    """

    return AgentEvaluationCase(
        case_id="rag_retrieval_golden_001",
        category="rag_retrieval_behavior",
        question="介绍 Golden Retriever",
        input_state={"top_k": 3},
        expected={
            "expected_dog_names": ["Golden Retriever"],
            "expected_filters": {"dog_name": "Golden Retriever"},
            "top1_hit": True,
            "empty_retrieval": False,
            "quality_is_usable": True,
        },
        tags=["rag", "dog_name"],
    )


def build_rag_result(
    *,
    error_message: str | None = None,
) -> RagEvalResult:
    """
    构建适配器测试使用的旧 RAG 检索结果。

    参数含义：
        error_message:
            可选执行错误；用于覆盖底层检索异常转换逻辑。

    返回值含义：
        RagEvalResult:
            包含单条金毛召回结果的旧 RAG 成绩。
    """

    return RagEvalResult(
        case_id="rag_retrieval_golden_001",
        question="介绍 Golden Retriever",
        expected_dog_names=["Golden Retriever"],
        expected_filters={"dog_name": "Golden Retriever"},
        parsed_filters={
            "dog_name": {"$eq": "Golden Retriever"},
        },
        retrieved_items=[
            RagEvalRetrievedItem(
                rank=1,
                chunk_id="golden-001",
                dog_name="Golden Retriever",
            )
        ],
        retrieved_dog_names=["Golden Retriever"],
        hit=True,
        hit_rank=1,
        top1_hit=True,
        filter_matched=True,
        empty_retrieval=False,
        passed=error_message is None,
        error_message=error_message,
        latency_ms=12.5,
        extra={
            "quality_is_usable": error_message is None,
            "quality_status": "usable" if error_message is None else "failed",
            "quality_score": 0.9 if error_message is None else 0.0,
            "quality_failure_type": "" if error_message is None else "empty",
            "quality_reasons": ["测试质量原因"],
            "quality_metrics": {"chunks_count": 1},
        },
    )


@pytest.mark.asyncio
async def test_should_convert_real_rag_result_to_unified_result() -> None:
    """
    测试旧 RAG 检索成绩会转换成统一检查项和输出摘要。

    参数含义：
        无。

    返回值含义：
        None。
    """

    fake_evaluator = FakeOfflineRetrievalEvaluator(build_rag_result())
    evaluator = RagRetrievalBehaviorEvaluator(
        retrieval_evaluator_builder=lambda: fake_evaluator,
    )

    result = await evaluator.evaluate_case(build_eval_case())

    assert result.passed is True
    assert result.latency_ms == 12.5
    assert result.output["hit_at_k"] is True
    assert result.output["retrieved_dog_names"] == ["Golden Retriever"]
    assert result.output["quality_score"] == 0.9
    assert result.output["quality_reasons"] == ["测试质量原因"]
    assert [check.check_name for check in result.checks] == [
        "expected_dog_names",
        "expected_filters",
        "top1_hit",
        "empty_retrieval",
        "quality_is_usable",
    ]
    assert all(check.passed for check in result.checks)
    assert fake_evaluator.cases[0].top_k == 3
    assert fake_evaluator.cases[0].tags == ["rag", "dog_name"]


@pytest.mark.asyncio
async def test_should_keep_rag_execution_error_in_unified_result() -> None:
    """
    测试底层真实检索错误会保留到统一结果并导致用例失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    fake_evaluator = FakeOfflineRetrievalEvaluator(
        build_rag_result(error_message="Chroma 检索失败")
    )
    evaluator = RagRetrievalBehaviorEvaluator(
        retrieval_evaluator_builder=lambda: fake_evaluator,
    )

    result = await evaluator.evaluate_case(build_eval_case())

    assert result.passed is False
    assert result.error_message == "Chroma 检索失败"


@pytest.mark.asyncio
async def test_should_reject_unsupported_expected_field() -> None:
    """
    测试 RAG 用例声明未知 expected 字段时返回结构化失败结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    fake_evaluator = FakeOfflineRetrievalEvaluator(build_rag_result())
    evaluator = RagRetrievalBehaviorEvaluator(
        retrieval_evaluator_builder=lambda: fake_evaluator,
    )
    eval_case = build_eval_case().model_copy(deep=True)
    eval_case.expected["unknown_field"] = True

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert result.checks == []
    assert "不支持的 expected 字段" in str(result.error_message)
