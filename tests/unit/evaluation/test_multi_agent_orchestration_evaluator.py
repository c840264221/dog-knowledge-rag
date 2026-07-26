from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import MultiAgentOrchestrationEvaluator
from src.evaluation.schemas import AgentEvaluationCase


DATASET_PATH = Path(
    "evaluation/datasets/multi_agent_orchestration_cases.json"
)


@pytest.mark.asyncio
async def test_orchestration_dataset_should_pass_evaluator() -> None:
    """
    检查全部总编排黄金用例是否通过确定性集成评估。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)

    results = await MultiAgentOrchestrationEvaluator().evaluate_many(
        eval_cases
    )

    assert len(results) == 4
    assert all(result.passed for result in results)


@pytest.mark.asyncio
async def test_orchestration_evaluator_should_reject_unknown_expected_field(
) -> None:
    """
    检查总编排评估器是否拒绝未定义的期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id="unknown_orchestration_field",
        category="multi_agent_orchestration",
        question="测试未知字段",
        input_state={},
        expected={"unknown_field": True},
    )

    result = await MultiAgentOrchestrationEvaluator().evaluate_case(
        eval_case
    )

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
