from typing import Any

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import MemoryRecallBehaviorEvaluator


class FakeVectorStoreTrace:
    """
    为记忆评估器测试提供固定向量检索调用轨迹。

    参数含义：
        无。

    返回值含义：
        FakeVectorStoreTrace:
            包含 user_id 和 active 状态过滤条件的轨迹对象。
    """

    def __init__(self) -> None:
        self.calls = [
            {
                "filter": {
                    "$and": [
                        {"user_id": {"$eq": "user_a"}},
                        {"status": {"$eq": "active"}},
                    ]
                }
            }
        ]


class FakeStoreTrace:
    """
    为记忆评估器测试提供固定 SQLite Store 调用轨迹。

    参数含义：
        无。

    返回值含义：
        FakeStoreTrace:
            包含一次回查记录的轨迹对象。
    """

    def __init__(self) -> None:
        self.calls = [{}]


class FakeMemoryRuntime:
    """
    为记忆评估器单元测试返回固定最终状态。

    参数含义：
        result_state:
            invoke 方法需要返回的记忆召回状态。

    返回值含义：
        FakeMemoryRuntime:
            具有固定存储轨迹和异步 invoke 方法的场景对象。
    """

    def __init__(self, result_state: dict[str, Any]) -> None:
        self.result_state = result_state
        self.vector_store = FakeVectorStoreTrace()
        self.store = FakeStoreTrace()
        self.ranker = object()

    async def invoke(self) -> dict[str, Any]:
        """
        返回初始化时配置的固定记忆召回状态。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                模拟的最终状态。
        """

        return dict(self.result_state)


@pytest.mark.asyncio
async def test_memory_recall_evaluator_should_check_applied_result() -> None:
    """
    测试评估器可以检查已采用记忆、语义分和隔离过滤条件。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = FakeMemoryRuntime(
        {
            "memory_context": "- 用户喜欢的狗狗：金毛寻回犬",
            "memory_recall_result": {
                "status": "applied",
                "candidate_count": 1,
                "threshold_passed_count": 1,
                "selected_count": 1,
                "selected_memory_ids": [1],
                "max_semantic_score": 0.91,
                "reason": "存在可用记忆。",
            },
        }
    )
    evaluator = MemoryRecallBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="memory_eval_001",
        category="memory_recall_behavior",
        question="我喜欢什么狗？",
        expected={
            "recall_status": "applied",
            "memory_context_contains": "金毛寻回犬",
            "selected_memory_ids": [1],
            "minimum_max_semantic_score": 0.9,
            "requested_user_id": "user_a",
            "requested_status": "active",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.failed_checks() == []


@pytest.mark.asyncio
async def test_memory_recall_evaluator_should_reject_unknown_field() -> None:
    """
    测试记忆评估器不会静默忽略未知期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = MemoryRecallBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="memory_eval_invalid_001",
        category="memory_recall_behavior",
        question="测试问题",
        expected={"unknown_field": True},
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
