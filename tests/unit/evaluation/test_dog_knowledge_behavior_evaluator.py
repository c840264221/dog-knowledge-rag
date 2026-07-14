from typing import Any

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import DogKnowledgeBehaviorEvaluator


class FakeTrace:
    """
    为评估器测试提供固定调用轨迹。

    参数含义：
        count:
            需要模拟的调用次数。

    返回值含义：
        FakeTrace:
            同时暴露 inputs、queries、calls 和 prompts 的简单对象。
    """

    def __init__(self, count: int) -> None:
        values = [{} for _ in range(count)]
        self.inputs = list(values)
        self.queries = list(values)
        self.calls = list(values)
        self.prompts = ["prompt" for _ in range(count)]


class FakeDogKnowledgeRuntime:
    """
    为行为评估器单元测试返回固定 DogKnowledgeAgent 最终状态。

    参数含义：
        result_state:
            invoke 方法需要返回的固定状态。

    返回值含义：
        FakeDogKnowledgeRuntime:
            具有固定调用轨迹和异步 invoke 方法的场景对象。
    """

    def __init__(self, result_state: dict[str, Any]) -> None:
        self.result_state = result_state
        self.parser = FakeTrace(1)
        self.retriever = FakeTrace(1)
        self.reranker = FakeTrace(1)
        self.llm_provider = FakeTrace(1)

    async def invoke(self) -> dict[str, Any]:
        """
        返回初始化时配置的固定最终状态。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                模拟的 DogKnowledgeAgent 最终状态。
        """

        return dict(self.result_state)


@pytest.mark.asyncio
async def test_dog_knowledge_evaluator_should_check_contract_and_behavior() -> None:
    """
    测试评估器可以同时检查响应字段、推荐结果和分层契约。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = FakeDogKnowledgeRuntime(
        {
            "dog_query_result": {},
            "dog_retrieval_result": {},
            "dog_generation_result": {},
            "dog_knowledge_pipeline_result": {},
            "dog_knowledge_answer_public": {},
            "dog_knowledge_answer": {
                "status": "success",
                "query_type": "recommendation",
                "is_fallback": False,
                "recommended_breeds": [
                    {
                        "breed_name": "Golden Retriever",
                    }
                ],
                "evidences": [
                    {
                        "evidence_id": "evidence-1",
                    }
                ],
            },
            "final_answer": "推荐金毛寻回犬。",
        }
    )
    evaluator = DogKnowledgeBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="dog_behavior_eval_001",
        category="dog_knowledge_behavior",
        question="新手适合养什么狗？",
        expected={
            "response_status": "success",
            "query_type": "recommendation",
            "expected_breed_names": ["Golden Retriever"],
            "min_evidence_count": 1,
            "required_layer_outputs": [
                "query",
                "retrieval",
                "generation",
                "pipeline",
                "answer",
                "public_answer",
            ],
            "final_answer_contains": "金毛",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.failed_checks() == []
    assert result.output["recommended_breed_names"] == [
        "Golden Retriever"
    ]


@pytest.mark.asyncio
async def test_dog_knowledge_evaluator_should_reject_unknown_field() -> None:
    """
    测试评估器不会静默忽略未知黄金期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = DogKnowledgeBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="dog_behavior_invalid_001",
        category="dog_knowledge_behavior",
        question="测试问题",
        expected={
            "unknown_field": True,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
