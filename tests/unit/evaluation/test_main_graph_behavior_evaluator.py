from typing import Any

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import MainGraphBehaviorEvaluator


class FakeTrace:
    """
    为主图评估器测试提供固定调用轨迹。

    参数含义：
        attribute_name:
            需要暴露的轨迹字段名称。
        count:
            轨迹记录数量。

    返回值含义：
        FakeTrace:
            暴露 inputs、queries、calls 中指定字段的测试对象。
    """

    def __init__(self, attribute_name: str, count: int) -> None:
        """
        初始化固定调用轨迹。

        参数含义：
            attribute_name:
                需要创建的轨迹字段名称。
            count:
                轨迹中的记录数量。

        返回值含义：
            None。
        """

        setattr(self, attribute_name, [{} for _ in range(count)])


class FakeMainGraphLLMProvider:
    """
    为主图评估器测试提供固定分类调用次数。

    参数含义：
        counts:
            LLM 调用分类到次数的映射。

    返回值含义：
        FakeMainGraphLLMProvider:
            支持 count_calls 方法的测试对象。
    """

    def __init__(self, counts: dict[str, int]) -> None:
        """
        初始化固定 LLM 调用次数。

        参数含义：
            counts:
                调用分类到次数的映射。

        返回值含义：
            None。
        """

        self.counts = dict(counts)

    def count_calls(self, call_type: str) -> int:
        """
        返回指定调用分类的固定次数。

        参数含义：
            call_type:
                需要统计的 LLM 调用分类。

        返回值含义：
            int:
                配置的调用次数；未配置时返回 0。
        """

        return self.counts.get(call_type, 0)


class FakeMainGraphRuntime:
    """
    为行为评估器单元测试返回固定 Main Graph 最终状态。

    参数含义：
        result_state:
            invoke 方法需要返回的固定主图状态。

    返回值含义：
        FakeMainGraphRuntime:
            具有固定依赖轨迹和异步 invoke 方法的场景对象。
    """

    def __init__(self, result_state: dict[str, Any]) -> None:
        """
        初始化固定主图状态和依赖轨迹。

        参数含义：
            result_state:
                invoke 返回的主图最终状态。

        返回值含义：
            None。
        """

        self.result_state = dict(result_state)
        self.llm_provider = FakeMainGraphLLMProvider(
            {
                "memory_extract": 1,
                "dog_answer": 1,
            }
        )
        self.dog_parser = FakeTrace("inputs", 1)
        self.dog_retriever = FakeTrace("queries", 1)
        self.dog_reranker = FakeTrace("calls", 1)
        self.tool_parser = FakeTrace("inputs", 0)
        self.tool_executor = FakeTrace("calls", 0)

    async def invoke(self) -> dict[str, Any]:
        """
        返回初始化时配置的固定最终状态。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                模拟的 Main Graph 最终 DogState。
        """

        return dict(self.result_state)


@pytest.mark.asyncio
async def test_main_graph_evaluator_should_check_route_answer_and_traces() -> None:
    """
    测试主图评估器可以同时检查路由、答案、状态字段和调用轨迹。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = FakeMainGraphRuntime(
        {
            "route_decision": {
                "route": "dog_knowledge_agent",
                "query_type": "dog_knowledge",
            },
            "next_agent": "dog_knowledge_agent",
            "final_answer": "金毛寻回犬的寿命通常为 10 到 12 年。",
            "dog_knowledge_answer": {},
        }
    )
    evaluator = MainGraphBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="main_graph_eval_001",
        category="main_graph_behavior",
        question="金毛寿命是多少？",
        expected={
            "route": "dog_knowledge_agent",
            "query_type": "dog_knowledge",
            "final_answer_contains": "10 到 12 年",
            "memory_extract_call_count": 1,
            "dog_retriever_call_count": 1,
            "dog_answer_call_count": 1,
            "tool_executor_call_count": 0,
            "required_state_fields": [
                "route_decision",
                "dog_knowledge_answer",
                "final_answer",
            ],
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.failed_checks() == []
    assert result.output["route"] == "dog_knowledge_agent"
    assert result.output["dog_retriever_call_count"] == 1


@pytest.mark.asyncio
async def test_main_graph_evaluator_should_reject_unknown_expected_field() -> None:
    """
    测试主图评估器不会静默忽略未知黄金期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = MainGraphBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="main_graph_invalid_001",
        category="main_graph_behavior",
        question="测试问题",
        expected={
            "route": "general_agent",
            "unknown_field": True,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)


@pytest.mark.asyncio
async def test_main_graph_evaluator_should_require_route_expectation() -> None:
    """
    测试主图黄金用例必须明确声明预期 RootAgent 路由。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = MainGraphBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="main_graph_missing_route_001",
        category="main_graph_behavior",
        question="测试问题",
        expected={
            "final_answer_contains": "测试",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "必须声明 expected.route" in str(result.error_message)
