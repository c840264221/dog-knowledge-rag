from typing import Any

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import ToolAgentBehaviorEvaluator
from src.evaluation.scenarios import ToolAgentScenarioRuntime


class FakeEvaluationGraph:
    """
    为 ToolAgent 行为评估器单元测试提供固定最终 state。

    参数含义：
        result_state:
            ainvoke 需要返回的固定 ToolAgent state。

    返回值含义：
        FakeEvaluationGraph:
            支持异步 ainvoke 的模拟图对象。
    """

    def __init__(self, result_state: dict[str, Any]) -> None:
        """
        初始化模拟评估图。

        参数含义：
            result_state:
                固定的 ToolAgent 最终 state。

        返回值含义：
            None。
        """

        self.result_state = result_state

    async def ainvoke(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        接收初始 state 并返回固定最终 state。

        参数含义：
            state:
                行为评估器传入的场景初始 state。

        返回值含义：
            dict[str, Any]:
                初始化时配置的固定最终 state。
        """

        assert "question" in state
        return dict(self.result_state)


class FakeTraceObject:
    """
    提供评估器读取的 inputs 或 calls 轨迹列表。

    参数含义：
        values:
            需要暴露给评估器的调用轨迹。

    返回值含义：
        FakeTraceObject:
            同时具有 inputs 和 calls 属性的简单对象。
    """

    def __init__(self, values: list[dict[str, Any]]) -> None:
        """
        初始化模拟调用轨迹。

        参数含义：
            values:
                解析器或执行器的调用记录列表。

        返回值含义：
            None。
        """

        self.inputs = list(values)
        self.calls = list(values)


def build_fake_runtime(
    result_state: dict[str, Any],
    executor_calls: list[dict[str, Any]],
) -> ToolAgentScenarioRuntime:
    """
    构造 ToolAgent 行为评估器单元测试使用的场景环境。

    参数含义：
        result_state:
            模拟 ToolAgent 子图最终 state。
        executor_calls:
            模拟工具执行轨迹。

    返回值含义：
        ToolAgentScenarioRuntime:
            包含模拟图和调用轨迹的场景运行环境。
    """

    return ToolAgentScenarioRuntime(
        graph=FakeEvaluationGraph(result_state),
        initial_state={
            "question": "今天几号？",
        },
        parser=FakeTraceObject([{"question": "今天几号？"}]),
        executor=FakeTraceObject(executor_calls),
        confirmation_prompts=[],
    )


@pytest.mark.asyncio
async def test_tool_agent_behavior_evaluator_should_build_passed_result() -> None:
    """
    测试 ToolAgent 实际行为符合黄金期望时生成通过结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = build_fake_runtime(
        result_state={
            "tool_agent_response": {
                "status": "completed",
                "permission": {
                    "status": "not_required",
                },
            },
            "tool_confirmation_required": False,
            "tool_call_validation_ok": True,
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "date",
                }
            ],
            "final_answer": "今天的日期是 2026-07-08。",
        },
        executor_calls=[
            {
                "tool_name": "date",
                "args": {},
            }
        ],
    )
    evaluator = ToolAgentBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="tool_date_eval_001",
        category="tool_behavior",
        question="今天几号？",
        expected={
            "response_status": "completed",
            "permission_status": "not_required",
            "executed_tool_names": ["date"],
            "tool_result_count": 1,
            "final_answer_contains": "2026-07-08",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["executed_tool_names"] == ["date"]
    assert result.failed_checks() == []


@pytest.mark.asyncio
async def test_tool_agent_behavior_evaluator_should_expose_mismatch() -> None:
    """
    测试工具执行轨迹不符合期望时暴露具体失败检查项。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = build_fake_runtime(
        result_state={
            "tool_agent_response": {
                "status": "completed",
                "permission": {
                    "status": "not_required",
                },
            },
            "tool_results": [],
            "final_answer": "没有执行工具。",
        },
        executor_calls=[],
    )
    evaluator = ToolAgentBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="tool_date_eval_002",
        category="tool_behavior",
        question="今天几号？",
        expected={
            "executed_tool_names": ["date"],
            "tool_result_count": 1,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert {
        check.check_name
        for check in result.failed_checks()
    } == {
        "executed_tool_names",
        "tool_result_count",
    }


@pytest.mark.asyncio
async def test_tool_agent_behavior_evaluator_should_reject_unknown_field() -> None:
    """
    测试 ToolAgent 行为评估器不会静默忽略未知期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = ToolAgentBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="tool_invalid_eval_001",
        category="tool_behavior",
        question="测试问题",
        expected={
            "unknown_field": True,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
