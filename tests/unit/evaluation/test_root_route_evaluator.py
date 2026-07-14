from typing import Any

import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import RootRouteEvaluator


def build_route_update(
    route: str = "tool_agent",
    query_type: str = "tool_request",
) -> dict[str, Any]:
    """
    构造 RootAgent 路由评估测试使用的局部 state 更新。

    参数含义：
        route:
            模拟的主图路由目标。
        query_type:
            模拟的 RootAgent 查询类型。

    返回值含义：
        dict[str, Any]:
            模拟路由节点返回的局部 state 更新。
    """

    return {
        "route_decision": {
            "route": route,
            "query_type": query_type,
            "confidence": 0.9,
            "requires_rag": False,
            "requires_tool": True,
            "requires_memory": True,
        },
        "next_agent": route,
        "current_agent": "root_agent",
    }


@pytest.mark.asyncio
async def test_root_route_evaluator_should_build_passed_result() -> None:
    """
    测试路由节点和主图条件边均符合预期时评估通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def fake_route_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        模拟主图异步路由节点并验证输入问题。

        参数含义：
            state:
                评估器构造的初始 state。

        返回值含义：
            dict[str, Any]:
                模拟的 RootAgent 局部 state 更新。
        """

        assert state["question"] == "今天成都天气怎么样？"
        return build_route_update()

    evaluator = RootRouteEvaluator(
        route_node=fake_route_node,
        route_resolver=lambda state: state["route_decision"]["route"],
    )
    eval_case = AgentEvaluationCase(
        case_id="root_route_weather_001",
        category="root_route",
        question="今天成都天气怎么样？",
        expected={
            "route": "tool_agent",
            "query_type": "tool_request",
            "requires_tool": True,
            "min_confidence": 0.8,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["graph_route"] == "tool_agent"
    assert result.failed_checks() == []


@pytest.mark.asyncio
async def test_root_route_evaluator_should_expose_route_mismatch() -> None:
    """
    测试主图条件边路由错误会形成可读的失败检查项。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = RootRouteEvaluator(
        route_node=lambda state: build_route_update(),
        route_resolver=lambda state: "general_agent",
    )
    eval_case = AgentEvaluationCase(
        case_id="root_route_weather_002",
        category="root_route",
        question="今天成都天气怎么样？",
        expected={
            "route": "tool_agent",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert [
        check.check_name
        for check in result.failed_checks()
    ] == ["graph_route"]


@pytest.mark.asyncio
async def test_root_route_evaluator_should_convert_exception_to_result() -> None:
    """
    测试单条路由异常会转换成失败结果而不是中断批量评估。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def failing_route_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        模拟执行失败的主图路由节点。

        参数含义：
            state:
                评估器构造的初始 state。

        返回值含义：
            dict[str, Any]:
                本函数固定抛出异常，不会正常返回该字典。
        """

        del state
        raise RuntimeError("路由节点执行失败")

    evaluator = RootRouteEvaluator(
        route_node=failing_route_node,
    )
    eval_case = AgentEvaluationCase(
        case_id="root_route_error_001",
        category="root_route",
        question="测试问题",
        expected={
            "route": "general_agent",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert result.checks == []
    assert result.error_message == "路由节点执行失败"


@pytest.mark.asyncio
async def test_root_route_evaluator_should_reject_unknown_expected_field() -> None:
    """
    测试 RootAgent 评估器不会静默忽略未知期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = RootRouteEvaluator(
        route_node=lambda state: build_route_update(),
    )
    eval_case = AgentEvaluationCase(
        case_id="root_route_invalid_001",
        category="root_route",
        question="测试问题",
        expected={
            "route": "general_agent",
            "unknown_field": "value",
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
