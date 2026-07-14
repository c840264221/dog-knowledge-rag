import pytest
from pydantic import ValidationError

from src.evaluation import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)


def test_agent_evaluation_case_should_normalize_common_fields() -> None:
    """
    测试统一评估用例会清洗文本和标签。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id=" root_route_weather_001 ",
        category=" root_route ",
        question=" 今天成都天气怎么样？ ",
        expected={
            "route": "tool_agent",
        },
        tags=[
            " root_agent ",
            "weather",
            "root_agent",
            " ",
        ],
    )

    assert eval_case.case_id == "root_route_weather_001"
    assert eval_case.category == "root_route"
    assert eval_case.question == "今天成都天气怎么样？"
    assert eval_case.tags == ["root_agent", "weather"]


def test_agent_evaluation_case_should_reject_empty_expected() -> None:
    """
    测试没有期望结果的评估用例会被拒绝。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="expected 不能为空",
    ):
        AgentEvaluationCase(
            case_id="invalid_case_001",
            category="root_route",
            question="测试问题",
            expected={},
        )


def test_evaluation_check_result_should_preserve_comparison_values() -> None:
    """
    测试单项检查结果会保存期望值、实际值和中文说明。

    参数含义：
        无。

    返回值含义：
        None。
    """

    check = EvaluationCheckResult(
        check_name=" route ",
        passed=False,
        expected="tool_agent",
        actual="general_agent",
        message=" 路由结果不符合预期。 ",
    )

    assert check.check_name == "route"
    assert check.expected == "tool_agent"
    assert check.actual == "general_agent"
    assert check.message == "路由结果不符合预期。"


def test_agent_evaluation_result_should_compute_passed_automatically() -> None:
    """
    测试全部检查通过且没有异常时自动判定整条用例通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = AgentEvaluationResult(
        case_id="root_route_weather_001",
        category="root_route",
        checks=[
            EvaluationCheckResult(
                check_name="route",
                passed=True,
                expected="tool_agent",
                actual="tool_agent",
            ),
            EvaluationCheckResult(
                check_name="query_type",
                passed=True,
                expected="tool_request",
                actual="tool_request",
            ),
        ],
        output={
            "route": "tool_agent",
            "query_type": "tool_request",
        },
    )

    assert result.passed is True
    assert result.failed_checks() == []
    assert result.model_dump()["passed"] is True


def test_agent_evaluation_result_should_fail_when_any_check_fails() -> None:
    """
    测试任意检查失败时整条用例自动失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    failed_check = EvaluationCheckResult(
        check_name="route",
        passed=False,
        expected="tool_agent",
        actual="general_agent",
    )
    result = AgentEvaluationResult(
        case_id="root_route_weather_001",
        category="root_route",
        checks=[failed_check],
    )

    assert result.passed is False
    assert result.failed_checks() == [failed_check]


def test_agent_evaluation_result_should_fail_when_runtime_error_exists() -> None:
    """
    测试存在运行异常时即使检查项通过也自动判定失败。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = AgentEvaluationResult(
        case_id="tool_weather_001",
        category="tool_call",
        checks=[
            EvaluationCheckResult(
                check_name="tool_name",
                passed=True,
                expected="weather",
                actual="weather",
            )
        ],
        error_message="工具执行超时",
    )

    assert result.passed is False


def test_evaluation_schemas_should_forbid_unknown_fields() -> None:
    """
    测试统一评估契约拒绝未声明字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        AgentEvaluationCase(
            case_id="invalid_case_002",
            category="root_route",
            question="测试问题",
            expected={
                "route": "general_agent",
            },
            unexpected_field="不允许的字段",
        )
