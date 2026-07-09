"""
V1.8.0 ToolAgent smoke checks 测试。

功能：
    测试 ToolAgent 契约冒烟检查逻辑本身是否稳定。

测试重点：
    1. smoke case 覆盖五种关键状态。
    2. 单个 smoke case 可以返回通过结果。
    3. 全部 smoke checks 可以通过。
"""

from __future__ import annotations

from src.agents.tool_agent.smoke.v180_smoke_checks import (
    assert_v180_tool_agent_smoke_checks,
    build_v180_tool_agent_smoke_cases,
    run_tool_agent_smoke_case,
    run_v180_tool_agent_smoke_checks,
)


def test_v180_tool_agent_smoke_cases_should_cover_expected_statuses() -> None:
    """
    测试 smoke case 覆盖关键状态。

    功能：
        确认 no_tool、pending_confirmation、completed、failed、cancelled
        都有对应 mock state。

    参数：
        无。

    返回值：
        None。
    """

    smoke_cases = build_v180_tool_agent_smoke_cases()

    assert {
        smoke_case.expected_status
        for smoke_case in smoke_cases
    } == {
        "no_tool",
        "pending_confirmation",
        "completed",
        "failed",
        "cancelled",
    }


def test_run_tool_agent_smoke_case_should_pass_single_case() -> None:
    """
    测试单个 smoke case 可以通过。

    功能：
        使用第一个 mock state 验证 run_tool_agent_smoke_case 返回通过结果。

    参数：
        无。

    返回值：
        None。
    """

    smoke_case = build_v180_tool_agent_smoke_cases()[0]

    result = run_tool_agent_smoke_case(
        smoke_case=smoke_case,
    )

    assert result.passed is True
    assert result.case_name == smoke_case.name
    assert result.status == smoke_case.expected_status


def test_run_v180_tool_agent_smoke_checks_should_pass_all_cases() -> None:
    """
    测试全部 smoke checks 通过。

    功能：
        直接运行全部 V1.8.0 ToolAgent 契约冒烟检查。

    参数：
        无。

    返回值：
        None。
    """

    results = run_v180_tool_agent_smoke_checks()

    assert results
    assert all(
        result.passed
        for result in results
    )


def test_assert_v180_tool_agent_smoke_checks_should_return_results() -> None:
    """
    测试断言式 smoke check 返回结果列表。

    功能：
        确认 assert_v180_tool_agent_smoke_checks 在全部通过时不会抛异常。

    参数：
        无。

    返回值：
        None。
    """

    results = assert_v180_tool_agent_smoke_checks()

    assert len(results) == 5
