"""
V1.8.0 ToolAgent 契约冒烟检查。

功能：
    使用 mock state 验证旧工具字段可以稳定转换成 tool_agent_response。

当前阶段：
    该模块不调用真实 LLM，不调用真实工具 API，不接入主图。
    它只验证 ToolAgent 契约和适配器输出形态是否稳定。

专业名词：
    Smoke Check：冒烟检查，用少量关键场景快速确认主能力没有明显断裂。
    Mock State：模拟状态，用测试数据模拟 LangGraph state。
    Contract：契约，模块之间约定好的输入输出格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_state_update,
)


REQUIRED_RESPONSE_KEYS = {
    "status",
    "intent",
    "planned_calls",
    "permission",
    "execution_records",
    "final_answer",
    "metadata",
}


@dataclass(frozen=True)
class ToolAgentSmokeCase:
    """
    ToolAgent 冒烟检查用例。

    功能：
        描述一个 mock state 场景，以及该场景期望得到的 ToolAgentResponse 状态。

    参数：
        name:
            用例名称。

        state:
            模拟 LangGraph state。

        expected_status:
            期望的 tool_agent_response.status。

        expected_permission_status:
            期望的 tool_agent_response.permission.status。

    返回值：
        ToolAgentSmokeCase:
            dataclass 数据对象本身，无额外计算逻辑。
    """

    name: str
    state: dict[str, Any]
    expected_status: str
    expected_permission_status: str


@dataclass(frozen=True)
class ToolAgentSmokeResult:
    """
    ToolAgent 冒烟检查结果。

    功能：
        保存单个 smoke case 的执行结果。

    参数：
        case_name:
            用例名称。

        passed:
            是否通过。

        status:
            实际 ToolAgentResponse 状态。

        message:
            结果说明。

    返回值：
        ToolAgentSmokeResult:
            dataclass 数据对象本身，无额外计算逻辑。
    """

    case_name: str
    passed: bool
    status: str
    message: str


def build_v180_tool_agent_smoke_cases() -> tuple[
    ToolAgentSmokeCase,
    ...
]:
    """
    构建 V1.8.0 ToolAgent 冒烟检查用例。

    功能：
        返回覆盖 no_tool、pending_confirmation、completed、failed、cancelled
        五种关键状态的 mock state。

    参数：
        无。

    返回值：
        tuple[ToolAgentSmokeCase, ...]:
            ToolAgent 契约冒烟检查用例集合。
    """

    return (
        ToolAgentSmokeCase(
            name="no_tool",
            state={
                "question": "你好",
            },
            expected_status="no_tool",
            expected_permission_status="not_required",
        ),
        ToolAgentSmokeCase(
            name="pending_weather_confirmation",
            state={
                "need_tool": True,
                "tool_calls": [
                    {
                        "name": "weather",
                        "args": {
                            "city": "成都",
                        },
                    }
                ],
                "tool_results": [],
                "tool_round": 1,
            },
            expected_status="pending_confirmation",
            expected_permission_status="pending",
        ),
        ToolAgentSmokeCase(
            name="weather_completed",
            state={
                "tool_results": [
                    {
                        "success": True,
                        "tool_name": "weather",
                        "content": "成都今天晴天。",
                        "latency": 1.2,
                    }
                ],
                "final_answer": "成都今天晴天。",
            },
            expected_status="completed",
            expected_permission_status="confirmed",
        ),
        ToolAgentSmokeCase(
            name="weather_failed",
            state={
                "tool_results": [
                    {
                        "success": False,
                        "tool_name": "weather",
                        "content": None,
                        "error": "天气 API 超时",
                    }
                ],
            },
            expected_status="failed",
            expected_permission_status="confirmed",
        ),
        ToolAgentSmokeCase(
            name="user_cancelled",
            state={
                "tool_confirmed": "n",
                "tool_results": [
                    "用户取消了工具调用。",
                ],
            },
            expected_status="cancelled",
            expected_permission_status="rejected",
        ),
    )


def run_tool_agent_smoke_case(
    smoke_case: ToolAgentSmokeCase,
) -> ToolAgentSmokeResult:
    """
    执行单个 ToolAgent 冒烟检查用例。

    功能：
        将 mock state 转换成 tool_agent_response，并验证输出结构和状态。

    参数：
        smoke_case:
            单个 ToolAgentSmokeCase。

    返回值：
        ToolAgentSmokeResult:
            单个用例的检查结果。
    """

    state_update = build_tool_agent_response_state_update(
        state=smoke_case.state,
    )
    response = state_update.get(
        TOOL_AGENT_RESPONSE_STATE_KEY,
    )

    if not isinstance(
        response,
        dict,
    ):
        return ToolAgentSmokeResult(
            case_name=smoke_case.name,
            passed=False,
            status="missing",
            message="tool_agent_response 缺失或不是 dict。",
        )

    missing_keys = REQUIRED_RESPONSE_KEYS - set(
        response.keys()
    )

    if missing_keys:
        return ToolAgentSmokeResult(
            case_name=smoke_case.name,
            passed=False,
            status=str(
                response.get(
                    "status",
                    "unknown",
                )
            ),
            message=f"tool_agent_response 缺少字段: {sorted(missing_keys)}",
        )

    actual_status = str(
        response.get(
            "status",
            "",
        )
    )
    permission = response.get(
        "permission",
        {},
    )
    actual_permission_status = str(
        permission.get(
            "status",
            "",
        )
    )

    if actual_status != smoke_case.expected_status:
        return ToolAgentSmokeResult(
            case_name=smoke_case.name,
            passed=False,
            status=actual_status,
            message=(
                "status 不符合预期，"
                f"expected={smoke_case.expected_status}, actual={actual_status}"
            ),
        )

    if actual_permission_status != smoke_case.expected_permission_status:
        return ToolAgentSmokeResult(
            case_name=smoke_case.name,
            passed=False,
            status=actual_status,
            message=(
                "permission.status 不符合预期，"
                f"expected={smoke_case.expected_permission_status}, "
                f"actual={actual_permission_status}"
            ),
        )

    return ToolAgentSmokeResult(
        case_name=smoke_case.name,
        passed=True,
        status=actual_status,
        message="通过",
    )


def run_v180_tool_agent_smoke_checks() -> list[
    ToolAgentSmokeResult
]:
    """
    执行 V1.8.0 ToolAgent 契约冒烟检查。

    功能：
        依次执行全部 mock state 场景，并返回检查结果列表。

    参数：
        无。

    返回值：
        list[ToolAgentSmokeResult]:
            全部 smoke case 的执行结果。
    """

    return [
        run_tool_agent_smoke_case(
            smoke_case=smoke_case,
        )
        for smoke_case in build_v180_tool_agent_smoke_cases()
    ]


def assert_v180_tool_agent_smoke_checks() -> list[
    ToolAgentSmokeResult
]:
    """
    执行并断言 V1.8.0 ToolAgent 冒烟检查。

    功能：
        如果任意 smoke case 失败，则抛出 AssertionError。
        如果全部通过，则返回结果列表。

    参数：
        无。

    返回值：
        list[ToolAgentSmokeResult]:
            全部 smoke case 的执行结果。
    """

    results = run_v180_tool_agent_smoke_checks()
    failed_results = [
        result
        for result in results
        if not result.passed
    ]

    if failed_results:
        failure_text = "\n".join(
            f"- {result.case_name}: {result.message}"
            for result in failed_results
        )
        raise AssertionError(
            f"V1.8.0 ToolAgent smoke check failed:\n{failure_text}"
        )

    return results
