"""
ToolAgent state adapter 测试。

功能：
    测试旧工具 state 字段是否可以转换成新的 ToolAgentResponse 契约。

测试重点：
    1. 无工具状态转换为 no_tool。
    2. 旧 tool_calls 转换为 planned_calls。
    3. 旧 tool_results 转换为 execution_records。
    4. 用户取消工具调用时转换为 cancelled。
    5. 输出 state update 时必须是普通 dict。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    build_tool_agent_response_state_update,
    normalize_tool_calls,
    normalize_tool_results,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


def test_build_response_should_return_no_tool_when_state_has_no_tool_data() -> None:
    """
    测试没有工具字段时返回 no_tool。

    功能：
        确认空 state 不会被误判为需要工具。

    参数：
        无。

    返回值：
        None。
    """

    response = build_tool_agent_response_from_state(
        state={
            "question": "你好",
        },
    )

    assert response.status == "no_tool"
    assert response.intent.need_tool is False
    assert response.planned_calls == []
    assert response.execution_records == []


def test_build_response_should_convert_tool_calls_to_planned_calls() -> None:
    """
    测试旧 tool_calls 转换为 planned_calls。

    功能：
        确认旧 state 中的 name / args 会进入底层 ToolCall，
        Agent 层只额外补充 call_id、requires_confirmation 和 reason。

    参数：
        无。

    返回值：
        None。
    """

    response = build_tool_agent_response_from_state(
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
    )

    assert response.status == "pending_confirmation"
    assert response.intent.need_tool is True
    assert response.intent.candidate_tools == [
        "weather",
    ]
    assert len(response.planned_calls) == 1
    assert response.planned_calls[0].call_id == "planned_1_weather"
    assert response.planned_calls[0].tool_call.name == "weather"
    assert response.planned_calls[0].tool_call.args == {
        "city": "成都",
    }
    assert response.planned_calls[0].requires_confirmation is True
    assert response.permission.status == "pending"


def test_build_response_should_convert_tool_results_to_execution_records() -> None:
    """
    测试旧 tool_results 转换为 execution_records。

    功能：
        确认 dict 形式的工具结果会被归一化为 ToolResult，
        再包装成 ToolAgentExecutionRecord。

    参数：
        无。

    返回值：
        None。
    """

    response = build_tool_agent_response_from_state(
        state={
            "tool_calls": [],
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "weather",
                    "content": {
                        "city": "成都",
                        "temperature": 30.8,
                    },
                    "latency": 1.25,
                }
            ],
            "final_answer": "今天成都天气晴。",
        },
    )

    assert response.status == "completed"
    assert response.intent.need_tool is True
    assert response.intent.candidate_tools == [
        "weather",
    ]
    assert response.permission.status == "confirmed"
    assert len(response.execution_records) == 1
    assert response.execution_records[0].call_id == "executed_1_weather"
    assert response.execution_records[0].duration_ms == 1250
    assert response.execution_records[0].tool_result.tool_name == "weather"
    assert response.final_answer == "今天成都天气晴。"


def test_build_response_should_mark_failed_when_tool_result_failed() -> None:
    """
    测试工具结果失败时返回 failed。

    功能：
        如果任意 ToolResult.success=False，ToolAgentResponse.status 应该是 failed。

    参数：
        无。

    返回值：
        None。
    """

    response = build_tool_agent_response_from_state(
        state={
            "tool_results": [
                {
                    "success": False,
                    "tool_name": "weather",
                    "content": None,
                    "error": "API 超时",
                }
            ],
        },
    )

    assert response.status == "failed"
    assert response.execution_records[0].tool_result.error == "API 超时"


def test_build_response_should_mark_cancelled_when_user_cancelled() -> None:
    """
    测试用户取消工具调用时返回 cancelled。

    功能：
        兼容旧 ask_confirm_tool_node 返回的“用户取消了工具调用。”字符串。

    参数：
        无。

    返回值：
        None。
    """

    response = build_tool_agent_response_from_state(
        state={
            "tool_confirmed": "n",
            "tool_results": [
                "用户取消了工具调用。",
            ],
        },
    )

    assert response.status == "cancelled"
    assert response.permission.status == "rejected"


def test_build_response_state_update_should_dump_plain_dict() -> None:
    """
    测试 state update 输出普通 dict。

    功能：
        确认适配器不会直接把 Pydantic 对象写回 state，
        避免 checkpoint 保存自定义对象。

    参数：
        无。

    返回值：
        None。
    """

    update = build_tool_agent_response_state_update(
        state={
            "tool_results": [
                ToolResult(
                    success=True,
                    tool_name="weather",
                    content="晴天",
                )
            ],
        },
    )

    assert TOOL_AGENT_RESPONSE_STATE_KEY in update
    assert isinstance(
        update[TOOL_AGENT_RESPONSE_STATE_KEY],
        dict,
    )
    assert (
        update[TOOL_AGENT_RESPONSE_STATE_KEY]["execution_records"][0]
        ["tool_result"]["content"]
        == "晴天"
    )


def test_normalize_tool_calls_should_skip_invalid_items() -> None:
    """
    测试非法 tool_calls 会被跳过。

    功能：
        适配器不应该因为旧 state 中出现坏数据就打断主链路。

    参数：
        无。

    返回值：
        None。
    """

    calls = normalize_tool_calls(
        [
            {
                "name": "weather",
                "args": {
                    "city": "成都",
                },
            },
            {
                "args": {
                    "city": "北京",
                },
            },
            "bad_call",
        ]
    )

    assert calls == [
        ToolCall(
            name="weather",
            args={
                "city": "成都",
            },
        )
    ]


def test_normalize_tool_results_should_convert_legacy_string() -> None:
    """
    测试旧字符串 tool_results 会被转换。

    功能：
        兼容历史遗留的字符串工具结果。

    参数：
        无。

    返回值：
        None。
    """

    results = normalize_tool_results(
        "用户取消了工具调用。"
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].tool_name == "legacy_tool_result_1"
    assert results[0].content == "用户取消了工具调用。"

