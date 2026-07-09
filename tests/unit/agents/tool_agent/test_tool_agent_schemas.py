"""
ToolAgent Agent 层 schema 测试。

功能：
    测试 ToolAgent 编排层数据结构是否稳定，并确认它们会组合底层 ToolCall / ToolResult。

测试重点：
    1. ToolAgentIntent 默认不需要工具。
    2. ToolAgentPlannedCall 组合底层 ToolCall。
    3. ToolAgentPermissionDecision 可以表达 pending / confirmed / rejected。
    4. ToolAgentExecutionRecord 组合底层 ToolResult。
    5. ToolAgentResponse 可以 model_dump 成普通 dict，便于 checkpoint 保存。
"""

from __future__ import annotations

from src.agents.tool_agent.contracts.schemas import (
    ToolAgentExecutionRecord,
    ToolAgentIntent,
    ToolAgentPermissionDecision,
    ToolAgentPlannedCall,
    ToolAgentResponse,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


def test_tool_agent_intent_should_use_safe_defaults() -> None:
    """
    测试 ToolAgentIntent 默认值。

    功能：
        确认默认情况下 ToolAgent 不会认为必须调用工具。

    参数：
        无。

    返回值：
        None。
    """

    intent = ToolAgentIntent()

    assert intent.need_tool is False
    assert intent.candidate_tools == []
    assert intent.reason == ""


def test_tool_agent_planned_call_should_wrap_tool_call() -> None:
    """
    测试 ToolAgentPlannedCall 组合底层 ToolCall。

    功能：
        确认 Agent 层计划调用不会重复定义工具 name / args，而是引用 ToolCall。

    参数：
        无。

    返回值：
        None。
    """

    tool_call = ToolCall(
        name="weather",
        args={
            "city": "成都",
            "date": "today",
        },
    )

    planned_call = ToolAgentPlannedCall(
        call_id="call_weather_001",
        tool_call=tool_call,
        requires_confirmation=True,
        reason="用户询问天气，需要调用天气工具。",
    )

    assert planned_call.call_id == "call_weather_001"
    assert planned_call.tool_call is tool_call
    assert planned_call.tool_call.name == "weather"
    assert planned_call.tool_call.args["city"] == "成都"
    assert planned_call.requires_confirmation is True


def test_tool_agent_permission_decision_should_represent_pending_status() -> None:
    """
    测试 ToolAgentPermissionDecision 可以表达待确认状态。

    功能：
        pending 表示工具调用需要用户确认，通常会配合 interrupt 使用。

    参数：
        无。

    返回值：
        None。
    """

    permission = ToolAgentPermissionDecision(
        status="pending",
        call_ids=[
            "call_weather_001",
        ],
        prompt="是否允许查询成都天气？",
        reason="天气工具需要访问外部 API。",
    )

    assert permission.status == "pending"
    assert permission.call_ids == [
        "call_weather_001",
    ]
    assert "成都天气" in permission.prompt


def test_tool_agent_execution_record_should_wrap_tool_result() -> None:
    """
    测试 ToolAgentExecutionRecord 组合底层 ToolResult。

    功能：
        确认 Agent 层执行记录不会重复定义工具结果字段，而是引用 ToolResult。

    参数：
        无。

    返回值：
        None。
    """

    tool_result = ToolResult(
        success=True,
        tool_name="weather",
        content={
            "city": "成都",
            "temperature": 30.8,
        },
        latency=1.2,
    )

    record = ToolAgentExecutionRecord(
        call_id="call_weather_001",
        tool_result=tool_result,
        duration_ms=1200,
        metadata={
            "source": "weather_api",
        },
    )

    assert record.call_id == "call_weather_001"
    assert record.tool_result is tool_result
    assert record.tool_result.success is True
    assert record.tool_result.tool_name == "weather"
    assert record.duration_ms == 1200


def test_tool_agent_response_should_dump_to_plain_dict() -> None:
    """
    测试 ToolAgentResponse 可以转换成普通 dict。

    功能：
        确认最终响应契约可以通过 model_dump 转成普通字典，
        避免 checkpoint 中直接保存不可序列化对象。

    参数：
        无。

    返回值：
        None。
    """

    response = ToolAgentResponse(
        status="completed",
        intent=ToolAgentIntent(
            need_tool=True,
            candidate_tools=[
                "weather",
            ],
            reason="用户询问天气。",
        ),
        planned_calls=[
            ToolAgentPlannedCall(
                call_id="call_weather_001",
                tool_call=ToolCall(
                    name="weather",
                    args={
                        "city": "成都",
                    },
                ),
                requires_confirmation=True,
            )
        ],
        permission=ToolAgentPermissionDecision(
            status="confirmed",
            call_ids=[
                "call_weather_001",
            ],
        ),
        execution_records=[
            ToolAgentExecutionRecord(
                call_id="call_weather_001",
                tool_result=ToolResult(
                    success=True,
                    tool_name="weather",
                    content="晴天",
                ),
            )
        ],
        final_answer="今天成都天气是晴天。",
    )

    dumped = response.model_dump()

    assert dumped["status"] == "completed"
    assert dumped["intent"]["need_tool"] is True
    assert dumped["planned_calls"][0]["tool_call"]["name"] == "weather"
    assert dumped["permission"]["status"] == "confirmed"
    assert dumped["execution_records"][0]["tool_result"]["content"] == "晴天"
    assert dumped["final_answer"] == "今天成都天气是晴天。"


def test_tool_agent_response_should_use_independent_default_lists() -> None:
    """
    测试 ToolAgentResponse 默认列表彼此独立。

    功能：
        确认 default_factory 没有共享可变列表，避免多个响应对象互相污染。

    参数：
        无。

    返回值：
        None。
    """

    first_response = ToolAgentResponse()
    second_response = ToolAgentResponse()

    first_response.planned_calls.append(
        ToolAgentPlannedCall(
            call_id="call_weather_001",
            tool_call=ToolCall(
                name="weather",
            ),
        )
    )

    assert len(first_response.planned_calls) == 1
    assert second_response.planned_calls == []

