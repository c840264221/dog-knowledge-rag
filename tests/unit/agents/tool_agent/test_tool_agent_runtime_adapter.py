"""
ToolAgent runtime adapter 测试。

功能：
    测试 ToolAgent 是否可以通过 runtime adapter 调用底层工具执行器。

测试重点：
    1. ToolCall 可以传给 executor.execute。
    2. 成功结果会转换成 ToolAgentExecutionRecord。
    3. 多个工具调用会按顺序执行。
    4. 执行异常会转换成 success=False 的 ToolResult。
    5. state update 输出普通 dict，兼容旧 tool_results 字段。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.state_adapter import (
    build_tool_agent_response_from_state,
)
from src.agents.tool_agent.contracts.schemas import ToolAgentExecutionRecord
from src.agents.tool_agent.adapters.runtime_adapter import (
    TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY,
    build_failed_tool_result,
    build_runtime_call_id,
    build_tool_agent_runtime_state_update,
    dump_tool_agent_execution_records_for_state,
    execute_tool_call_with_runtime,
    execute_tool_calls_with_runtime,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


class FakeExecutor:
    """
    测试用工具执行器。

    功能：
        模拟 ToolExecutor.execute，记录调用参数并返回预设 ToolResult。

    参数：
        should_fail:
            是否模拟执行失败。

    返回值：
        FakeExecutor:
            测试用工具执行器实例。
    """

    def __init__(
        self,
        should_fail: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.calls: list[tuple[str, dict]] = []

    async def execute(
        self,
        tool_name: str,
        args: dict,
    ) -> ToolResult:
        """
        模拟执行工具。

        功能：
            记录工具名称和参数。should_fail=True 时抛出异常，
            否则返回成功 ToolResult。

        参数：
            tool_name:
                工具名称。

            args:
                工具参数。

        返回值：
            ToolResult:
                模拟工具执行结果。
        """

        self.calls.append(
            (
                tool_name,
                args,
            )
        )

        if self.should_fail:
            raise RuntimeError(
                "模拟工具失败"
            )

        return ToolResult(
            success=True,
            tool_name=tool_name,
            content={
                "args": args,
            },
            latency=0.25,
            retry_count=1,
        )


@pytest.mark.asyncio
async def test_execute_tool_call_with_runtime_should_return_execution_record() -> None:
    """
    测试单个工具调用成功时返回执行记录。

    功能：
        确认 runtime adapter 会把 ToolCall 传给 executor.execute，
        并将 ToolResult 包装成 ToolAgentExecutionRecord。

    参数：
        无。

    返回值：
        None。
    """

    executor = FakeExecutor()

    execution_record = await execute_tool_call_with_runtime(
        tool_call=ToolCall(
            name="weather",
            args={
                "city": "成都",
            },
        ),
        executor=executor,
        call_id="planned_1_weather",
    )

    assert executor.calls == [
        (
            "weather",
            {
                "city": "成都",
            },
        )
    ]
    assert execution_record.call_id == "planned_1_weather"
    assert execution_record.tool_result.success is True
    assert execution_record.tool_result.tool_name == "weather"
    assert execution_record.tool_result.retry_count == 1
    assert execution_record.duration_ms is not None


@pytest.mark.asyncio
async def test_execute_tool_calls_with_runtime_should_run_in_order() -> None:
    """
    测试多个工具调用按顺序执行。

    功能：
        确认当前主线采用顺序执行，不引入并发和批量治理复杂度。

    参数：
        无。

    返回值：
        None。
    """

    executor = FakeExecutor()

    execution_records = await execute_tool_calls_with_runtime(
        tool_calls=[
            ToolCall(
                name="date",
                args={},
            ),
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            ),
        ],
        executor=executor,
    )

    assert executor.calls == [
        (
            "date",
            {},
        ),
        (
            "weather",
            {
                "city": "成都",
            },
        ),
    ]
    assert [
        record.call_id
        for record in execution_records
    ] == [
        "runtime_1_date",
        "runtime_2_weather",
    ]


@pytest.mark.asyncio
async def test_execute_tool_call_with_runtime_should_convert_exception_to_failed_result() -> None:
    """
    测试工具执行异常会转换成失败结果。

    功能：
        确认底层 executor 抛错时，runtime adapter 返回 success=False 的 ToolResult。

    参数：
        无。

    返回值：
        None。
    """

    execution_record = await execute_tool_call_with_runtime(
        tool_call=ToolCall(
            name="weather",
            args={
                "city": "成都",
            },
        ),
        executor=FakeExecutor(
            should_fail=True,
        ),
    )

    assert execution_record.tool_result.success is False
    assert execution_record.tool_result.tool_name == "weather"
    assert "模拟工具失败" in str(
        execution_record.tool_result.error
    )
    assert (
        execution_record.tool_result.metadata["error_type"]
        == "RuntimeError"
    )


@pytest.mark.asyncio
async def test_build_tool_agent_runtime_state_update_should_return_plain_dict() -> None:
    """
    测试运行时 state update 输出普通 dict。

    功能：
        确认输出同时包含旧 tool_results 字段和新的 ToolAgent 执行记录字段。

    参数：
        无。

    返回值：
        None。
    """

    update = await build_tool_agent_runtime_state_update(
        tool_calls=[
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            )
        ],
        executor=FakeExecutor(),
    )

    assert update["tool_results"][0]["tool_name"] == "weather"
    assert TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY in update
    assert (
        update[TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY][0]["tool_result"]
        ["content"]["args"]["city"]
        == "成都"
    )


def test_dump_tool_agent_execution_records_for_state_should_return_plain_dict() -> None:
    """
    测试执行记录 dump 成普通 dict。

    功能：
        确认不会把 Pydantic 对象直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    failed_result = build_failed_tool_result(
        tool_call=ToolCall(
            name="weather",
            args={},
        ),
        error=RuntimeError(
            "失败",
        ),
    )

    records = dump_tool_agent_execution_records_for_state(
        execution_records=[
            # 这里复用 runtime adapter 生成的字段结构，避免测试依赖真实执行器。
            ToolAgentExecutionRecord(
                call_id="runtime_1_weather",
                tool_result=failed_result,
                duration_ms=1,
            )
        ],
    )

    assert isinstance(
        records[0],
        dict,
    )
    assert records[0]["tool_result"]["success"] is False


def test_build_runtime_call_id_should_use_index_and_tool_name() -> None:
    """
    测试运行时调用 ID 格式。

    功能：
        确认调用 ID 稳定可读，方便后续关联计划和执行记录。

    参数：
        无。

    返回值：
        None。
    """

    assert (
        build_runtime_call_id(
            tool_call=ToolCall(
                name="weather",
                args={},
            ),
            index=3,
        )
        == "runtime_3_weather"
    )


@pytest.mark.asyncio
async def test_runtime_state_update_should_be_compatible_with_tool_agent_state_adapter() -> None:
    """
    测试 runtime_adapter 输出兼容 state_adapter。

    功能：
        确认 build_tool_agent_runtime_state_update 返回的 tool_results，
        可以继续被 build_tool_agent_response_from_state 识别为 completed 状态。

    参数：
        无。

    返回值：
        None。
    """

    update = await build_tool_agent_runtime_state_update(
        tool_calls=[
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            )
        ],
        executor=FakeExecutor(),
    )

    response = build_tool_agent_response_from_state(
        state={
            "tool_results": update["tool_results"],
            "final_answer": "今天成都天气晴。",
        },
    )

    assert response.status == "completed"
    assert response.intent.need_tool is True
    assert response.intent.candidate_tools == [
        "weather",
    ]
    assert response.execution_records[0].tool_result.tool_name == "weather"
    assert response.final_answer == "今天成都天气晴。"


@pytest.mark.asyncio
async def test_failed_runtime_state_update_should_be_compatible_with_state_adapter() -> None:
    """
    测试失败 runtime_adapter 输出兼容 state_adapter。

    功能：
        确认 executor 执行失败后生成的 success=False tool_results，
        可以继续被 build_tool_agent_response_from_state 识别为 failed 状态。

    参数：
        无。

    返回值：
        None。
    """

    update = await build_tool_agent_runtime_state_update(
        tool_calls=[
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            )
        ],
        executor=FakeExecutor(
            should_fail=True,
        ),
    )

    response = build_tool_agent_response_from_state(
        state={
            "tool_results": update["tool_results"],
        },
    )

    assert response.status == "failed"
    assert response.execution_records[0].tool_result.success is False
    assert "模拟工具失败" in str(
        response.execution_records[0].tool_result.error
    )
