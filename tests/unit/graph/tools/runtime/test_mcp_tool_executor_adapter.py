"""
MCP 工具执行适配器测试。

功能：
    验证内部 ToolCall 可以通过 Mock MCP Client 执行，并转换成内部 ToolResult。
"""

from __future__ import annotations

import pytest

from src.graph.tools.runtime.mcp_tool_executor_adapter import (
    build_default_mock_mcp_client,
    execute_mcp_tool_call,
    execute_mcp_tool_calls,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.mcp.mock_client import MockMcpClient


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_should_return_success_result() -> None:
    """
    测试 MCP 工具执行成功时返回 ToolResult。

    功能：
        注册一个 Mock MCP 工具返回值，然后通过 ToolCall 调用它，
        验证返回结果被转换成内部 ToolResult。

    参数：
        无。

    返回值：
        None。
    """

    client = MockMcpClient()
    client.register_tool_result(
        name="mcp_echo",
        result={
            "text": "hello",
        },
    )

    result = await execute_mcp_tool_call(
        tool_call=ToolCall(
            name="mcp_echo",
            args={
                "text": "hello",
            },
        ),
        mcp_client=client,
    )

    assert result.success is True
    assert result.tool_name == "mcp_echo"
    assert result.content == {
        "text": "hello",
    }
    assert result.error is None
    assert result.latency is not None
    assert result.metadata["source"] == "mcp"
    assert result.metadata["adapter"] == "mcp_tool_executor_adapter"


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_should_pass_name_and_args_to_client() -> None:
    """
    测试 ToolCall 的名称和参数会传给 MCP Client。

    功能：
        通过 MockMcpClient.call_records 检查 adapter 是否正确传递工具名和参数。

    参数：
        无。

    返回值：
        None。
    """

    client = MockMcpClient()
    client.register_tool_result(
        name="mcp_search",
        result=[
            "result-1",
        ],
    )

    await execute_mcp_tool_call(
        tool_call=ToolCall(
            name="mcp_search",
            args={
                "query": "golden retriever",
                "limit": 3,
            },
        ),
        mcp_client=client,
    )

    call_record = client.get_last_call_record()

    assert call_record is not None
    assert call_record.name == "mcp_search"
    assert call_record.arguments == {
        "query": "golden retriever",
        "limit": 3,
    }


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_should_return_failed_result_when_tool_missing() -> None:
    """
    测试 MCP 工具不存在时返回失败 ToolResult。

    功能：
        MockMcpClient 未注册工具时会抛出 KeyError，
        adapter 应该捕获异常并转换成 success=False 的 ToolResult。

    参数：
        无。

    返回值：
        None。
    """

    client = MockMcpClient()

    result = await execute_mcp_tool_call(
        tool_call=ToolCall(
            name="mcp_missing",
            args={},
        ),
        mcp_client=client,
    )

    assert result.success is False
    assert result.tool_name == "mcp_missing"
    assert result.content is None
    assert "Mock MCP 工具不存在" in str(
        result.error
    )
    assert result.metadata["source"] == "mcp"
    assert result.metadata["error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_should_return_failed_result_when_client_invalid() -> None:
    """
    测试 MCP Client 缺少 call_tool 方法时返回失败 ToolResult。

    功能：
        防止错误的 MCP client 对象直接导致上层流程崩溃。

    参数：
        无。

    返回值：
        None。
    """

    result = await execute_mcp_tool_call(
        tool_call=ToolCall(
            name="mcp_echo",
            args={},
        ),
        mcp_client=object(),
    )

    assert result.success is False
    assert result.tool_name == "mcp_echo"
    assert "call_tool" in str(
        result.error
    )
    assert result.metadata["error_type"] == "TypeError"


@pytest.mark.asyncio
async def test_execute_mcp_tool_calls_should_execute_in_order() -> None:
    """
    测试批量 MCP 工具调用按顺序执行。

    功能：
        当前 MVP 采用顺序执行，测试用于确认返回顺序和调用顺序一致。

    参数：
        无。

    返回值：
        None。
    """

    client = MockMcpClient()
    client.register_tool_result(
        name="mcp_first",
        result="first",
    )
    client.register_tool_result(
        name="mcp_second",
        result="second",
    )

    results = await execute_mcp_tool_calls(
        tool_calls=[
            ToolCall(
                name="mcp_first",
                args={},
            ),
            ToolCall(
                name="mcp_second",
                args={},
            ),
        ],
        mcp_client=client,
    )

    assert [
        result.content
        for result in results
    ] == [
        "first",
        "second",
    ]
    assert [
        call_record.name
        for call_record in client.call_records
    ] == [
        "mcp_first",
        "mcp_second",
    ]


def test_build_default_mock_mcp_client_should_return_empty_client() -> None:
    """
    测试默认 Mock MCP Client 工厂函数。

    功能：
        验证 build_default_mock_mcp_client 返回可用的空 MockMcpClient。

    参数：
        无。

    返回值：
        None。
    """

    client = build_default_mock_mcp_client()

    assert isinstance(
        client,
        MockMcpClient,
    )
    assert client.call_records == []
