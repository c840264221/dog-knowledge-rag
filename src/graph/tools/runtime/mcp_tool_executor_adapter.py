"""
MCP 工具执行适配器。

功能：
    将项目内部 ToolCall 转换为 MCP Client 调用，并把 MCP 返回值转换成 ToolResult。

设计原则：
    1. ToolAgent 仍然只认识内部 ToolCall / ToolResult。
    2. MCP 原始返回不直接进入 DogState，而是统一包成 ToolResult。
    3. 当前阶段使用 MockMcpClient 学习执行适配，不连接真实 MCP Server。

专业名词：
    Adapter：
        适配器，用来隔离外部协议对象和内部业务对象。
    ToolCall：
        工具调用请求，包含工具名和参数。
    ToolResult：
        工具执行结果，包含成功状态、内容、错误和元数据。
"""

from __future__ import annotations

import time
from typing import Any

from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult
from src.mcp.mock_client import MockMcpClient


async def execute_mcp_tool_call(
    tool_call: ToolCall,
    mcp_client: Any,
) -> ToolResult:
    """
    执行单个 MCP 工具调用。

    功能：
        将内部 ToolCall 转换成 mcp_client.call_tool(name, arguments)，
        并将 MCP 返回值转换成内部 ToolResult。
        如果 MCP 调用抛出异常，则返回 success=False 的 ToolResult。

    参数：
        tool_call:
            内部工具调用请求，包含 name 和 args。

        mcp_client:
            MCP 客户端对象。当前测试使用 MockMcpClient，
            只要求对象具有 async call_tool(name, arguments) 方法。

    返回值：
        ToolResult:
            内部标准工具执行结果。
    """

    started_at = time.perf_counter()

    try:
        content = await call_mcp_client_tool(
            mcp_client=mcp_client,
            tool_name=tool_call.name,
            arguments=tool_call.args,
        )
    except Exception as exc:
        return build_failed_mcp_tool_result(
            tool_call=tool_call,
            error=exc,
            latency=build_latency_seconds(
                started_at=started_at,
            ),
        )

    return ToolResult(
        success=True,
        tool_name=tool_call.name,
        content=content,
        latency=build_latency_seconds(
            started_at=started_at,
        ),
        retry_count=0,
        metadata={
            "source": "mcp",
            "adapter": "mcp_tool_executor_adapter",
        },
    )


async def call_mcp_client_tool(
    mcp_client: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """
    调用 MCP Client 的工具方法。

    功能：
        统一调用 mcp_client.call_tool。
        单独抽出该函数，方便后续真实 MCP Client 和 Mock MCP Client 共用入口。

    参数：
        mcp_client:
            MCP 客户端对象。

        tool_name:
            MCP 工具名称。

        arguments:
            MCP 工具参数。

    返回值：
        Any:
            MCP Client 返回的原始内容。
    """

    call_tool = getattr(
        mcp_client,
        "call_tool",
        None,
    )

    if not callable(
        call_tool
    ):
        raise TypeError(
            "mcp_client 必须提供 call_tool 方法。"
        )

    return await call_tool(
        tool_name,
        dict(
            arguments
            or {}
        ),
    )


async def execute_mcp_tool_calls(
    tool_calls: list[ToolCall],
    mcp_client: Any,
) -> list[ToolResult]:
    """
    顺序执行多个 MCP 工具调用。

    功能：
        逐个调用 execute_mcp_tool_call，并返回 ToolResult 列表。
        当前 MVP 使用顺序执行，避免引入并发、幂等和批量错误处理复杂度。

    参数：
        tool_calls:
            内部工具调用请求列表。

        mcp_client:
            MCP 客户端对象。

    返回值：
        list[ToolResult]:
            MCP 工具执行结果列表。
    """

    results: list[ToolResult] = []

    for tool_call in tool_calls:
        results.append(
            await execute_mcp_tool_call(
                tool_call=tool_call,
                mcp_client=mcp_client,
            )
        )

    return results


def build_failed_mcp_tool_result(
    tool_call: ToolCall,
    error: Exception,
    latency: float,
) -> ToolResult:
    """
    构建 MCP 工具失败结果。

    功能：
        将 MCP Client 抛出的异常转换成 success=False 的 ToolResult，
        避免异常直接穿透到 ToolAgent 主流程。

    参数：
        tool_call:
            当前执行失败的工具调用请求。

        error:
            MCP Client 抛出的异常。

        latency:
            当前调用耗时，单位秒。

    返回值：
        ToolResult:
            标准失败工具结果。
    """

    return ToolResult(
        success=False,
        tool_name=tool_call.name,
        content=None,
        error=str(
            error
        ),
        latency=latency,
        retry_count=0,
        metadata={
            "source": "mcp",
            "adapter": "mcp_tool_executor_adapter",
            "error_type": type(
                error
            ).__name__,
        },
    )


def build_latency_seconds(
    started_at: float,
) -> float:
    """
    计算工具调用耗时。

    功能：
        使用 time.perf_counter 计算从 started_at 到当前时间的耗时秒数。

    参数：
        started_at:
            调用开始时间。

    返回值：
        float:
            工具调用耗时，单位秒。
    """

    return time.perf_counter() - started_at


def build_default_mock_mcp_client() -> MockMcpClient:
    """
    构建默认 Mock MCP Client。

    功能：
        提供一个便捷工厂函数，方便 smoke 或测试创建空的 MockMcpClient。

    参数：
        无。

    返回值：
        MockMcpClient:
            空的 Mock MCP 客户端。
    """

    return MockMcpClient()
