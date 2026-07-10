"""
ToolAgent 工具运行时适配器。

功能：
    将 ToolAgent 的 ToolCall（工具调用请求）桥接到已有 ToolExecutor（工具执行器）。

设计原则：
    1. ToolAgent 负责工具调用编排，不直接实现超时、重试、追踪、日志。
    2. 超时、重试、追踪、日志继续复用底层 ToolExecutor 和 MiddlewarePipeline。
    3. 本模块只做数据结构转换和失败兜底。
    4. 输出普通 dict，避免 checkpoint 保存自定义对象。

专业名词：
    Runtime：运行时，负责真实执行工具以及管理执行过程。
    Middleware：中间件，负责日志、追踪、重试、超时等横切能力。
    Adapter：适配器，用来隔离 ToolAgent 契约和底层执行器契约。
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from src.agents.tool_agent.contracts.schemas import ToolAgentExecutionRecord
from src.graph.tools.runtime.mcp_tool_executor_adapter import (
    execute_mcp_tool_call,
)
from src.graph.tools.runtime.tool_executor import ToolExecutor
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY = "tool_agent_runtime_execution_records"


async def execute_tool_call_with_runtime(
    tool_call: ToolCall,
    executor: Any | None = None,
    mcp_client: Any | None = None,
    tool_catalog: list[Mapping[str, Any]] | None = None,
    call_id: str | None = None,
) -> ToolAgentExecutionRecord:
    """
    通过运行时执行单个 ToolCall。

    功能：
        根据工具目录中的 source 字段判断工具来源。
        local 工具交给已有 ToolExecutor.execute。
        mcp 工具交给 MCP 工具执行适配器。
        如果执行失败，本函数会返回 success=False 的 ToolAgentExecutionRecord。

    参数：
        tool_call:
            工具调用请求，包含工具名称和工具参数。

        executor:
            工具执行器。默认创建项目已有 ToolExecutor。
            测试时可以传入 fake executor，避免执行真实工具。

        mcp_client:
            MCP Client（模型上下文协议客户端）。
            当工具目录标记 source=mcp 时，会通过它调用 MCP 工具。

        tool_catalog:
            ToolAgent 工具目录。用于通过工具名读取 source 字段，
            判断当前工具应该走 local 还是 mcp。

        call_id:
            ToolAgent 计划调用 ID。为空时根据工具名生成兜底 ID。

    返回值：
        ToolAgentExecutionRecord:
            ToolAgent 标准工具执行记录。
    """

    resolved_call_id = call_id or build_runtime_call_id(
        tool_call=tool_call,
        index=1,
    )
    tool_source = resolve_tool_source(
        tool_name=tool_call.name,
        tool_catalog=tool_catalog,
    )
    started_at = time.perf_counter()

    if tool_source == "mcp":
        tool_result = await execute_mcp_tool_call(
            tool_call=tool_call,
            mcp_client=mcp_client,
        )
    else:
        resolved_executor = executor or ToolExecutor()

        try:
            tool_result = await resolved_executor.execute(
                tool_call.name,
                tool_call.args,
            )
        except Exception as exc:
            tool_result = build_failed_tool_result(
                tool_call=tool_call,
                error=exc,
            )

    duration_ms = int(
        (
            time.perf_counter()
            - started_at
        )
        * 1000
    )

    return ToolAgentExecutionRecord(
        call_id=resolved_call_id,
        tool_result=tool_result,
        duration_ms=duration_ms,
        metadata={
            "source": "tool_agent_runtime_adapter",
            "tool_source": tool_source,
        },
    )


async def execute_tool_calls_with_runtime(
    tool_calls: list[ToolCall],
    executor: Any | None = None,
    mcp_client: Any | None = None,
    tool_catalog: list[Mapping[str, Any]] | None = None,
) -> list[ToolAgentExecutionRecord]:
    """
    通过运行时顺序执行多个 ToolCall。

    功能：
        逐个执行工具调用，并把每次结果转换成 ToolAgentExecutionRecord。
        每个工具会根据工具目录 source 字段分别走 local 或 mcp 执行分支。
        当前主线先采用顺序执行，避免引入并发执行、批量确认、幂等等 Plus 复杂度。

    参数：
        tool_calls:
            待执行的工具调用列表。

        executor:
            工具执行器。默认创建项目已有 ToolExecutor。

        mcp_client:
            MCP Client。用于执行 source=mcp 的工具。

        tool_catalog:
            ToolAgent 工具目录，用于按工具名判断工具来源。

    返回值：
        list[ToolAgentExecutionRecord]:
            ToolAgent 标准工具执行记录列表。
    """

    execution_records: list[ToolAgentExecutionRecord] = []

    for index, tool_call in enumerate(
        tool_calls,
        start=1,
    ):
        execution_records.append(
            await execute_tool_call_with_runtime(
                tool_call=tool_call,
                executor=executor,
                mcp_client=mcp_client,
                tool_catalog=tool_catalog,
                call_id=build_runtime_call_id(
                    tool_call=tool_call,
                    index=index,
                ),
            )
        )

    return execution_records


def dump_tool_agent_execution_records_for_state(
    execution_records: list[ToolAgentExecutionRecord],
) -> list[dict[str, Any]]:
    """
    将 ToolAgentExecutionRecord 列表转换成普通 dict 列表。

    功能：
        避免把 Pydantic 对象直接写入 LangGraph state 或 checkpoint。

    参数：
        execution_records:
            ToolAgent 工具执行记录列表。

    返回值：
        list[dict[str, Any]]:
            可写入 state 的普通字典列表。
    """

    return [
        execution_record.model_dump()
        for execution_record in execution_records
    ]


async def build_tool_agent_runtime_state_update(
    tool_calls: list[ToolCall],
    executor: Any | None = None,
    mcp_client: Any | None = None,
    tool_catalog: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    执行工具调用并构建可写回 state 的运行时结果。

    功能：
        执行 ToolCall 列表，并返回两个字段：
        1. tool_results：兼容旧工具 state_adapter 的底层工具结果列表。
        2. tool_agent_runtime_execution_records：ToolAgent 标准执行记录列表。

    参数：
        tool_calls:
            待执行的工具调用列表。

        executor:
            工具执行器。默认创建项目已有 ToolExecutor。

        mcp_client:
            MCP Client。用于执行工具目录中 source=mcp 的工具。

        tool_catalog:
            ToolAgent 工具目录。runtime adapter 通过它识别工具来源。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的普通字典。
    """

    execution_records = await execute_tool_calls_with_runtime(
        tool_calls=tool_calls,
        executor=executor,
        mcp_client=mcp_client,
        tool_catalog=tool_catalog,
    )

    return {
        "tool_results": [
            execution_record.tool_result.model_dump()
            for execution_record in execution_records
        ],
        TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY: (
            dump_tool_agent_execution_records_for_state(
                execution_records=execution_records,
            )
        ),
    }


def build_failed_tool_result(
    tool_call: ToolCall,
    error: Exception,
) -> ToolResult:
    """
    构建失败工具结果。

    功能：
        当底层 ToolExecutor 抛出异常时，转换成 success=False 的 ToolResult，
        让 ToolAgent 可以继续生成结构化失败响应。

    参数：
        tool_call:
            当前执行失败的工具调用。

        error:
            底层执行异常。

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
        retry_count=0,
        metadata={
            "source": "tool_agent_runtime_adapter",
            "error_type": type(
                error
            ).__name__,
        },
    )


def build_runtime_call_id(
    tool_call: ToolCall,
    index: int,
) -> str:
    """
    构建运行时调用 ID。

    功能：
        根据工具名和序号生成稳定可读的调用 ID，方便关联计划调用和执行记录。

    参数：
        tool_call:
            工具调用请求。

        index:
            当前工具调用序号，从 1 开始。

    返回值：
        str:
            运行时调用 ID。
    """

    return f"runtime_{index}_{tool_call.name}"


def resolve_tool_source(
    tool_name: str,
    tool_catalog: list[Mapping[str, Any]] | None = None,
) -> str:
    """
    根据工具目录解析工具来源。

    功能：
        遍历 ToolAgent 工具目录，找到 name 等于 tool_name 的条目，
        并读取它的 source 字段。
        当前只识别 mcp，其余情况统一兜底为 local，保证旧本地工具链路稳定。

    参数：
        tool_name:
            当前待执行的工具名称。

        tool_catalog:
            ToolAgent 工具目录列表。每个条目通常包含 name 和 source 字段。

    返回值：
        str:
            工具来源。mcp 表示 MCP 工具，local 表示本地工具或未知来源。
    """

    if not tool_catalog:
        return "local"

    for tool_item in tool_catalog:
        if not isinstance(
            tool_item,
            Mapping,
        ):
            continue

        if str(
            tool_item.get(
                "name",
                "",
            )
        ) != tool_name:
            continue

        source = str(
            tool_item.get(
                "source",
                "local",
            )
            or "local"
        ).strip()

        if source == "mcp":
            return "mcp"

        return "local"

    return "local"
