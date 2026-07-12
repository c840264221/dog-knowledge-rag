"""
ToolAgent 工具目录条目适配器测试。

功能：
    验证本地 ToolMetadata 和 MCP 工具定义都能转换成 ToolCatalogItem。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.tool_catalog_item_adapter import (
    MCP_TOOL_SOURCE,
    build_tool_catalog_item_from_mcp_tool,
    build_tool_catalog_item_from_tool_metadata,
    build_tool_catalog_items_from_mcp_tools,
    dump_tool_catalog_item_for_state,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.implementations.weather_tool import WeatherTool
from src.mcp.sqlite.tool_definitions import (
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    build_sqlite_run_readonly_query_tool_definition,
)


def test_build_tool_catalog_item_from_tool_metadata_should_create_local_item() -> None:
    """
    测试本地工具元数据转换成工具目录条目。

    功能：
        确认 ToolMetadata 的配置被保留，input_schema 默认为空。

    参数：
        无。

    返回值：
        None。
    """

    item = build_tool_catalog_item_from_tool_metadata(
        metadata=ToolMetadata(
            name="weather",
            description="查询天气",
            timeout=8,
            retries=2,
            require_confirm=True,
        )
    )

    assert item.name == "weather"
    assert item.description == "查询天气"
    assert item.timeout == 8
    assert item.retries == 2
    assert item.require_confirm is True
    assert item.input_schema == {}
    assert item.source == "local"


def test_build_local_tool_catalog_item_should_keep_input_schema() -> None:
    """
    测试本地工具参数契约会进入统一工具目录。

    功能：
        确认天气工具的 city 必填要求不会在本地工具适配过程中丢失。

    参数：
        无。

    返回值：
        None。
    """

    item = build_tool_catalog_item_from_tool_metadata(
        metadata=WeatherTool.metadata
    )

    assert item.input_schema["required"] == [
        "city",
    ]
    assert item.input_schema["properties"]["city"]["type"] == "string"


def test_build_tool_catalog_item_from_mcp_tool_should_keep_input_schema() -> None:
    """
    测试 MCP 工具定义转换成工具目录条目。

    功能：
        确认 MCP input_schema、确认配置和来源标记会被保留。

    参数：
        无。

    返回值：
        None。
    """

    item = build_tool_catalog_item_from_mcp_tool(
        mcp_tool=build_sqlite_run_readonly_query_tool_definition(),
    )

    assert item.name == SQLITE_RUN_READONLY_QUERY_TOOL_NAME
    assert item.source == MCP_TOOL_SOURCE
    assert item.require_confirm is True
    assert item.retries == 0
    assert item.input_schema["properties"]["database_name"]["type"] == "string"
    assert item.input_schema["properties"]["sql"]["type"] == "string"


def test_dump_tool_catalog_item_for_state_should_return_plain_dict() -> None:
    """
    测试工具目录条目转普通 dict。

    功能：
        确认 adapter 不会把 Pydantic 对象直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    item = build_tool_catalog_item_from_mcp_tool(
        mcp_tool=build_sqlite_run_readonly_query_tool_definition(),
    )
    dumped = dump_tool_catalog_item_for_state(
        item=item,
    )

    assert isinstance(
        dumped,
        dict,
    )
    assert dumped["name"] == SQLITE_RUN_READONLY_QUERY_TOOL_NAME
    assert dumped["source"] == MCP_TOOL_SOURCE


def test_build_tool_catalog_items_from_mcp_tools_should_sort_by_name() -> None:
    """
    测试 MCP 工具目录条目按名称排序。

    功能：
        确认批量转换结果稳定，方便日志、测试和调试。

    参数：
        无。

    返回值：
        None。
    """

    items = build_tool_catalog_items_from_mcp_tools(
        mcp_tools=[
            build_sqlite_run_readonly_query_tool_definition(),
        ],
    )

    assert [
        item.name
        for item in items
    ] == sorted(
        item.name
        for item in items
    )
