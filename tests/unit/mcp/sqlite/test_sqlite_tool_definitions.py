"""
SQLite MCP 工具定义测试。

功能：
    验证 SQLite 只读能力被正确声明成 Mock MCP 工具定义。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.mcp_tool_schema_adapter import (
    build_tool_metadata_list_from_mcp_tools,
)
from src.mcp.sqlite.tool_definitions import (
    SQLITE_DESCRIBE_TABLE_TOOL_NAME,
    SQLITE_LIST_TABLES_TOOL_NAME,
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
    build_sqlite_mcp_tool_definitions,
)


def test_build_sqlite_mcp_tool_definitions_should_return_four_tools() -> None:
    """
    测试 SQLite MCP 工具定义数量和名称。

    功能：
        验证 SQLite MCP Readonly MVP 暴露四个预期工具。

    参数：
        无。

    返回值：
        None。
    """

    tool_definitions = build_sqlite_mcp_tool_definitions()

    assert [
        tool.name
        for tool in tool_definitions
    ] == [
        SQLITE_LIST_TABLES_TOOL_NAME,
        SQLITE_DESCRIBE_TABLE_TOOL_NAME,
        SQLITE_SELECT_ROWS_TOOL_NAME,
        SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    ]


def test_sqlite_fixed_query_tools_should_not_require_confirmation() -> None:
    """
    测试固定只读查询工具不需要确认。

    功能：
        list_tables、describe_table、select_rows 都是固定 SQL 形态，
        当前 MVP 默认不需要用户确认。

    参数：
        无。

    返回值：
        None。
    """

    tool_definitions = build_sqlite_mcp_tool_definitions()
    confirmation_by_name = {
        tool.name: tool.annotations["require_confirm"]
        for tool in tool_definitions
    }

    assert confirmation_by_name[SQLITE_LIST_TABLES_TOOL_NAME] is False
    assert confirmation_by_name[SQLITE_DESCRIBE_TABLE_TOOL_NAME] is False
    assert confirmation_by_name[SQLITE_SELECT_ROWS_TOOL_NAME] is False


def test_sqlite_readonly_query_tool_should_require_confirmation() -> None:
    """
    测试自由只读 SQL 查询工具需要确认。

    功能：
        run_readonly_query 允许用户输入 SQL，风险高于固定查询，
        因此工具定义中 require_confirm 应该为 True。

    参数：
        无。

    返回值：
        None。
    """

    tool_definitions = build_sqlite_mcp_tool_definitions()
    query_tool = next(
        tool
        for tool in tool_definitions
        if tool.name == SQLITE_RUN_READONLY_QUERY_TOOL_NAME
    )

    assert query_tool.annotations["require_confirm"] is True
    assert query_tool.annotations["timeout"] == 8
    assert query_tool.annotations["retries"] == 0


def test_sqlite_tool_definitions_should_have_input_schema() -> None:
    """
    测试 SQLite MCP 工具定义包含 input_schema。

    功能：
        每个工具都应该声明参数结构，方便后续 ToolAgent 或 LLM 理解如何调用。

    参数：
        无。

    返回值：
        None。
    """

    tool_definitions = build_sqlite_mcp_tool_definitions()

    for tool_definition in tool_definitions:
        assert tool_definition.input_schema["type"] == "object"
        assert tool_definition.input_schema["properties"]
        assert tool_definition.input_schema["required"]


def test_sqlite_tool_definitions_should_convert_to_tool_metadata() -> None:
    """
    测试 SQLite MCP 工具定义可以转换成 ToolMetadata。

    功能：
        复用 V1.9.2 的 MCP schema adapter，
        验证 SQLite MCP 工具定义能进入内部工具元数据标准。

    参数：
        无。

    返回值：
        None。
    """

    metadata_items = build_tool_metadata_list_from_mcp_tools(
        mcp_tools=build_sqlite_mcp_tool_definitions(),
    )

    metadata_by_name = {
        metadata.name: metadata
        for metadata in metadata_items
    }

    assert set(
        metadata_by_name
    ) == {
        SQLITE_LIST_TABLES_TOOL_NAME,
        SQLITE_DESCRIBE_TABLE_TOOL_NAME,
        SQLITE_SELECT_ROWS_TOOL_NAME,
        SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    }
    assert metadata_by_name[
        SQLITE_RUN_READONLY_QUERY_TOOL_NAME
    ].require_confirm is True
    assert metadata_by_name[
        SQLITE_SELECT_ROWS_TOOL_NAME
    ].require_confirm is False
    assert metadata_by_name[
        SQLITE_LIST_TABLES_TOOL_NAME
    ].retries == 0
