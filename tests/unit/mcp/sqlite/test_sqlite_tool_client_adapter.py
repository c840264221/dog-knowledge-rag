"""
SQLite MCP 工具客户端适配器测试。

功能：
    验证 SQLiteMcpToolClientAdapter 能把 MCP 工具名分发给 SQLite 只读客户端。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.graph.tools.runtime.mcp_tool_executor_adapter import (
    execute_mcp_tool_call,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.mcp.sqlite.client import SQLiteMcpReadonlyClient
from src.mcp.sqlite.tool_client_adapter import SQLiteMcpToolClientAdapter
from src.mcp.sqlite.tool_definitions import (
    SQLITE_DESCRIBE_TABLE_TOOL_NAME,
    SQLITE_LIST_TABLES_TOOL_NAME,
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
)


def create_test_database(
    database_path: Path,
) -> None:
    """
    创建测试 SQLite 数据库。

    功能：
        在临时路径中创建 dogs 表，并插入两行测试数据。

    参数：
        database_path:
            SQLite 数据库文件路径。

    返回值：
        None。
    """

    connection = sqlite3.connect(
        database_path
    )

    try:
        connection.execute(
            """
            CREATE TABLE dogs (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                size TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO dogs(name, size) VALUES (?, ?)",
            [
                (
                    "Golden Retriever",
                    "large",
                ),
                (
                    "Poodle",
                    "medium",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def build_test_adapter(
    tmp_path: Path,
) -> SQLiteMcpToolClientAdapter:
    """
    构建测试 SQLite MCP 工具客户端适配器。

    功能：
        创建临时 SQLite 数据库，构建只读客户端，再包成 MCP 工具客户端适配器。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        SQLiteMcpToolClientAdapter:
            测试用 MCP 工具客户端适配器。
    """

    database_path = tmp_path / "dogs.sqlite3"
    create_test_database(
        database_path=database_path,
    )
    readonly_client = SQLiteMcpReadonlyClient(
        allowed_databases={
            "dogs_db": database_path,
        },
        default_limit=1,
        max_limit=2,
    )

    return SQLiteMcpToolClientAdapter(
        readonly_client=readonly_client,
    )


@pytest.mark.asyncio
async def test_call_tool_should_list_tables(
    tmp_path: Path,
) -> None:
    """
    测试 call_tool 分发到 list_tables。

    功能：
        通过 sqlite_list_tables 工具名获取数据库用户表列表。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )

    result = await adapter.call_tool(
        name=SQLITE_LIST_TABLES_TOOL_NAME,
        arguments={
            "database_name": "dogs_db",
        },
    )

    assert result.tables == [
        "dogs",
    ]


@pytest.mark.asyncio
async def test_call_tool_should_describe_table(
    tmp_path: Path,
) -> None:
    """
    测试 call_tool 分发到 describe_table。

    功能：
        通过 sqlite_describe_table 工具名获取表字段结构。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )

    result = await adapter.call_tool(
        name=SQLITE_DESCRIBE_TABLE_TOOL_NAME,
        arguments={
            "database_name": "dogs_db",
            "table_name": "dogs",
        },
    )

    assert [
        column.name
        for column in result.columns
    ] == [
        "id",
        "name",
        "size",
    ]


@pytest.mark.asyncio
async def test_call_tool_should_select_rows(
    tmp_path: Path,
) -> None:
    """
    测试 call_tool 分发到 select_rows。

    功能：
        通过 sqlite_select_rows 工具名读取表数据。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )

    result = await adapter.call_tool(
        name=SQLITE_SELECT_ROWS_TOOL_NAME,
        arguments={
            "database_name": "dogs_db",
            "table_name": "dogs",
            "limit": 1,
        },
    )

    assert result.rows == [
        {
            "id": 1,
            "name": "Golden Retriever",
            "size": "large",
        }
    ]


@pytest.mark.asyncio
async def test_call_tool_should_run_readonly_query(
    tmp_path: Path,
) -> None:
    """
    测试 call_tool 分发到 run_readonly_query。

    功能：
        通过 sqlite_run_readonly_query 工具名执行只读 SQL。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )

    result = await adapter.call_tool(
        name=SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
        arguments={
            "database_name": "dogs_db",
            "sql": "SELECT name FROM dogs ORDER BY id",
            "limit": 2,
        },
    )

    assert result.row_count == 2
    assert result.rows == [
        {
            "name": "Golden Retriever",
        },
        {
            "name": "Poodle",
        },
    ]


@pytest.mark.asyncio
async def test_call_tool_should_reject_unknown_tool(
    tmp_path: Path,
) -> None:
    """
    测试未知 SQLite MCP 工具会被拒绝。

    功能：
        call_tool 只允许分发 SQLite MCP 工具定义中的工具名。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )

    with pytest.raises(
        KeyError,
    ):
        await adapter.call_tool(
            name="unknown_sqlite_tool",
            arguments={},
        )


@pytest.mark.asyncio
async def test_adapter_should_work_with_mcp_tool_executor(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP 适配器可以接入通用 MCP 工具执行器。

    功能：
        使用 ToolCall 调用 execute_mcp_tool_call，
        验证返回结果被包装成内部 ToolResult。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    adapter = build_test_adapter(
        tmp_path=tmp_path,
    )
    tool_call = ToolCall(
        name=SQLITE_SELECT_ROWS_TOOL_NAME,
        args={
            "database_name": "dogs_db",
            "table_name": "dogs",
            "limit": 1,
        },
    )

    result = await execute_mcp_tool_call(
        tool_call=tool_call,
        mcp_client=adapter,
    )

    assert result.success is True
    assert result.tool_name == SQLITE_SELECT_ROWS_TOOL_NAME
    assert result.content.rows[0]["name"] == "Golden Retriever"
    assert result.metadata["source"] == "mcp"
