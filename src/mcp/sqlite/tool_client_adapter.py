"""
SQLite MCP 工具客户端适配器。

功能：
    将 SQLiteMcpReadonlyClient 包装成 MCP 风格的工具客户端。

设计原则：
    1. SQLiteMcpReadonlyClient 继续负责真实 SQLite 只读查询。
    2. 本适配器只负责把工具名和参数分发到对应方法。
    3. 上层 MCP 执行适配器只需要调用 call_tool，不需要了解 SQLite 内部方法名。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.mcp.sqlite.client import SQLiteMcpReadonlyClient
from src.mcp.sqlite.tool_definitions import (
    SQLITE_DESCRIBE_TABLE_TOOL_NAME,
    SQLITE_LIST_TABLES_TOOL_NAME,
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
)


class SQLiteMcpToolClientAdapter:
    """
    SQLite MCP 工具客户端适配器。

    功能：
        提供 async call_tool(name, arguments) 统一入口，
        并将 MCP 工具名分发给 SQLiteMcpReadonlyClient 的具体方法。

    参数：
        readonly_client:
            SQLite MCP 只读客户端，负责真实数据库访问。

    返回值：
        SQLiteMcpToolClientAdapter:
            可被 execute_mcp_tool_call 调用的 MCP 风格客户端对象。
    """

    def __init__(
        self,
        readonly_client: SQLiteMcpReadonlyClient,
    ) -> None:
        """
        初始化 SQLite MCP 工具客户端适配器。

        功能：
            保存 SQLiteMcpReadonlyClient 实例，供 call_tool 分发调用。

        参数：
            readonly_client:
                SQLite MCP 只读客户端。

        返回值：
            None。
        """

        self.readonly_client = readonly_client

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        调用 SQLite MCP 工具。

        功能：
            根据 MCP 工具名调用对应的 SQLiteMcpReadonlyClient 方法。

        参数：
            name:
                MCP 工具名称。

            arguments:
                MCP 工具参数字典。

        返回值：
            Any:
                SQLiteMcpReadonlyClient 返回的结构化结果对象。

        异常：
            KeyError:
                工具名不属于 SQLite MCP 工具集合时抛出。
        """

        safe_arguments = dict(
            arguments
            or {}
        )

        if name == SQLITE_LIST_TABLES_TOOL_NAME:
            return self.readonly_client.list_tables(
                database_name=safe_arguments["database_name"],
            )

        if name == SQLITE_DESCRIBE_TABLE_TOOL_NAME:
            return self.readonly_client.describe_table(
                database_name=safe_arguments["database_name"],
                table_name=safe_arguments["table_name"],
            )

        if name == SQLITE_SELECT_ROWS_TOOL_NAME:
            return self.readonly_client.select_rows(
                database_name=safe_arguments["database_name"],
                table_name=safe_arguments["table_name"],
                limit=safe_arguments.get(
                    "limit"
                ),
            )

        if name == SQLITE_RUN_READONLY_QUERY_TOOL_NAME:
            return self.readonly_client.run_readonly_query(
                database_name=safe_arguments["database_name"],
                sql=safe_arguments["sql"],
                limit=safe_arguments.get(
                    "limit"
                ),
            )

        raise KeyError(
            f"未知 SQLite MCP 工具: {name}"
        )
