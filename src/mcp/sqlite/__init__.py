"""
SQLite MCP Readonly 模块。

功能：
    提供学习用 SQLite MCP 只读客户端和安全校验能力。

当前阶段：
    V1.9.4 只实现底层只读客户端，不接入 ToolAgent 和真实 MCP Server。
"""

from src.mcp.sqlite.client import (
    SQLiteMcpReadonlyClient,
)
from src.mcp.sqlite.readonly_guard import (
    ensure_readonly_sql,
    normalize_sql_limit,
)
from src.mcp.sqlite.schemas import (
    SQLiteColumnInfo,
    SQLiteDescribeTableResult,
    SQLiteListTablesResult,
    SQLiteQueryResult,
    SQLiteSelectRowsResult,
)
from src.mcp.sqlite.tool_definitions import (
    SQLITE_DESCRIBE_TABLE_TOOL_NAME,
    SQLITE_LIST_TABLES_TOOL_NAME,
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
    build_sqlite_mcp_tool_definitions,
)
from src.mcp.sqlite.tool_client_adapter import (
    SQLiteMcpToolClientAdapter,
)

__all__ = [
    "SQLITE_DESCRIBE_TABLE_TOOL_NAME",
    "SQLITE_LIST_TABLES_TOOL_NAME",
    "SQLITE_RUN_READONLY_QUERY_TOOL_NAME",
    "SQLITE_SELECT_ROWS_TOOL_NAME",
    "SQLiteColumnInfo",
    "SQLiteDescribeTableResult",
    "SQLiteListTablesResult",
    "SQLiteMcpReadonlyClient",
    "SQLiteMcpToolClientAdapter",
    "SQLiteQueryResult",
    "SQLiteSelectRowsResult",
    "build_sqlite_mcp_tool_definitions",
    "ensure_readonly_sql",
    "normalize_sql_limit",
]
