"""
SQLite MCP 工具定义。

功能：
    将 SQLiteMcpReadonlyClient 的只读能力声明成 Mock MCP 工具定义。

设计原则：
    1. 本模块只描述工具，不执行工具。
    2. 固定只读工具默认不需要确认。
    3. 自由 SQL 查询工具风险更高，需要用户确认。
    4. 输出 MockMcpToolDefinition，后续可通过 MCP schema adapter 转成 ToolMetadata。
"""

from __future__ import annotations

from src.mcp.schemas import MockMcpToolDefinition


SQLITE_LIST_TABLES_TOOL_NAME = "sqlite_list_tables"
SQLITE_DESCRIBE_TABLE_TOOL_NAME = "sqlite_describe_table"
SQLITE_SELECT_ROWS_TOOL_NAME = "sqlite_select_rows"
SQLITE_RUN_READONLY_QUERY_TOOL_NAME = "sqlite_run_readonly_query"

SQLITE_FIXED_TOOL_TIMEOUT = 5
SQLITE_QUERY_TOOL_TIMEOUT = 8
SQLITE_TOOL_RETRIES = 0


def build_sqlite_mcp_tool_definitions() -> list[MockMcpToolDefinition]:
    """
    构建 SQLite MCP 工具定义列表。

    功能：
        声明 SQLite MCP Readonly Client 提供的四个只读工具：
        1. sqlite_list_tables
        2. sqlite_describe_table
        3. sqlite_select_rows
        4. sqlite_run_readonly_query

    参数：
        无。

    返回值：
        list[MockMcpToolDefinition]:
            SQLite MCP 工具定义列表。
    """

    return [
        build_sqlite_list_tables_tool_definition(),
        build_sqlite_describe_table_tool_definition(),
        build_sqlite_select_rows_tool_definition(),
        build_sqlite_run_readonly_query_tool_definition(),
    ]


def build_sqlite_list_tables_tool_definition() -> MockMcpToolDefinition:
    """
    构建 sqlite_list_tables 工具定义。

    功能：
        声明列出 SQLite 数据库用户表的工具。

    参数：
        无。

    返回值：
        MockMcpToolDefinition:
            sqlite_list_tables 工具定义。
    """

    return MockMcpToolDefinition(
        name=SQLITE_LIST_TABLES_TOOL_NAME,
        description="列出允许访问的 SQLite 数据库中的用户表。",
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名，例如 memory 或 checkpoint。",
                },
            },
            "required": [
                "database_name",
            ],
        },
        annotations=build_sqlite_tool_annotations(
            require_confirm=False,
            timeout=SQLITE_FIXED_TOOL_TIMEOUT,
        ),
    )


def build_sqlite_describe_table_tool_definition() -> MockMcpToolDefinition:
    """
    构建 sqlite_describe_table 工具定义。

    功能：
        声明查看 SQLite 表字段结构的工具。

    参数：
        无。

    返回值：
        MockMcpToolDefinition:
            sqlite_describe_table 工具定义。
    """

    return MockMcpToolDefinition(
        name=SQLITE_DESCRIBE_TABLE_TOOL_NAME,
        description="查看允许访问的 SQLite 数据库中某张表的字段结构。",
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                },
                "table_name": {
                    "type": "string",
                    "description": "要查看结构的表名。",
                },
            },
            "required": [
                "database_name",
                "table_name",
            ],
        },
        annotations=build_sqlite_tool_annotations(
            require_confirm=False,
            timeout=SQLITE_FIXED_TOOL_TIMEOUT,
        ),
    )


def build_sqlite_select_rows_tool_definition() -> MockMcpToolDefinition:
    """
    构建 sqlite_select_rows 工具定义。

    功能：
        声明读取 SQLite 表前 N 行的固定只读工具。

    参数：
        无。

    返回值：
        MockMcpToolDefinition:
            sqlite_select_rows 工具定义。
    """

    return MockMcpToolDefinition(
        name=SQLITE_SELECT_ROWS_TOOL_NAME,
        description="查看允许访问的 SQLite 数据库中某张表的前 N 行数据。",
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                },
                "table_name": {
                    "type": "string",
                    "description": "要读取数据的表名。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少行，实际会受到客户端 max_limit 限制。",
                    "default": 20,
                    "minimum": 1,
                },
            },
            "required": [
                "database_name",
                "table_name",
            ],
        },
        annotations=build_sqlite_tool_annotations(
            require_confirm=False,
            timeout=SQLITE_FIXED_TOOL_TIMEOUT,
        ),
    )


def build_sqlite_run_readonly_query_tool_definition() -> MockMcpToolDefinition:
    """
    构建 sqlite_run_readonly_query 工具定义。

    功能：
        声明执行自由只读 SQL 的工具。
        该工具允许用户传入 SQL，风险高于固定查询，因此默认需要确认。

    参数：
        无。

    返回值：
        MockMcpToolDefinition:
            sqlite_run_readonly_query 工具定义。
    """

    return MockMcpToolDefinition(
        name=SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
        description="执行允许访问的 SQLite 数据库上的只读 SELECT/WITH 查询。",
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                },
                "sql": {
                    "type": "string",
                    "description": "只读 SQL，只允许 SELECT 或 WITH 查询。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少行，实际会受到客户端 max_limit 限制。",
                    "default": 20,
                    "minimum": 1,
                },
            },
            "required": [
                "database_name",
                "sql",
            ],
        },
        annotations=build_sqlite_tool_annotations(
            require_confirm=True,
            timeout=SQLITE_QUERY_TOOL_TIMEOUT,
        ),
    )


def build_sqlite_tool_annotations(
    require_confirm: bool,
    timeout: int,
) -> dict[str, object]:
    """
    构建 SQLite MCP 工具 annotations。

    功能：
        统一生成 SQLite MCP 工具的确认、超时和重试配置。

    参数：
        require_confirm:
            该工具是否需要用户确认。

        timeout:
            工具超时时间。

    返回值：
        dict[str, object]:
            Mock MCP annotations 字典。
    """

    return {
        "require_confirm": require_confirm,
        "timeout": timeout,
        "retries": SQLITE_TOOL_RETRIES,
    }
