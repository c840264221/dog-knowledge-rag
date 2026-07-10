"""
SQLite MCP Provider 测试。

功能：
    验证 SQLiteMcpProvider 能统一提供 SQLite MCP 客户端、工具客户端适配器和工具定义。
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
    SQLITE_LIST_TABLES_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
)
from src.settings.mcp import SQLiteMcpSettings
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.sqlite_mcp_provider import SQLiteMcpProvider


def create_test_database(
    database_path: Path,
) -> None:
    """
    创建测试 SQLite 数据库。

    功能：
        在临时路径创建 dogs 表，并插入一条测试数据。

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
                name TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO dogs(name) VALUES (?)",
            (
                "Golden Retriever",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def build_test_provider(
    tmp_path: Path,
) -> SQLiteMcpProvider:
    """
    构建测试 SQLite MCP Provider。

    功能：
        创建临时 SQLite 数据库，并将其注册进 Provider 白名单。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        SQLiteMcpProvider:
            测试用 SQLite MCP Provider。
    """

    database_path = tmp_path / "dogs.sqlite3"
    create_test_database(
        database_path=database_path,
    )

    return SQLiteMcpProvider(
        allowed_databases={
            "dogs_db": database_path,
        },
        default_limit=1,
        max_limit=2,
    )


def test_sqlite_mcp_provider_should_build_clients(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP Provider 能创建客户端对象。

    功能：
        验证 readonly_client 和 tool_client 的类型正确。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    provider = build_test_provider(
        tmp_path=tmp_path,
    )

    assert isinstance(
        provider.readonly_client,
        SQLiteMcpReadonlyClient,
    )
    assert isinstance(
        provider.tool_client,
        SQLiteMcpToolClientAdapter,
    )


def test_sqlite_mcp_provider_should_cache_clients(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP Provider 会缓存客户端对象。

    功能：
        多次访问 readonly_client 和 tool_client 时应返回同一个实例。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    provider = build_test_provider(
        tmp_path=tmp_path,
    )

    assert provider.readonly_client is provider.readonly_client
    assert provider.tool_client is provider.tool_client


def test_sqlite_mcp_provider_should_return_tool_definitions(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP Provider 能提供工具定义。

    功能：
        验证 provider.tool_definitions 包含 SQLite MCP 工具名称。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    provider = build_test_provider(
        tmp_path=tmp_path,
    )

    tool_names = {
        tool_definition.name
        for tool_definition in provider.tool_definitions
    }

    assert SQLITE_LIST_TABLES_TOOL_NAME in tool_names
    assert SQLITE_SELECT_ROWS_TOOL_NAME in tool_names


def test_sqlite_mcp_provider_should_accept_settings_values(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP Provider 能接收 settings 配置值。

    功能：
        使用 SQLiteMcpSettings 构造 Provider，
        验证数据库白名单和 limit 配置会传入 provider。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    database_path = tmp_path / "dogs.sqlite3"
    create_test_database(
        database_path=database_path,
    )
    sqlite_settings = SQLiteMcpSettings(
        allowed_databases={
            "dogs_db": str(
                database_path
            ),
        },
        default_limit=1,
        max_limit=2,
    )
    provider = SQLiteMcpProvider(
        allowed_databases=sqlite_settings.allowed_databases,
        default_limit=sqlite_settings.default_limit,
        max_limit=sqlite_settings.max_limit,
    )

    assert provider.allowed_databases == {
        "dogs_db": str(
            database_path
        ),
    }
    assert provider.readonly_client.default_limit == 1
    assert provider.readonly_client.max_limit == 2


@pytest.mark.asyncio
async def test_sqlite_mcp_provider_should_work_with_container_lifecycle(
    tmp_path: Path,
) -> None:
    """
    测试 SQLite MCP Provider 可以接入 RuntimeContainer 生命周期。

    功能：
        将 provider 注册进容器，执行 startup 和 shutdown，确认流程不报错。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    container = RuntimeContainer()
    provider = build_test_provider(
        tmp_path=tmp_path,
    )
    container.register(
        "sqlite_mcp",
        provider,
    )

    await container.startup()
    assert container.get(
        "sqlite_mcp"
    ).tool_client is provider.tool_client
    await container.shutdown()


@pytest.mark.asyncio
async def test_sqlite_mcp_provider_tool_client_should_work_with_executor(
    tmp_path: Path,
) -> None:
    """
    测试 Provider 提供的 tool_client 可以接入 MCP 工具执行器。

    功能：
        使用 execute_mcp_tool_call 调用 provider.tool_client，
        验证返回标准 ToolResult。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    provider = build_test_provider(
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
        mcp_client=provider.tool_client,
    )

    assert result.success is True
    assert result.content.rows == [
        {
            "id": 1,
            "name": "Golden Retriever",
        }
    ]
