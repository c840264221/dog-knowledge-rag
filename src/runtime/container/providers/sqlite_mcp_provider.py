"""
SQLite MCP Provider。

功能：
    将 SQLite MCP 只读客户端、工具客户端适配器和工具定义统一交给 RuntimeContainer 管理。

设计原则：
    1. Provider 负责创建和持有 SQLite MCP 相关对象。
    2. 业务层后续通过 container.get("sqlite_mcp") 获取能力。
    3. 当前阶段只接入容器，不接入 ToolAgent 主链路。
"""

from __future__ import annotations

from pathlib import Path

from src.logger import logger
from src.mcp.schemas import MockMcpToolDefinition
from src.mcp.sqlite.client import SQLiteMcpReadonlyClient
from src.mcp.sqlite.tool_client_adapter import SQLiteMcpToolClientAdapter
from src.mcp.sqlite.tool_definitions import (
    build_sqlite_mcp_tool_definitions,
)


class SQLiteMcpProvider:
    """
    SQLite MCP 服务提供者。

    功能：
        统一创建和管理 SQLiteMcpReadonlyClient、SQLiteMcpToolClientAdapter
        和 SQLite MCP 工具定义。

    参数：
        allowed_databases:
            SQLite 数据库白名单，key 是数据库别名，value 是数据库文件路径。

        default_limit:
            默认返回行数。

        max_limit:
            最大返回行数。

    返回值：
        SQLiteMcpProvider:
            可注册进 RuntimeContainer 的 Provider 对象。
    """

    def __init__(
        self,
        allowed_databases: dict[str, str | Path] | None = None,
        default_limit: int = 20,
        max_limit: int = 100,
    ) -> None:
        """
        初始化 SQLite MCP Provider。

        功能：
            保存 SQLite MCP 配置，并准备懒加载内部客户端对象。

        参数：
            allowed_databases:
                SQLite 数据库白名单。当前如果不传，则默认没有可访问数据库。

            default_limit:
                默认返回行数。

            max_limit:
                最大返回行数。

        返回值：
            None。
        """

        self.allowed_databases = dict(
            allowed_databases
            or {}
        )
        self.default_limit = default_limit
        self.max_limit = max_limit
        self._readonly_client: SQLiteMcpReadonlyClient | None = None
        self._tool_client: SQLiteMcpToolClientAdapter | None = None
        self._tool_definitions: list[MockMcpToolDefinition] | None = None

    @property
    def readonly_client(
        self,
    ) -> SQLiteMcpReadonlyClient:
        """
        获取 SQLite MCP 只读客户端。

        功能：
            懒加载 SQLiteMcpReadonlyClient，统一管理 SQLite 只读查询能力。

        参数：
            无。

        返回值：
            SQLiteMcpReadonlyClient:
                SQLite MCP 只读客户端。
        """

        if self._readonly_client is None:
            logger.info(
                "初始化 SQLiteMcpReadonlyClient..."
            )

            self._readonly_client = SQLiteMcpReadonlyClient(
                allowed_databases=self.allowed_databases,
                default_limit=self.default_limit,
                max_limit=self.max_limit,
            )

        return self._readonly_client

    @property
    def tool_client(
        self,
    ) -> SQLiteMcpToolClientAdapter:
        """
        获取 SQLite MCP 工具客户端适配器。

        功能：
            懒加载 SQLiteMcpToolClientAdapter，
            对外提供统一的 call_tool(name, arguments) MCP 风格入口。

        参数：
            无。

        返回值：
            SQLiteMcpToolClientAdapter:
                SQLite MCP 工具客户端适配器。
        """

        if self._tool_client is None:
            logger.info(
                "初始化 SQLiteMcpToolClientAdapter..."
            )

            self._tool_client = SQLiteMcpToolClientAdapter(
                readonly_client=self.readonly_client,
            )

        return self._tool_client

    @property
    def tool_definitions(
        self,
    ) -> list[MockMcpToolDefinition]:
        """
        获取 SQLite MCP 工具定义列表。

        功能：
            懒加载 SQLite MCP 工具定义，供后续 ToolAgent 工具目录适配使用。

        参数：
            无。

        返回值：
            list[MockMcpToolDefinition]:
                SQLite MCP 工具定义列表。
        """

        if self._tool_definitions is None:
            logger.info(
                "初始化 SQLite MCP 工具定义..."
            )

            self._tool_definitions = build_sqlite_mcp_tool_definitions()

        return self._tool_definitions

    async def startup(
        self,
    ) -> None:
        """
        启动 SQLite MCP Provider。

        功能：
            提前初始化 SQLite MCP 只读客户端、工具客户端适配器和工具定义。

        参数：
            无。

        返回值：
            None。
        """

        _ = self.readonly_client
        _ = self.tool_client
        _ = self.tool_definitions

        logger.info(
            "SQLiteMcpProvider 启动完成"
        )

    async def shutdown(
        self,
    ) -> None:
        """
        关闭 SQLite MCP Provider。

        功能：
            当前 SQLiteMcpReadonlyClient 每次查询都使用短连接，
            Provider 暂无需要主动关闭的长连接资源。

        参数：
            无。

        返回值：
            None。
        """

        logger.info(
            "SQLiteMcpProvider 已关闭"
        )
