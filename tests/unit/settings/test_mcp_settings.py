"""
MCP settings 测试。

功能：
    验证 MCP 配置对象的默认值和 SQLite MCP 配置结构。
"""

from __future__ import annotations

from src.settings.mcp import McpSettings, SQLiteMcpSettings


def test_sqlite_mcp_settings_should_use_safe_defaults() -> None:
    """
    测试 SQLite MCP 配置默认值。

    功能：
        验证默认启用 SQLite MCP，但默认没有数据库白名单。

    参数：
        无。

    返回值：
        None。
    """

    sqlite_settings = SQLiteMcpSettings()

    assert sqlite_settings.enabled is True
    assert sqlite_settings.allowed_databases == {}
    assert sqlite_settings.default_limit == 20
    assert sqlite_settings.max_limit == 100


def test_mcp_settings_should_include_sqlite_settings() -> None:
    """
    测试 MCP 总配置包含 SQLite MCP 配置。

    功能：
        验证 McpSettings 会创建 sqlite 子配置对象。

    参数：
        无。

    返回值：
        None。
    """

    mcp_settings = McpSettings()

    assert isinstance(
        mcp_settings.sqlite,
        SQLiteMcpSettings,
    )
