"""
ToolAgent 工具目录节点测试。

功能：
    验证 tool_catalog_node 能把本地工具和 MCP 工具目录写入 state update。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
)
from src.agents.tool_agent.nodes.tool_catalog_node import (
    TOOL_AGENT_ALLOWED_DATABASES_STATE_KEY,
    build_tool_agent_tool_catalog_node,
    resolve_mcp_tool_definitions,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.sqlite.tool_definitions import (
    SQLITE_RUN_READONLY_QUERY_TOOL_NAME,
    SQLITE_SELECT_ROWS_TOOL_NAME,
    build_sqlite_mcp_tool_definitions,
)


class FakeTool:
    """
    测试用工具对象。

    功能：
        模拟真实工具对象，只提供 metadata 字段。

    参数：
        metadata:
            工具元数据。

    返回值：
        FakeTool:
            测试用工具对象。
    """

    def __init__(
        self,
        metadata: ToolMetadata,
    ) -> None:
        self.metadata = metadata


class FakeRegistry:
    """
    测试用工具注册表。

    功能：
        模拟 ToolRegistry，只提供 tools 字段。

    参数：
        tools:
            工具名到工具对象的映射。

    返回值：
        FakeRegistry:
            测试用工具注册表。
    """

    def __init__(
        self,
        tools: dict[str, FakeTool],
    ) -> None:
        self.tools = tools


class FakeSQLiteMcpProvider:
    """
    测试用 SQLite MCP Provider。

    功能：
        模拟真实 provider，只提供 tool_definitions 属性。

    参数：
        无。

    返回值：
        FakeSQLiteMcpProvider:
            测试用 provider。
    """

    allowed_databases = {
        "memory": "chroma_memory_db/chroma.sqlite3",
        "rag": "chroma_db/chroma.sqlite3",
    }

    @property
    def tool_definitions(
        self,
    ):
        """
        获取测试用 SQLite MCP 工具定义。

        功能：
            返回真实 SQLite MCP 工具定义，模拟 provider 行为。

        参数：
            无。

        返回值：
            list[MockMcpToolDefinition]:
                SQLite MCP 工具定义列表。
        """

        return build_sqlite_mcp_tool_definitions()


def build_fake_registry() -> FakeRegistry:
    """
    构建测试用本地工具注册表。

    功能：
        创建包含 weather 和 date 的工具注册表。

    参数：
        无。

    返回值：
        FakeRegistry:
            测试用本地工具注册表。
    """

    return FakeRegistry(
        tools={
            "weather": FakeTool(
                ToolMetadata(
                    name="weather",
                    description="查询天气",
                    require_confirm=True,
                )
            ),
            "date": FakeTool(
                ToolMetadata(
                    name="date",
                    description="查询日期",
                    require_confirm=False,
                )
            ),
        }
    )


@pytest.mark.asyncio
async def test_tool_catalog_node_should_return_local_tool_catalog() -> None:
    """
    测试工具目录节点返回本地工具目录。

    功能：
        未传 MCP 工具定义时，节点应返回 weather 和 date 工具目录。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_catalog_node(
        tool_registry=build_fake_registry(),
        runtime_context_getter=lambda: None,
    )

    update = await node(
        {
            "question": "今天几号？",
        }
    )
    catalog = update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY]
    names = {
        item["name"]
        for item in catalog
    }

    assert names == {
        "weather",
        "date",
    }


@pytest.mark.asyncio
async def test_tool_catalog_node_should_include_mcp_tool_definitions() -> None:
    """
    测试工具目录节点可以包含 MCP 工具定义。

    功能：
        显式传入 SQLite MCP 工具定义时，节点应返回本地工具 + SQLite MCP 工具。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_catalog_node(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
        runtime_context_getter=lambda: None,
    )

    update = await node(
        {
            "question": "查看数据库表",
        }
    )
    catalog = update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY]
    names = {
        item["name"]
        for item in catalog
    }

    assert "weather" in names
    assert "date" in names
    assert SQLITE_SELECT_ROWS_TOOL_NAME in names
    assert SQLITE_RUN_READONLY_QUERY_TOOL_NAME in names


@pytest.mark.asyncio
async def test_tool_catalog_node_should_read_mcp_definitions_from_provider() -> None:
    """
    测试工具目录节点可以从 provider 读取 MCP 工具定义。

    功能：
        未显式传入 mcp_tool_definitions 时，
        节点应尝试读取 sqlite_mcp_provider.tool_definitions。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_catalog_node(
        tool_registry=build_fake_registry(),
        sqlite_mcp_provider=FakeSQLiteMcpProvider(),
        runtime_context_getter=lambda: None,
    )

    update = await node(
        {}
    )
    catalog = update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY]
    names = {
        item["name"]
        for item in catalog
    }

    assert SQLITE_SELECT_ROWS_TOOL_NAME in names
    assert SQLITE_RUN_READONLY_QUERY_TOOL_NAME in names
    assert update[TOOL_AGENT_ALLOWED_DATABASES_STATE_KEY] == {
        "memory": "memory",
        "rag": "rag",
    }

    sqlite_item = next(
        item
        for item in catalog
        if item["name"] == SQLITE_SELECT_ROWS_TOOL_NAME
    )
    database_name_schema = sqlite_item["input_schema"]["properties"][
        "database_name"
    ]

    assert database_name_schema["enum"] == [
        "memory",
        "rag",
    ]
    assert database_name_schema["x-requires-explicit-user-input"] is True
    assert "memory、rag" in database_name_schema["description"]


def test_resolve_mcp_tool_definitions_should_prefer_explicit_definitions() -> None:
    """
    测试 MCP 工具定义解析优先使用显式参数。

    功能：
        同时传入 mcp_tool_definitions 和 provider 时，
        resolve_mcp_tool_definitions 应返回显式传入的对象。

    参数：
        无。

    返回值：
        None。
    """

    explicit_definitions = build_sqlite_mcp_tool_definitions()

    resolved = resolve_mcp_tool_definitions(
        mcp_tool_definitions=explicit_definitions,
        sqlite_mcp_provider=FakeSQLiteMcpProvider(),
    )

    assert resolved is explicit_definitions


@pytest.mark.asyncio
async def test_tool_catalog_node_should_return_plain_state_update() -> None:
    """
    测试工具目录节点返回普通 state update。

    功能：
        确认节点输出可以安全写入 LangGraph state，
        不会直接保存 ToolMetadata 或 MockMcpToolDefinition 对象。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_catalog_node(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
        runtime_context_getter=lambda: None,
    )

    update = await node(
        {}
    )
    catalog = update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY]

    assert isinstance(
        update,
        dict,
    )
    assert isinstance(
        catalog,
        list,
    )
    assert all(
        isinstance(
            item,
            dict,
        )
        for item in catalog
    )
