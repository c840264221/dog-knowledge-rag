"""
ToolAgent MCP 工具目录适配测试。

功能：
    验证 ToolAgent 可以把本地工具目录和 MCP 工具定义合并成统一工具目录。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    build_tool_agent_tool_catalog_state_update_with_mcp,
    build_tool_agent_tool_catalog_with_mcp,
    merge_tool_catalog_items,
    merge_tool_metadata_items,
)
from src.agents.tool_agent.contracts.tool_catalog_item_schema import (
    ToolCatalogItem,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.schemas import MockMcpToolDefinition
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
            测试用工具实例。
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
            工具名称到工具对象的映射。

    返回值：
        FakeRegistry:
            测试用工具注册表实例。
    """

    def __init__(
        self,
        tools: dict[str, FakeTool],
    ) -> None:
        self.tools = tools


def build_fake_registry() -> FakeRegistry:
    """
    构建测试用本地工具注册表。

    功能：
        创建包含 weather 和 date 的本地工具注册表。

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
                    timeout=8,
                    retries=2,
                    require_confirm=True,
                )
            ),
            "date": FakeTool(
                ToolMetadata(
                    name="date",
                    description="查询日期",
                    timeout=3,
                    retries=1,
                    require_confirm=False,
                )
            ),
        }
    )


def test_merge_tool_metadata_items_should_keep_base_when_name_conflicts() -> None:
    """
    测试合并工具元数据时本地工具优先。

    功能：
        当 base 和 extra 中出现同名工具时，保留 base 中的工具元数据。

    参数：
        无。

    返回值：
        None。
    """

    merged_items = merge_tool_metadata_items(
        base_metadata_items=[
            ToolMetadata(
                name="weather",
                description="本地天气工具",
                timeout=8,
                retries=2,
                require_confirm=True,
            )
        ],
        extra_metadata_items=[
            ToolMetadata(
                name="weather",
                description="MCP 天气工具",
                timeout=5,
                retries=0,
                require_confirm=False,
            )
        ],
    )

    assert len(
        merged_items
    ) == 1
    assert merged_items[0].description == "本地天气工具"
    assert merged_items[0].require_confirm is True


def test_merge_tool_catalog_items_should_keep_base_when_name_conflicts() -> None:
    """
    测试合并统一工具目录条目时本地工具优先。

    功能：
        当本地工具和 MCP 工具同名时，保留本地 ToolCatalogItem。

    参数：
        无。

    返回值：
        None。
    """

    merged_items = merge_tool_catalog_items(
        base_catalog_items=[
            ToolCatalogItem(
                name="weather",
                description="本地天气工具",
                timeout=8,
                retries=2,
                require_confirm=True,
                source="local",
            )
        ],
        extra_catalog_items=[
            ToolCatalogItem(
                name="weather",
                description="MCP 天气工具",
                timeout=5,
                retries=0,
                require_confirm=False,
                source="mcp",
            )
        ],
    )

    assert len(
        merged_items
    ) == 1
    assert merged_items[0].description == "本地天气工具"
    assert merged_items[0].source == "local"


def test_build_tool_agent_tool_catalog_with_mcp_should_include_local_and_mcp_tools() -> None:
    """
    测试工具目录包含本地工具和 MCP 工具。

    功能：
        构建 ToolAgent 工具目录时，同时包含 weather/date 和 SQLite MCP 工具。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )
    names = {
        item["name"]
        for item in catalog
    }

    assert "weather" in names
    assert "date" in names
    assert SQLITE_SELECT_ROWS_TOOL_NAME in names
    assert SQLITE_RUN_READONLY_QUERY_TOOL_NAME in names


def test_build_tool_agent_tool_catalog_with_mcp_should_keep_mcp_input_schema() -> None:
    """
    测试 MCP 工具 input_schema 能进入工具目录。

    功能：
        确认 SQLite MCP 工具的参数结构会保留到 ToolAgent 工具目录中。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )
    catalog_by_name = {
        item["name"]: item
        for item in catalog
    }
    sqlite_select_rows = catalog_by_name[SQLITE_SELECT_ROWS_TOOL_NAME]

    assert sqlite_select_rows["source"] == "mcp"
    assert sqlite_select_rows["input_schema"]["properties"]["database_name"][
        "type"
    ] == "string"
    assert sqlite_select_rows["input_schema"]["properties"]["table_name"][
        "type"
    ] == "string"


def test_build_tool_agent_tool_catalog_with_mcp_should_keep_mcp_confirmation_flag() -> None:
    """
    测试 MCP 工具确认标记能保留。

    功能：
        sqlite_run_readonly_query 允许用户输入 SQL，工具定义中 require_confirm=True，
        合并到 ToolAgent 工具目录后也应保持 True。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )
    catalog_by_name = {
        item["name"]: item
        for item in catalog
    }

    assert catalog_by_name[
        SQLITE_RUN_READONLY_QUERY_TOOL_NAME
    ]["require_confirm"] is True
    assert catalog_by_name[
        SQLITE_SELECT_ROWS_TOOL_NAME
    ]["require_confirm"] is False


def test_build_tool_agent_tool_catalog_with_mcp_should_return_plain_dicts() -> None:
    """
    测试工具目录输出普通 dict。

    功能：
        确认合并后的工具目录可以安全写入 LangGraph state，
        不会直接保存 Pydantic 对象。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )

    assert all(
        isinstance(
            item,
            dict,
        )
        for item in catalog
    )


def test_build_tool_agent_tool_catalog_with_mcp_should_not_allow_mcp_override_local_tool() -> None:
    """
    测试 MCP 工具不能覆盖本地同名工具。

    功能：
        如果 MCP 工具定义中出现 weather，同名本地工具仍然优先保留。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=[
            MockMcpToolDefinition(
                name="weather",
                description="MCP 天气工具",
                annotations={
                    "timeout": 1,
                    "retries": 0,
                    "require_confirm": False,
                },
            )
        ],
    )
    catalog_by_name = {
        item["name"]: item
        for item in catalog
    }

    assert catalog_by_name["weather"]["description"] == "查询天气"
    assert catalog_by_name["weather"]["timeout"] == 8
    assert catalog_by_name["weather"]["require_confirm"] is True


def test_build_tool_agent_tool_catalog_with_mcp_should_sort_by_tool_name() -> None:
    """
    测试合并后的工具目录按名称排序。

    功能：
        保证工具目录输出稳定，方便测试、日志和调试。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )

    assert [
        item["name"]
        for item in catalog
    ] == sorted(
        item["name"]
        for item in catalog
    )


def test_build_tool_agent_tool_catalog_state_update_with_mcp_should_use_standard_key() -> None:
    """
    测试 MCP 工具目录 state update 使用标准 key。

    功能：
        验证带 MCP 的工具目录会被包装成
        {"tool_agent_tool_catalog": [...]} 格式。

    参数：
        无。

    返回值：
        None。
    """

    update = build_tool_agent_tool_catalog_state_update_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
    )

    assert set(
        update
    ) == {
        TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    }


def test_build_tool_agent_tool_catalog_state_update_with_mcp_should_include_sqlite_tools() -> None:
    """
    测试 MCP 工具目录 state update 包含 SQLite 工具。

    功能：
        验证 state update 中的工具目录既包含本地工具，
        也包含 SQLite MCP 工具。

    参数：
        无。

    返回值：
        None。
    """

    update = build_tool_agent_tool_catalog_state_update_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
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


def test_build_tool_agent_tool_catalog_state_update_with_mcp_should_return_plain_state() -> None:
    """
    测试 MCP 工具目录 state update 是普通 dict。

    功能：
        确认返回值可以安全写入 LangGraph state，
        不会把 ToolMetadata 或 MockMcpToolDefinition 直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    update = build_tool_agent_tool_catalog_state_update_with_mcp(
        tool_registry=build_fake_registry(),
        mcp_tool_definitions=build_sqlite_mcp_tool_definitions(),
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
