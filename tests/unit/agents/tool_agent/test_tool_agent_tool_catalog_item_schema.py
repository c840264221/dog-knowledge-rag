"""
ToolAgent 工具目录条目 Schema 测试。

功能：
    验证 ToolCatalogItem 字段默认值和可序列化输出。
"""

from __future__ import annotations

from src.agents.tool_agent.contracts.tool_catalog_item_schema import (
    ToolCatalogItem,
)


def test_tool_catalog_item_should_have_safe_defaults() -> None:
    """
    测试 ToolCatalogItem 默认值。

    功能：
        确认最小工具目录条目也能生成稳定字段。

    参数：
        无。

    返回值：
        None。
    """

    item = ToolCatalogItem(
        name="date",
    )

    assert item.description == ""
    assert item.timeout == 5
    assert item.retries == 3
    assert item.require_confirm is False
    assert item.input_schema == {}
    assert item.source == "local"


def test_tool_catalog_item_should_dump_plain_dict() -> None:
    """
    测试 ToolCatalogItem 可转普通 dict。

    功能：
        确认工具目录条目可以安全写入 LangGraph state。

    参数：
        无。

    返回值：
        None。
    """

    item = ToolCatalogItem(
        name="sqlite_select_rows",
        description="查看表数据",
        timeout=5,
        retries=0,
        require_confirm=False,
        input_schema={
            "type": "object",
        },
        source="mcp",
    )

    assert item.model_dump() == {
        "name": "sqlite_select_rows",
        "description": "查看表数据",
        "timeout": 5,
        "retries": 0,
        "require_confirm": False,
        "input_schema": {
            "type": "object",
        },
        "source": "mcp",
    }
