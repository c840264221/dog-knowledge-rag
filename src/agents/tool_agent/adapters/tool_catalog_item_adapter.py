"""
ToolAgent 工具目录条目适配器。

功能：
    将本地 ToolMetadata 和 MCP MockMcpToolDefinition 转换成统一 ToolCatalogItem。

设计说明：
    ToolAgent 后续只需要读取 ToolCatalogItem，就不用关心工具来自本地注册表
    还是 MCP Server。这样可以降低 prompt 构建、日志和 state 写入的复杂度。

专业名词：
    Adapter：
        适配器，用来隔离不同数据结构之间的差异。
    ToolMetadata：
        本地工具元数据。
    MCP：
        Model Context Protocol，模型上下文协议。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.agents.tool_agent.adapters.mcp_tool_schema_adapter import (
    DEFAULT_MCP_REQUIRE_CONFIRM,
    DEFAULT_MCP_TOOL_RETRIES,
    DEFAULT_MCP_TOOL_TIMEOUT,
    read_bool_annotation,
    read_int_annotation,
)
from src.agents.tool_agent.contracts.tool_catalog_item_schema import (
    ToolCatalogItem,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.schemas import MockMcpToolDefinition


LOCAL_TOOL_SOURCE = "local"
MCP_TOOL_SOURCE = "mcp"


def build_tool_catalog_item_from_tool_metadata(
    metadata: ToolMetadata,
    source: str = LOCAL_TOOL_SOURCE,
) -> ToolCatalogItem:
    """
    将本地 ToolMetadata 转换成 ToolCatalogItem。

    功能：
        读取本地工具名称、描述、超时、重试和确认配置，
        构建 ToolAgent 统一工具目录条目。

    参数：
        metadata:
            本地工具元数据。

        source:
            工具来源标记，默认 local。

    返回值：
        ToolCatalogItem:
            统一工具目录条目。
    """

    return ToolCatalogItem(
        name=metadata.name,
        description=metadata.description,
        timeout=metadata.timeout,
        retries=metadata.retries,
        require_confirm=metadata.require_confirm,
        input_schema=dict(
            metadata.input_schema
        ),
        source=source,
    )


def build_tool_catalog_item_from_mcp_tool(
    mcp_tool: MockMcpToolDefinition,
    default_timeout: int = DEFAULT_MCP_TOOL_TIMEOUT,
    default_retries: int = DEFAULT_MCP_TOOL_RETRIES,
    default_require_confirm: bool = DEFAULT_MCP_REQUIRE_CONFIRM,
) -> ToolCatalogItem:
    """
    将 Mock MCP 工具定义转换成 ToolCatalogItem。

    功能：
        读取 MCP 工具的 name、description、input_schema 和 annotations，
        构建 ToolAgent 统一工具目录条目。

    参数：
        mcp_tool:
            Mock MCP 工具定义。

        default_timeout:
            annotations 中没有 timeout 时使用的默认值。

        default_retries:
            annotations 中没有 retries 时使用的默认值。

        default_require_confirm:
            annotations 中没有 require_confirm 时使用的默认值。

    返回值：
        ToolCatalogItem:
            统一工具目录条目。
    """

    annotations = mcp_tool.annotations

    return ToolCatalogItem(
        name=mcp_tool.name,
        description=mcp_tool.description,
        timeout=read_int_annotation(
            annotations=annotations,
            key="timeout",
            default=default_timeout,
        ),
        retries=read_int_annotation(
            annotations=annotations,
            key="retries",
            default=default_retries,
        ),
        require_confirm=read_bool_annotation(
            annotations=annotations,
            key="require_confirm",
            default=default_require_confirm,
        ),
        input_schema=dict(
            mcp_tool.input_schema
        ),
        source=MCP_TOOL_SOURCE,
    )


def build_tool_catalog_items_from_metadata(
    metadata_items: Iterable[ToolMetadata],
    source: str = LOCAL_TOOL_SOURCE,
) -> list[ToolCatalogItem]:
    """
    批量转换本地工具元数据。

    功能：
        将多个 ToolMetadata 转换成 ToolCatalogItem 列表，
        并按工具名排序，保证输出稳定。

    参数：
        metadata_items:
            本地工具元数据集合。

        source:
            工具来源标记，默认 local。

    返回值：
        list[ToolCatalogItem]:
            按 name 排序后的统一工具目录条目列表。
    """

    catalog_items = [
        build_tool_catalog_item_from_tool_metadata(
            metadata=metadata,
            source=source,
        )
        for metadata in metadata_items
    ]

    return sorted(
        catalog_items,
        key=lambda item: item.name,
    )


def build_tool_catalog_items_from_mcp_tools(
    mcp_tools: Iterable[MockMcpToolDefinition],
) -> list[ToolCatalogItem]:
    """
    批量转换 MCP 工具定义。

    功能：
        将多个 MockMcpToolDefinition 转换成 ToolCatalogItem 列表，
        并按工具名排序，保证输出稳定。

    参数：
        mcp_tools:
            Mock MCP 工具定义集合。

    返回值：
        list[ToolCatalogItem]:
            按 name 排序后的统一工具目录条目列表。
    """

    catalog_items = [
        build_tool_catalog_item_from_mcp_tool(
            mcp_tool=mcp_tool,
        )
        for mcp_tool in mcp_tools
    ]

    return sorted(
        catalog_items,
        key=lambda item: item.name,
    )


def dump_tool_catalog_item_for_state(
    item: ToolCatalogItem,
) -> dict[str, Any]:
    """
    将 ToolCatalogItem 转换成可写入 state 的普通 dict。

    功能：
        避免 LangGraph checkpoint 中保存 Pydantic 自定义对象，
        只保存普通 JSON 风格字典。

    参数：
        item:
            统一工具目录条目。

    返回值：
        dict[str, Any]:
            普通字典格式的工具目录条目。
    """

    return item.model_dump()


def dump_tool_catalog_items_for_state(
    items: Iterable[ToolCatalogItem],
) -> list[dict[str, Any]]:
    """
    批量转换 ToolCatalogItem 列表。

    功能：
        将统一工具目录条目列表转换成可写入 state 的普通 dict 列表。

    参数：
        items:
            统一工具目录条目集合。

    返回值：
        list[dict[str, Any]]:
            可写入 state 的普通字典列表。
    """

    return [
        dump_tool_catalog_item_for_state(
            item=item,
        )
        for item in items
    ]
