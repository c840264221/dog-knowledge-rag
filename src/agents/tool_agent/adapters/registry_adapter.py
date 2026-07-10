"""
ToolAgent 工具注册表适配器。

功能：
    为 ToolAgent（工具智能体）提供读取底层工具注册表的稳定入口。

设计原则：
    1. ToolAgent 不直接关心底层 ToolRegistry 的内部存储结构。
    2. 底层 ToolMetadata 仍然是单个工具的标准描述。
    3. 本模块只做读取和转换，不执行真实工具。
    4. 输出普通 dict，方便写入 state，并避免 checkpoint 保存自定义对象。

专业名词：
    Registry：注册表，用来保存当前系统已注册的工具。
    Adapter：适配器，用来隔离两层模块之间的数据结构差异。
    Metadata：元数据，用来描述工具名称、说明、超时、重试等配置。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.tool_catalog_item_adapter import (
    build_tool_catalog_items_from_mcp_tools,
    build_tool_catalog_items_from_metadata,
    dump_tool_catalog_items_for_state,
)
from src.agents.tool_agent.contracts.tool_catalog_item_schema import (
    ToolCatalogItem,
)
from src.graph.tools.registry.default_registry import (
    registry as default_tool_registry,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.schemas import MockMcpToolDefinition


TOOL_AGENT_TOOL_CATALOG_STATE_KEY = "tool_agent_tool_catalog"


def list_registered_tool_metadata(
    tool_registry: Any = default_tool_registry,
) -> list[ToolMetadata]:
    """
    列出工具注册表中的工具元数据。

    功能：
        从底层 ToolRegistry 中读取所有已注册工具，并提取每个工具的 ToolMetadata。
        返回结果按工具名排序，保证测试和调试输出稳定。

    参数：
        tool_registry:
            工具注册表对象。默认使用项目全局 registry。
            只要求对象具有 tools 字段，方便测试时传入假的 registry。

    返回值：
        list[ToolMetadata]:
            已注册工具的元数据列表。
    """

    resolved_registry = tool_registry or default_tool_registry

    raw_tools = getattr(
        resolved_registry,
        "tools",
        {},
    )

    if not isinstance(
        raw_tools,
        Mapping,
    ):
        return []

    metadata_items: list[ToolMetadata] = []

    for tool in raw_tools.values():
        metadata = getattr(
            tool,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            ToolMetadata,
        ):
            metadata_items.append(
                metadata
            )

    return sorted(
        metadata_items,
        key=lambda item: item.name,
    )


def get_registered_tool_metadata(
    tool_name: str,
    tool_registry: Any = default_tool_registry,
) -> ToolMetadata | None:
    """
    根据工具名读取单个工具元数据。

    功能：
        调用底层 ToolRegistry.get_tool 获取工具对象，并返回它的 ToolMetadata。
        如果工具不存在或 metadata 类型不正确，则返回 None。

    参数：
        tool_name:
            工具名称，例如 weather、date。

        tool_registry:
            工具注册表对象。默认使用项目全局 registry。

    返回值：
        ToolMetadata | None:
            找到合法工具时返回 ToolMetadata，否则返回 None。
    """

    if not tool_name:
        return None

    resolved_registry = tool_registry or default_tool_registry

    get_tool = getattr(
        resolved_registry,
        "get_tool",
        None,
    )

    if not callable(
        get_tool
    ):
        return None

    tool = get_tool(
        tool_name
    )
    metadata = getattr(
        tool,
        "metadata",
        None,
    )

    if isinstance(
        metadata,
        ToolMetadata,
    ):
        return metadata

    return None


def dump_tool_metadata_for_agent(
    metadata: ToolMetadata,
) -> dict[str, Any]:
    """
    将 ToolMetadata 转换成 ToolAgent 可写入 state 的普通 dict。

    功能：
        把底层工具元数据转换成普通字典，避免上层 state 直接保存 Pydantic 对象。

    参数：
        metadata:
            单个工具的 ToolMetadata 元数据对象。

    返回值：
        dict[str, Any]:
            普通字典格式的工具描述。
    """

    return metadata.model_dump()


def build_tool_agent_tool_catalog(
    tool_registry: Any = default_tool_registry,
) -> list[dict[str, Any]]:
    """
    构建 ToolAgent 工具目录。

    功能：
        读取所有已注册工具的 ToolMetadata，并转换成普通 dict 列表。
        该列表用于后续 ToolAgent 规划工具调用时了解可用工具。

    参数：
        tool_registry:
            工具注册表对象。默认使用项目全局 registry。

    返回值：
        list[dict[str, Any]]:
            工具目录列表。每个元素包含 name、description、timeout、retries、require_confirm。
    """

    metadata_items = list_registered_tool_metadata(
        tool_registry=tool_registry or default_tool_registry,
    )

    return [
        dump_tool_metadata_for_agent(
            metadata=metadata,
        )
        for metadata in metadata_items
    ]


def merge_tool_metadata_items(
    base_metadata_items: Iterable[ToolMetadata],
    extra_metadata_items: Iterable[ToolMetadata] | None = None,
) -> list[ToolMetadata]:
    """
    合并工具元数据列表。

    功能：
        将本地工具元数据和扩展工具元数据合并成一个列表。
        如果出现同名工具，本地工具优先，扩展工具不会覆盖已有工具。
        最终按工具名排序，保证输出稳定。

    参数：
        base_metadata_items:
            基础工具元数据列表，通常来自本地 ToolRegistry。

        extra_metadata_items:
            扩展工具元数据列表，通常来自 MCP 工具定义转换结果。

    返回值：
        list[ToolMetadata]:
            合并、去重并按名称排序后的工具元数据列表。
    """

    metadata_by_name: dict[str, ToolMetadata] = {}

    for metadata in base_metadata_items:
        metadata_by_name[metadata.name] = metadata

    for metadata in extra_metadata_items or []:
        if metadata.name not in metadata_by_name:
            metadata_by_name[metadata.name] = metadata

    return sorted(
        metadata_by_name.values(),
        key=lambda item: item.name,
    )


def merge_tool_catalog_items(
    base_catalog_items: Iterable[ToolCatalogItem],
    extra_catalog_items: Iterable[ToolCatalogItem] | None = None,
) -> list[ToolCatalogItem]:
    """
    合并统一工具目录条目列表。

    功能：
        将本地工具目录条目和 MCP 工具目录条目合并成一个列表。
        如果出现同名工具，本地工具优先，扩展工具不会覆盖已有工具。
        最终按工具名排序，保证输出稳定。

    参数：
        base_catalog_items:
            基础工具目录条目列表，通常来自本地 ToolRegistry。

        extra_catalog_items:
            扩展工具目录条目列表，通常来自 MCP 工具定义转换结果。

    返回值：
        list[ToolCatalogItem]:
            合并、去重并按名称排序后的统一工具目录条目列表。
    """

    catalog_by_name: dict[str, ToolCatalogItem] = {}

    for item in base_catalog_items:
        catalog_by_name[item.name] = item

    for item in extra_catalog_items or []:
        if item.name not in catalog_by_name:
            catalog_by_name[item.name] = item

    return sorted(
        catalog_by_name.values(),
        key=lambda item: item.name,
    )


def build_tool_agent_tool_catalog_with_mcp(
    tool_registry: Any = default_tool_registry,
    mcp_tool_definitions: Iterable[MockMcpToolDefinition] | None = None,
) -> list[dict[str, Any]]:
    """
    构建包含 MCP 工具的 ToolAgent 工具目录。

    功能：
        读取本地工具注册表中的 ToolMetadata，
        再将本地工具和 MCP 工具统一转换成 ToolCatalogItem，
        最后合并成本地工具 + MCP 工具的普通 dict 目录。

    参数：
        tool_registry:
            本地工具注册表对象。默认使用项目全局默认注册表。

        mcp_tool_definitions:
            MCP 工具定义集合。为空时只返回本地工具目录。

    返回值：
        list[dict[str, Any]]:
            ToolAgent 可写入 state 的工具目录列表。
    """

    local_metadata_items = list_registered_tool_metadata(
        tool_registry=tool_registry or default_tool_registry,
    )
    local_catalog_items = build_tool_catalog_items_from_metadata(
        metadata_items=local_metadata_items,
    )
    mcp_catalog_items = build_tool_catalog_items_from_mcp_tools(
        mcp_tools=mcp_tool_definitions or [],
    )
    merged_catalog_items = merge_tool_catalog_items(
        base_catalog_items=local_catalog_items,
        extra_catalog_items=mcp_catalog_items,
    )

    return dump_tool_catalog_items_for_state(
        items=merged_catalog_items,
    )


def build_tool_agent_tool_catalog_state_update_with_mcp(
    tool_registry: Any = default_tool_registry,
    mcp_tool_definitions: Iterable[MockMcpToolDefinition] | None = None,
) -> dict[str, Any]:
    """
    构建包含 MCP 工具目录的 state 更新。

    功能：
        将本地工具和 MCP 工具合并后的工具目录，
        包装成 {"tool_agent_tool_catalog": [...]} 格式，
        方便后续 ToolAgent 节点直接返回给 LangGraph 合并 state。

    参数：
        tool_registry:
            本地工具注册表对象。默认使用项目全局默认注册表。

        mcp_tool_definitions:
            MCP 工具定义集合。为空时只返回本地工具目录更新。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的普通字典。
    """

    return {
        TOOL_AGENT_TOOL_CATALOG_STATE_KEY: build_tool_agent_tool_catalog_with_mcp(
            tool_registry=tool_registry or default_tool_registry,
            mcp_tool_definitions=mcp_tool_definitions,
        )
    }


def build_tool_agent_tool_catalog_state_update(
    tool_registry: Any = default_tool_registry,
) -> dict[str, Any]:
    """
    构建可写回 LangGraph state 的工具目录更新。

    功能：
        将 ToolAgent 工具目录包装成 {"tool_agent_tool_catalog": [...]} 格式，
        方便后续节点直接返回给 LangGraph 合并 state。

    参数：
        tool_registry:
            工具注册表对象。默认使用项目全局 registry。

    返回值：
        dict[str, Any]:
            可写回 state 的普通字典。
    """

    return {
        TOOL_AGENT_TOOL_CATALOG_STATE_KEY: build_tool_agent_tool_catalog(
            tool_registry=tool_registry or default_tool_registry,
        )
    }


def tool_requires_confirmation(
    tool_name: str,
    tool_registry: Any = default_tool_registry,
) -> bool:
    """
    判断工具是否需要用户确认。

    功能：
        根据工具注册表中的 ToolMetadata.require_confirm 判断工具调用前是否需要确认。
        工具不存在时保守返回 False，避免当前适配层制造额外阻塞。

    参数：
        tool_name:
            工具名称。

        tool_registry:
            工具注册表对象。默认使用项目全局 registry。

    返回值：
        bool:
            True 表示该工具需要用户确认，False 表示不需要或工具不存在。
    """

    metadata = get_registered_tool_metadata(
        tool_name=tool_name,
        tool_registry=tool_registry or default_tool_registry,
    )

    if metadata is None:
        return False

    return bool(
        metadata.require_confirm
    )
