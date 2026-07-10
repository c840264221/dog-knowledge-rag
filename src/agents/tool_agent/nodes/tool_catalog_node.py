"""
ToolAgent 工具目录节点。

功能：
    在 ToolAgent 链路开始时构建工具目录，并写入 LangGraph state。

设计原则：
    1. 节点只负责生成工具目录，不执行工具。
    2. 本地工具来自 ToolRegistry。
    3. MCP 工具定义可以通过参数直接传入，也可以从 SQLiteMcpProvider 获取。
    4. 输出普通 dict，避免 checkpoint 保存自定义对象。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    build_tool_agent_tool_catalog_state_update_with_mcp,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.mcp.schemas import MockMcpToolDefinition
from src.runtime.context import runtime_ctx


ToolCatalogNode = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]
TOOL_AGENT_ALLOWED_DATABASES_STATE_KEY = "tool_agent_allowed_databases"


def build_tool_agent_tool_catalog_node(
    tool_registry: Any | None = None,
    mcp_tool_definitions: Iterable[MockMcpToolDefinition] | None = None,
    sqlite_mcp_provider: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolCatalogNode:
    """
    构建 ToolAgent 工具目录节点。

    功能：
        创建一个 async node，用于生成 ToolAgent 可用工具目录。
        节点会合并本地工具和 MCP 工具，并返回 state update。

    参数：
        tool_registry:
            本地工具注册表。为空时 registry_adapter 会使用默认工具注册表。

        mcp_tool_definitions:
            MCP 工具定义集合。优先级高于 sqlite_mcp_provider。

        sqlite_mcp_provider:
            SQLite MCP Provider。未显式传入 mcp_tool_definitions 时，
            会尝试读取 sqlite_mcp_provider.tool_definitions。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。

    返回值：
        ToolCatalogNode:
            async 工具目录节点函数，接收 state，返回工具目录 state update。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def tool_agent_tool_catalog_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        构建并返回 ToolAgent 工具目录 state update。

        功能：
            1. 写入当前运行时 node 信息。
            2. 解析 MCP 工具定义来源。
            3. 构建本地工具 + MCP 工具目录。
            4. 返回 {"tool_agent_tool_catalog": [...]}。

        参数：
            state:
                当前 LangGraph state。此节点当前不依赖具体 state 字段。

        返回值：
            dict[str, Any]:
                可合并进 LangGraph state 的工具目录更新。
        """

        write_tool_catalog_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        resolved_mcp_tool_definitions = resolve_mcp_tool_definitions(
            mcp_tool_definitions=mcp_tool_definitions,
            sqlite_mcp_provider=sqlite_mcp_provider,
        )
        update = build_tool_agent_tool_catalog_state_update_with_mcp(
            tool_registry=tool_registry,
            mcp_tool_definitions=resolved_mcp_tool_definitions,
        )
        allowed_database_aliases = read_sqlite_allowed_database_aliases(
            sqlite_mcp_provider=sqlite_mcp_provider,
        )
        update = attach_allowed_database_aliases_to_catalog_update(
            update=update,
            allowed_database_aliases=allowed_database_aliases,
        )

        log_tool_agent_state(
            node_name="tool_catalog",
            event="tool_catalog_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "tool_catalog_count": len(
                    update.get(
                        "tool_agent_tool_catalog",
                        [],
                    )
                ),
                "has_mcp_tool_definitions": bool(
                    resolved_mcp_tool_definitions
                ),
                "allowed_database_aliases": allowed_database_aliases,
            },
        )

        return update

    return tool_agent_tool_catalog_node


def read_sqlite_allowed_database_aliases(
    sqlite_mcp_provider: Any | None = None,
) -> list[str]:
    """
    读取 SQLite MCP 允许访问的数据库别名。

    功能：
        从 sqlite_mcp_provider.allowed_databases 中读取白名单 key，
        只返回字符串别名并排序，方便写入 state、prompt 和日志。

    参数：
        sqlite_mcp_provider:
            SQLite MCP Provider，可能提供 allowed_databases 属性。

    返回值：
        list[str]:
            排序后的数据库白名单别名列表，例如 ["memory", "rag"]。
    """

    if sqlite_mcp_provider is None:
        return []

    raw_allowed_databases = getattr(
        sqlite_mcp_provider,
        "allowed_databases",
        {},
    )

    if not isinstance(
        raw_allowed_databases,
        Mapping,
    ):
        return []

    return sorted(
        str(
            database_name
        )
        for database_name in raw_allowed_databases
        if str(
            database_name
        ).strip()
    )


def attach_allowed_database_aliases_to_catalog_update(
    update: Mapping[str, Any],
    allowed_database_aliases: list[str],
) -> dict[str, Any]:
    """
    将数据库白名单别名附加到工具目录 state update。

    功能：
        1. 把别名列表写入 tool_agent_allowed_databases，方便日志和调试。
        2. 给 MCP SQLite 工具的 database_name 参数补充 enum / allowed_values，
           让 LLM 和校验器都能看到合法取值。

    参数：
        update:
            原始工具目录 state update。

        allowed_database_aliases:
            SQLite MCP 允许访问的数据库别名列表。

    返回值：
        dict[str, Any]:
            增强后的 state update。
    """

    enhanced_update = dict(
        update
    )
    raw_catalog = enhanced_update.get(
        TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
        [],
    )

    if not isinstance(
        raw_catalog,
        list,
    ):
        raw_catalog = []

    enhanced_update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY] = [
        attach_allowed_database_aliases_to_catalog_item(
            raw_item=raw_item,
            allowed_database_aliases=allowed_database_aliases,
        )
        for raw_item in raw_catalog
    ]
    enhanced_update[TOOL_AGENT_ALLOWED_DATABASES_STATE_KEY] = {
        alias: alias
        for alias in allowed_database_aliases
    }

    return enhanced_update


def attach_allowed_database_aliases_to_catalog_item(
    raw_item: Any,
    allowed_database_aliases: list[str],
) -> Any:
    """
    给单个工具目录条目的 database_name 参数补充白名单信息。

    功能：
        只处理 dict 格式工具目录，且只在 input_schema.properties.database_name
        存在时补充 enum 和 allowed_values。

    参数：
        raw_item:
            单个工具目录条目。

        allowed_database_aliases:
            SQLite MCP 允许访问的数据库别名列表。

    返回值：
        Any:
            增强后的工具目录条目；非法结构原样返回。
    """

    if not isinstance(
        raw_item,
        Mapping,
    ):
        return raw_item

    item = dict(
        raw_item
    )
    input_schema = item.get(
        "input_schema",
        {},
    )

    if not isinstance(
        input_schema,
        Mapping,
    ):
        return item

    schema_copy = dict(
        input_schema
    )
    properties = schema_copy.get(
        "properties",
        {},
    )

    if not isinstance(
        properties,
        Mapping,
    ):
        return item

    properties_copy = dict(
        properties
    )
    database_name_schema = properties_copy.get(
        "database_name",
        {},
    )

    if not isinstance(
        database_name_schema,
        Mapping,
    ):
        return item

    database_schema_copy = dict(
        database_name_schema
    )
    database_schema_copy["allowed_values"] = list(
        allowed_database_aliases
    )
    database_schema_copy["enum"] = list(
        allowed_database_aliases
    )

    if allowed_database_aliases:
        database_schema_copy["description"] = (
            "数据库白名单别名，只能使用以下值之一："
            + "、".join(
                allowed_database_aliases
            )
            + "。"
        )

    properties_copy["database_name"] = database_schema_copy
    schema_copy["properties"] = properties_copy
    item["input_schema"] = schema_copy

    return item


def resolve_mcp_tool_definitions(
    mcp_tool_definitions: Iterable[MockMcpToolDefinition] | None = None,
    sqlite_mcp_provider: Any | None = None,
) -> Iterable[MockMcpToolDefinition] | None:
    """
    解析 MCP 工具定义来源。

    功能：
        优先使用显式传入的 mcp_tool_definitions。
        如果没有显式传入，则尝试读取 sqlite_mcp_provider.tool_definitions。

    参数：
        mcp_tool_definitions:
            显式传入的 MCP 工具定义集合。

        sqlite_mcp_provider:
            SQLite MCP Provider，可能提供 tool_definitions 属性。

    返回值：
        Iterable[MockMcpToolDefinition] | None:
            MCP 工具定义集合；没有可用来源时返回 None。
    """

    if mcp_tool_definitions is not None:
        return mcp_tool_definitions

    if sqlite_mcp_provider is None:
        return None

    return getattr(
        sqlite_mcp_provider,
        "tool_definitions",
        None,
    )


def write_tool_catalog_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具目录节点运行时事件。

    功能：
        如果存在 RuntimeContext，则记录当前 node 和 timeline 事件。
        如果不存在，则静默跳过，保证单元测试可独立运行。

    参数：
        runtime_context:
            当前请求 RuntimeContext，可能为 None。

    返回值：
        None。
    """

    if runtime_context is None:
        return

    runtime_context.state().set_node(
        "tool_agent_tool_catalog_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_catalog_node",
    )
