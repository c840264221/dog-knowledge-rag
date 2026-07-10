"""
MCP 工具 Schema 适配器。

功能：
    将 Mock MCP 工具描述转换成项目内部 ToolMetadata。

设计原则：
    1. ToolAgent 只依赖内部 ToolMetadata，不直接依赖 MCP 原始对象。
    2. 当前阶段只适配 name、description、timeout、retries、require_confirm。
    3. MCP input_schema 暂不写入 ToolMetadata，避免提前扩大内部工具契约。
    4. 输出 Pydantic ToolMetadata，后续可以复用现有 registry_adapter。

专业名词：
    Adapter：
        适配器，用来隔离外部协议对象和内部业务对象。
    ToolMetadata：
        工具元数据，项目内部描述工具的标准对象。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.schemas import MockMcpToolDefinition


DEFAULT_MCP_TOOL_TIMEOUT = 5
DEFAULT_MCP_TOOL_RETRIES = 3
DEFAULT_MCP_REQUIRE_CONFIRM = False


def build_tool_metadata_from_mcp_tool(
    mcp_tool: MockMcpToolDefinition,
    default_timeout: int = DEFAULT_MCP_TOOL_TIMEOUT,
    default_retries: int = DEFAULT_MCP_TOOL_RETRIES,
    default_require_confirm: bool = DEFAULT_MCP_REQUIRE_CONFIRM,
) -> ToolMetadata:
    """
    将单个 Mock MCP 工具定义转换成 ToolMetadata。

    功能：
        读取 MockMcpToolDefinition 的基础字段和 annotations，
        构建项目内部工具元数据对象。

    参数：
        mcp_tool:
            Mock MCP 工具定义。

        default_timeout:
            annotations 中没有 timeout 时使用的默认超时时间。

        default_retries:
            annotations 中没有 retries 时使用的默认重试次数。

        default_require_confirm:
            annotations 中没有 require_confirm 时使用的默认确认配置。

    返回值：
        ToolMetadata:
            项目内部工具元数据。
    """

    annotations = mcp_tool.annotations

    return ToolMetadata(
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
    )


def build_tool_metadata_list_from_mcp_tools(
    mcp_tools: Iterable[MockMcpToolDefinition],
) -> list[ToolMetadata]:
    """
    批量转换 Mock MCP 工具定义。

    功能：
        将多个 MockMcpToolDefinition 转换成 ToolMetadata 列表，
        并按工具名排序，保证测试和调试输出稳定。

    参数：
        mcp_tools:
            Mock MCP 工具定义集合。

    返回值：
        list[ToolMetadata]:
            按 name 排序后的内部工具元数据列表。
    """

    metadata_items = [
        build_tool_metadata_from_mcp_tool(
            mcp_tool=mcp_tool,
        )
        for mcp_tool in mcp_tools
    ]

    return sorted(
        metadata_items,
        key=lambda item: item.name,
    )


def dump_mcp_tool_metadata_for_agent(
    metadata: ToolMetadata,
) -> dict[str, Any]:
    """
    将 MCP 来源的 ToolMetadata 转换成普通 dict。

    功能：
        复用 ToolMetadata.model_dump，输出可写入 state 或报告的普通字典。

    参数：
        metadata:
            ToolMetadata 工具元数据。

    返回值：
        dict[str, Any]:
            普通字典格式的工具元数据。
    """

    return metadata.model_dump()


def build_mcp_tool_catalog_for_agent(
    mcp_tools: Iterable[MockMcpToolDefinition],
) -> list[dict[str, Any]]:
    """
    构建 ToolAgent 可读的 MCP 工具目录。

    功能：
        将 Mock MCP 工具定义批量转换成 ToolMetadata，
        再转换成普通 dict 列表，方便后续写入 state。

    参数：
        mcp_tools:
            Mock MCP 工具定义集合。

    返回值：
        list[dict[str, Any]]:
            ToolAgent 可读的 MCP 工具目录。
    """

    return [
        dump_mcp_tool_metadata_for_agent(
            metadata=metadata,
        )
        for metadata in build_tool_metadata_list_from_mcp_tools(
            mcp_tools=mcp_tools,
        )
    ]


def read_bool_annotation(
    annotations: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    """
    从 annotations 中读取布尔配置。

    功能：
        支持 bool 和常见字符串布尔值，例如 "true"、"false"、"yes"、"no"。
        无法识别时返回默认值。

    参数：
        annotations:
            MCP 工具附加配置。

        key:
            需要读取的字段名。

        default:
            字段不存在或无法识别时的默认值。

    返回值：
        bool:
            解析后的布尔值。
    """

    value = annotations.get(
        key,
        default,
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        normalized_value = value.strip().lower()

        if normalized_value in {
            "1",
            "true",
            "yes",
            "y",
        }:
            return True

        if normalized_value in {
            "0",
            "false",
            "no",
            "n",
        }:
            return False

    return default


def read_int_annotation(
    annotations: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    """
    从 annotations 中读取整数配置。

    功能：
        支持 int 和可以转换成 int 的字符串。
        字段不存在、转换失败或值小于 0 时返回默认值。

    参数：
        annotations:
            MCP 工具附加配置。

        key:
            需要读取的字段名。

        default:
            字段不存在或无法识别时的默认值。

    返回值：
        int:
            解析后的整数。
    """

    value = annotations.get(
        key,
        default,
    )

    try:
        parsed_value = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default

    if parsed_value < 0:
        return default

    return parsed_value
