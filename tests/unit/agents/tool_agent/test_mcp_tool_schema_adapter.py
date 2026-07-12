"""
MCP 工具 Schema 适配器测试。

功能：
    验证 Mock MCP 工具定义可以稳定转换成项目内部 ToolMetadata。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.mcp_tool_schema_adapter import (
    build_mcp_tool_catalog_for_agent,
    build_tool_metadata_from_mcp_tool,
    build_tool_metadata_list_from_mcp_tools,
    read_bool_annotation,
    read_int_annotation,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.mcp.schemas import MockMcpToolDefinition


def test_build_tool_metadata_from_mcp_tool_should_map_basic_fields() -> None:
    """
    测试单个 Mock MCP 工具基础字段映射。

    功能：
        验证 name、description 和默认 timeout/retries/require_confirm 会正确进入 ToolMetadata。

    参数：
        无。

    返回值：
        None。
    """

    mcp_tool = MockMcpToolDefinition(
        name="mcp_echo",
        description="回显输入文本。",
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                }
            },
        },
    )

    metadata = build_tool_metadata_from_mcp_tool(
        mcp_tool=mcp_tool,
    )

    assert isinstance(
        metadata,
        ToolMetadata,
    )
    assert metadata.name == "mcp_echo"
    assert metadata.description == "回显输入文本。"
    assert metadata.timeout == 5
    assert metadata.retries == 3
    assert metadata.require_confirm is False


def test_build_tool_metadata_from_mcp_tool_should_read_annotations() -> None:
    """
    测试 annotations 覆盖默认配置。

    功能：
        验证 require_confirm、timeout、retries 可以从 Mock MCP annotations 中读取。

    参数：
        无。

    返回值：
        None。
    """

    mcp_tool = MockMcpToolDefinition(
        name="mcp_database_query",
        description="执行只读数据库查询。",
        annotations={
            "require_confirm": True,
            "timeout": 12,
            "retries": 1,
        },
    )

    metadata = build_tool_metadata_from_mcp_tool(
        mcp_tool=mcp_tool,
    )

    assert metadata.name == "mcp_database_query"
    assert metadata.require_confirm is True
    assert metadata.timeout == 12
    assert metadata.retries == 1


def test_build_tool_metadata_from_mcp_tool_should_parse_string_annotations() -> None:
    """
    测试字符串格式 annotations 可以被解析。

    功能：
        MCP 外部配置可能以字符串传入，本测试验证适配器能把常见字符串转换成内部类型。

    参数：
        无。

    返回值：
        None。
    """

    mcp_tool = MockMcpToolDefinition(
        name="mcp_safe_search",
        description="执行安全搜索。",
        annotations={
            "require_confirm": "yes",
            "timeout": "8",
            "retries": "2",
        },
    )

    metadata = build_tool_metadata_from_mcp_tool(
        mcp_tool=mcp_tool,
    )

    assert metadata.require_confirm is True
    assert metadata.timeout == 8
    assert metadata.retries == 2


def test_build_tool_metadata_from_mcp_tool_should_fallback_invalid_annotations() -> None:
    """
    测试非法 annotations 会回退默认值。

    功能：
        当 timeout/retries 无法转换为合法整数时，适配器应该使用默认值，
        避免外部 MCP 配置污染内部工具契约。

    参数：
        无。

    返回值：
        None。
    """

    mcp_tool = MockMcpToolDefinition(
        name="mcp_invalid_config",
        description="带非法配置的工具。",
        annotations={
            "require_confirm": "unknown",
            "timeout": "slow",
            "retries": -1,
        },
    )

    metadata = build_tool_metadata_from_mcp_tool(
        mcp_tool=mcp_tool,
        default_timeout=6,
        default_retries=2,
        default_require_confirm=False,
    )

    assert metadata.require_confirm is False
    assert metadata.timeout == 6
    assert metadata.retries == 2


def test_build_tool_metadata_list_from_mcp_tools_should_sort_by_name() -> None:
    """
    测试批量转换结果按工具名排序。

    功能：
        稳定排序可以让测试、日志和调试报告输出更可预测。

    参数：
        无。

    返回值：
        None。
    """

    metadata_items = build_tool_metadata_list_from_mcp_tools(
        mcp_tools=[
            MockMcpToolDefinition(
                name="mcp_z_tool",
            ),
            MockMcpToolDefinition(
                name="mcp_a_tool",
            ),
        ],
    )

    assert [
        metadata.name
        for metadata in metadata_items
    ] == [
        "mcp_a_tool",
        "mcp_z_tool",
    ]


def test_build_mcp_tool_catalog_for_agent_should_return_plain_dicts() -> None:
    """
    测试构建 ToolAgent 可读的 MCP 工具目录。

    功能：
        验证输出是普通 dict 列表，避免把 Pydantic 对象直接写入 LangGraph state。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_mcp_tool_catalog_for_agent(
        mcp_tools=[
            MockMcpToolDefinition(
                name="mcp_echo",
                description="回显输入。",
                annotations={
                    "require_confirm": False,
                    "timeout": 4,
                    "retries": 0,
                },
            )
        ],
    )

    assert catalog == [
        {
            "name": "mcp_echo",
            "description": "回显输入。",
            "timeout": 4,
            "retries": 0,
            "require_confirm": False,
            "input_schema": {},
        }
    ]


def test_read_bool_annotation_should_support_common_values() -> None:
    """
    测试布尔 annotations 读取工具函数。

    功能：
        验证常见字符串布尔值能被稳定转换。

    参数：
        无。

    返回值：
        None。
    """

    assert read_bool_annotation(
        annotations={
            "value": "true",
        },
        key="value",
        default=False,
    ) is True
    assert read_bool_annotation(
        annotations={
            "value": "no",
        },
        key="value",
        default=True,
    ) is False
    assert read_bool_annotation(
        annotations={
            "value": "unknown",
        },
        key="value",
        default=True,
    ) is True


def test_read_int_annotation_should_reject_invalid_values() -> None:
    """
    测试整数 annotations 读取工具函数。

    功能：
        验证字符串整数可以转换，非法值和负数会回退默认值。

    参数：
        无。

    返回值：
        None。
    """

    assert read_int_annotation(
        annotations={
            "value": "10",
        },
        key="value",
        default=5,
    ) == 10
    assert read_int_annotation(
        annotations={
            "value": "bad",
        },
        key="value",
        default=5,
    ) == 5
    assert read_int_annotation(
        annotations={
            "value": -1,
        },
        key="value",
        default=5,
    ) == 5
