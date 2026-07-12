"""
ToolAgent 工具目录条目 Schema。

功能：
    定义 ToolAgent 写入 state 的统一工具目录条目。

设计说明：
    ToolMetadata 是本地工具自己的元数据；
    MockMcpToolDefinition 是 MCP 工具的外部描述；
    ToolCatalogItem 是 ToolAgent 统一读取的工具目录视图。

专业名词：
    Schema：
        数据结构 / 数据模型，用来约束字段格式。
    Tool Catalog：
        工具目录，告诉 ToolAgent 当前有哪些工具可用。
    input_schema：
        输入参数结构，告诉 LLM 调用工具时 args 应该包含哪些字段。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCatalogItem(BaseModel):
    """
    ToolAgent 统一工具目录条目。

    功能：
        统一描述本地工具和 MCP 工具，使 ToolAgent prompt、日志、state
        都可以读取同一种工具目录格式。

    参数：
        name:
            工具名称，例如 weather、date、sqlite_select_rows。

        description:
            工具描述，用于告诉 LLM 这个工具能做什么。

        timeout:
            工具超时时间，单位由底层运行时约定。

        retries:
            工具失败后的重试次数。

        require_confirm:
            是否需要用户确认。

        input_schema:
            工具输入参数结构。本地工具和 MCP 工具都通过该字段向 ToolAgent
            暴露参数名称、类型、说明和必填要求；无参数工具使用空字典。

        source:
            工具来源，例如 local 表示本地工具，mcp 表示 MCP 工具。

    返回值：
        ToolCatalogItem:
            Pydantic 数据对象，可通过 model_dump 转成普通 dict。
    """

    name: str = Field(
        description="工具名称",
    )
    description: str = Field(
        default="",
        description="工具描述",
    )
    timeout: int = Field(
        default=5,
        description="工具超时时间",
    )
    retries: int = Field(
        default=3,
        description="工具重试次数",
    )
    require_confirm: bool = Field(
        default=False,
        description="是否需要用户确认",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="工具输入参数结构",
    )
    source: str = Field(
        default="local",
        description="工具来源",
    )
