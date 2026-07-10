"""
Mock MCP 数据结构。

功能：
    定义 V1.9.2 阶段用于学习和测试的 MCP 工具描述对象。

设计说明：
    这里的 MockMcpToolDefinition 不是最终真实 MCP SDK 对象。
    它用于模拟 MCP Server 暴露的工具描述，帮助项目先完成
    MCP Tool -> ToolMetadata 的适配链路。

专业名词：
    MCP：
        Model Context Protocol，模型上下文协议。
    Tool Definition：
        工具定义，描述一个外部工具的名称、说明、参数结构和附加配置。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MockMcpToolDefinition(BaseModel):
    """
    Mock MCP 工具定义。

    功能：
        模拟 MCP Server 暴露的一条工具描述。
        当前只用于 schema adapter 单元测试，不直接进入 ToolAgent state。

    参数：
        name:
            MCP 工具名称。

        description:
            MCP 工具描述，用于说明工具用途。

        input_schema:
            MCP 工具参数结构。当前阶段只保留在 Mock MCP 对象中，
            不写入 ToolMetadata，避免提前改内部工具标准。

        annotations:
            MCP 工具附加配置。当前 adapter 会从这里读取：
            require_confirm、timeout、retries。

    返回值：
        MockMcpToolDefinition:
            Pydantic 数据对象，可通过 model_dump 转成普通 dict。
    """

    name: str = Field(
        description="MCP 工具名称",
    )
    description: str = Field(
        default="",
        description="MCP 工具描述",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="MCP 工具参数结构",
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="MCP 工具附加配置",
    )
