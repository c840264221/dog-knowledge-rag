"""
默认工具注册表。

功能：
    集中管理项目默认本地工具的注册流程。

设计原则：
    1. ToolRegistry 只负责注册表数据结构。
    2. 本模块负责装配默认工具实例。
    3. 后续 MCP 工具或动态工具可以在独立模块中扩展，不污染 ToolRegistry 类定义。
"""

from __future__ import annotations

from src.graph.tools.implementations.date_tool import DateTool
from src.graph.tools.implementations.weather_tool import WeatherTool
from src.graph.tools.registry.tool_registry import ToolRegistry


def register_default_tools(
    tool_registry: ToolRegistry,
) -> ToolRegistry:
    """
    注册默认本地工具。

    功能：
        向传入的 ToolRegistry 注册当前项目默认启用的本地工具：
        1. WeatherTool：天气查询工具。
        2. DateTool：日期查询工具。

    参数：
        tool_registry:
            工具注册表实例。

    返回值：
        ToolRegistry:
            已注册默认工具的同一个工具注册表实例。
    """

    tool_registry.register(
        WeatherTool()
    )
    tool_registry.register(
        DateTool()
    )

    return tool_registry


def build_default_tool_registry() -> ToolRegistry:
    """
    构建默认工具注册表。

    功能：
        创建新的 ToolRegistry，并注册项目默认本地工具。

    参数：
        无。

    返回值：
        ToolRegistry:
            包含默认本地工具的工具注册表。
    """

    tool_registry = ToolRegistry()

    return register_default_tools(
        tool_registry=tool_registry,
    )


default_tool_registry = build_default_tool_registry()

# 兼容旧代码中使用 registry 变量名的场景。
registry = default_tool_registry
