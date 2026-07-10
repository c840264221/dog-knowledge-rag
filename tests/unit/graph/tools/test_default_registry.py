"""
默认工具注册表测试。

功能：
    验证 ToolRegistry 类定义和默认工具注册流程已经解耦。
"""

from __future__ import annotations

from src.graph.tools.registry.default_registry import (
    build_default_tool_registry,
    register_default_tools,
)
from src.graph.tools.registry.tool_registry import ToolRegistry


def test_tool_registry_should_be_empty_by_default() -> None:
    """
    测试 ToolRegistry 默认不注册具体工具。

    功能：
        验证 ToolRegistry 只负责数据结构，不在初始化时产生工具注册副作用。

    参数：
        无。

    返回值：
        None。
    """

    tool_registry = ToolRegistry()

    assert tool_registry.tools == {}


def test_register_default_tools_should_register_local_tools() -> None:
    """
    测试默认工具注册函数。

    功能：
        验证 register_default_tools 会向传入 registry 注册 weather 和 date。

    参数：
        无。

    返回值：
        None。
    """

    tool_registry = ToolRegistry()

    returned_registry = register_default_tools(
        tool_registry=tool_registry,
    )

    assert returned_registry is tool_registry
    assert set(
        tool_registry.tools
    ) == {
        "weather",
        "date",
    }


def test_build_default_tool_registry_should_return_registered_registry() -> None:
    """
    测试构建默认工具注册表。

    功能：
        验证 build_default_tool_registry 会返回包含默认本地工具的注册表。

    参数：
        无。

    返回值：
        None。
    """

    tool_registry = build_default_tool_registry()

    assert tool_registry.get_tool(
        "weather"
    ) is not None
    assert tool_registry.get_tool(
        "date"
    ) is not None


def test_build_default_tool_registry_should_create_new_registry_each_time() -> None:
    """
    测试每次构建都会创建新的注册表实例。

    功能：
        避免测试或运行时代码意外共享同一个临时 registry 实例。

    参数：
        无。

    返回值：
        None。
    """

    first_registry = build_default_tool_registry()
    second_registry = build_default_tool_registry()

    assert first_registry is not second_registry
    assert first_registry.tools is not second_registry.tools
