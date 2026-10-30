"""多服务分发的关键检查；使用内存替身，不连接外部服务或写账。"""

import asyncio

import pytest

from src.mcp.schemas import MockMcpToolDefinition
from src.mcp.tool_dispatcher import McpToolDispatcher


class FakeMcpProvider:
    """提供统一目录和客户端接口的测试替身，不包含具体业务服务分支。"""

    def __init__(self, label, tool_names, *, enabled=True, allowed_databases=None):
        """保存测试配置；参数为服务标签、工具名、开关及可选白名单；无返回值。"""
        self.label = label
        self.enabled = enabled
        self.allowed_databases = allowed_databases or {}
        self.catalog_reads = 0
        self.startup_count = 0
        self.calls = []
        self._definitions = [MockMcpToolDefinition(name=name, description="测试工具", input_schema={})
                             for name in tool_names]

    @property
    def tool_client(self):
        """提供统一客户端入口；无参数；返回自身，不创建连接。"""
        return self

    @property
    def tool_definitions(self):
        """读取测试目录；无参数；返回工具定义并记录读取次数。"""
        self.catalog_reads += 1
        return self._definitions

    async def startup(self):
        """记录生命周期调用；无参数、无返回值，分发器不应主动调用此函数。"""
        self.startup_count += 1

    async def call_tool(self, name, arguments):
        """记录模拟调用；参数为名称和业务参数；返回服务标签用于检查路由归属。"""
        self.calls.append((name, arguments))
        return {"service": self.label, "tool": name, "arguments": arguments}


def test_three_services_share_one_dispatcher():
    """验证三服务共存和第三服务扩展；无参数；无返回值，不改分发逻辑即可合并及调用。"""
    providers = {name: FakeMcpProvider(name, [f"{name}_tool"])
                 for name in ("first", "second", "third")}
    dispatcher = McpToolDispatcher(providers)
    asyncio.run(dispatcher.startup())
    assert {tool.name for tool in dispatcher.tool_definitions} == {"first_tool", "second_tool", "third_tool"}
    for name, provider in providers.items():
        result = asyncio.run(dispatcher.call_tool(f"{name}_tool", {"value": 1}))
        assert result["service"] == name
        assert provider.calls == [(f"{name}_tool", {"value": 1})]
        assert provider.startup_count == 0
    assert dispatcher.tool_client is dispatcher


def test_disabled_and_unknown_services_do_not_execute():
    """验证停用和未知工具边界；无参数；无返回值，停用服务不读取目录，未知调用不分配客户端。"""
    disabled = FakeMcpProvider("disabled", ["blocked_tool"], enabled=False)
    active = FakeMcpProvider("active", ["active_tool"])
    dispatcher = McpToolDispatcher({"disabled": disabled, "active": active})
    assert [tool.name for tool in dispatcher.tool_definitions] == ["active_tool"]
    with pytest.raises(ValueError):
        asyncio.run(dispatcher.call_tool("blocked_tool", {}))
    assert disabled.catalog_reads == 0
    assert disabled.calls == active.calls == []


def test_duplicate_tools_and_conflicting_database_aliases_are_rejected():
    """验证目录冲突不静默覆盖；无参数；无返回值，重名工具及不同资源的同名白名单均拒绝。"""
    first = FakeMcpProvider("one", ["search"], allowed_databases={"records": "one.sqlite3"})
    second = FakeMcpProvider("two", ["search"])
    with pytest.raises(ValueError, match="search"):
        asyncio.run(McpToolDispatcher({"one": first, "two": second}).startup())
    third = FakeMcpProvider("three", ["other_tool"], allowed_databases={"records": "two.sqlite3"})
    with pytest.raises(ValueError, match="records"):
        asyncio.run(McpToolDispatcher({"one": first, "three": third}).startup())
