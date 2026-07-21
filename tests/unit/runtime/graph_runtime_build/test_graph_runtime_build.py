"""
GraphRuntimeService 主图构建测试。

功能：
    用轻量 mock 验证 GraphRuntimeService 是否把运行时 Provider
    正确注入到 ToolAgent 构建入口。
"""

from __future__ import annotations

import pytest

from src.runtime.services import graph_runtime_service
from src.runtime.services.graph_runtime_service import GraphRuntimeService


class FakeCheckpointProvider:
    """
    测试用 CheckpointProvider。

    功能：
        只提供 manager 字段，用于验证 GraphRuntimeService 会把
        checkpoint_provider.manager 传给 ToolAgent。

    参数：
        无。

    返回值：
        FakeCheckpointProvider:
            测试用检查点 Provider。
    """

    def __init__(self) -> None:
        self.manager = object()


class FakeSQLiteMcpProvider:
    """
    测试用 SQLite MCP Provider。

    功能：
        作为占位对象验证 GraphRuntimeService 会把 sqlite_mcp_provider
        继续传给 ToolAgent。

    参数：
        无。

    返回值：
        FakeSQLiteMcpProvider：
            测试用 SQLite MCP Provider。
    """


class FakeStateGraph:
    """
    测试用 StateGraph（状态图）假对象。

    功能：
        记录主图节点和边的注册，避免单元测试真实编译 LangGraph。

    参数：
        state_schema：主图使用的 DogState。

    返回值：
        FakeStateGraph：可供 GraphRuntimeService 调用的假图。
    """

    def __init__(self, state_schema) -> None:
        self.state_schema = state_schema
        self.nodes = {}

    def add_node(self, name, node) -> None:
        """记录节点；name 是节点名，node 是节点对象；无返回值。"""
        self.nodes[name] = node

    def set_entry_point(self, name) -> None:
        """记录图入口；name 是入口节点名；无返回值。"""
        self.entry_point = name

    def add_edge(self, start, end) -> None:
        """接收普通边的起点和终点；本测试不需记录；无返回值。"""

    def add_conditional_edges(self, source, path, path_map) -> None:
        """接收条件边、路由函数和映射；本测试不需记录；无返回值。"""

    def compile(self, checkpointer=None):
        """
        模拟编译主图。

        参数：
            checkpointer：GraphRuntimeService 传入的 LangGraph 检查点存储。

        返回值：
            FakeStateGraph：直接返回当前假图便于断言。
        """

        self.checkpointer = checkpointer
        return self


def test_graph_runtime_should_pass_sqlite_mcp_provider_to_tool_agent(
    monkeypatch,
) -> None:
    """
    测试 GraphRuntimeService 会把 SQLite MCP Provider 注入 ToolAgent。

    功能：
        monkeypatch build_tool_agent_graph，捕获 GraphRuntimeService
        调用 ToolAgent 构建函数时传入的关键参数。

    参数：
        monkeypatch:
            pytest 提供的 monkeypatch fixture（测试夹具），
            用来替换模块中的 build_tool_agent_graph。

    返回值：
        None。
    """

    captured_kwargs = {}

    def fake_build_tool_agent_graph(
        **kwargs,
    ):
        """
        模拟 ToolAgent 图构建函数。

        功能：
            记录调用参数，并返回假节点，避免测试真实编译 LangGraph 子图。

        参数：
            **kwargs:
                GraphRuntimeService 传入 ToolAgent 的构建参数。

        返回值：
            str:
                假 ToolAgent 节点。
        """

        captured_kwargs.update(
            kwargs
        )
        return "fake_tool_agent"

    monkeypatch.setattr(
        graph_runtime_service,
        "build_tool_agent_graph",
        fake_build_tool_agent_graph,
    )

    llm_provider = object()
    checkpoint_provider = FakeCheckpointProvider()
    sqlite_mcp_provider = FakeSQLiteMcpProvider()
    tool_parser = object()

    service = GraphRuntimeService(
        llm_provider=llm_provider,
        checkpoint_provider=checkpoint_provider,
        sqlite_mcp_provider=sqlite_mcp_provider,
        tool_parser=tool_parser,
    )

    tool_agent = service._build_tool_agent_node()

    assert tool_agent == "fake_tool_agent"
    assert captured_kwargs["llm_provider"] is llm_provider
    assert captured_kwargs["checkpoint_manager"] is checkpoint_provider.manager
    assert captured_kwargs["sqlite_mcp_provider"] is sqlite_mcp_provider
    assert captured_kwargs["parser"] is tool_parser
    assert captured_kwargs["interrupt_func"] is graph_runtime_service.interrupt


@pytest.mark.asyncio
async def test_graph_runtime_should_inject_memory_extract_node_dependencies(
        monkeypatch,
) -> None:
    """
    测试 GraphRuntimeService 在构图时注入记忆抽取节点依赖。

    功能：
        替换所有子图构建入口，捕获 build_memory_extract_node 参数，
        验证节点使用 GraphRuntimeService 已持有的 Provider，不需要自己获取 Container。

    参数：
        monkeypatch：pytest 提供的临时替换测试夹具。

    返回值：
        None。
    """

    captured_kwargs = {}
    injected_node = object()

    def fake_build_memory_extract_node(**kwargs):
        """
        记录记忆抽取节点的构建参数。

        参数：**kwargs 是 GraphRuntimeService 注入的依赖。
        返回值：object，测试用节点占位对象。
        """

        captured_kwargs.update(kwargs)
        return injected_node

    monkeypatch.setattr(graph_runtime_service, "StateGraph", FakeStateGraph)
    monkeypatch.setattr(
        graph_runtime_service,
        "build_memory_extract_node",
        fake_build_memory_extract_node,
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_dog_knowledge_agent",
        lambda **kwargs: "dog_agent",
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_general_qa_agent",
        lambda **kwargs: "general_agent",
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_tool_agent_graph",
        lambda **kwargs: "tool_agent",
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_integrated_dog_knowledge_entry_node",
        lambda delegate_node: delegate_node,
    )
    monkeypatch.setattr(
        GraphRuntimeService,
        "_build_multi_agent_node",
        lambda self, **kwargs: "multi_agent",
    )

    llm_provider = object()
    memory_provider = object()
    checkpoint_provider = FakeCheckpointProvider()
    service = GraphRuntimeService(
        llm_provider=llm_provider,
        memory_provider=memory_provider,
        checkpoint_provider=checkpoint_provider,
    )

    graph = await service._build_graph()

    assert captured_kwargs == {
        "llm_provider": llm_provider,
        "memory_provider": memory_provider,
        "checkpoint_manager": checkpoint_provider.manager,
    }
    assert graph.nodes["memory_extract"] is injected_node
    assert graph.nodes["multi_agent"] == "multi_agent"
