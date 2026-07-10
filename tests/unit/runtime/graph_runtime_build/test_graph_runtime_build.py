"""
GraphRuntimeService 主图构建测试。

功能：
    用轻量 mock 验证 GraphRuntimeService 是否把运行时 Provider
    正确注入到 ToolAgent 构建入口。
"""

from __future__ import annotations

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
        FakeSQLiteMcpProvider:
            测试用 SQLite MCP Provider。
    """


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

    service = GraphRuntimeService(
        llm_provider=llm_provider,
        checkpoint_provider=checkpoint_provider,
        sqlite_mcp_provider=sqlite_mcp_provider,
    )

    tool_agent = service._build_tool_agent_node()

    assert tool_agent == "fake_tool_agent"
    assert captured_kwargs["llm_provider"] is llm_provider
    assert captured_kwargs["checkpoint_manager"] is checkpoint_provider.manager
    assert captured_kwargs["sqlite_mcp_provider"] is sqlite_mcp_provider
    assert captured_kwargs["interrupt_func"] is graph_runtime_service.interrupt
