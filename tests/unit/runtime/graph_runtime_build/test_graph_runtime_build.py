"""
GraphRuntimeService 主图构建测试。

功能：
    用轻量 mock 验证 GraphRuntimeService 是否把运行时 Provider
    正确注入到 ToolAgent 构建入口。
"""

from __future__ import annotations

from types import SimpleNamespace

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


class FakeRedisProvider:
    """提供确定启用状态和共享客户端的 RedisProvider 测试替身。"""

    def __init__(self, *, enabled: bool) -> None:
        """
        初始化 Redis 开关和测试客户端。

        参数含义：
            enabled:
                是否允许 GraphRuntimeService 组装长任务应用服务。

        返回值含义：
            None。
        """

        self.enabled = enabled
        self._client = object()
        self.client_reads = 0

    @property
    def client(self):
        """
        记录客户端读取次数并返回固定测试对象。

        返回值含义：
            object:
                Store 和 Stream 应共同复用的 Redis 客户端。
        """

        self.client_reads += 1
        return self._client


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


def test_multi_agent_scheduler_options_should_follow_runtime_settings(
    monkeypatch,
) -> None:
    """
    测试多 Agent Scheduler 参数是否来自 Runtime Settings。

    参数：
        monkeypatch:
            pytest 提供的临时属性替换工具。

    返回值：
        None。
    """

    runtime_settings = graph_runtime_service.settings.runtime
    monkeypatch.setattr(runtime_settings, "enable_timeout", True)
    monkeypatch.setattr(runtime_settings, "enable_retry", True)
    monkeypatch.setattr(
        runtime_settings,
        "multi_agent_maximum_parallel_steps",
        3,
    )
    monkeypatch.setattr(
        runtime_settings,
        "multi_agent_step_timeout_seconds",
        45.0,
    )
    monkeypatch.setattr(
        runtime_settings,
        "multi_agent_maximum_step_attempts",
        4,
    )
    monkeypatch.setattr(
        runtime_settings,
        "multi_agent_inline_budget_seconds",
        2.5,
    )

    options = graph_runtime_service._build_multi_agent_scheduler_options()

    assert options == {
        "maximum_parallel_steps": 3,
        "step_timeout_seconds": 45.0,
        "maximum_step_attempts": 4,
        "inline_budget_seconds": 2.5,
    }


def test_multi_agent_scheduler_options_should_respect_disabled_controls(
    monkeypatch,
) -> None:
    """
    测试关闭全局开关时多 Agent 超时和重试是否回退到兼容行为。

    参数：
        monkeypatch:
            pytest 提供的临时属性替换工具。

    返回值：
        None。
    """

    runtime_settings = graph_runtime_service.settings.runtime
    monkeypatch.setattr(runtime_settings, "enable_timeout", False)
    monkeypatch.setattr(runtime_settings, "enable_retry", False)
    monkeypatch.setattr(
        runtime_settings,
        "multi_agent_inline_budget_seconds",
        0.0,
    )

    options = graph_runtime_service._build_multi_agent_scheduler_options()

    assert options["step_timeout_seconds"] is None
    assert options["maximum_step_attempts"] == 1
    assert options["inline_budget_seconds"] is None


def test_graph_runtime_should_build_shared_redis_long_task_service() -> None:
    """
    验证启用 Redis 时 Store 和 Stream 复用同一个 Provider 客户端。

    返回值含义：
        None。
    """

    redis_provider = FakeRedisProvider(enabled=True)
    runtime = GraphRuntimeService(redis_provider=redis_provider)

    application_service = (
        runtime._build_long_task_application_service()
    )

    assert application_service is not None
    assert application_service._store._redis is redis_provider._client
    assert (
        application_service._queue_publisher._redis
        is redis_provider._client
    )
    assert redis_provider.client_reads == 1


def test_graph_runtime_should_skip_disabled_redis_handoff() -> None:
    """
    验证 Redis 禁用时不读取客户端并保持长任务交接依赖为空。

    返回值含义：
        None。
    """

    redis_provider = FakeRedisProvider(enabled=False)
    runtime = GraphRuntimeService(redis_provider=redis_provider)

    application_service = (
        runtime._build_long_task_application_service()
    )

    assert application_service is None
    assert redis_provider.client_reads == 0


def test_graph_runtime_should_inject_long_task_service_into_entry_node(
    monkeypatch,
) -> None:
    """
    验证多智能体节点构建时把长任务交接服务传给入口节点。

    参数含义：
        monkeypatch:
            pytest 临时替换工具，用于隔离 Planner、Worker 和编排器构建。

    返回值含义：
        None。
    """

    captured_entry_kwargs = {}
    handoff_service = object()

    monkeypatch.setattr(
        graph_runtime_service,
        "PlannerAgent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_default_skill_runtime",
        lambda: object(),
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_graph_agent_workers",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "MultiAgentTaskScheduler",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "ResultAggregator",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "MultiAgentOrchestrator",
        lambda **kwargs: object(),
    )

    def fake_build_multi_agent_entry_node(**kwargs):
        """
        记录入口节点构建参数并返回固定节点。

        参数含义：
            **kwargs:
                GraphRuntimeService 注入入口节点的全部依赖。

        返回值含义：
            str:
                测试使用的固定多智能体节点。
        """

        captured_entry_kwargs.update(kwargs)
        return "multi_agent_node"

    monkeypatch.setattr(
        graph_runtime_service,
        "build_multi_agent_entry_node",
        fake_build_multi_agent_entry_node,
    )
    runtime = GraphRuntimeService()
    monkeypatch.setattr(
        runtime,
        "_build_long_task_application_service",
        lambda: handoff_service,
    )

    node = runtime._build_multi_agent_node(
        dog_knowledge_agent=SimpleNamespace(ainvoke=object()),
        general_agent=SimpleNamespace(ainvoke=object()),
    )

    assert node == "multi_agent_node"
    assert (
        captured_entry_kwargs["long_task_application_service"]
        is handoff_service
    )


def test_default_container_should_inject_registered_redis_provider() -> None:
    """
    验证默认容器把同一个 RedisProvider 注入 GraphRuntimeService。

    返回值含义：
        None。
    """

    from src.runtime.container.init import container

    graph_runtime = container.get("graph_runtime")

    assert graph_runtime.redis_provider is container.get("redis")


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
    injected_skill_node = object()
    injected_guard_node = object()
    shared_skill_runtime = object()
    captured_skill_kwargs = {}
    captured_guard_kwargs = {}

    def fake_build_skill_prepare_node(**kwargs):
        """记录主图向 Skill 节点注入的宠物档案服务。"""

        captured_skill_kwargs.update(kwargs)
        return injected_skill_node

    monkeypatch.setattr(
        graph_runtime_service,
        "build_skill_prepare_node",
        fake_build_skill_prepare_node,
    )
    monkeypatch.setattr(
        graph_runtime_service,
        "build_default_skill_runtime",
        lambda: shared_skill_runtime,
    )

    def fake_build_task_relation_guard_node(**kwargs):
        """记录主图向任务关系门卫注入的 Skill 运行器。"""

        captured_guard_kwargs.update(kwargs)
        return injected_guard_node

    monkeypatch.setattr(
        graph_runtime_service,
        "build_task_relation_guard_node",
        fake_build_task_relation_guard_node,
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
    pet_profile_service = object()

    class FakeMemoryProvider:
        """只提供宠物档案服务的主图构建测试替身。"""

        def __init__(self) -> None:
            self.pet_profile_service = pet_profile_service

    memory_provider = FakeMemoryProvider()
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
        "pet_profile_service": pet_profile_service,
        "checkpoint_manager": checkpoint_provider.manager,
    }
    assert graph.nodes["memory_extract"] is injected_node
    assert graph.nodes["task_relation_guard"] is injected_guard_node
    assert captured_guard_kwargs == {
        "skill_runtime": shared_skill_runtime,
    }
    assert graph.entry_point == "task_relation_guard"
    assert graph.nodes["skill_prepare"] is injected_skill_node
    assert captured_skill_kwargs == {
        "skill_runtime": shared_skill_runtime,
        "pet_profile_service": pet_profile_service
    }
    assert graph.nodes["multi_agent"] == "multi_agent"


def test_graph_runtime_should_cancel_registered_multi_agent_task() -> None:
    """
    检查 GraphRuntimeService 会把外部取消请求转交给运行中任务登记表。

    参数含义：无。
    返回值含义：None。
    """

    service = GraphRuntimeService()
    task_id = "multi_agent_task_runtime_cancel"
    token = service._multi_agent_cancellation_registry.register(task_id)

    assert service.cancel_multi_agent_task(task_id) is True
    assert token.is_cancelled is True
    assert service.cancel_multi_agent_task("unknown_task") is False
