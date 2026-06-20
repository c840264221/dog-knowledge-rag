"""
runtime context 单元测试。

runtime context（运行时上下文）：
用于在一次请求执行过程中统一保存 trace_id、user_id、session_id、component、
request_scope、registry、hook_manager、metadata 等运行时信息。

scope（作用域）：
表示某一类运行时数据或能力的管理范围，例如 memory scope、retrieval scope、state scope。

registry（注册表）：
用于统一注册和获取 runtime service（运行时服务）。

hook manager（钩子管理器）：
用于管理运行时 hook（钩子），方便在关键执行点插入扩展逻辑。

lifecycle（生命周期）：
服务从 startup（启动）到 shutdown（关闭）的过程。
"""

import uuid

import pytest

from src.runtime.context.runtime_context import RuntimeContext

from src.runtime.context.request_scope import RequestScope

from src.runtime.scopes.memory_scope import MemoryScope

from src.runtime.scopes.retrieval_scope import RetrievalScope

from src.runtime.scopes.metrics_scope import MetricsScope

from src.runtime.scopes.state_scope import StateScope

from src.runtime.scopes.timeline_scope import TimelineScope

from src.runtime.services.runtime_service_registry import (
    RuntimeServiceRegistry,
)

from src.runtime.hooks.hook_manager import RuntimeHookManager


class AsyncLifecycleService:
    """
    测试用异步生命周期服务。

    AsyncLifecycleService（异步生命周期服务）：
    用于测试 RuntimeContext.startup 和 RuntimeContext.shutdown
    是否会调用 registry（注册表）中服务的 startup / shutdown 方法。
    """

    def __init__(self):
        """
        初始化测试服务。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """
        self.started = False
        self.shutdown_called = False
        self.events = []

    async def startup(self):
        """
        启动测试服务。

        参数：
            无。

        返回值：
            None：无业务返回值，只修改服务内部状态。
        """
        self.started = True
        self.events.append("startup")

    async def shutdown(self):
        """
        关闭测试服务。

        参数：
            无。

        返回值：
            None：无业务返回值，只修改服务内部状态。
        """
        self.shutdown_called = True
        self.events.append("shutdown")


class SyncLifecycleService:
    """
    测试用同步生命周期服务。

    SyncLifecycleService（同步生命周期服务）：
    用于测试 RuntimeContext.startup 和 RuntimeContext.shutdown
    是否可以兼容普通 def 定义的 startup / shutdown 方法。
    """

    def __init__(self):
        """
        初始化同步测试服务。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """
        self.started = False
        self.shutdown_called = False
        self.events = []

    def startup(self):
        """
        同步启动测试服务。

        参数：
            无。

        返回值：
            None：无业务返回值，只修改服务内部状态。
        """
        self.started = True
        self.events.append("startup")

    def shutdown(self):
        """
        同步关闭测试服务。

        参数：
            无。

        返回值：
            None：无业务返回值，只修改服务内部状态。
        """
        self.shutdown_called = True
        self.events.append("shutdown")


def test_runtime_context_basic_fields(
    sample_user_id,
    sample_session_id,
):
    """
    测试 RuntimeContext 基础字段是否可以正常构造。

    参数：
        sample_user_id：pytest fixture，测试用户 ID。
        sample_session_id：pytest fixture，测试会话 ID。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    trace_id = str(uuid.uuid4())

    runtime_context = RuntimeContext(
        trace_id=trace_id,
        user_id=sample_user_id,
        session_id=sample_session_id,
        component="test_component",
    )

    assert runtime_context.trace_id == trace_id
    assert runtime_context.user_id == "test_user"
    assert runtime_context.session_id == "test_session"
    assert runtime_context.component == "test_component"


def test_runtime_context_default_fields_should_be_none_or_empty():
    """
    测试 RuntimeContext 默认字段是否符合预期。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    assert runtime_context.trace_id is None
    assert runtime_context.user_id is None
    assert runtime_context.session_id is None
    assert runtime_context.component is None
    assert runtime_context.current_span is None
    assert runtime_context.error is None
    assert runtime_context.metadata == {}


def test_runtime_context_metadata_should_be_isolated_between_instances():
    """
    测试不同 RuntimeContext 实例之间的 metadata 是否相互隔离。

    metadata（元数据）：
    用于保存请求执行过程中的附加信息。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_context = RuntimeContext()
    second_context = RuntimeContext()

    first_context.metadata["source"] = "first"

    assert first_context.metadata["source"] == "first"
    assert "source" not in second_context.metadata


def test_runtime_context_should_create_core_runtime_objects():
    """
    测试 RuntimeContext 是否会自动创建核心运行时对象。

    核心运行时对象包括：
    - RequestScope（请求作用域）
    - RuntimeServiceRegistry（运行时服务注册表）
    - RuntimeHookManager（运行时钩子管理器）

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    assert isinstance(
        runtime_context.request_scope,
        RequestScope,
    )

    assert isinstance(
        runtime_context.registry,
        RuntimeServiceRegistry,
    )

    assert isinstance(
        runtime_context.hook_manager,
        RuntimeHookManager,
    )


def test_runtime_context_should_register_memory_scope():
    """
    测试 RuntimeContext 初始化后是否注册 MemoryScope。

    MemoryScope（记忆作用域）：
    用于管理当前请求中的 memory（记忆）相关运行时数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    memory_scope = runtime_context.service(
        MemoryScope,
    )

    assert isinstance(
        memory_scope,
        MemoryScope,
    )


def test_runtime_context_should_register_retrieval_scope():
    """
    测试 RuntimeContext 初始化后是否注册 RetrievalScope。

    RetrievalScope（检索作用域）：
    用于管理 RAG 检索过程中的运行时数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    retrieval_scope = runtime_context.service(
        RetrievalScope,
    )

    assert isinstance(
        retrieval_scope,
        RetrievalScope,
    )


def test_runtime_context_should_register_metrics_scope():
    """
    测试 RuntimeContext 初始化后是否注册 MetricsScope。

    MetricsScope（指标作用域）：
    用于管理 latency、token、命中次数等统计指标数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    metrics_scope = runtime_context.service(
        MetricsScope,
    )

    assert isinstance(
        metrics_scope,
        MetricsScope,
    )


def test_runtime_context_should_register_state_scope():
    """
    测试 RuntimeContext 初始化后是否注册 StateScope。

    StateScope（状态作用域）：
    用于管理当前图执行过程中的 state（状态）相关数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    state_scope = runtime_context.service(
        StateScope,
    )

    assert isinstance(
        state_scope,
        StateScope,
    )


def test_runtime_context_should_register_timeline_scope():
    """
    测试 RuntimeContext 初始化后是否注册 TimelineScope。

    TimelineScope（时间线作用域）：
    用于记录当前请求执行过程中的 timeline event（时间线事件）。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    timeline_scope = runtime_context.service(
        TimelineScope,
    )

    assert isinstance(
        timeline_scope,
        TimelineScope,
    )


def test_runtime_context_state_helper_should_return_state_scope():
    """
    测试 RuntimeContext.state() 是否可以返回 StateScope。

    helper method（辅助方法）：
    为常用 service 获取逻辑提供更简洁的访问方式。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    state_scope = runtime_context.state()

    assert isinstance(
        state_scope,
        StateScope,
    )

    assert state_scope is runtime_context.service(
        StateScope,
    )


def test_runtime_context_timeline_helper_should_return_timeline_scope():
    """
    测试 RuntimeContext.timeline() 是否可以返回 TimelineScope。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    timeline_scope = runtime_context.timeline()

    assert isinstance(
        timeline_scope,
        TimelineScope,
    )

    assert timeline_scope is runtime_context.service(
        TimelineScope,
    )


def test_runtime_context_hooks_should_return_hook_manager():
    """
    测试 RuntimeContext.hooks() 是否可以返回 RuntimeHookManager。

    hooks（钩子）：
    用于在运行时关键节点插入扩展逻辑，例如 before_node、after_node、on_error。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()

    hook_manager = runtime_context.hooks()

    assert isinstance(
        hook_manager,
        RuntimeHookManager,
    )

    assert hook_manager is runtime_context.hook_manager


@pytest.mark.asyncio
async def test_runtime_context_startup_should_call_registered_service_startup():
    """
    测试 RuntimeContext.startup 是否会调用已注册服务的 startup 方法。

    startup（启动）：
    服务初始化资源的生命周期方法。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()
    lifecycle_service = AsyncLifecycleService()

    runtime_context.registry.register(
        lifecycle_service,
    )

    await runtime_context.startup()

    assert lifecycle_service.started is True
    assert lifecycle_service.events == ["startup"]


@pytest.mark.asyncio
async def test_runtime_context_shutdown_should_call_registered_service_shutdown():
    """
    测试 RuntimeContext.shutdown 是否会调用已注册服务的 shutdown 方法。

    shutdown（关闭）：
    服务释放资源的生命周期方法。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()
    lifecycle_service = AsyncLifecycleService()

    runtime_context.registry.register(
        lifecycle_service,
    )

    await runtime_context.shutdown()

    assert lifecycle_service.shutdown_called is True
    assert lifecycle_service.events == ["shutdown"]

@pytest.mark.asyncio
async def test_runtime_context_startup_should_call_sync_service_startup():
    """
    测试 RuntimeContext.startup 是否兼容同步 service.startup。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()
    lifecycle_service = SyncLifecycleService()

    runtime_context.registry.register(
        lifecycle_service,
    )

    await runtime_context.startup()

    assert lifecycle_service.started is True
    assert lifecycle_service.events == ["startup"]


@pytest.mark.asyncio
async def test_runtime_context_shutdown_should_call_sync_service_shutdown():
    """
    测试 RuntimeContext.shutdown 是否兼容同步 service.shutdown。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_context = RuntimeContext()
    lifecycle_service = SyncLifecycleService()

    runtime_context.registry.register(
        lifecycle_service,
    )

    await runtime_context.shutdown()

    assert lifecycle_service.shutdown_called is True
    assert lifecycle_service.events == ["shutdown"]