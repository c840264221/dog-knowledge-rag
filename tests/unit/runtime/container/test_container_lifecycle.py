"""
container lifecycle 单元测试。

container（容器）：统一管理项目中的 provider（提供者）和 service（服务）。
lifecycle（生命周期）：对象从启动 startup 到关闭 shutdown 的过程。
startup（启动）：初始化 provider / service 所需资源。
shutdown（关闭）：释放 provider / service 占用的资源。
isolated container（隔离容器）：测试中单独创建的容器实例，避免影响真实全局 container。
sync（同步）：普通函数调用，执行完成后直接返回结果。
async（异步）：使用 async / await 的函数调用，可以等待异步任务完成。
"""

import inspect

import pytest


class SyncLifecycleProvider:
    """
    测试用同步生命周期 Provider。

    SyncLifecycleProvider（同步生命周期提供者）：用于测试 container 是否能调用同步 startup / shutdown 方法。
    """

    def __init__(self):
        """
        初始化同步测试 Provider。

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
        启动同步 Provider。

        参数：
            无。

        返回值：
            None：无返回值。
        """
        self.started = True
        self.events.append("startup")

    def shutdown(self):
        """
        关闭同步 Provider。

        参数：
            无。

        返回值：
            None：无返回值。
        """
        self.shutdown_called = True
        self.events.append("shutdown")


class AsyncLifecycleProvider:
    """
    测试用异步生命周期 Provider。

    AsyncLifecycleProvider（异步生命周期提供者）：用于测试 container 是否能调用异步 startup / shutdown 方法。
    """

    def __init__(self):
        """
        初始化异步测试 Provider。

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
        异步启动 Provider。

        参数：
            无。

        返回值：
            None：无返回值。
        """
        self.started = True
        self.events.append("startup")

    async def shutdown(self):
        """
        异步关闭 Provider。

        参数：
            无。

        返回值：
            None：无返回值。
        """
        self.shutdown_called = True
        self.events.append("shutdown")


async def call_maybe_async(func):
    """
    调用可能是同步也可能是异步的函数。

    maybe async（可能异步）：表示函数可能直接返回普通结果，也可能返回 coroutine（协程）。
    coroutine（协程）：异步函数调用后得到的对象，需要 await 才会真正执行。
    awaitable（可等待对象）：可以被 await 等待的对象。

    参数：
        func：需要调用的函数对象，可能是同步函数，也可能是异步函数。

    返回值：
        object：函数执行后的返回值。格式取决于被调用函数本身。
    """
    result = func()

    if inspect.isawaitable(result):
        return await result

    return result


@pytest.fixture()
def isolated_container():
    """
    创建测试专用隔离 container。

    fixture（测试夹具）：pytest 中用于准备测试环境或测试依赖的函数。
    isolated container（隔离容器）：独立于真实全局 container 的新容器实例。

    参数：
        无。

    返回值：
        object：container 的新实例，类型与真实全局 container 相同。
    """
    from src.runtime.container.init import container as global_container

    container_class = type(global_container)

    return container_class()


@pytest.mark.asyncio
async def test_container_startup_should_call_sync_provider_startup(
    isolated_container,
):
    """
    测试 container.startup 是否会调用同步 Provider 的 startup 方法。

    参数：
        isolated_container：pytest fixture，测试专用隔离容器。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """
    provider_key = "test.sync_lifecycle_provider"
    provider = SyncLifecycleProvider()

    isolated_container.register(provider_key, provider)

    await call_maybe_async(isolated_container.startup)

    assert provider.started is True
    assert provider.events == ["startup"]


@pytest.mark.asyncio
async def test_container_shutdown_should_call_sync_provider_shutdown(
    isolated_container,
):
    """
    测试 container.shutdown 是否会调用同步 Provider 的 shutdown 方法。

    参数：
        isolated_container：pytest fixture，测试专用隔离容器。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """
    provider_key = "test.sync_lifecycle_provider"
    provider = SyncLifecycleProvider()

    isolated_container.register(provider_key, provider)

    await call_maybe_async(isolated_container.startup)
    await call_maybe_async(isolated_container.shutdown)

    assert provider.shutdown_called is True
    assert provider.events == ["startup", "shutdown"]


@pytest.mark.asyncio
async def test_container_startup_should_call_async_provider_startup(
    isolated_container,
):
    """
    测试 container.startup 是否会调用异步 Provider 的 startup 方法。

    参数：
        isolated_container：pytest fixture，测试专用隔离容器。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """
    provider_key = "test.async_lifecycle_provider"
    provider = AsyncLifecycleProvider()

    isolated_container.register(provider_key, provider)

    await call_maybe_async(isolated_container.startup)

    assert provider.started is True
    assert provider.events == ["startup"]


@pytest.mark.asyncio
async def test_container_shutdown_should_call_async_provider_shutdown(
    isolated_container,
):
    """
    测试 container.shutdown 是否会调用异步 Provider 的 shutdown 方法。

    参数：
        isolated_container：pytest fixture，测试专用隔离容器。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """
    provider_key = "test.async_lifecycle_provider"
    provider = AsyncLifecycleProvider()

    isolated_container.register(provider_key, provider)

    await call_maybe_async(isolated_container.startup)
    await call_maybe_async(isolated_container.shutdown)

    assert provider.shutdown_called is True
    assert provider.events == ["startup", "shutdown"]