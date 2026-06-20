"""
runtime service registry 单元测试。

RuntimeServiceRegistry（运行时服务注册表）：
用于统一注册、保存和获取 runtime service（运行时服务）。

runtime service（运行时服务）：
在一次 Agent 执行过程中被 RuntimeContext 管理的服务对象，
例如 MemoryScope、RetrievalScope、MetricsScope、StateScope、TimelineScope。

registry（注册表）：
用于维护 service_type（服务类型）到 service_instance（服务实例）的映射关系。
"""


from src.runtime.services.runtime_service_registry import (
    RuntimeServiceRegistry,
)


class FakeMemoryService:
    """
    测试用记忆服务。

    FakeMemoryService（假记忆服务）：
    用于模拟 MemoryScope 这类运行时服务，避免测试依赖真实业务逻辑。
    """

    def __init__(self, name: str):
        """
        初始化假记忆服务。

        参数：
            name：服务名称，字符串格式。

        返回值：
            None：构造函数无返回值。
        """
        self.name = name


class FakeRetrievalService:
    """
    测试用检索服务。

    FakeRetrievalService（假检索服务）：
    用于模拟 RetrievalScope 这类运行时服务。
    """

    def __init__(self, name: str):
        """
        初始化假检索服务。

        参数：
            name：服务名称，字符串格式。

        返回值：
            None：构造函数无返回值。
        """
        self.name = name


def test_registry_can_be_created():
    """
    测试 RuntimeServiceRegistry 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()

    assert registry is not None


def test_registry_can_register_and_get_service():
    """
    测试 registry 是否可以注册并获取 service。

    register（注册）：
    把 service 实例保存到 registry 中。

    get（获取）：
    根据 service 类型从 registry 中取回对应实例。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()
    service = FakeMemoryService(
        name="memory_service",
    )

    registry.register(
        service,
    )

    resolved_service = registry.get(
        FakeMemoryService,
    )

    assert resolved_service is service
    assert resolved_service.name == "memory_service"


def test_registry_get_should_return_same_instance():
    """
    测试 registry 多次 get 是否返回同一个 service 实例。

    same instance（同一个实例）：
    表示两次获取到的是内存中的同一个对象。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()
    service = FakeMemoryService(
        name="memory_service",
    )

    registry.register(
        service,
    )

    first_resolved_service = registry.get(
        FakeMemoryService,
    )

    second_resolved_service = registry.get(
        FakeMemoryService,
    )

    assert first_resolved_service is service
    assert second_resolved_service is service
    assert first_resolved_service is second_resolved_service


def test_registry_can_register_multiple_services():
    """
    测试 registry 是否可以注册多个不同类型的 service。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()

    memory_service = FakeMemoryService(
        name="memory_service",
    )

    retrieval_service = FakeRetrievalService(
        name="retrieval_service",
    )

    registry.register(
        memory_service,
    )

    registry.register(
        retrieval_service,
    )

    assert registry.get(FakeMemoryService) is memory_service
    assert registry.get(FakeRetrievalService) is retrieval_service


def test_registry_all_services_should_return_registered_services():
    """
    测试 registry.all_services 是否可以返回已注册的所有 service。

    all_services（所有服务）：
    返回 registry 当前保存的全部 service 实例。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()

    memory_service = FakeMemoryService(
        name="memory_service",
    )

    retrieval_service = FakeRetrievalService(
        name="retrieval_service",
    )

    registry.register(
        memory_service,
    )

    registry.register(
        retrieval_service,
    )

    services = list(
        registry.all_services()
    )

    assert memory_service in services
    assert retrieval_service in services
    assert len(services) == 2


def test_registry_all_services_should_keep_register_order():
    """
    测试 registry.all_services 是否保持注册顺序。

    register order（注册顺序）：
    service 被 register 进入 registry 的先后顺序。

    为什么要测这个：
    RuntimeContext.shutdown 使用 reversed(list(registry.all_services()))，
    如果 all_services 顺序稳定，shutdown 才能按照注册反序关闭服务。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    registry = RuntimeServiceRegistry()

    memory_service = FakeMemoryService(
        name="memory_service",
    )

    retrieval_service = FakeRetrievalService(
        name="retrieval_service",
    )

    registry.register(
        memory_service,
    )

    registry.register(
        retrieval_service,
    )

    services = list(
        registry.all_services()
    )

    assert services == [
        memory_service,
        retrieval_service,
    ]