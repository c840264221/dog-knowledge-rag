"""
container/provider 单元测试。

container（容器）：统一管理项目中的 provider（提供者）和 runtime service（运行时服务）。
provider（提供者）：负责创建、保存、提供某类服务实例。
DI Container（Dependency Injection Container，依赖注入容器）：用于统一注册、获取、管理依赖对象的容器。
lifecycle（生命周期）：对象从启动 startup 到关闭 shutdown 的管理过程。
"""


class FakeProvider:
    """
    测试用假 Provider。

    FakeProvider（假提供者）：专门用于测试的简单 provider，
    不依赖真实数据库、LLM、向量库或外部服务。
    """

    def __init__(self, name: str):
        """
        初始化假 Provider。

        参数：
            name：provider 名称，字符串格式。

        返回值：
            None：构造函数无返回值。
        """
        self.name = name


def test_container_init_module_can_be_imported():
    """
    测试 container init 模块是否可以正常导入。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    import src.runtime.container.init as container_init

    assert container_init is not None


def test_container_object_exists():
    """
    测试 container 对象是否存在。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    import src.runtime.container.init as container_init

    assert hasattr(container_init, "container")


def test_container_object_is_not_none():
    """
    测试 container 对象是否不为空。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.container.init import container

    assert container is not None


def test_container_public_api_should_match_expected_methods():
    """
    测试 container 是否暴露预期的公开 API。

    public API（公开接口）：外部模块可以调用的方法或属性。
    expected methods（预期方法）：当前架构中 container 应该提供的方法。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.container.init import container

    expected_methods = {
        "get",
        "register",
        "startup",
        "shutdown",
    }

    public_attrs = {
        attr
        for attr in dir(container)
        if not attr.startswith("_")
    }

    assert expected_methods.issubset(public_attrs)


def test_container_public_api_methods_are_callable():
    """
    测试 container 的公开 API 是否都是可调用方法。

    callable（可调用对象）：可以像函数一样执行的对象，例如普通函数、方法、实现了 __call__ 的对象。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.container.init import container

    assert callable(container.get)
    assert callable(container.register)
    assert callable(container.startup)
    assert callable(container.shutdown)


def test_container_register_and_get_provider():
    """
    测试 container 是否可以注册并获取 provider。

    register（注册）：把 provider 保存到 container 中。
    get（获取）：根据 key 从 container 中取回 provider。
    key（键）：用于标识 provider 的字符串名称。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.container.init import container

    provider_key = "test.fake_provider"
    fake_provider = FakeProvider(name="fake_provider")

    container.register(provider_key, fake_provider)

    resolved_provider = container.get(provider_key)

    assert resolved_provider is fake_provider
    assert resolved_provider.name == "fake_provider"


def test_container_get_same_provider_should_return_same_instance():
    """
    测试多次 get 同一个 provider key 是否返回同一个实例。

    same instance（同一个实例）：两次获取到的是内存中的同一个对象。
    singleton-like behavior（类似单例行为）：同一个 key 对应同一个对象实例。

    参数：
        无。

    返回值：
        None：无返回值。pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.container.init import container

    provider_key = "test.same_instance_provider"
    fake_provider = FakeProvider(name="same_instance_provider")

    container.register(provider_key, fake_provider)

    first_resolved_provider = container.get(provider_key)
    second_resolved_provider = container.get(provider_key)

    assert first_resolved_provider is fake_provider
    assert second_resolved_provider is fake_provider
    assert first_resolved_provider is second_resolved_provider