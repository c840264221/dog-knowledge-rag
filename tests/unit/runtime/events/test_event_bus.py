"""
event bus 单元测试。

EventBus（事件总线）：
用于在系统内部发布事件和监听事件，让 trace、timeline、logger、tool middleware 等模块解耦。

event（事件）：
系统运行过程中发生的一次动作或状态变化，例如工具开始、工具成功、节点开始、节点结束。

event_type（事件类型）：
当前 EventBus 使用事件对象的 class 类型作为事件类型，例如 ToolStartedEvent。

listener（监听器）：
订阅某类事件的对象。当前 EventBus 要求 listener 必须实现 async handle(event) 方法。

emit（发送事件）：
把事件对象发送给订阅了该事件类型的所有 listener。

subscribe（订阅事件）：
把 listener 注册到某个 event_type 下。
"""

import pytest

from src.runtime.events.event_bus import (
    EventBus,
    event_bus,
)


class FakeToolStartedEvent:
    """
    测试用工具开始事件。

    FakeToolStartedEvent（假工具开始事件）：
    用于模拟真实 ToolStartedEvent，避免单元测试依赖真实业务事件类。
    """

    def __init__(self, tool_name: str):
        """
        初始化测试事件。

        参数：
            tool_name：工具名称，字符串格式。

        返回值：
            None：构造函数无返回值。
        """
        self.tool_name = tool_name


class FakeToolSucceededEvent:
    """
    测试用工具成功事件。

    FakeToolSucceededEvent（假工具成功事件）：
    用于测试不同事件类型之间是否互不干扰。
    """

    def __init__(self, tool_name: str):
        """
        初始化测试事件。

        参数：
            tool_name：工具名称，字符串格式。

        返回值：
            None：构造函数无返回值。
        """
        self.tool_name = tool_name


class FakeAsyncListener:
    """
    测试用异步监听器。

    FakeAsyncListener（假异步监听器）：
    用于模拟真实 listener，内部记录收到过的 event。
    """

    def __init__(self):
        """
        初始化测试监听器。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """
        self.handled_events = []

    async def handle(self, event):
        """
        处理事件。

        参数：
            event：EventBus 发送过来的事件对象。

        返回值：
            None：无业务返回值，只把事件保存到 handled_events。
        """
        self.handled_events.append(event)


class FakeSyncListener:
    """
    测试用同步监听器。

    FakeSyncListener（假同步监听器）：
    用于测试 EventBus.emit 是否可以兼容普通 def handle(event)。
    """

    def __init__(self):
        """
        初始化同步测试监听器。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.handled_events = []

    def handle(self, event):
        """
        同步处理事件。

        参数：
            event：EventBus 发送过来的事件对象。

        返回值：
            None：无业务返回值，只把事件保存到 handled_events。
        """

        self.handled_events.append(event)


class FakeFailingListener:
    """
    测试用失败监听器。

    FakeFailingListener（假失败监听器）：
    用于测试某个 listener 报错时，EventBus 是否能隔离异常。
    """

    def __init__(
        self,
        error_message: str = "listener failed",
    ):
        """
        初始化失败监听器。

        参数：
            error_message：
                抛出异常时使用的错误信息。

        返回值：
            None：构造函数无返回值。
        """

        self.error_message = error_message

    async def handle(
        self,
        event,
    ):
        """
        处理事件并主动抛出异常。

        参数：
            event：
                EventBus 发送过来的事件对象。

        返回值：
            None：本方法会抛出 RuntimeError，不会正常返回。
        """

        raise RuntimeError(
            self.error_message
        )


def test_event_bus_can_be_created():
    """
    测试 EventBus 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    assert bus is not None
    assert bus.listeners is not None
    assert len(bus.listeners) == 0


def test_global_event_bus_should_exist():
    """
    测试全局 event_bus 对象是否存在。

    global event_bus（全局事件总线）：
    项目中可以复用的默认 EventBus 实例。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert event_bus is not None
    assert isinstance(event_bus, EventBus)


def test_event_bus_subscribe_should_register_listener():
    """
    测试 subscribe 是否可以注册 listener。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()
    listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        listener,
    )

    assert FakeToolStartedEvent in bus.listeners
    assert listener in bus.listeners[FakeToolStartedEvent]


@pytest.mark.asyncio
async def test_event_bus_emit_should_call_listener_handle():
    """
    测试 emit 是否会调用对应 listener 的 handle 方法。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()
    listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert len(listener.handled_events) == 1
    assert listener.handled_events[0] is event
    assert listener.handled_events[0].tool_name == "date"


@pytest.mark.asyncio
async def test_event_bus_emit_should_call_multiple_listeners():
    """
    测试 emit 是否会调用同一事件类型下的多个 listener。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    first_listener = FakeAsyncListener()
    second_listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        first_listener,
    )

    bus.subscribe(
        FakeToolStartedEvent,
        second_listener,
    )

    event = FakeToolStartedEvent(
        tool_name="weather",
    )

    await bus.emit(event)

    assert first_listener.handled_events == [event]
    assert second_listener.handled_events == [event]


@pytest.mark.asyncio
async def test_event_bus_emit_should_only_notify_matching_event_type():
    """
    测试 emit 是否只通知匹配事件类型的 listener。

    当前 EventBus 使用 type(event) 作为事件类型。
    因此 FakeToolStartedEvent 和 FakeToolSucceededEvent 是两个不同事件类型。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    started_listener = FakeAsyncListener()
    succeeded_listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        started_listener,
    )

    bus.subscribe(
        FakeToolSucceededEvent,
        succeeded_listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert started_listener.handled_events == [event]
    assert succeeded_listener.handled_events == []


@pytest.mark.asyncio
async def test_event_bus_emit_without_listener_should_not_raise_error():
    """
    测试没有 listener 的事件 emit 时是否不会报错。

    no listener（无监听器）：
    表示当前 event_type 没有任何订阅者。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert bus.listeners.get(FakeToolStartedEvent, []) == []


@pytest.mark.asyncio
async def test_event_bus_emit_should_keep_event_object_identity():
    """
    测试 listener 收到的 event 是否是原始 event 对象。

    object identity（对象身份）：
    使用 is 判断两个变量是否指向内存中的同一个对象。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()
    listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert listener.handled_events[0] is event


@pytest.mark.asyncio
async def test_event_bus_emit_should_support_sync_listener():
    """
    测试 EventBus.emit 是否支持同步 listener。

    sync listener（同步监听器）：
    handle 方法使用普通 def 定义，调用后会立即执行并返回 None。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()
    listener = FakeSyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert listener.handled_events == [event]


@pytest.mark.asyncio
async def test_event_bus_emit_should_support_sync_and_async_listeners_together():
    """
    测试 EventBus.emit 是否可以同时支持同步 listener 和异步 listener。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    sync_listener = FakeSyncListener()
    async_listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        sync_listener,
    )

    bus.subscribe(
        FakeToolStartedEvent,
        async_listener,
    )

    event = FakeToolStartedEvent(
        tool_name="weather",
    )

    await bus.emit(event)

    assert sync_listener.handled_events == [event]
    assert async_listener.handled_events == [event]

@pytest.mark.asyncio
async def test_event_bus_emit_should_continue_when_listener_failed():
    """
    测试某个 listener 报错时，EventBus 默认是否继续执行后续 listener。

    fault isolation（故障隔离）：
    一个 listener 失败，不影响其他 listener 继续处理事件。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus()

    failing_listener = FakeFailingListener()
    normal_listener = FakeAsyncListener()

    bus.subscribe(
        FakeToolStartedEvent,
        failing_listener,
    )

    bus.subscribe(
        FakeToolStartedEvent,
        normal_listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    await bus.emit(event)

    assert normal_listener.handled_events == [event]


@pytest.mark.asyncio
async def test_event_bus_emit_should_raise_error_in_strict_mode():
    """
    测试 strict mode 下 listener 报错时是否会向外抛出异常。

    strict mode（严格模式）：
    listener 报错时不吞掉异常，而是直接 raise 给调用方。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    bus = EventBus(
        raise_listener_errors=True,
    )

    failing_listener = FakeFailingListener(
        error_message="strict listener failed",
    )

    bus.subscribe(
        FakeToolStartedEvent,
        failing_listener,
    )

    event = FakeToolStartedEvent(
        tool_name="date",
    )

    with pytest.raises(
        RuntimeError,
        match="strict listener failed",
    ):
        await bus.emit(event)