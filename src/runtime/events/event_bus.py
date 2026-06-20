from collections import defaultdict

import inspect

from src.logger import logger


class EventBus:

    def __init__(
        self,
        raise_listener_errors: bool = False,
    ):
        """
        初始化事件总线。

        功能：
        - 创建 listeners 监听器注册表
        - 支持按事件类型注册多个 listener
        - 支持配置 listener 报错时是否继续执行

        参数：
            raise_listener_errors：
                是否在 listener 报错时继续向外抛出异常。
                False 表示记录错误并继续执行后续 listener。
                True 表示 listener 报错后立即抛出异常。

        返回值：
            None：构造函数无返回值。

        专业名词：
            EventBus（事件总线）：
                用于在系统内部发布事件和监听事件，降低模块之间的直接依赖。

            listener（监听器）：
                订阅事件并处理事件的对象。

            fault isolation（故障隔离）：
                某个模块失败时，不影响其他模块继续运行。
        """

        self.listeners = defaultdict(list)
        self.raise_listener_errors = raise_listener_errors

    def subscribe(
        self,
        event_type,
        listener,
    ):
        """
        订阅指定类型的事件。

        功能：
        - 将 listener 注册到 event_type 对应的监听器列表中
        - 当 emit 收到该类型事件时，会依次调用这些 listener

        参数：
            event_type：
                事件类型，通常是事件类，例如 ToolStartEvent。

            listener：
                监听器对象，必须实现 handle(event) 方法。

        返回值：
            None：无业务返回值，只修改 listeners 注册表。
        """

        self.listeners[event_type].append(listener)

    async def emit(
        self,
        event,
    ):
        """
        发送事件。

        功能：
        - 根据 type(event) 获取事件类型
        - 找到订阅该事件类型的所有 listener
        - 依次调用 listener.handle(event)
        - 同时兼容同步 handle 和异步 handle
        - 默认隔离单个 listener 的异常，避免影响后续 listener

        参数：
            event：
                事件对象，例如 ToolStartEvent、ToolSuccessEvent 等。

        返回值：
            None：无业务返回值，只负责通知监听器。

        专业名词：
            emit（发送事件）：
                把事件对象发送给所有订阅者。

            sync listener（同步监听器）：
                handle 方法使用普通 def 定义的监听器。

            async listener（异步监听器）：
                handle 方法使用 async def 定义的监听器。

            awaitable（可等待对象）：
                可以被 await 等待的对象，例如 coroutine、Task、Future。

            coroutine（协程）：
                async 函数调用后返回的异步执行对象。
        """

        event_type = type(event)

        listeners = self.listeners.get(
            event_type,
            [],
        )

        for listener in listeners:

            try:
                result = listener.handle(
                    event
                )

                if inspect.isawaitable(result):
                    await result

            except Exception as exc:
                logger.exception(
                    f"EventBus listener 执行失败: "
                    f"event_type={event_type.__name__}, "
                    f"listener={listener.__class__.__name__}, "
                    f"error={exc}"
                )

                if self.raise_listener_errors:
                    raise


event_bus = EventBus()