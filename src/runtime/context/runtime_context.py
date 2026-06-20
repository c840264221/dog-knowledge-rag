from dataclasses import dataclass, field

from typing import Any

from src.runtime.context.request_scope import (
    RequestScope
)

from src.runtime.scopes.retrieval_scope import (
    RetrievalScope
)

from src.runtime.scopes.memory_scope import (
    MemoryScope
)

from src.runtime.services.runtime_service_registry import (
    RuntimeServiceRegistry
)

from src.runtime.scopes.metrics_scope import (
    MetricsScope
)

from src.runtime.hooks.hook_manager import RuntimeHookManager

from src.runtime.scopes.state_scope import (
    StateScope
)


from src.runtime.scopes.timeline_scope import (
    TimelineScope
)

import inspect


@dataclass
class RuntimeContext:

    # =========================
    # Request Info
    # =========================

    trace_id: str | None = None

    user_id: str | None = None

    session_id: str | None = None

    component: str | None = None

    # =========================
    # Runtime State
    # =========================

    current_span: Any = None

    error: str | None = None

    # =========================
    # Metadata
    # =========================

    metadata: dict = field(
        default_factory=dict
    )


    # 请求作用域
    request_scope: RequestScope = field(
        default_factory=RequestScope
    )

    registry: RuntimeServiceRegistry = field(
        default_factory=RuntimeServiceRegistry
    )

    hook_manager: RuntimeHookManager = field(
        default_factory=RuntimeHookManager
    )


    def __post_init__(self):


        # 注册记忆相关作用域
        memory_scope = MemoryScope(
            self.request_scope
        )

        self.registry.register(
            memory_scope
        )

        # 注册检索生成相关作用域
        retrieval_scope = RetrievalScope(
            self.request_scope
        )

        self.registry.register(
            retrieval_scope
        )

        # 注册统计数据相关作用域
        metrics_scope = MetricsScope(
            self.request_scope
        )

        self.registry.register(
            metrics_scope
        )

        # 注册state相关作用域
        state_scope = StateScope()

        self.registry.register(
            state_scope
        )

        # 注册时间线相关作用域
        timeline_scope = TimelineScope()

        self.registry.register(
            timeline_scope
        )

    async def _call_lifecycle_method(
            self,
            service,
            method_name: str,
    ):
        """
        调用运行时服务的生命周期方法。

        功能：
        - 根据 method_name 获取 service 上对应的生命周期方法
        - 如果 service 没有该方法，则直接跳过
        - 如果生命周期方法是同步函数，则直接执行
        - 如果生命周期方法返回 awaitable（可等待对象），则 await 等待执行完成
        - 统一兼容同步 Service 和异步 Service

        参数：
            service：运行时服务实例，例如 MemoryScope、RetrievalScope、TimelineScope 等。
            method_name：生命周期方法名称，字符串格式，例如 startup 或 shutdown。

        返回值：
            None：无业务返回值，只负责执行生命周期方法。

        专业名词：
            lifecycle method（生命周期方法）：
                服务启动或关闭时自动执行的方法。

            awaitable（可等待对象）：
                可以被 await 等待的对象，例如 coroutine、Task、Future。

            coroutine（协程）：
                async 函数调用后返回的异步执行对象。

            sync service（同步服务）：
                使用普通 def startup / def shutdown 的服务。

            async service（异步服务）：
                使用 async def startup / async def shutdown 的服务。
        """

        lifecycle_method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(lifecycle_method):
            return

        result = lifecycle_method()

        if inspect.isawaitable(result):
            await result

    def service(self,service_type):
        return self.registry.get(
            service_type
        )

    def hooks(self):

        return self.hook_manager

    def state(self):

        from src.runtime.scopes.state_scope import (
            StateScope
        )

        return self.service(
            StateScope
        )

    def timeline(self):

        return self.service(
            TimelineScope
        )

    async def startup(self):
        """
        启动 RuntimeContext 中注册的所有运行时服务。

        功能：
        - 遍历 registry 中的所有 runtime service（运行时服务）
        - 如果服务实现了 startup 方法，则自动调用
        - 同时兼容同步 startup 和异步 startup

        参数：
            无。

        返回值：
            None：无业务返回值，只执行启动流程。
        """

        for service in self.registry.all_services():
            await self._call_lifecycle_method(
                service=service,
                method_name="startup",
            )

    async def shutdown(self):
        """
        关闭 RuntimeContext 中注册的所有运行时服务。

        功能：
        - 按注册顺序的反方向遍历所有 runtime service（运行时服务）
        - 如果服务实现了 shutdown 方法，则自动调用
        - 同时兼容同步 shutdown 和异步 shutdown

        参数：
            无。

        返回值：
            None：无业务返回值，只执行关闭流程。
        """

        for service in reversed(
                list(self.registry.all_services())
        ):
            await self._call_lifecycle_method(
                service=service,
                method_name="shutdown",
            )