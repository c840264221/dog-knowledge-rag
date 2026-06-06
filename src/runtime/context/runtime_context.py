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

        for service in self.registry.all_services():

            if hasattr(service, "startup"):
                await service.startup()

    async def shutdown(self):

        for service in reversed(
                list(self.registry.all_services())
        ):

            if hasattr(service, "shutdown"):
                await service.shutdown()