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

from src.runtime.state.state_scope import (
    StateScope
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

    current_agent: str | None = None

    retry_count: int = 0

    error: str | None = None

    # =========================
    # Metadata
    # =========================

    metadata: dict = field(
        default_factory=dict
    )

    # =========================
    # Runtime Data
    # =========================

    runtime_data: dict = field(
        default_factory=dict
    )

    # =========================
    # Services
    # =========================

    services: dict = field(
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

        retrieval_scope = RetrievalScope(
            self.request_scope
        )

        memory_scope = MemoryScope(
            self.request_scope
        )

        metrics_scope = MetricsScope(
            self.request_scope
        )

        state_scope = StateScope()

        self.registry.register(
            memory_scope
        )

        self.registry.register(
            retrieval_scope
        )

        self.registry.register(
            metrics_scope
        )

        self.registry.register(
            state_scope
        )

    def service(self,service_type):
        return self.registry.get(
            service_type
        )

    def hooks(self):

        return self.hook_manager

    def state(self):

        from src.runtime.state.state_scope import (
            StateScope
        )

        return self.service(
            StateScope
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