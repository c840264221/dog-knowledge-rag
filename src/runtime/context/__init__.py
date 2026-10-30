from src.runtime.context.runtime_context import (
    RuntimeContext
)

from src.runtime.context.context_manager import (
    RuntimeContextManager
)
from src.runtime.context.time_context import RuntimeTimeContext


runtime_ctx = RuntimeContextManager()


def get_runtime_time_context() -> dict[str, str]:
    """读取统一时间环境；无参数；有请求上下文时复用快照，旧独立调用无上下文时采样宿主时间，返回独立字典。"""
    context = runtime_ctx.get()
    snapshot = getattr(context, "time_context", None)
    if not isinstance(snapshot, RuntimeTimeContext):
        snapshot = RuntimeTimeContext.capture()
    return snapshot.as_dict()
