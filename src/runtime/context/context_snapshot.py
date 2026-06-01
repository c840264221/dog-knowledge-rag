from copy import deepcopy

from src.runtime.context.runtime_context import (
    RuntimeContext
)


def clone_runtime_context(ctx: RuntimeContext) -> RuntimeContext:

    """
    克隆 RuntimeContext

    用于:
    - asyncio task propagation
    - background task
    - nested runtime
    """

    return deepcopy(ctx)