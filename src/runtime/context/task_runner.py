import asyncio

from src.runtime.context import runtime_ctx

from src.runtime.context.context_snapshot import (
    clone_runtime_context
)


class ContextAwareTaskRunner:

    async def create_task(self,coro):

        """
        创建自动继承 RuntimeContext 的 task
        """

        parent_ctx = runtime_ctx.get()

        child_ctx = clone_runtime_context(
            parent_ctx
        )

        async def wrapped():

            runtime_ctx.set(child_ctx)

            return await coro

        return asyncio.create_task(
            wrapped()
        )


runtime_tasks = (
    ContextAwareTaskRunner()
)