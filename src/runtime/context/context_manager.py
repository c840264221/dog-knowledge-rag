from contextvars import ContextVar

from src.runtime.context.runtime_context import (
    RuntimeContext
)


_runtime_context_var = ContextVar(
    "runtime_context",
    default=RuntimeContext()
)


class RuntimeContextManager:

    @staticmethod
    def set(ctx: RuntimeContext):

        _runtime_context_var.set(ctx)

    @staticmethod
    def get() -> RuntimeContext:

        return _runtime_context_var.get()

    @staticmethod
    def clear():

        _runtime_context_var.set(None)

    @staticmethod
    def get_scope():
        return _runtime_context_var.get().request_scope

    async def create(self, ctx):
        self.set(ctx)

        await ctx.startup()

        return ctx

    async def destroy(self):
        ctx = self.get()

        if ctx:
            await ctx.shutdown()

        self.clear()