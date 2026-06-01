from src.runtime.hooks.base_hook import (
    BaseHook
)

from src.runtime.context import runtime_ctx

from src.logger import logger


class ToolCounterHook(BaseHook):

    async def execute(

        self,

        *args,

        **kwargs
    ):

        ctx = runtime_ctx.get()

        count = ctx.runtime_data.get(
            "tool_count",
            0
        )

        count += 1

        ctx.runtime_data[
            "tool_count"
        ] = count

        logger.info(

            f"[Hook] "

            f"当前请求已调用 "

            f"{count} 次Tool"
        )