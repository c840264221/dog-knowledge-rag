from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError
)

from src.graph.tools.runtime.middleware.base_middleware import (
    BaseMiddleware
)

from src.graph.tools.errors.tool_errors import ToolTimeoutError

import asyncio


class TimeoutMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        timeout = ctx.tool.metadata.timeout

        # 同步运行下的超时设置
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #
        #     future = executor.submit(next_func)

        try:
            return await asyncio.wait_for(
                next_func(),
                timeout=timeout
            )

        except asyncio.TimeoutError:

            raise ToolTimeoutError(
                ctx.tool.metadata.name
            )

    def execute(self,ctx, func,timeout):

        with ThreadPoolExecutor(max_workers=1) as executor:

            future = executor.submit(func)

            try:

                return future.result(
                    timeout=timeout
                )

            except TimeoutError:
                ctx.error = "timeout"
                raise ToolTimeoutError(
                    ctx.tool.metadata.name
                )