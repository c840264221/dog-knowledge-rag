import asyncio

from src.logger import logger


def safe_execute_tool(func,args=None,timeout=5):
    logger.debug(f"args: {args}, args_type:{type(args)},args is not None---{args is not None}")
    try:

        # 异步工具调用
        if asyncio.iscoroutinefunction(func):

            result = asyncio.run(
                asyncio.wait_for(
                    func(args) if len(args) != 0 else func(),
                    timeout=timeout
                )
            )

        # 同步工具调用
        else:
            result = (
                func(args)
                if len(args) != 0
                else func()
            )

        return result

    except asyncio.TimeoutError:

        logger.warning(
            f"工具执行超时: {func.__name__}"
        )

        return "工具执行超时"

    except Exception as e:

        logger.exception(
            f"工具执行失败: {e}"
        )

        return f"工具执行失败: {str(e)}"


import time
import traceback

from src.graph.tools.registry.tool_registry import (
    registry
)

from src.graph.tools.runtime.middleware.logging_middleware import (
    LoggingMiddleware
)

from src.graph.tools.runtime.middleware.retry_middleware import (
    RetryMiddleware
)

from src.graph.tools.runtime.middleware.timeout_middleware import (
    TimeoutMiddleware
)

from src.graph.tools.schemas.tool_result_schema import (
    ToolResult
)

from src.graph.tools.runtime.tool_context import (
    ToolContext
)

from src.graph.tools.runtime.middleware.async_middleware import AsyncMiddleware

from src.graph.tools.errors.tool_errors import (
    ToolNotFoundError,
    ToolExecutionError
)

from src.graph.tools.runtime.middleware_pipeline import MiddlewarePipeline

from src.logger import logger


class ToolExecutor:

    def __init__(self):

        self.middlewares = [
            LoggingMiddleware(),

            RetryMiddleware(),

            TimeoutMiddleware(),

            AsyncMiddleware()
        ]

        self.pipeline = MiddlewarePipeline(self.middlewares)

    async def execute(self,tool_name,args):

        tool = registry.get_tool(tool_name)

        if not tool:
            logger.error(f"工具不存在: {tool_name}")

            raise ToolNotFoundError(
                f"{tool_name}"
            )

        ctx = ToolContext(
            tool,
            args
        )

        async def final_func():
            return await ctx.invoke()

        # self.logging.before(ctx)

        try:

            # result = self.retry.run(
            #     ctx,
            #     lambda:
            #         self.timeout.execute(
            #             ctx,
            #             lambda:
            #                 self.async_runner.execute(
            #                     ctx
            #                 ),
            #
            #             timeout=tool.metadata.timeout
            #     ),
            #
            #     retries=tool.metadata.retries
            # )
            #
            # ctx.result = result

            # self.logging.after(ctx)

            result = await self.pipeline.run(ctx,final_func)

            return ToolResult(

                success=True,

                tool_name=tool_name,

                content=result,

                latency=ctx.latency,

                retry_count=ctx.retry_count
            )

        except Exception as e:

            ctx.error = str(e)

            # self.logging.on_error(ctx,e)

            raise ToolExecutionError(
                str(e)
            )