import asyncio

from src.logger import logger

import time

# import src.runtime.trace.trace_ctx as trace_ctx


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

from src.graph.tools.registry.default_registry import (
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

from src.graph.tools.runtime.middleware.trace_middleware import (
    TraceMiddleware
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

from src.runtime.trace.init import trace_manager

from src.settings import settings


class ToolExecutor:

    def __init__(self):

        # self.middlewares = [
        #     LoggingMiddleware(),
        #
        #     TraceMiddleware(),
        #
        #     RetryMiddleware(),
        #
        #     TimeoutMiddleware(),
        #
        #     AsyncMiddleware()
        # ]

        # 用工程化配置文件来控制中间件的开关
        middlewares = []

        if settings.runtime.enable_logging_middleware:
            middlewares.append(LoggingMiddleware())

        if settings.runtime.enable_trace_middleware:
            middlewares.append(TraceMiddleware())

        if settings.runtime.enable_retry_middleware:
            middlewares.append(RetryMiddleware())

        if settings.runtime.enable_timeout_middleware:
            middlewares.append(TimeoutMiddleware())

        if settings.runtime.enable_async_middleware:
            middlewares.append(AsyncMiddleware())


        self.pipeline = MiddlewarePipeline(middlewares)

    async def execute(self,tool_name,args):

        tool = registry.get_tool(tool_name)

        if not tool:
            logger.error(f"工具不存在: {tool_name}")

            raise ToolNotFoundError(
                f"{tool_name}"
            )

        # 启用runtime上下文管理  废弃trace_ctx
        from src.runtime.context import runtime_ctx

        runtime_context = runtime_ctx.get()

        trace_id = runtime_context.trace_id

        ctx = ToolContext(
            tool=tool,
            args=args,
            trace_id=trace_id
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

            await runtime_context.hooks().emit(
                "tool.before"
            )

            result = await self.pipeline.run(ctx,final_func)


            return ToolResult(

                success=True,

                tool_name=tool_name,

                content=result,

                latency=ctx.current_span.latency if ctx.current_span else None,

                retry_count=ctx.retry_count
            )

        except Exception as e:

            ctx.error = str(e)

            # self.logging.on_error(ctx,e)

            raise ToolExecutionError(
                str(e)
            )
