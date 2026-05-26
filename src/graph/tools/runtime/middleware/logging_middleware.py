import time
from src.logger import logger
from src.graph.tools.runtime.middleware.base_middleware import BaseMiddleware

from src.runtime.events.event_types import (
    ToolStartEvent,
    ToolSuccessEvent,
    ToolErrorEvent
)

from src.runtime.events.event_bus import event_bus


class LoggingMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        # bound_logger = logger.bind(component="middleware.logging")
        # bound_logger.info(f"⭐Tool start: {ctx.tool.metadata.name}")

        start_event = ToolStartEvent(ctx)

        await event_bus.emit(start_event)

        # span = ctx.current_span
        #
        # logger.info(
        #
        #     f"[Tool Start] "
        #
        #     f"tool={ctx.tool.metadata.name} "
        #
        #     f"trace_id={ctx.trace_id}"
        # )

        try:

            result = await next_func()

            ctx.result = result

            success_event = ToolSuccessEvent(ctx)

            await event_bus.emit(success_event)


            # logger.info(
            #
            #     f"[Tool Success] "
            #
            #     f"tool={ctx.tool.metadata.name} "
            #
            #     f"latency={span.latency}s "
            #
            #     f"trace_id={ctx.trace_id}"
            # )

            return result

        except Exception as e:

            # logger.error(
            #
            #     f"[Tool Error] "
            #
            #     f"tool={ctx.tool.metadata.name} "
            #
            #     f"error={span.error} "
            #
            #     f"latency={span.latency}s "
            #
            #     f"trace_id={ctx.trace_id}"
            # )
            ctx.error = str(e)
            error_event = ToolErrorEvent(ctx)
            await event_bus.emit(error_event)

            raise

    def before(self, ctx):
        ctx.latency = round(time.time()- ctx.start_time,3)

        logger.info(
            f"[Tool Start] "
            f"{ctx.tool.metadata.name} "
            f"trace={ctx.trace_id}"
        )

    def after(self, ctx):
        ctx.latency = round(time.time()- ctx.start_time,3)

        logger.info(
            f"[Tool Success] "
            f"{ctx.tool.metadata.name} "
            f"latency={ctx.latency}s "
            f"retry={ctx.retry_count}"
        )

    def on_error(self, ctx, e):

        ctx.latency = round(time.time()- ctx.start_time,3)

        logger.error(

            f"[Tool Error] "

            f"{ctx.tool.metadata.name} "

            f"error={ctx.error} "

            f"retry={ctx.retry_count} "

            f"latency={ctx.latency}s "

            f"trace={ctx.trace_id}"
        )