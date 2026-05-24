import time
from src.logger import logger
from src.graph.tools.runtime.middleware.base_middleware import BaseMiddleware


class LoggingMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        logger.info(
            f"[Tool Start] "
            f"{ctx.tool.metadata.name}"
        )

        try:
            result = await next_func()
            ctx.latency = round(time.time()- ctx.start_time,3)

            logger.info(

                f"[Tool Success] "

                f"{ctx.tool.metadata.name} "

                f"latency={ctx.latency}s"
            )

            return result

        except Exception as e:

            ctx.latency = round(time.time()- ctx.start_time,3)

            logger.error(

                f"[Tool Error] "

                f"{ctx.tool.metadata.name} "

                f"error={str(e)}"
            )

            raise e

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