import time
from src.logger import logger


class LoggingMiddleware:

    def before(self, ctx):
        logger.info(
            f"[Tool Start] "
            f"{ctx.tool.metadata.name}"
        )

    def after(self, ctx):
        ctx.latency = round(time.time()- ctx.start_time,3)

        logger.info(

            f"[Tool Success] "

            f"{ctx.tool.metadata.name} "

            f"latency={ctx.latency}s"
        )

    def on_error(self, ctx, e):

        ctx.latency = round(time.time()- ctx.start_time,3)

        logger.error(

            f"[Tool Error] "

            f"{ctx.tool.metadata.name} "

            f"error={str(e)} "

            f"latency={ctx.latency}s"
        )