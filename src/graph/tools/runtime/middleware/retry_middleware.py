import time

import asyncio

from src.logger import logger

from src.graph.tools.errors.base import (
    BaseRuntimeError
)

from src.graph.tools.runtime.middleware.base_middleware import (
    BaseMiddleware
)

from src.settings import settings


class RetryMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        retries = ctx.tool.metadata.retries or settings.runtime.max_retries

        last_error = None

        for i in range(retries):

            try:

                ctx.retry_count = i

                return await next_func()

            except BaseRuntimeError as e:

                last_error = e

                ctx.error = str(e)

                if not e.retryable:
                    raise e

                logger.warning(
                    f"重试中 "
                    f"{i + 1}/{retries}"
                )

                await asyncio.sleep(
                    settings.runtime.retry_delay
                )

        raise last_error

    def run(self,ctx,func,retries=3):

        last_error = None

        for i in range(retries):

            try:

                ctx.retry_count = i

                return func()

            except BaseRuntimeError as e:

                last_error = e

                ctx.error = str(e)

                if not e.retryable:
                    raise e

                logger.warning(
                    f"重试中 "
                    f"{i + 1}/{retries} "
                    f"error={e.message}"
                )


                time.sleep(1)

        raise last_error