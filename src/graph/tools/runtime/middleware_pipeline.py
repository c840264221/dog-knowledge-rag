from sqlalchemy.util import await_only
import time


class MiddlewarePipeline:

    def __init__(self, middlewares):
        self.middlewares = middlewares

    async def run(self,ctx,final_func):

        try:
            async def build_chain(index):
                # 最后一层
                if index == len(self.middlewares):

                    return await final_func()

                middleware = self.middlewares[index]

                return await middleware.process(
                        ctx,
                        lambda :build_chain(index + 1)
                    )

            return await build_chain(0)
        finally:
            ctx.latency = time.time() - ctx.start_time