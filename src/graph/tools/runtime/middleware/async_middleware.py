import asyncio
from src.graph.tools.runtime.middleware.base_middleware import BaseMiddleware


class AsyncMiddleware(BaseMiddleware):

    async def execute_async(self,tool,args):

        # 异步工具

        if asyncio.iscoroutinefunction(
            tool.run
        ):

            return await tool.run(args)

        # 同步工具
        else:
            # 如果是同步工具可能会阻塞事件循环 放入to_thread中
            return await asyncio.to_thread(tool.run, args)
            # return tool.run(args)

    async def process(self,ctx,next_func):

        return await next_func()
