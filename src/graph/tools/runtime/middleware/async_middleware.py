import asyncio


class AsyncMiddleware:

    async def execute_async(self,tool,args):

        # 异步工具

        if asyncio.iscoroutinefunction(
            tool.run
        ):

            return await tool.run(args)

        # 同步工具

        return tool.run(args)

    def execute(self,tool,args):

        return asyncio.run(self.execute_async(tool,args))