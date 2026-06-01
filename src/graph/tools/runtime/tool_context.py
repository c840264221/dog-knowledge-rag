import time
import uuid
from typing import Any
from typing import Optional, Dict
import asyncio
from src.runtime.trace.init import trace_manager
# from src.runtime.trace import trace_ctx


class ToolContext:

    def __init__(self,tool,args, trace_id: Optional[str] = None):
        # 工具对象
        self.tool = tool

        # 工具参数
        self.args: Dict[str, Any] = args

        # 开始时间
        self.start_time = time.time()

        # trace id
        self.trace_id = trace_id

        self.span_id: str | None = None

        # 当前节点
        self.current_span = None

        # 重试次数
        self.retry_count: int = 0

        # 工具结果
        self.result: Optional[Any] = None

        # 错误信息
        self.error: Optional[str] = None

        # 耗时
        # self.latency: Optional[float] = None

        # 启用runtime上下文管理  废弃trace_ctx
        from src.runtime.context import runtime_ctx

        runtime_context = runtime_ctx.get()
        runtime_context.trace_id = self.trace_id

    async def invoke(self):
        if asyncio.iscoroutinefunction(self.tool.run):
            return await self.tool.run(
                self.args
            )
        else:
            return await asyncio.to_thread(
                self.tool.run,
                self.args
            )