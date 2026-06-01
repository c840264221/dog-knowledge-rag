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


        start_event = ToolStartEvent(ctx)

        await event_bus.emit(start_event)

        try:

            result = await next_func()

            ctx.result = result

            success_event = ToolSuccessEvent(ctx)

            await event_bus.emit(success_event)

            return result

        except Exception as e:

            ctx.error = str(e)
            error_event = ToolErrorEvent(ctx)
            await event_bus.emit(error_event)

            raise
