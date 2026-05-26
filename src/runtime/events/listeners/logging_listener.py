from src.logger import logger

from src.runtime.events.event_types import (
    ToolStartEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
    SpanStartEvent,
    SpanEndEvent
)

from src.runtime.trace.init import (
    trace_manager
)


class LoggingListener:

    async def handle(self, event):

        if isinstance(event, ToolStartEvent):
        # if isinstance(event, SpanStartEvent):

            logger.info(

                f"[Tool Start] "

                f"tool={event.ctx.tool.metadata.name} "

                f"trace_id={event.trace_id}"
            )

        elif isinstance(event, ToolSuccessEvent):

            span = event.ctx.current_span

            logger.info(

                f"[Tool Success] "

                f"tool={event.ctx.tool.metadata.name} "

                # f"latency={root_span.latency}s "
                f"latency={span.latency if span else None}s "

                f"trace_id={event.trace_id}"
            )

        elif isinstance(event, ToolErrorEvent):

            span = event.ctx.current_span

            logger.error(

                f"[Tool Error] "
                
                f"tool={event.ctx.tool.metadata.name} "
                
                f"error={span.error if span else None} "
                
                f"latency={span.latency if span else None}s "
                
                f"trace_id={event.ctx.trace_id}"
            )