from src.logger import logger

from src.runtime.events.event_types import (
    ToolStartEvent,
    ToolSuccessEvent,
    ToolErrorEvent,
)

from src.runtime.trace.init import trace_manager

from src.runtime.context import runtime_ctx
from src.runtime.scopes.metrics_scope import MetricsScope


class LoggingListener:

    async def handle(self, event):

        if isinstance(event, ToolStartEvent):
            
            logger.info(

                f"[Tool Start] "

                f"tool={event.ctx.tool.metadata.name} "

                f"trace_id={event.trace_id}"
            )

        elif isinstance(event, ToolSuccessEvent):
            span = trace_manager.span_map.get(
                event.span_id
            )

            logger.info(

                f"[Tool Success] "

                f"tool={event.ctx.tool.metadata.name} "

                # f"latency={root_span.latency}s "
                f"latency={span.latency if span else None}s "

                f"trace_id={event.trace_id}"
            )

        elif isinstance(event, ToolErrorEvent):

            span = trace_manager.span_map.get(
                event.span_id
            )

            logger.error(

                f"[Tool Error] "
                
                f"tool={event.ctx.tool.metadata.name} "
                
                f"error={span.error if span else None} "
                
                f"latency={span.latency if span else None}s "
                
                f"trace_id={event.ctx.trace_id}"
            )