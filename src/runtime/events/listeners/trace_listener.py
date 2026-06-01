from src.logger import logger
from src.runtime.context import runtime_ctx
from src.runtime.trace.init import (
    trace_manager
)

from src.runtime.events.event_types import (
    SpanStartEvent,
    SpanEndEvent
)

from src.runtime.trace.trace_span import TraceSpan


class TraceListener:

    async def handle(self, event):

        if isinstance(event, SpanStartEvent):

            span = trace_manager.create_span(

                trace_id=event.trace_id,

                name=event.span_name,

                parent_span=event.parent_span
            )

            # 回填给 Middleware
            event.span = span

            logger.info(

                f"[Span Start] "

                f"{span.name} "

                f"span_id={span.span_id}"
            )

        # ========= Span End =========

        elif isinstance(event, SpanEndEvent):

            trace_manager.finish_span(

                event.span_id,

                status=event.status,

                error=event.error

            )

            span = trace_manager.span_map.get(

                event.span_id

            )

            if not span:
                return

            logger.info(

                f"[Span End] "


                f"{span.name} "


                f"status={span.status} "


                f"latency={span.latency}s"

            )