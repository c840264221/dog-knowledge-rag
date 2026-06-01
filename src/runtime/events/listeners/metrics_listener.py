from src.runtime.context import runtime_ctx
from src.runtime.scopes.metrics_scope import MetricsScope
from src.runtime.events.event_types import (
    ToolSuccessEvent,
    ToolErrorEvent,
)

from src.runtime.trace.init import trace_manager



class MetricsListener:

    async def handle(self, event):

        runtime_context = runtime_ctx.get()

        metrics = runtime_context.service(
            MetricsScope
        )

        span = trace_manager.span_map.get(
            event.span_id
        )

        if isinstance(event,ToolSuccessEvent):

            metrics.increment(
                "tool_count"
            )

            metrics.update(
                "tool_latency",
                metrics.get_metrics().get(
                    "tool_latency",
                    0
                ) + span.latency
            )


        elif isinstance(event,ToolErrorEvent):

            metrics.increment(
                "error_count"
            )