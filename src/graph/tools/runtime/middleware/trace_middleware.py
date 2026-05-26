from src.runtime.trace.init import trace_manager

from src.graph.tools.runtime.middleware.base_middleware import BaseMiddleware

from src.runtime.events.event_bus import (
    event_bus
)

from src.runtime.events.event_types import (
    SpanStartEvent,
    SpanEndEvent
)


class TraceMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        # start_event = SpanStartEvent(
        #
        #     trace_id=ctx.trace_id,
        #
        #     span_name=ctx.tool.metadata.name,
        #
        #     parent_span=ctx.current_span if ctx.current_span else None,
        # )

        start_event = SpanStartEvent(ctx)

        await event_bus.emit(start_event)

        # span = trace_manager.create_span(
        #
        #     trace_id=ctx.trace_id,
        #
        #     name=ctx.tool.metadata.name,
        #
        #     parent_span=ctx.current_span
        # )

        # 从 trace_manager 拿 span
        span = trace_manager.span_map.get(
            start_event.span_id
        )

        previous_span = ctx.current_span

        ctx.current_span = span

        try:

            result = await next_func()

            await event_bus.emit(

                SpanEndEvent(
                    ctx=ctx,

                    span_id=start_event.span_id,

                    status="success"
                )
            )

            # span.finish(
            #     status="success"
            # )

            return result

        except Exception as e:

            # span.finish(
            #
            #     status="error",
            #
            #     error=str(e)
            # )
            ctx.error = str(e)

            await event_bus.emit(
                SpanEndEvent(
                    ctx=ctx,
                    span_id=start_event.span_id,
                    status="error",
                )
            )

            raise e
        finally:
            ctx.current_span = previous_span
