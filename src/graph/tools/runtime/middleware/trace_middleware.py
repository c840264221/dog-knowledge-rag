from src.graph.tools.runtime.middleware.base_middleware import BaseMiddleware

from src.runtime.events.event_bus import (
    event_bus
)

from src.runtime.events.event_types import (
    SpanStartEvent,
    SpanEndEvent
)

from src.settings import settings

from src.runtime.context import runtime_ctx

from src.logger import logger


class TraceMiddleware(BaseMiddleware):

    async def process(self,ctx,next_func):

        # 通过配置来控制是否进行trace
        if not settings.runtime.enable_trace:
            return await next_func()

        # start_event = SpanStartEvent(
        #
        #     trace_id=ctx.trace_id,
        #
        #     span_name=ctx.tool.metadata.name,
        #
        #     parent_span=ctx.current_span if ctx.current_span else None,
        # )

        # span = trace_manager.create_span(
        #
        #     trace_id=ctx.trace_id,
        #
        #     name=ctx.tool.metadata.name,
        #
        #     parent_span=ctx.current_span
        # )

        # 从 trace_manager 拿 span
        # span = trace_manager.span_map.get(
        #     start_event.span_id
        # )


        runtime_context = runtime_ctx.get()

        if runtime_context is None:
            logger.warning(
                "TraceMiddleware 未获取到 RuntimeContext，"
                "本次工具调用跳过 trace。"
            )

            return await next_func()

        previous_span = runtime_context.current_span


        start_event = SpanStartEvent(ctx)

        await event_bus.emit(start_event)

        span = start_event.span

        if span is None:
            logger.warning(
                "SpanStartEvent 未被 listener 回填 span，"
                "本次工具调用跳过 trace。"
            )

            return await next_func()

        runtime_context.current_span = span

        ctx.current_span = span

        ctx.span_id = span.span_id

        try:

            result = await next_func()

            await event_bus.emit(

                SpanEndEvent(
                    ctx=ctx,

                    span_id=span.span_id,

                    status="success"
                )
            )


            return result

        except Exception as e:
            ctx.error = str(e)

            await event_bus.emit(
                SpanEndEvent(
                    ctx=ctx,
                    span_id=span.span_id,
                    status="error",
                )
            )

            raise e

        finally:
            runtime_context.current_span = previous_span
