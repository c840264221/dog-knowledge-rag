from src.runtime.events.event_bus import (
    event_bus
)

from src.runtime.events.event_types import (

    ToolStartEvent,

    ToolSuccessEvent,

    ToolErrorEvent,

    SpanStartEvent,

    SpanEndEvent
)

from src.runtime.events.listeners.logging_listener import (
    LoggingListener
)

from src.runtime.events.listeners.trace_listener import (
    TraceListener
)

from src.runtime.events.listeners.metrics_listener import (
    MetricsListener
)


logging_listener = LoggingListener()

tracing_listener = TraceListener()

metrics_listener = MetricsListener()

event_bus.subscribe(
    ToolStartEvent,
    logging_listener
)

event_bus.subscribe(
    ToolSuccessEvent,
    logging_listener
)

event_bus.subscribe(
    ToolErrorEvent,
    logging_listener
)

event_bus.subscribe(
    SpanStartEvent,
    tracing_listener
)

event_bus.subscribe(
    SpanEndEvent,
    tracing_listener
)

event_bus.subscribe(
    ToolSuccessEvent,
    metrics_listener
)

event_bus.subscribe(
    ToolErrorEvent,
    metrics_listener
)