from src.settings.base import BaseAppSettings


class TraceSettings(BaseAppSettings):

    # enable_trace: bool = True

    # enable_span: bool = True
    #
    # enable_event_bus: bool = True
    #
    # enable_langsmith: bool = False

    max_span_depth: int = 10

    export_trace: bool = False

    trace_exporter: str = "console"

    trace_batch_size: int = 100