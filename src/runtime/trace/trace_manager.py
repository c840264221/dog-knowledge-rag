import uuid

from src.runtime.trace.trace_span import (
    TraceSpan
)


class TraceManager:

    def __init__(self):

        self.traces = {}

        self.span_map = {}

    def create_trace(self, trace_id:str):


        # root span id
        root_span_id = str(uuid.uuid4())

        root_span = TraceSpan(
            span_id=root_span_id,

            name="root",

            trace_id=trace_id,

            parent_span_id=None
        )

        self.traces[trace_id] = root_span

        self.span_map[root_span_id] = root_span

        return root_span

    def create_span(self,span_id,trace_id,name,parent_span=None):

        span = TraceSpan(
            span_id=span_id,
            name=name,
            trace_id=trace_id,
            parent_span_id=(
                parent_span.span_id
                if parent_span
                else None
            )
        )

        if parent_span:

            parent_span.add_child(span)
        self.span_map[span.span_id] = span

        return span