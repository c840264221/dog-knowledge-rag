import uuid

from src.runtime.trace.trace_span import (
    TraceSpan
)

from src.runtime.trace.trace import (
    Trace
)

class TraceManager:

    def __init__(self):

        self.trace_map = {}

        self.span_map = {}

    def create_trace(self, trace_id:str):
        #
        #
        # # root span id
        # root_span_id = str(uuid.uuid4())
        #
        # root_span = TraceSpan(
        #     span_id=root_span_id,
        #
        #     name="root",
        #
        #     trace_id=trace_id,
        #
        #     parent_span=None
        # )
        #
        # self.traces[trace_id] = root_span
        #
        # self.span_map[root_span_id] = root_span
        #
        # return root_span

        trace = Trace(trace_id)

        self.trace_map[trace_id] = trace

        return trace

    def create_span(self,trace_id,name,parent_span=None):

        span = TraceSpan(
            name=name,
            trace_id=trace_id,
            parent_span=(
                parent_span
                if parent_span
                else None
            )
        )

        self.span_map[span.span_id] = span

        trace = self.trace_map[trace_id]

        # root span
        if parent_span is None:

            trace.add_root_span(span)

        # child span
        else:

            parent_span.add_child(span)

        return span

    def finish_span(self,span_id,status="success",error=None):

        span = self.span_map.get(span_id)

        if span:
            span.finish(
                status=status,
                error=error
            )

    def get_trace(self, trace_id):

        return self.trace_map.get(trace_id)