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

    def ensure_trace(self, trace_id: str):
        """
        确保指定 trace_id 对应的 Trace 存在。

        功能：
            如果 trace_map 中已经存在 trace_id，则直接返回已有 Trace。
            如果不存在，则创建一个新的 Trace 并注册到 trace_map。
            主要用于 checkpoint resume（检查点恢复）场景，避免内存中的
            trace 数据丢失后 create_span 直接失败。

        参数：
            trace_id：
                当前请求链路追踪 ID。

        返回值：
            Trace：
                已存在或新创建的 Trace 对象。
        """

        trace = self.trace_map.get(
            trace_id
        )

        if trace is not None:
            return trace

        return self.create_trace(
            trace_id
        )

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

        trace = self.ensure_trace(
            trace_id
        )

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
