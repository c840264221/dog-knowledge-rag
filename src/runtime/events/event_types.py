from src.runtime.events.base_event import (
    BaseEvent
)

import uuid


class ToolStartEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.tool_name = ctx.tool.metadata.name


class ToolSuccessEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.tool_name = ctx.tool.metadata.name

        self.span_id = ctx.span_id


class ToolErrorEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.tool_name = ctx.tool.metadata.name

        self.error = ctx.error

        self.span_id = ctx.span_id

class SpanStartEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.span_name = ctx.tool.metadata.name

        self.parent_span = ctx.current_span

        # Listener回填
        self.span = None


class SpanEndEvent(BaseEvent):

    def __init__(self,ctx,span_id,status="success"):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.span_id = span_id

        self.status = status

        self.error = ctx.error