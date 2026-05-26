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


class ToolErrorEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.tool_name = ctx.tool.metadata.name

        self.error = ctx.error

class SpanStartEvent(BaseEvent):

    def __init__(self,ctx):

        super().__init__()

        self.ctx = ctx

        self.span_id = str(uuid.uuid4())

        self.trace_id = ctx.trace_id

        self.span_name = ctx.tool.metadata.name

        self.parent_span = ctx.current_span


class SpanEndEvent(BaseEvent):

    def __init__(self,ctx,span_id,status="success"):

        super().__init__()

        self.ctx = ctx

        self.trace_id = ctx.trace_id

        self.span_id = span_id

        self.status = status

        self.error = ctx.error