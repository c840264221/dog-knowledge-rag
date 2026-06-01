import time
import uuid
from typing import Optional

class TraceSpan:

    def __init__(self,name: str,trace_id: str,parent_span=None):

        # 当前步骤ID
        self.span_id = str(
            uuid.uuid4()
        )

        # 整个请求的唯一ID
        self.trace_id = trace_id

        self.parent_span = parent_span

        # 父节点ID  形成树形结构
        self.parent_span_id = (
            parent_span.span_id
            if parent_span
            else None
        )

        self.name = name

        # 子节点
        self.children = []

        self.status = "running"

        self.error: Optional[str] = None

        self.start_time = time.time()

        self.end_time = None

        self.latency = None


    def finish(self, status:str="success", error: Optional[str] = None):

        self.end_time = time.time()

        self.latency = round(
            self.end_time - self.start_time,
            3
        )

        self.status = status

        self.error = error

    def add_child(self, span):

        self.children.append(span)

    def to_dict(self):
        return {

            "trace_id": self.trace_id,

            "span_id": self.span_id,

            "parent_span_id": self.parent_span_id,

            "name": self.name,

            "status": self.status,

            "error": self.error,

            "latency": self.latency,

            "children": [
                child.to_dict()
                for child in self.children
            ]
        }