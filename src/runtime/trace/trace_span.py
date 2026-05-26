import time
import uuid
from typing import Optional

class TraceSpan:

    def __init__(self,span_id:str,name: str,trace_id: str,parent_span_id: str = None):

        # 当前步骤ID
        # self.span_id = str(
        #     uuid.uuid4()
        # )
        self.span_id = span_id

        # 整个请求的唯一ID
        self.trace_id = trace_id

        # 父节点ID  形成树形结构
        self.parent_span_id = parent_span_id

        self.name = name

        self.start_time = time.time()

        self.end_time = None

        self.latency = None

        self.status = "running"

        self.error: Optional[str] = None

        self.children = []

        # 子节点
        self.children = []

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