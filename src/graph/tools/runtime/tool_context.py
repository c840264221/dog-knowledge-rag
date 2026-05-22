import time
import uuid
from typing import Any
from typing import Optional, Dict


class ToolContext:

    def __init__(self,tool,args):
        # 工具对象
        self.tool = tool

        # 工具参数
        self.args: Dict[str, Any] = {}

        # 开始时间
        self.start_time: float = time.time()

        # trace id
        self.trace_id: str = str(
            uuid.uuid4()
        )

        # 重试次数
        self.retry_count: int = 0

        # 工具结果
        self.result: Optional[Any] = None

        # 错误信息
        self.error: Optional[str] = None

        # 耗时
        self.latency: Optional[float] = None