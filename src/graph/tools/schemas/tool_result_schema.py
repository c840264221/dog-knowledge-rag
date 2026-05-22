from pydantic import BaseModel
from typing import Any
from typing import Optional
from typing import Dict


class ToolResult(BaseModel):

    # 是否成功
    success: bool

    # 工具名称
    tool_name: str

    # 工具返回内容
    content: Any = None

    # 错误信息
    error: Optional[str] = None

    # 耗时（秒）
    latency: Optional[float] = None

    # 重试次数
    retry_count: int = 0

    # 附加元数据
    metadata: Dict[str, Any] = {}