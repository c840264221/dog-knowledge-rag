from pydantic import BaseModel, Field
from typing import Literal
from dataclasses import dataclass


from datetime import datetime

class MemoryOutput(BaseModel):
    """
    用于定义 LLM 输出的内容。

    作用：
    - 校验 LLM 判断出来的记忆是否值得保存
    - 约束 LLM 输出字段格式
    """

    should_save: bool

    confidence: float = Field(
        ge=0,
        le=1
    )

    memory_type: Literal[
        "favorite_dog",
        "preference",
        "dislike"
    ]

    content: str

    reason: str


@dataclass
class MemoryRecord:
    """
    MemoryRecord：数据库中的记忆实体。

    作用：
    - 表示 SQLite user_memory 表中的一条记录
    - 后续也可以作为同步到 Chroma 的数据来源
    """

    id: int | None

    user_id: str

    memory_type: str

    content: str

    confidence: float

    strength: float

    status: str = "active"

    created_at: str | None = None

    last_seen: str | None = None