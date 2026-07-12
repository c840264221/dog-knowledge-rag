from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

from pydantic import BaseModel, Field


MemoryType: TypeAlias = Literal[
    "favorite_dog",
    "preference",
    "dislike",
    "hobby",
    "profile",
]

MemoryStatus: TypeAlias = Literal[
    "active",
    "inactive",
]

MemorySource: TypeAlias = Literal[
    "conversation",
    "tool",
    "manual",
    "system",
]

VALID_MEMORY_TYPES = frozenset(
    get_args(MemoryType)
)

VALID_MEMORY_SOURCES = frozenset(
    get_args(MemorySource)
)


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

    memory_type: MemoryType

    content: str

    reason: str

    importance: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )


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

    memory_type: MemoryType

    content: str

    confidence: float

    strength: float

    status: MemoryStatus = "active"

    created_at: str | None = None

    last_seen: str | None = None

    source: MemorySource = "conversation"

    importance: float = 0.5

    updated_at: str | None = None

    expires_at: str | None = None
