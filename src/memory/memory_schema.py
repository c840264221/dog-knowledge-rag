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

MemoryRecallStatus: TypeAlias = Literal[
    "applied",
    "empty",
    "failed",
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


class MemoryRecallResult(BaseModel):
    """
    Memory Recall Result（记忆召回结果）数据契约。

    功能：
        记录记忆召回是否成功应用、候选数量、语义门槛和最终采用的记忆。
        该对象用于服务内部校验，写入 LangGraph state 前需转换为普通 dict。

    参数：
        status：召回状态，applied 表示已采用，empty 表示无可用记忆，failed 表示召回异常。
        memory_context：可直接注入答案 Prompt（提示词）的记忆文本。
        candidate_count：Chroma 初步语义检索返回的候选数量。
        threshold_passed_count：通过最低语义相关性门槛的候选数量。
        selected_count：去重、排序后最终采用的记忆数量。
        semantic_threshold：本次召回使用的最低语义相关分。
        max_semantic_score：最终采用记忆中的最高语义相关分。
        selected_memory_ids：最终采用的 SQLite 记忆 ID 列表。
        reason：对本次召回结果的中文说明。

    返回值：
        MemoryRecallResult：经 Pydantic（数据校验库）验证的记忆召回结果。
    """

    status: MemoryRecallStatus
    memory_context: str = "暂无用户记忆"
    candidate_count: int = Field(default=0, ge=0)
    threshold_passed_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    semantic_threshold: float = Field(default=0.0, ge=0, le=1)
    max_semantic_score: float | None = Field(default=None, ge=0, le=1)
    selected_memory_ids: list[int] = Field(default_factory=list)
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
