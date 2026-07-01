"""
Deprecated legacy query parsing schemas.

兼容性说明：
    该模块属于旧版 query_parse（查询解析）链路，仅为历史代码兼容保留。
    V1.7.1 之后，新的 RootAgent 主路由链路不应该再使用这里的结果模型。

新代码要求：
    RootAgent 路由结果请使用 src.agents.root_agent.schemas.RootRouteDecision。
    RAG 查询结构请使用新的 RAG schema / builder 路径。
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class Intent(str, Enum):

    # CHAT = "chat"

    RECOMMEND = "recommend"

    ASK_INFO = "ask_info"

    GENERAL = "general"

    # TOOL_CALL = "tool_call"

class QueryParseResult(BaseModel):
    intent: str = Field(default_factory=Intent.GENERAL.value)

    filters: dict = Field(default_factory=dict)

    tags: list[str] = Field(default_factory=list)

    features: list[str] = Field(default_factory=list)

    dog_name: str | None = None
