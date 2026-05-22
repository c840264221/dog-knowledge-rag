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