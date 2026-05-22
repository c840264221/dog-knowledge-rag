from pydantic import BaseModel, Field
from typing import Literal


class MemoryOutput(BaseModel):

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