from typing import Any

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):

    name: str

    description: str

    timeout: int = 5

    retries: int = 3

    require_confirm: bool = False

    input_schema: dict[str, Any] = Field(
        default_factory=dict
    )
