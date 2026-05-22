from pydantic import BaseModel


class ToolMetadata(BaseModel):

    name: str

    description: str

    timeout: int = 5

    retries: int = 3

    require_confirm: bool = False