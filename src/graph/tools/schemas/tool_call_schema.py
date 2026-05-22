from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List,
    Dict,
    Any
)


class ToolCall(BaseModel):

    name: str = Field(
        description="工具名称"
    )

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数"
    )


class ToolParseResult(BaseModel):

    need_tool: bool = Field(
        description="是否需要调用工具"
    )

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="工具调用列表"
    )

    response: str = Field(
        default="",
        description="无需工具时直接回复用户"
    )