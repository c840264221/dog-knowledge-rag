from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryParseResult(BaseModel):
    intent: str = Field(description="查询意图")
    filters: Dict = Field(description="结构过滤条件",default_factory=dict)
    tags: List[str] = Field(description="语义标签",default_factory=list)
    features: List[str] = Field(description="结构特征",default_factory=list)
    dog_name: Optional[str] = Field(default=None, description="狗品种")