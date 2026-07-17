"""
多 Agent 结果聚合器统一导入入口。

功能：
    导出结果聚合服务、统一异常和 LLM 聚合草稿结构。
"""

from src.agents.collaboration.aggregator.agent import (
    ResultAggregationError,
    ResultAggregator,
)
from src.agents.collaboration.aggregator.schemas import (
    ResultAggregationDraft,
)

__all__ = [
    "ResultAggregationDraft",
    "ResultAggregationError",
    "ResultAggregator",
]
