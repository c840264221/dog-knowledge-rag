"""
多 Agent 总编排器统一导入入口。

功能：
    导出完整多 Agent 流程入口和带失败阶段的统一异常。
"""

from src.agents.collaboration.orchestrator.orchestrator import (
    MultiAgentOrchestrationError,
    MultiAgentOrchestrator,
)

__all__ = [
    "MultiAgentOrchestrationError",
    "MultiAgentOrchestrator",
]
