"""
多 Agent 协作契约的统一导入入口。

功能：
    集中导出任务计划、任务步骤、步骤结果和最终协作结果，避免调用方依赖
    contracts 目录中的具体文件结构。
"""

from src.agents.collaboration.contracts.schemas import (
    AgentCollaborationStatus,
    AgentTaskPlan,
    AgentTaskPlanStatus,
    AgentTaskResult,
    AgentTaskResultStatus,
    AgentTaskStep,
    AgentTaskStepStatus,
    MultiAgentTaskResult,
)

__all__ = [
    "AgentCollaborationStatus",
    "AgentTaskPlan",
    "AgentTaskPlanStatus",
    "AgentTaskResult",
    "AgentTaskResultStatus",
    "AgentTaskStep",
    "AgentTaskStepStatus",
    "MultiAgentTaskResult",
]
