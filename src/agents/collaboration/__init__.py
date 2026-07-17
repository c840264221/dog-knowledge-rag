"""
多 Agent 协作模块。

功能：
    提供复杂任务拆解、步骤执行和结果汇总共同使用的标准数据结构，
    并导出负责生成任务计划的 PlannerAgent。
"""

from src.agents.collaboration.contracts import (
    AgentCollaborationStatus,
    AgentTaskPlan,
    AgentTaskPlanStatus,
    AgentTaskResult,
    AgentTaskResultStatus,
    AgentTaskStep,
    AgentTaskStepStatus,
    MultiAgentTaskResult,
)
from src.agents.collaboration.aggregator import (
    ResultAggregationDraft,
    ResultAggregationError,
    ResultAggregator,
)
from src.agents.collaboration.planner import (
    PlannerAgent,
    PlannerGenerationError,
)
from src.agents.collaboration.scheduler import (
    MultiAgentTaskScheduler,
    WorkerHandler,
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
    "ResultAggregationDraft",
    "ResultAggregationError",
    "ResultAggregator",
    "PlannerAgent",
    "PlannerGenerationError",
    "MultiAgentTaskScheduler",
    "WorkerHandler",
]
