"""
多 Agent 协作模块。

功能：
    统一导出复杂任务的数据结构、计划生成、依赖调度、结果聚合和总编排
    入口，调用方不需要了解各子目录的内部文件位置。
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
from src.agents.collaboration.orchestrator import (
    MultiAgentOrchestrationError,
    MultiAgentOrchestrator,
)
from src.agents.collaboration.scheduler import (
    MultiAgentTaskScheduler,
    WorkerHandler,
)
from src.agents.collaboration.workers import (
    AgentStateBuilder,
    AgentStateRunner,
    GraphAgentWorkerAdapter,
    build_default_agent_state,
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
    "MultiAgentOrchestrationError",
    "MultiAgentOrchestrator",
    "WorkerHandler",
    "AgentStateBuilder",
    "AgentStateRunner",
    "GraphAgentWorkerAdapter",
    "build_default_agent_state",
]
