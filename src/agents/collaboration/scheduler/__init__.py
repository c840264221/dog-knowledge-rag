"""
多 Agent 任务调度器统一导入入口。

功能：
    导出依赖驱动调度器和 Worker 函数类型，调用方无需依赖内部文件位置。
"""

from src.agents.collaboration.scheduler.scheduler import (
    MultiAgentTaskCancellationRegistry,
    MultiAgentTaskCancellationToken,
    MultiAgentTaskScheduler,
    WorkerHandler,
    build_multi_agent_task_id,
)

__all__ = [
    "MultiAgentTaskCancellationRegistry",
    "MultiAgentTaskCancellationToken",
    "MultiAgentTaskScheduler",
    "WorkerHandler",
    "build_multi_agent_task_id",
]
