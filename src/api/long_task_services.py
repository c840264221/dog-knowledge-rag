"""持久化长任务的 API 查询服务。"""

from __future__ import annotations

from src.api.schemas import (
    LongTaskInteractionResponse,
    LongTaskStatusResponse,
    LongTaskStepStatusResponse,
)
from src.runtime.long_tasks.contracts import LongTask
from src.runtime.long_tasks.store import LongTaskStore


class LongTaskApiQueryService:
    """
    按任务归属读取 LongTask，并转换为前端安全的公开响应。

    参数含义：
        store:
            提供最新权威 LongTask 快照的持久化存储。

    返回值含义：
        LongTaskApiQueryService:
            可以执行只读长任务状态查询的 API 应用服务。
    """

    def __init__(self, store: LongTaskStore) -> None:
        self._store = store

    async def get_status(
        self,
        *,
        task_id: str,
        user_id: str,
    ) -> LongTaskStatusResponse | None:
        """
        查询属于指定用户的长任务，并过滤内部运行字段。

        参数含义：
            task_id:
                需要查询的长任务唯一编号。
            user_id:
                当前调用方声明的用户编号，用于任务归属校验。

        返回值含义：
            LongTaskStatusResponse | None:
                找到且任务属于当前用户时返回公开状态；任务不存在或不属于
                当前用户时统一返回 None，避免泄露其他用户任务是否存在。
        """

        task = await self._store.load(task_id)
        if task is None or task.user_id != user_id:
            return None
        return build_long_task_status_response(task)


def build_long_task_status_response(
    task: LongTask,
) -> LongTaskStatusResponse:
    """
    把内部 LongTask 快照转换成公开 API 状态契约。

    参数含义：
        task:
            从 Store 加载并通过契约校验的权威任务快照。

    返回值含义：
        LongTaskStatusResponse:
            已移除输入原文、Worker 领取信息和内部 metadata 的响应对象。
    """

    interaction = task.pending_interaction
    return LongTaskStatusResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        objective=task.goal.objective,
        status=task.status,
        execution_mode=task.execution_mode,
        progression_mode=task.progression_mode,
        version=task.version,
        total_step_count=len(task.steps),
        completed_step_count=sum(
            step.status in {"completed", "skipped"}
            for step in task.steps
        ),
        steps=[
            LongTaskStepStatusResponse(
                step_id=step.step_id,
                title=step.title,
                status=step.status,
                depends_on=list(step.depends_on),
                waiting_reason=step.waiting_reason,
                output_summary=step.output_summary,
                attempt_count=step.attempt_count,
            )
            for step in task.steps
        ],
        pending_interaction=(
            LongTaskInteractionResponse(
                interaction_id=interaction.interaction_id,
                interaction_type=interaction.interaction_type,
                prompt=interaction.prompt,
                allowed_actions=list(interaction.allowed_actions),
                input_contract=dict(interaction.input_contract),
            )
            if interaction is not None
            else None
        ),
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )
