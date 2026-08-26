"""协调长任务业务动作与快照存储的应用服务。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskQueueMessage,
)
from src.runtime.long_tasks.interaction_service import (
    InvalidLongTaskInteractionError,
    LongTaskApprovalAction,
    LongTaskInteractionService,
    LongTaskMissingInputAction,
)
from src.runtime.long_tasks.store import (
    LongTaskNotFoundError,
    LongTaskStore,
)


class LongTaskQueuePublisher(Protocol):
    """定义恢复长任务时发布轻量队列通知所需的最小能力。"""

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        发布一条长任务通知并返回运输层消息编号。

        参数含义：
            message:
                包含任务版本和 Ready Step 提示的轻量通知。

        返回值含义：
            str:
                Redis Stream 或测试替身生成的消息编号。
        """

        ...


class LongTaskApplicationService:
    """
    根据 task_id 加载任务、执行领域动作并以乐观锁保存结果。

    功能：
        为统一执行入口提供面向用例的异步 API，隐藏 LongTask 字段联动、
        Store 加载和 expected_version 保存细节。当前编排创建任务、人工批准
        和输入恢复，不负责 Event、Artifact 或 Worker 内部调度。

    参数含义：
        store:
            保存长任务最新快照的 LongTaskStore 实现。
        interaction_service:
            可选人工交互纯业务服务；默认创建无状态
            LongTaskInteractionService，测试可以注入。
        queue_publisher:
            可选轻量队列通知发布器；提交 durable 任务的用户输入时必须提供。

    返回值含义：
        LongTaskApplicationService:
            可由未来 ExecutionGateway 调用的长任务用例服务。
    """

    def __init__(
        self,
        store: LongTaskStore,
        *,
        interaction_service: LongTaskInteractionService | None = None,
        queue_publisher: LongTaskQueuePublisher | None = None,
    ) -> None:
        self._store = store
        self._interaction_service = (
            interaction_service or LongTaskInteractionService()
        )
        self._queue_publisher = queue_publisher

    async def create_task(self, task: LongTask) -> LongTask:
        """
        通过 Store 原子创建一份初始长任务。

        参数含义：
            task:
                Planner 或统一入口构建的版本 1 任务快照。

        返回值含义：
            LongTask:
                Store 创建成功后的任务快照。
        """

        return await self._store.create(task)

    async def handoff_paused_collaboration_task(
        self,
        task: LongTask,
    ) -> LongTask:
        """
        保存预算暂停后转换出的 durable 长任务并发布首次执行通知。

        功能：
            为多智能体请求内执行升级到后台长任务提供内部交接边界。
            本方法先完成全部快照校验，再创建版本 1 任务，最后把当前
            Ready Step 编号作为轻量提示发布到队列。

        参数含义：
            task:
                已由协作结果适配器构建、尚未写入 Store 的 LongTask。

        返回值含义：
            LongTask:
                Store 原子创建成功后的版本 1 durable 任务快照。
        """

        publisher = self._queue_publisher
        if publisher is None:
            raise RuntimeError("长任务提交队列尚未配置")
        ready_step_ids = self._validate_paused_collaboration_handoff(task)
        saved_task = await self._store.create(task)
        source_correlation_id = task.metadata.get(
            "source_collaboration_id"
        )
        await publisher.publish(
            LongTaskQueueMessage(
                task_id=saved_task.task_id,
                task_version=saved_task.version,
                reason="submitted",
                ready_step_ids=ready_step_ids,
                correlation_id=(
                    str(source_correlation_id)
                    if source_correlation_id is not None
                    else None
                ),
            )
        )
        return saved_task

    async def wait_for_approval(
        self,
        *,
        task_id: str,
        source_step_ids: Sequence[str],
        target_step_ids: Sequence[str],
        prompt: str,
    ) -> LongTask:
        """
        加载指定任务，建立人工批准边界并原子保存新版本。

        参数含义：
            task_id:
                准备暂停的长任务唯一编号。
            source_step_ids:
                已完成且导致本次批准请求产生的步骤编号。
            target_step_ids:
                用户批准后准备激活的 Ready 步骤编号。
            prompt:
                展示给用户的批准问题。

        返回值含义：
            LongTask:
                已持久化为 awaiting_input 的最新任务快照。
        """

        current_task = await self._require_task(task_id)
        awaiting_task = self._interaction_service.wait_for_approval(
            current_task,
            source_step_ids=source_step_ids,
            target_step_ids=target_step_ids,
            prompt=prompt,
        )
        return await self._store.save(
            awaiting_task,
            expected_version=current_task.version,
        )

    async def resume_after_approval(
        self,
        *,
        task_id: str,
        interaction_id: str,
        action: LongTaskApprovalAction,
    ) -> LongTask:
        """
        加载等待任务，消费指定交互并原子保存继续或取消结果。

        参数含义：
            task_id:
                需要恢复或取消的长任务唯一编号。
            interaction_id:
                用户当前响应所对应的等待交互编号。
            action:
                ``continue`` 表示继续目标步骤，``cancel`` 表示取消任务。

        返回值含义：
            LongTask:
                已持久化为 running 或 cancelled 的最新任务快照。
        """

        current_task = await self._require_task(task_id)
        resumed_task = self._interaction_service.resume_after_approval(
            current_task,
            interaction_id=interaction_id,
            action=action,
        )
        return await self._store.save(
            resumed_task,
            expected_version=current_task.version,
        )

    async def respond_to_missing_input(
        self,
        *,
        task_id: str,
        user_id: str,
        interaction_id: str,
        action: LongTaskMissingInputAction,
        answers: Mapping[str, Any],
    ) -> LongTask:
        """
        校验任务归属，消费缺少输入交互，保存后发布后台恢复通知。

        参数含义：
            task_id:
                需要恢复或取消的持久化长任务编号。
            user_id:
                当前调用方声明的用户编号，用于任务归属校验。
            interaction_id:
                前端正在响应的等待交互编号。
            action:
                submit_input 表示保存回答并恢复，cancel 表示取消任务。
            answers:
                以等待 Step 编号为键的用户回答映射；取消时应为空。

        返回值含义：
            LongTask:
                已通过乐观锁保存的 running 或 cancelled 最新任务快照。
        """

        current_task = await self._require_task(task_id)
        if current_task.user_id != user_id:
            raise LongTaskNotFoundError("没有找到对应的长任务")
        if action == "submit_input":
            if current_task.execution_mode != "durable":
                raise InvalidLongTaskInteractionError(
                    "当前 MVP 只恢复 durable 长任务"
                )
            if self._queue_publisher is None:
                raise RuntimeError("长任务恢复队列尚未配置")

        interaction = current_task.pending_interaction
        target_step_ids = (
            list(interaction.target_step_ids)
            if interaction is not None
            else []
        )
        resumed_task = (
            self._interaction_service.respond_to_missing_input(
                current_task,
                interaction_id=interaction_id,
                action=action,
                answers=answers,
            )
        )
        saved_task = await self._store.save(
            resumed_task,
            expected_version=current_task.version,
        )
        if action == "submit_input":
            publisher = self._queue_publisher
            if publisher is None:
                raise RuntimeError("长任务恢复队列尚未配置")
            await publisher.publish(
                LongTaskQueueMessage(
                    task_id=saved_task.task_id,
                    task_version=saved_task.version,
                    reason="resumed",
                    ready_step_ids=target_step_ids,
                    correlation_id=interaction_id,
                )
            )
        return saved_task

    async def _require_task(self, task_id: str) -> LongTask:
        """
        加载必须存在的任务，并把空结果转换成明确业务异常。

        参数含义：
            task_id:
                需要加载的长任务唯一编号。

        返回值含义：
            LongTask:
                Store 中最新且已通过契约校验的任务快照。
        """

        task = await self._store.load(task_id)
        if task is None:
            raise LongTaskNotFoundError(f"长任务不存在: {task_id}")
        return task

    @staticmethod
    def _validate_paused_collaboration_handoff(
        task: LongTask,
    ) -> list[str]:
        """
        校验任务处于当前 MVP 可以首次持久化并入队的安全边界。

        参数含义：
            task:
                准备从请求内多智能体执行交接到后台 Worker 的快照。

        返回值含义：
            list[str]:
                按任务步骤顺序提取的 Ready Step 编号。
        """

        if task.execution_mode != "durable":
            raise ValueError("协作交接任务必须使用 durable 执行模式")
        if task.status != "running":
            raise ValueError("协作交接任务必须处于 running 状态")
        if task.version != 1:
            raise ValueError("协作交接只能创建版本 1 的首次快照")
        if (
            task.metadata.get("source_type")
            != "collaboration_budget_handoff"
        ):
            raise ValueError("任务不是预算暂停产生的协作交接快照")
        if task.pending_interaction is not None:
            raise ValueError("协作交接任务不能携带待处理人工交互")
        if task.active_step_ids:
            raise ValueError("协作交接前不能存在活动执行批次")

        claimed_step_ids = [
            step.step_id
            for step in task.steps
            if step.claim_id is not None
        ]
        if claimed_step_ids:
            raise ValueError(
                "协作交接前 Step 不能持有 Worker 租约: "
                f"{claimed_step_ids}"
            )
        invalid_migrated_step_ids = [
            step.step_id
            for step in task.steps
            if step.metadata.get("migrated_from_inline_result") is True
            and step.status not in {"completed", "skipped"}
        ]
        if invalid_migrated_step_ids:
            raise ValueError(
                "请求内历史 Step 必须已经完成或跳过: "
                f"{invalid_migrated_step_ids}"
            )

        ready_step_ids = [
            step.step_id
            for step in task.steps
            if step.status == "ready"
        ]
        if not ready_step_ids:
            raise ValueError("协作交接任务至少需要一个 Ready Step")
        return ready_step_ids
