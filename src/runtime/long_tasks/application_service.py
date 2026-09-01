"""协调长任务业务动作与快照存储的应用服务。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskEventActorType,
    LongTaskQueueMessage,
)
from src.runtime.long_tasks.commit_service import (
    build_commit_request,
    build_long_task_fingerprint_view,
    calculate_request_fingerprint,
    supports_reliable_commit,
)
from src.runtime.long_tasks.interaction_service import (
    InvalidLongTaskInteractionError,
    LongTaskApprovalAction,
    LongTaskInteractionService,
    LongTaskMissingInputAction,
)
from src.runtime.long_tasks.store import (
    LongTaskCommitFingerprintConflictError,
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
        Store 加载和 expected_version 保存细节。可靠 Store 会把业务事件、
        回执和可选队列消息与快照一起提交；本服务不负责 Artifact 或
        Worker 内部调度。

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

        fingerprint_payload = {
            "action": "create_task",
            "task": build_long_task_fingerprint_view(task),
        }
        return await self._create_business_task(
            task=task,
            commit_id=f"create:{task.task_id}",
            fingerprint_payload=fingerprint_payload,
            actor_type="system",
            actor_id="long_task_application_service",
            correlation_id=task.thread_id,
        )

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
        source_correlation_id = task.metadata.get(
            "source_collaboration_id"
        )
        correlation_id = (
            str(source_correlation_id)
            if source_correlation_id is not None
            else None
        )
        queue_message = LongTaskQueueMessage(
            task_id=task.task_id,
            task_version=task.version,
            reason="submitted",
            ready_step_ids=ready_step_ids,
            correlation_id=correlation_id,
        )
        fingerprint_payload = {
            "action": "handoff_paused_collaboration_task",
            "task_id": task.task_id,
            "source_collaboration_id": correlation_id,
            "ready_step_ids": ready_step_ids,
        }
        saved_task = await self._create_business_task(
            task=task,
            commit_id=f"handoff:{task.task_id}",
            fingerprint_payload=fingerprint_payload,
            actor_type="system",
            actor_id="collaboration_handoff",
            correlation_id=correlation_id,
            queue_message=queue_message,
        )
        if not supports_reliable_commit(self._store):
            await publisher.publish(queue_message)
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

        fingerprint_payload = {
            "action": "wait_for_approval",
            "task_id": task_id,
            "source_step_ids": list(source_step_ids),
            "target_step_ids": list(target_step_ids),
            "prompt": prompt,
        }
        commit_id = _derived_commit_id(
            "wait-approval",
            fingerprint_payload,
        )
        committed_task = await self._load_idempotent_task_if_committed(
            task_id=task_id,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
        )
        if committed_task is not None:
            return committed_task
        current_task = await self._require_task(task_id)
        awaiting_task = self._interaction_service.wait_for_approval(
            current_task,
            source_step_ids=source_step_ids,
            target_step_ids=target_step_ids,
            prompt=prompt,
        )
        return await self._update_business_task(
            previous_task=current_task,
            next_task=awaiting_task,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
            actor_type="system",
            actor_id="long_task_application_service",
            correlation_id=commit_id,
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

        fingerprint_payload = {
            "action": "resume_after_approval",
            "task_id": task_id,
            "interaction_id": interaction_id,
            "approval_action": action,
        }
        commit_id = f"approval:{interaction_id}:{action}"
        committed_task = await self._load_idempotent_task_if_committed(
            task_id=task_id,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
        )
        if committed_task is not None:
            return committed_task
        current_task = await self._require_task(task_id)
        resumed_task = self._interaction_service.resume_after_approval(
            current_task,
            interaction_id=interaction_id,
            action=action,
        )
        return await self._update_business_task(
            previous_task=current_task,
            next_task=resumed_task,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
            actor_type="user",
            actor_id=current_task.user_id,
            correlation_id=interaction_id,
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

        fingerprint_payload = {
            "action": "respond_to_missing_input",
            "task_id": task_id,
            "user_id": user_id,
            "interaction_id": interaction_id,
            "input_action": action,
            "answers": dict(answers),
        }
        commit_id = f"interaction:{interaction_id}:{action}"
        current_task = await self._require_task(task_id)
        if current_task.user_id != user_id:
            raise LongTaskNotFoundError("没有找到对应的长任务")
        committed_task = await self._load_idempotent_task_if_committed(
            task_id=task_id,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
        )
        if committed_task is not None:
            return committed_task
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
        queue_message = None
        if action == "submit_input":
            queue_message = LongTaskQueueMessage(
                task_id=resumed_task.task_id,
                task_version=resumed_task.version,
                reason="resumed",
                ready_step_ids=target_step_ids,
                correlation_id=interaction_id,
            )
        saved_task = await self._update_business_task(
            previous_task=current_task,
            next_task=resumed_task,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
            actor_type="user",
            actor_id=user_id,
            correlation_id=interaction_id,
            queue_message=queue_message,
        )
        if queue_message is not None and not supports_reliable_commit(
            self._store
        ):
            publisher = self._queue_publisher
            if publisher is None:
                raise RuntimeError("长任务恢复队列尚未配置")
            await publisher.publish(queue_message)
        return saved_task

    async def _create_business_task(
        self,
        *,
        task: LongTask,
        commit_id: str,
        fingerprint_payload: Mapping[str, Any],
        actor_type: LongTaskEventActorType,
        actor_id: str | None,
        correlation_id: str | None,
        queue_message: LongTaskQueueMessage | None = None,
    ) -> LongTask:
        """
        使用可靠提交创建业务任务，并为旧 Store 保留 create 兼容路径。

        参数含义：
            task：准备创建的版本 1 快照。
            commit_id/fingerprint_payload：稳定幂等身份与关键业务输入。
            actor_type/actor_id：直接触发创建的可信主体。
            correlation_id：创建链路编号。
            queue_message：需要与快照一起写入的可选通知。

        返回值含义：
            LongTask：已经成为权威状态的版本 1 任务。
        """

        if supports_reliable_commit(self._store):
            request = build_commit_request(
                previous_task=None,
                next_task=task,
                commit_id=commit_id,
                fingerprint_payload=fingerprint_payload,
                actor_type=actor_type,
                actor_id=actor_id,
                correlation_id=correlation_id,
                queue_message=queue_message,
            )
            await self._store.commit(request)  # type: ignore[attr-defined]
            authoritative_task = await self._store.load(task.task_id)
            if authoritative_task is None:
                raise LongTaskNotFoundError(
                    "提交回执已经产生但权威任务不存在: "
                    f"{task.task_id}"
                )
            return authoritative_task
        return await self._store.create(task)

    async def _update_business_task(
        self,
        *,
        previous_task: LongTask,
        next_task: LongTask,
        commit_id: str,
        fingerprint_payload: Mapping[str, Any],
        actor_type: LongTaskEventActorType,
        actor_id: str | None,
        correlation_id: str | None,
        queue_message: LongTaskQueueMessage | None = None,
    ) -> LongTask:
        """
        使用可靠提交更新业务任务，并为旧 Store 保留 save 兼容路径。

        参数含义：
            previous_task/next_task：业务变化前后的任务快照。
            commit_id/fingerprint_payload：稳定幂等身份与关键业务输入。
            actor_type/actor_id：直接触发更新的可信主体。
            correlation_id：本次更新链路编号。
            queue_message：需要与快照一起写入的可选通知。

        返回值含义：
            LongTask：已经持久化的新版本权威任务。
        """

        if supports_reliable_commit(self._store):
            request = build_commit_request(
                previous_task=previous_task,
                next_task=next_task,
                commit_id=commit_id,
                fingerprint_payload=fingerprint_payload,
                actor_type=actor_type,
                actor_id=actor_id,
                correlation_id=correlation_id,
                queue_message=queue_message,
            )
            await self._store.commit(request)  # type: ignore[attr-defined]
            authoritative_task = await self._store.load(next_task.task_id)
            if authoritative_task is None:
                raise LongTaskNotFoundError(
                    "提交回执已经产生但权威任务不存在: "
                    f"{next_task.task_id}"
                )
            return authoritative_task
        return await self._store.save(
            next_task,
            expected_version=previous_task.version,
        )

    async def _load_idempotent_task_if_committed(
        self,
        *,
        task_id: str,
        commit_id: str,
        fingerprint_payload: Mapping[str, Any],
    ) -> LongTask | None:
        """
        在重新执行领域迁移前查询同一逻辑提交是否已经成功。

        参数含义：
            task_id/commit_id：准备重试的任务和稳定提交编号。
            fingerprint_payload：本次重试的规范化关键业务输入。

        返回值含义：
            LongTask | None：已提交时返回最新快照，否则返回 None。
        """

        load_receipt = getattr(self._store, "load_receipt", None)
        if not callable(load_receipt):
            return None
        receipt = await load_receipt(
            task_id=task_id,
            commit_id=commit_id,
        )
        if receipt is None:
            return None
        fingerprint = calculate_request_fingerprint(
            fingerprint_payload
        )
        if receipt.request_fingerprint != fingerprint:
            raise LongTaskCommitFingerprintConflictError(
                "commit_id 已用于不同业务动作: "
                f"task_id={task_id}, commit_id={commit_id}"
            )
        task = await self._store.load(task_id)
        if task is None:
            raise LongTaskNotFoundError(
                f"回执存在但长任务不存在: {task_id}"
            )
        return task

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


def _derived_commit_id(
    prefix: str,
    fingerprint_payload: Mapping[str, Any],
) -> str:
    """
    根据稳定业务输入派生不包含随机时间的提交编号。

    参数含义：
        prefix：用于区分业务动作的可读前缀。
        fingerprint_payload：能够唯一描述该逻辑动作的关键输入。

    返回值含义：
        str：可在网络重试时重新生成的稳定 commit_id。
    """

    fingerprint = calculate_request_fingerprint(
        fingerprint_payload
    )
    return f"{prefix}:{fingerprint.removeprefix('sha256:')[:24]}"
