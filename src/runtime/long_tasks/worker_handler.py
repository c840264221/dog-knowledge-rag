"""Redis Stream 长任务消息到 Step 执行与批次提交的最小业务桥接层。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import uuid4

from src.runtime.long_tasks.claim_service import LongTaskStepClaimService
from src.runtime.long_tasks.commit_service import supports_reliable_commit
from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskExecutionContext,
    LongTaskQueueMessage,
    LongTaskStep,
)
from src.runtime.long_tasks.runtime_driver import LongTaskRuntimeDriver
from src.runtime.long_tasks.store import (
    LongTaskNotFoundError,
    LongTaskStore,
)


LongTaskStepExecutor = Callable[
    [LongTask, LongTaskStep, str],
    Awaitable[LongTaskBatchStepResult],
]
IdFactory = Callable[[], str]


class _LongTaskQueuePublisher(Protocol):
    """定义 Handler 发布后续长任务通知所需的最小接口。"""

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        发布一条轻量长任务通知。

        参数含义：
            message:
                包含最新任务版本和 Ready Step 的后续执行通知。

        返回值含义：
            str：运输层为本次通知生成的消息编号。
        """

        ...


class InvalidLongTaskStepExecutionResultError(ValueError):
    """表示注入执行器返回了错误 Step 或错误领取令牌的结果。"""


def _default_claim_id_factory() -> str:
    """
    生成一次 Step 执行使用的默认领取令牌。

    返回值含义：
        str：带 claim_ 前缀的随机 UUID 字符串。
    """

    return f"claim_{uuid4()}"


def _default_batch_id_factory() -> str:
    """
    生成一次 Worker 执行批次的默认编号。

    返回值含义：
        str：带 batch_ 前缀的随机 UUID 字符串。
    """

    return f"batch_{uuid4()}"


class LongTaskQueueMessageHandler:
    """
    把一条轻量队列通知桥接到 Step 领取、执行和批次结果提交。

    参数含义：
        store:
            用于加载最新权威 LongTask 的 Store。
        step_executor:
            接收 LongTask、已领取 Step 和 claim_id，并异步返回单步结果的
            可注入执行器；真实 Agent/Tool 实现留给后续适配器。
        worker_name:
            当前后台进程在 Stream Consumer Group 中使用的稳定名称。
        lease_duration_ms:
            每个 Step 首次领取时获得的租约有效毫秒数。
        claim_service:
            可选 Step 领取服务；默认与当前 Store 组合创建。
        runtime_driver:
            可选批次结果 Driver；默认与当前 Store 组合创建。
        queue_publisher:
            可选轻量队列发布器；任务产生下一批 Ready Step 时必须配置，
            正式 Worker 运行时复用当前 Redis Stream。
        claim_id_factory:
            可选领取令牌工厂，测试可以注入确定值。
        batch_id_factory:
            可选批次编号工厂，测试可以注入确定值。

    返回值含义：
        LongTaskQueueMessageHandler:
            符合 LongTaskMessageHandler 签名、可注入 Stream Worker 的对象。
    """

    def __init__(
        self,
        *,
        store: LongTaskStore,
        step_executor: LongTaskStepExecutor,
        worker_name: str,
        lease_duration_ms: int = 30_000,
        claim_service: LongTaskStepClaimService | None = None,
        runtime_driver: LongTaskRuntimeDriver | None = None,
        queue_publisher: _LongTaskQueuePublisher | None = None,
        claim_id_factory: IdFactory | None = None,
        batch_id_factory: IdFactory | None = None,
    ) -> None:
        normalized_worker = str(worker_name or "").strip()
        if not normalized_worker:
            raise ValueError("worker_name 不能为空")
        if lease_duration_ms < 1:
            raise ValueError("lease_duration_ms 必须大于等于 1")
        self._store = store
        self._step_executor = step_executor
        self._worker_name = normalized_worker
        self._lease_duration_ms = lease_duration_ms
        self._claim_service = claim_service or LongTaskStepClaimService(store)
        self._runtime_driver = runtime_driver or LongTaskRuntimeDriver(store)
        self._queue_publisher = queue_publisher
        self._claim_id_factory = (
            claim_id_factory or _default_claim_id_factory
        )
        self._batch_id_factory = (
            batch_id_factory or _default_batch_id_factory
        )

    async def __call__(self, message: LongTaskQueueMessage) -> None:
        """
        处理一条队列消息；安全完成或确认消息陈旧时正常返回。

        参数含义：
            message:
                Stream 中恢复出的轻量长任务通知。

        返回值含义：
            None：批次结果已成功保存，或消息已经没有可执行 Step。
            领取、执行或提交失败时异常继续向上抛出，让 Worker 不执行
            ACK，并由 Pending 恢复链路稍后重新处理。
        """

        task = await self._store.load(message.task_id)
        if task is None:
            raise LongTaskNotFoundError(
                f"队列消息引用的长任务不存在: {message.task_id}"
            )
        if task.status != "running" or task.execution_mode != "durable":
            return

        candidate_step_ids = _select_candidate_step_ids(
            task=task,
            hinted_step_ids=message.ready_step_ids,
        )
        if not candidate_step_ids:
            await self._publish_continuation_if_ready(
                task=task,
                correlation_id=message.correlation_id,
            )
            return

        claim_id_by_step_id: dict[str, str] = {}
        claimed_task = task
        for step_id in candidate_step_ids:
            claim_id = self._claim_id_factory()
            claimed_task = await self._claim_service.claim_step(
                task_id=task.task_id,
                step_id=step_id,
                worker_name=self._worker_name,
                claim_id=claim_id,
                lease_duration_ms=self._lease_duration_ms,
            )
            claim_id_by_step_id[step_id] = claim_id

        claimed_steps_by_id = {
            step.step_id: step for step in claimed_task.steps
        }
        started_at = time.monotonic()
        step_results = await asyncio.gather(
            *(
                self._step_executor(
                    claimed_task,
                    claimed_steps_by_id[step_id],
                    claim_id_by_step_id[step_id],
                )
                for step_id in candidate_step_ids
            )
        )
        for step_id, result in zip(
            candidate_step_ids,
            step_results,
            strict=True,
        ):
            _require_matching_execution_result(
                expected_step_id=step_id,
                expected_claim_id=claim_id_by_step_id[step_id],
                result=result,
            )

        elapsed_ms = (time.monotonic() - started_at) * 1000
        saved_task = await self._runtime_driver.handle_batch_result(
            task_id=task.task_id,
            batch_result=LongTaskBatchResult(
                batch_id=self._batch_id_factory(),
                task_id=task.task_id,
                actor_type="worker",
                actor_id=self._worker_name,
                step_results=list(step_results),
            ),
            execution_context=LongTaskExecutionContext(
                elapsed_ms=elapsed_ms,
                inline_budget_ms=1,
            ),
            publish_ready_message=supports_reliable_commit(self._store),
            correlation_id=message.correlation_id,
        )
        if not supports_reliable_commit(self._store):
            await self._publish_continuation_if_ready(
                task=saved_task,
                correlation_id=message.correlation_id,
            )

    async def _publish_continuation_if_ready(
        self,
        *,
        task: LongTask,
        correlation_id: str | None,
    ) -> None:
        """
        为仍在运行且存在 Ready Step 的最新任务发布继续通知。

        功能：
            正常批次保存后立即唤醒下一批；如果保存成功但首次发布失败，
            原消息重新投递时也能根据最新任务补发通知。

        参数含义：
            task:
                Store 中已经保存或刚重新加载的最新权威任务快照。
            correlation_id:
                上一条消息携带的可选链路编号；为空时使用 task_id。

        返回值含义：
            None：没有后续工作时保持空操作；成功时通知已发布。需要发布
            却未配置发布器，或运输层失败时抛出异常，使原消息不被 ACK。
        """

        if task.status != "running" or task.execution_mode != "durable":
            return
        ready_step_ids = [
            step.step_id for step in task.steps if step.status == "ready"
        ]
        if not ready_step_ids:
            return
        publisher = self._queue_publisher
        if publisher is None:
            raise RuntimeError("长任务 Handler 尚未配置后续队列发布器")
        await publisher.publish(
            LongTaskQueueMessage(
                task_id=task.task_id,
                task_version=task.version,
                reason="continued",
                ready_step_ids=ready_step_ids,
                correlation_id=correlation_id or task.task_id,
            )
        )


def _select_candidate_step_ids(
    *,
    task: LongTask,
    hinted_step_ids: list[str],
) -> list[str]:
    """
    从消息提示中筛选最新任务里仍可首次领取或租约接管的 Step。

    参数含义：
        task:
            Store 中加载到的最新权威任务快照。
        hinted_step_ids:
            QueueMessage 发布时记录的 Ready Step 编号提示。

    返回值含义：
        list[str]：保持消息顺序、当前状态为 ready 或 running 的步骤编号。
    """

    steps_by_id = {step.step_id: step for step in task.steps}
    return [
        step_id
        for step_id in hinted_step_ids
        if step_id in steps_by_id
        and steps_by_id[step_id].status in {"ready", "running"}
    ]


def _require_matching_execution_result(
    *,
    expected_step_id: str,
    expected_claim_id: str,
    result: LongTaskBatchStepResult,
) -> None:
    """
    校验注入执行器没有把其他 Step 或旧领取令牌的结果混入批次。

    参数含义：
        expected_step_id:
            本次交给执行器的步骤编号。
        expected_claim_id:
            本次领取该步骤时生成的唯一令牌。
        result:
            执行器返回的单步骤批次结果。

    返回值含义：
        None：步骤和领取令牌一致时正常返回，否则抛出明确异常。
    """

    if result.step_id != expected_step_id:
        raise InvalidLongTaskStepExecutionResultError(
            "执行器返回了错误 Step 结果: "
            f"expected={expected_step_id}, actual={result.step_id}"
        )
    if result.claim_id != expected_claim_id:
        raise InvalidLongTaskStepExecutionResultError(
            "执行器返回了错误 claim_id: "
            f"step_id={expected_step_id}"
        )
