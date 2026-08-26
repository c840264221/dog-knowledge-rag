"""LongTask QueueMessage 真实业务桥接 Handler 单元测试。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pytest

from src.runtime.long_tasks import (
    InvalidLongTaskStepExecutionResultError,
    LongTask,
    LongTaskBatchStepResult,
    LongTaskGoal,
    LongTaskQueueMessage,
    LongTaskQueueMessageHandler,
    LongTaskStep,
    LongTaskVersionConflictError,
)


class InMemoryHandlerStore:
    """为 Worker Handler 测试提供带乐观锁语义的内存 Store。"""

    def __init__(self, task: LongTask) -> None:
        self.task = task
        self.save_count = 0

    async def create(self, task: LongTask) -> LongTask:
        """保存并返回初始测试任务。"""

        self.task = task
        return task

    async def load(self, task_id: str) -> LongTask | None:
        """任务编号匹配时返回当前权威快照。"""

        return self.task if self.task.task_id == task_id else None

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """比较旧版本后保存恰好递增一次的新任务快照。"""

        if self.task.version != expected_version:
            raise LongTaskVersionConflictError("测试Handler版本冲突")
        if task.version != expected_version + 1:
            raise ValueError("新任务版本必须递增 1")
        self.task = task
        self.save_count += 1
        return task


class RecordingQueuePublisher:
    """记录 Handler 发布的后续批次通知。"""

    def __init__(self) -> None:
        """初始化空消息列表。"""

        self.messages: list[LongTaskQueueMessage] = []

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        保存一条测试通知并返回固定消息编号。

        参数含义：
            message：Handler 根据最新 Ready Step 构建的继续通知。

        返回值含义：
            str：模拟 Redis Stream 生成的消息编号。
        """

        self.messages.append(message)
        return f"1000-{len(self.messages)}"


class FailingOnceQueuePublisher(RecordingQueuePublisher):
    """第一次发布失败、第二次成功的运输层替身。"""

    def __init__(self) -> None:
        """初始化一次待触发失败。"""

        super().__init__()
        self.attempt_count = 0

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        第一次模拟网络失败，后续调用记录消息。

        参数含义：
            message：准备发布的后续批次通知。

        返回值含义：
            str：第二次及以后发布成功时的模拟消息编号。
        """

        self.attempt_count += 1
        if self.attempt_count == 1:
            raise ConnectionError("模拟 continued 消息首次发布失败")
        return await super().publish(message)

def build_durable_task(
    *,
    status: Literal["running", "completed"] = "running",
) -> LongTask:
    """
    构建包含两个并发 Ready Step 的后台任务。

    参数含义：
        status：任务整体状态，测试陈旧消息时可传 completed。

    返回值含义：
        LongTask：可供 Handler 加载和领取的任务快照。
    """

    step_status = "completed" if status == "completed" else "ready"
    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="生成健康计划",
            objective="并发生成饮食和运动建议",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="生成饮食建议",
                assigned_agent="diet_agent",
                status=step_status,
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成运动建议",
                assigned_agent="exercise_agent",
                status=step_status,
            ),
        ],
        status=status,
        execution_mode="durable",
    )


def build_message() -> LongTaskQueueMessage:
    """
    构建提示两个 Ready Step 的轻量队列消息。

    返回值含义：
        LongTaskQueueMessage：当前 Handler 测试使用的合法消息。
    """

    return LongTaskQueueMessage(
        task_id="task_001",
        task_version=1,
        reason="continued",
        ready_step_ids=["step_1", "step_2"],
    )


def build_dependent_durable_task() -> LongTask:
    """
    构建第一步完成后第二步才会 Ready 的后台任务。

    返回值含义：
        LongTask：用于验证 continued 消息发布和失败补发的依赖任务。
    """

    return LongTask(
        task_id="task_dependent",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="先读取档案再生成建议",
            objective="按依赖完成两个步骤",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_dependent",
                title="读取档案",
                assigned_agent="profile_agent",
                status="ready",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_dependent",
                title="生成建议",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            ),
        ],
        status="running",
        execution_mode="durable",
    )


def build_dependent_message() -> LongTaskQueueMessage:
    """
    构建只提示依赖任务第一步的初始消息。

    返回值含义：
        LongTaskQueueMessage：第一批执行和失败重投共用的消息。
    """

    return LongTaskQueueMessage(
        task_id="task_dependent",
        task_version=1,
        reason="submitted",
        ready_step_ids=["step_1"],
        correlation_id="trace-dependent",
    )


def fixed_ids(*values: str) -> Iterator[str]:
    """
    按顺序返回测试使用的固定领取或批次编号。

    参数含义：
        values：准备依次返回的编号。

    返回值含义：
        Iterator[str]：每次 next() 返回一个固定编号的迭代器。
    """

    return iter(values)


@pytest.mark.asyncio
async def test_handler_should_claim_execute_and_commit_ready_batch() -> None:
    """验证Handler会先领取两个Step，再汇总结果并完成任务。"""

    store = InMemoryHandlerStore(build_durable_task())
    claim_ids = fixed_ids("claim-1", "claim-2")
    executed: list[tuple[str, str, list[str]]] = []

    async def executor(
        task: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """记录执行时两个Step均已领取，并返回成功结果。"""

        executed.append((step.step_id, claim_id, list(task.active_step_ids)))
        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="completed",
            output_summary=f"{step.title}完成",
            claim_id=claim_id,
        )

    await LongTaskQueueMessageHandler(
        store=store,
        step_executor=executor,
        worker_name="worker-1",
        claim_id_factory=lambda: next(claim_ids),
        batch_id_factory=lambda: "batch-1",
    )(build_message())

    assert [item[0] for item in executed] == ["step_1", "step_2"]
    assert all(item[2] == ["step_1", "step_2"] for item in executed)
    assert store.task.status == "completed"
    assert [step.status for step in store.task.steps] == [
        "completed",
        "completed",
    ]
    assert all(step.claim_id is None for step in store.task.steps)
    assert store.task.version == 4
    assert store.save_count == 3


@pytest.mark.asyncio
async def test_handler_should_ignore_message_without_actionable_steps() -> None:
    """验证任务已完成时旧消息会成为安全空操作。"""

    store = InMemoryHandlerStore(build_durable_task(status="completed"))
    executed = False

    async def executor(
        _: LongTask,
        __: LongTaskStep,
        ___: str,
    ) -> LongTaskBatchStepResult:
        """陈旧消息场景下不应调用执行器。"""

        nonlocal executed
        executed = True
        raise AssertionError("不应执行陈旧消息")

    await LongTaskQueueMessageHandler(
        store=store,
        step_executor=executor,
        worker_name="worker-1",
    )(build_message())

    assert executed is False
    assert store.save_count == 0


@pytest.mark.asyncio
async def test_handler_should_republish_ready_steps_for_stale_hint() -> None:
    """验证旧提示不会重复执行，并会为当前Ready步骤补发通知。"""

    store = InMemoryHandlerStore(build_durable_task())
    publisher = RecordingQueuePublisher()
    executed = False

    async def executor(
        _: LongTask,
        __: LongTaskStep,
        ___: str,
    ) -> LongTaskBatchStepResult:
        """消息没有匹配当前任务的Step时不应调用执行器。"""

        nonlocal executed
        executed = True
        raise AssertionError("不应执行消息中的未知Step")

    message = build_message().model_copy(
        update={"ready_step_ids": ["step_already_removed"]}
    )
    await LongTaskQueueMessageHandler(
        store=store,
        step_executor=executor,
        worker_name="worker-1",
        queue_publisher=publisher,
    )(message)

    assert executed is False
    assert store.save_count == 0
    assert len(publisher.messages) == 1
    assert publisher.messages[0].reason == "continued"
    assert publisher.messages[0].ready_step_ids == ["step_1", "step_2"]


@pytest.mark.asyncio
async def test_handler_should_reject_executor_result_with_wrong_claim() -> None:
    """验证执行器不能提交不属于当前领取批次的结果。"""

    store = InMemoryHandlerStore(build_durable_task())
    claim_ids = fixed_ids("claim-1", "claim-2")

    async def executor(
        _: LongTask,
        step: LongTaskStep,
        __: str,
    ) -> LongTaskBatchStepResult:
        """模拟执行器错误返回旧领取令牌。"""

        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="completed",
            claim_id="claim-old",
        )

    with pytest.raises(
        InvalidLongTaskStepExecutionResultError,
        match="错误 claim_id",
    ):
        await LongTaskQueueMessageHandler(
            store=store,
            step_executor=executor,
            worker_name="worker-1",
            claim_id_factory=lambda: next(claim_ids),
        )(build_message())

    assert [step.status for step in store.task.steps] == [
        "running",
        "running",
    ]
    assert store.save_count == 2


@pytest.mark.asyncio
async def test_handler_should_publish_next_ready_batch() -> None:
    """验证一个后台批次保存后会立即发布下一批Ready步骤。"""

    store = InMemoryHandlerStore(build_dependent_durable_task())
    publisher = RecordingQueuePublisher()

    async def executor(
        _: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """完成第一步，使依赖它的第二步进入Ready。"""

        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="completed",
            output_summary="档案读取完成",
            claim_id=claim_id,
        )

    await LongTaskQueueMessageHandler(
        store=store,
        step_executor=executor,
        worker_name="worker-1",
        queue_publisher=publisher,
        claim_id_factory=lambda: "claim-step-1",
        batch_id_factory=lambda: "batch-step-1",
    )(build_dependent_message())

    assert [step.status for step in store.task.steps] == [
        "completed",
        "ready",
    ]
    assert len(publisher.messages) == 1
    continued = publisher.messages[0]
    assert continued.task_id == "task_dependent"
    assert continued.task_version == store.task.version
    assert continued.reason == "continued"
    assert continued.ready_step_ids == ["step_2"]
    assert continued.correlation_id == "trace-dependent"


@pytest.mark.asyncio
async def test_handler_should_republish_after_first_publish_failure() -> None:
    """验证保存成功但发布失败后，原消息重投可以补发下一批。"""

    store = InMemoryHandlerStore(build_dependent_durable_task())
    publisher = FailingOnceQueuePublisher()
    execution_count = 0

    async def executor(
        _: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """记录第一步只被真正执行一次。"""

        nonlocal execution_count
        execution_count += 1
        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="completed",
            claim_id=claim_id,
        )

    handler = LongTaskQueueMessageHandler(
        store=store,
        step_executor=executor,
        worker_name="worker-1",
        queue_publisher=publisher,
        claim_id_factory=lambda: "claim-step-1",
        batch_id_factory=lambda: "batch-step-1",
    )

    with pytest.raises(ConnectionError, match="首次发布失败"):
        await handler(build_dependent_message())

    assert execution_count == 1
    assert [step.status for step in store.task.steps] == [
        "completed",
        "ready",
    ]

    await handler(build_dependent_message())

    assert execution_count == 1
    assert publisher.attempt_count == 2
    assert len(publisher.messages) == 1
    assert publisher.messages[0].ready_step_ids == ["step_2"]
