"""长任务后台 Worker 最小运行入口测试。"""

from __future__ import annotations

import asyncio

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskBatchStepResult,
    LongTaskGoal,
    LongTaskQueueMessage,
    LongTaskStep,
    LongTaskStreamEntry,
    LongTaskVersionConflictError,
    RedisLongTaskStream,
    run_long_task_worker,
)


class InMemoryWorkerRuntimeStore:
    """保存后台运行入口测试使用的最新 LongTask 快照。"""

    def __init__(self, task: LongTask) -> None:
        self.task = task

    async def create(self, task: LongTask) -> LongTask:
        """保存并返回初始任务。"""

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
        """按照生产 Store 的乐观锁规则保存下一版本快照。"""

        if self.task.version != expected_version:
            raise LongTaskVersionConflictError("测试 Store 版本冲突")
        if task.version != expected_version + 1:
            raise ValueError("新任务版本必须递增 1")
        self.task = task
        return task


class RecordingLongTaskStream(RedisLongTaskStream):
    """投递一条消息并记录 Consumer 与 ACK 的内存 Stream。"""

    def __init__(self, message: LongTaskQueueMessage) -> None:
        self._entry = LongTaskStreamEntry(
            stream_id="1000-0",
            message=message,
        )
        self._delivered = False
        self.consumer_names: list[str] = []
        self.acknowledged_ids: list[str] = []

    async def ensure_consumer_group(self) -> None:
        """测试中消费者组视为已经存在。"""

    async def claim_stale(
        self,
        *,
        consumer_name: str,
        min_idle_time_ms: int,
        count: int,
    ) -> list[LongTaskStreamEntry]:
        """当前测试没有需要恢复的 Pending 消息。"""

        return []

    async def read_new(
        self,
        *,
        consumer_name: str,
        count: int,
        block_ms: int,
    ) -> list[LongTaskStreamEntry]:
        """第一次读取返回消息，并记录实际 Consumer 名称。"""

        self.consumer_names.append(consumer_name)
        if self._delivered:
            return []
        self._delivered = True
        return [self._entry]

    async def acknowledge(self, stream_id: str) -> int:
        """记录成功处理后确认的 Stream Entry 编号。"""

        self.acknowledged_ids.append(stream_id)
        return 1


def build_durable_task() -> LongTask:
    """
    构建一个包含单个 Ready Step 的后台任务。

    返回值含义：
        LongTask：可由后台 Worker 领取并完成的版本 1 任务。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="生成饮食建议",
            objective="生成今天的狗狗饮食建议",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="生成饮食建议",
                assigned_agent="diet_agent",
                status="ready",
            )
        ],
        status="running",
        execution_mode="durable",
    )


@pytest.mark.asyncio
async def test_runtime_should_use_same_worker_identity_and_ack_result() -> None:
    """验证后台入口用同一身份领取消息、执行 Step 并完成 ACK。"""

    store = InMemoryWorkerRuntimeStore(build_durable_task())
    stream = RecordingLongTaskStream(
        LongTaskQueueMessage(
            task_id="task_001",
            task_version=1,
            reason="submitted",
            ready_step_ids=["step_1"],
        )
    )
    stop_event = asyncio.Event()
    observed_claimed_by: list[str | None] = []

    async def executor(
        _: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """记录 Step 领取身份并返回成功结果。"""

        observed_claimed_by.append(step.claimed_by)
        stop_event.set()
        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="completed",
            output_summary="饮食建议已生成",
            claim_id=claim_id,
        )

    await run_long_task_worker(
        stream=stream,
        store=store,
        step_executor=executor,
        worker_name="worker-runtime-1",
        stop_event=stop_event,
        block_ms=1,
        min_idle_time_ms=1,
        recovery_interval_ms=1,
        lease_duration_ms=1000,
    )

    assert stream.consumer_names == ["worker-runtime-1"]
    assert observed_claimed_by == ["worker-runtime-1"]
    assert stream.acknowledged_ids == ["1000-0"]
    assert store.task.status == "completed"
    assert store.task.version == 3


@pytest.mark.asyncio
async def test_runtime_should_ack_persisted_waiting_interaction() -> None:
    """验证等待提示保存成功后消息会 ACK，前端可查询权威交互。"""

    store = InMemoryWorkerRuntimeStore(build_durable_task())
    stream = RecordingLongTaskStream(
        LongTaskQueueMessage(
            task_id="task_001",
            task_version=1,
            reason="submitted",
            ready_step_ids=["step_1"],
        )
    )
    stop_event = asyncio.Event()

    async def executor(
        _: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """返回需要用户补充信息的步骤结果。"""

        stop_event.set()
        return LongTaskBatchStepResult(
            step_id=step.step_id,
            status="awaiting_input",
            waiting_reason="missing_input",
            user_prompt="请补充狗狗年龄。",
            claim_id=claim_id,
        )

    await run_long_task_worker(
        stream=stream,
        store=store,
        step_executor=executor,
        worker_name="worker-runtime-1",
        stop_event=stop_event,
        block_ms=1,
        min_idle_time_ms=1,
        recovery_interval_ms=1,
        lease_duration_ms=1000,
    )

    assert stream.acknowledged_ids == ["1000-0"]
    assert store.task.status == "awaiting_input"
    assert store.task.active_step_ids == ["step_1"]
    assert store.task.pending_interaction is not None
    assert store.task.pending_interaction.prompt == "请补充狗狗年龄。"
    assert store.task.steps[0].claim_id is None
    assert store.task.version == 3
