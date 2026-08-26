"""LongTask Step 领取、租约和幂等令牌单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskGoal,
    LongTaskStep,
    LongTaskStepAttemptsExhaustedError,
    LongTaskStepClaimConflictError,
    LongTaskStepClaimService,
    LongTaskVersionConflictError,
)


class InMemoryClaimStore:
    """为领取服务测试提供带乐观锁语义的内存 Store。"""

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
        """比较旧版本后保存版本恰好递增一次的新快照。"""

        if self.task.version != expected_version:
            raise LongTaskVersionConflictError("测试领取版本冲突")
        if task.version != expected_version + 1:
            raise ValueError("领取后的 Task 版本必须递增 1")
        self.task = task
        self.save_count += 1
        return task


def build_durable_task(*, max_attempts: int = 2) -> LongTask:
    """
    构建包含一个 Ready Step 的后台运行任务。

    参数含义：
        max_attempts：当前步骤允许执行的最大次数。

    返回值含义：
        LongTask：可以被 Worker 领取的 durable 任务。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="生成健康计划",
            objective="生成健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                status="ready",
                max_attempts=max_attempts,
            )
        ],
        status="running",
        execution_mode="durable",
    )


@pytest.mark.asyncio
async def test_claim_should_persist_worker_token_and_lease() -> None:
    """验证首次领取会原子保存 Worker、令牌、租约和运行状态。"""

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    store = InMemoryClaimStore(build_durable_task())

    task = await LongTaskStepClaimService(store).claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now,
    )

    step = task.steps[0]
    assert task.version == 2
    assert task.active_step_ids == ["step_1"]
    assert step.status == "running"
    assert step.claimed_by == "worker-1"
    assert step.claim_id == "claim-001"
    assert step.lease_expires_at == now + timedelta(seconds=30)
    assert step.attempt_count == 1


@pytest.mark.asyncio
async def test_same_active_claim_should_be_idempotent() -> None:
    """验证相同有效领取重复提交不会再次递增版本或执行次数。"""

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    store = InMemoryClaimStore(build_durable_task())
    service = LongTaskStepClaimService(store)
    first = await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now,
    )
    repeated = await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now + timedelta(seconds=1),
    )

    assert repeated == first
    assert repeated.version == 2
    assert repeated.steps[0].attempt_count == 1
    assert store.save_count == 1


@pytest.mark.asyncio
async def test_other_worker_should_not_take_active_lease() -> None:
    """验证租约有效时其他 Worker 不能接管同一 Step。"""

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    store = InMemoryClaimStore(build_durable_task())
    service = LongTaskStepClaimService(store)
    await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now,
    )

    with pytest.raises(
        LongTaskStepClaimConflictError,
        match="有效租约",
    ):
        await service.claim_step(
            task_id="task_001",
            step_id="step_1",
            worker_name="worker-2",
            claim_id="claim-002",
            lease_duration_ms=30_000,
            now=now + timedelta(seconds=10),
        )


@pytest.mark.asyncio
async def test_expired_lease_should_allow_reclaim() -> None:
    """验证租约过期后其他 Worker 可以用新令牌接管。"""

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    store = InMemoryClaimStore(build_durable_task())
    service = LongTaskStepClaimService(store)
    await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now,
    )
    reclaimed = await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-2",
        claim_id="claim-002",
        lease_duration_ms=30_000,
        now=now + timedelta(seconds=31),
    )

    step = reclaimed.steps[0]
    assert reclaimed.version == 3
    assert step.claimed_by == "worker-2"
    assert step.claim_id == "claim-002"
    assert step.attempt_count == 2


@pytest.mark.asyncio
async def test_expired_lease_should_respect_max_attempts() -> None:
    """验证租约过期也不能突破步骤最大执行次数。"""

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    store = InMemoryClaimStore(build_durable_task(max_attempts=1))
    service = LongTaskStepClaimService(store)
    await service.claim_step(
        task_id="task_001",
        step_id="step_1",
        worker_name="worker-1",
        claim_id="claim-001",
        lease_duration_ms=30_000,
        now=now,
    )

    with pytest.raises(LongTaskStepAttemptsExhaustedError):
        await service.claim_step(
            task_id="task_001",
            step_id="step_1",
            worker_name="worker-2",
            claim_id="claim-002",
            lease_duration_ms=30_000,
            now=now + timedelta(seconds=31),
        )
