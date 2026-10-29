"""长任务重试调度器单轮执行测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskCommitReceipt,
    LongTaskGoal,
    LongTaskStep,
)
from src.runtime.long_tasks.failure_contracts import (
    LongTaskRetryCancellationRequest,
    LongTaskRetryDispatchRequest,
    LongTaskRetryPlan,
)
from src.runtime.long_tasks.retry_scheduler import LongTaskRetryScheduler


NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


class FakeRetrySchedulerStore:
    """记录调度器调用并模拟一个成功计划和一个竞争失败计划。"""

    def __init__(self, task: LongTask, plan: LongTaskRetryPlan) -> None:
        """保存测试任务、已领取计划和调用结果。"""

        self.task = task
        self.plan = plan
        self.claim_calls: list[tuple[str, str, str]] = []
        self.commit_requests: list[LongTaskRetryDispatchRequest] = []
        self.cancellation_requests: list[
            LongTaskRetryCancellationRequest
        ] = []

    async def load_due_retry_plan_ids(
        self,
        *,
        due_at: datetime,
        limit: int = 100,
    ) -> list[str]:
        """返回两个到期计划编号。"""

        assert due_at == NOW
        assert limit == 10
        return ["PLAN-1", "PLAN-2"]

    async def claim_due_retry_plan(
        self,
        *,
        retry_plan_id: str,
        dispatcher_id: str,
        dispatch_claim_id: str,
        now: datetime,
        lease_seconds: float,
    ) -> LongTaskRetryPlan | None:
        """让 PLAN-1 领取成功，让 PLAN-2 模拟被其他调度器抢到。"""

        assert now == NOW
        assert lease_seconds == 30
        self.claim_calls.append(
            (retry_plan_id, dispatcher_id, dispatch_claim_id)
        )
        return self.plan if retry_plan_id == "PLAN-1" else None

    async def load(self, task_id: str) -> LongTask | None:
        """返回计划对应的当前权威失败任务。"""

        assert task_id == self.task.task_id
        return self.task

    async def commit_retry_dispatch(
        self,
        request: LongTaskRetryDispatchRequest,
    ) -> LongTaskCommitReceipt:
        """记录发布请求并返回模拟的成功提交回执。"""

        self.commit_requests.append(request)
        return LongTaskCommitReceipt(
            commit_id=request.commit_id,
            request_fingerprint=request.request_fingerprint,
            submission_fingerprint=request.submission_fingerprint,
            task_id=request.task.task_id,
            task_version=request.task.version,
            event_ids=["EVENT-1"],
            event_sequence_start=1,
            event_sequence_end=1,
            queue_message_id="MESSAGE-1",
            retry_plan_id=request.retry_plan_id,
            committed_at=NOW,
        )

    async def cancel_claimed_retry_plan(
        self,
        request: LongTaskRetryCancellationRequest,
    ) -> LongTaskRetryPlan:
        """
        记录调度器提交的取消请求并返回模拟的已取消计划。

        参数：
            request:
                携带计划编号、发布权身份和取消原因的请求。

        返回值：
            LongTaskRetryPlan：清空发布权并写入取消事实的计划。
        """

        self.cancellation_requests.append(request)
        return self.plan.model_copy(
            update={
                "status": "cancelled",
                "dispatcher_id": None,
                "dispatch_claim_id": None,
                "dispatch_lease_expires_at": None,
                "cancellation_reason": request.cancellation_reason,
                "cancelled_at": request.cancelled_at,
            }
        )


def build_failed_task_and_claimed_plan(
) -> tuple[LongTask, LongTaskRetryPlan]:
    """
    构建调度器测试使用的失败任务和已领取计划。

    参数：
        无。

    返回值：
        tuple[LongTask, LongTaskRetryPlan]：Task V8 和 dispatching 计划。
    """

    task = LongTask(
        task_id="TASK-1",
        user_id="USER-1",
        thread_id="THREAD-1",
        goal=LongTaskGoal(
            original_request="查询狗狗资料",
            objective="整理狗狗资料",
        ),
        steps=[
            LongTaskStep(
                step_id="STEP-1",
                task_id="TASK-1",
                title="读取资料",
                assigned_agent="dog_knowledge_agent",
                status="failed",
                attempt_count=1,
                max_attempts=3,
                version=4,
                last_error_code="UPSTREAM_TIMEOUT",
                last_error_message="调用知识服务超时",
            )
        ],
        status="failed",
        execution_mode="durable",
        version=8,
    )
    plan = LongTaskRetryPlan(
        retry_plan_id="PLAN-1",
        task_id="TASK-1",
        step_id="STEP-1",
        failure_task_version=8,
        failure_step_version=4,
        failed_attempt_count=1,
        next_attempt_count=2,
        max_attempts=3,
        scheduled_at=NOW,
        status="dispatching",
        error_code="UPSTREAM_TIMEOUT",
        failure_commit_id="FAILURE-COMMIT-1",
        source_claim_id="CLAIM-1",
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id="DISPATCH-CLAIM-1",
        dispatch_lease_expires_at=NOW + timedelta(seconds=30),
        created_at=NOW - timedelta(minutes=1),
    )
    return task, plan


@pytest.mark.asyncio
async def test_scheduler_should_dispatch_claimed_plan_and_skip_busy_plan(
) -> None:
    """
    验证单轮调度会发布领取成功的计划并跳过竞争失败的计划。

    参数：
        无。

    返回值：
        None：调用顺序、新状态和轮询汇总符合预期时测试通过。
    """

    task, plan = build_failed_task_and_claimed_plan()
    store = FakeRetrySchedulerStore(task, plan)
    claim_ids = iter(["DISPATCH-CLAIM-1", "DISPATCH-CLAIM-2"])
    scheduler = LongTaskRetryScheduler(
        store,  # type: ignore[arg-type]
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id_factory=lambda: next(claim_ids),
        clock=lambda: NOW,
    )

    result = await scheduler.dispatch_due_retries(
        due_at=NOW,
        limit=10,
        lease_seconds=30,
    )

    assert result.due_plan_ids == ["PLAN-1", "PLAN-2"]
    assert result.claimed_plan_ids == ["PLAN-1"]
    assert result.dispatched_plan_ids == ["PLAN-1"]
    assert result.skipped_plan_ids == ["PLAN-2"]
    assert result.failed_reasons == {}
    assert len(result.receipts) == 1
    assert store.claim_calls == [
        ("PLAN-1", "DISPATCHER-A", "DISPATCH-CLAIM-1"),
        ("PLAN-2", "DISPATCHER-A", "DISPATCH-CLAIM-2"),
    ]
    assert len(store.commit_requests) == 1
    request = store.commit_requests[0]
    assert request.task.version == 9
    assert request.task.status == "queued"
    assert request.task.steps[0].status == "ready"
    assert request.queue_message is not None
    assert request.queue_message.reason == "retry"


@pytest.mark.asyncio
async def test_scheduler_should_cancel_plan_when_task_version_changed(
) -> None:
    """
    验证计划引用的任务版本过期后会完成取消而不是反复等待调度。

    参数：
        无。

    返回值：
        None：计划进入取消汇总且没有发布消息时测试通过。
    """

    task, plan = build_failed_task_and_claimed_plan()
    stale_task = task.model_copy(update={"version": 9})
    store = FakeRetrySchedulerStore(stale_task, plan)
    scheduler = LongTaskRetryScheduler(
        store,  # type: ignore[arg-type]
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id_factory=lambda: "DISPATCH-CLAIM-1",
        clock=lambda: NOW,
    )

    result = await scheduler.dispatch_due_retries(
        due_at=NOW,
        limit=10,
        lease_seconds=30,
    )

    assert result.claimed_plan_ids == ["PLAN-1"]
    assert result.dispatched_plan_ids == []
    assert result.cancelled_plan_ids == ["PLAN-1"]
    assert result.cancellation_reasons == {
        "PLAN-1": "当前任务版本已经不再符合重试计划"
    }
    assert result.skipped_plan_ids == ["PLAN-2"]
    assert result.failed_reasons == {}
    assert store.commit_requests == []
    assert len(store.cancellation_requests) == 1
