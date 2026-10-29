"""长任务失败提交准备服务单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskGoal,
    LongTaskStep,
)
from src.runtime.long_tasks.failure_service import (
    build_long_task_failure_commit_request,
    prepare_long_task_failure_commit,
)


NOW = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)


def build_running_task(
    *,
    attempt_count: int = 1,
    max_attempts: int = 3,
) -> LongTask:
    """
    构建失败提交测试使用的运行中任务。

    参数：
        attempt_count:
            当前步骤已经开始执行的次数。
        max_attempts:
            当前步骤允许的最大执行次数。

    返回值：
        LongTask：持有 CLAIM-1、Task V7、Step V3 的运行中任务。
    """

    step = LongTaskStep(
        step_id="STEP-1",
        task_id="TASK-1",
        title="检索狗狗资料",
        assigned_agent="dog_knowledge_agent",
        status="running",
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        version=3,
        claimed_by="WORKER-1",
        claim_id="CLAIM-1",
        lease_expires_at=NOW + timedelta(minutes=5),
    )
    return LongTask(
        task_id="TASK-1",
        user_id="USER-1",
        thread_id="THREAD-1",
        goal=LongTaskGoal(
            original_request="查询边境牧羊犬资料",
            objective="整理边境牧羊犬资料",
        ),
        steps=[step],
        status="running",
        execution_mode="durable",
        active_step_ids=["STEP-1"],
        version=7,
    )


def build_failed_batch_result(
    *,
    claim_id: str = "CLAIM-1",
) -> LongTaskBatchResult:
    """
    构建 Worker 返回的单步骤失败结果。

    参数：
        claim_id:
            产生结果时 Worker 持有的领取令牌。

    返回值：
        LongTaskBatchResult：包含 Trace、Span 和错误事实的失败批次。
    """

    return LongTaskBatchResult(
        batch_id="BATCH-1",
        task_id="TASK-1",
        actor_type="worker",
        actor_id="WORKER-1",
        trace_id="TRACE-1",
        step_results=[
            LongTaskBatchStepResult(
                step_id="STEP-1",
                status="failed",
                error_code="UPSTREAM_TIMEOUT",
                error_message="调用知识服务超时",
                claim_id=claim_id,
                span_id="SPAN-1",
            )
        ],
    )


def test_transient_failure_should_prepare_failed_snapshot_and_retry_plan(
) -> None:
    """
    验证可重试失败会一起准备新快照、决定和重试计划。

    参数：
        无。

    返回值：
        None：版本、错误事实和计划引用全部正确时测试通过。
    """

    preparation = prepare_long_task_failure_commit(
        task=build_running_task(),
        batch_result=build_failed_batch_result(),
        failure_category="transient",
        failure_commit_id="FAILURE-COMMIT-1",
        now=NOW,
        base_delay_seconds=10,
        max_delay_seconds=60,
    )

    failed_step = preparation.failed_task.steps[0]
    assert preparation.expected_task_version == 7
    assert preparation.failed_task.version == 8
    assert preparation.failed_task.status == "failed"
    assert preparation.failed_task.active_step_ids == []
    assert failed_step.version == 4
    assert failed_step.status == "failed"
    assert failed_step.claim_id is None
    assert failed_step.last_error_code == "UPSTREAM_TIMEOUT"
    assert failed_step.last_trace_id == "TRACE-1"
    assert failed_step.last_span_id == "SPAN-1"
    assert preparation.decision.source_claim_id == "CLAIM-1"
    assert preparation.retry_plan is not None
    assert preparation.retry_plan.failure_task_version == 8
    assert preparation.retry_plan.failure_step_version == 4
    assert preparation.dead_letter_record is None

    request = build_long_task_failure_commit_request(preparation)
    assert request.operation == "update"
    assert request.expected_version == 7
    assert request.task.version == 8
    assert request.failure_action == "retry"
    assert request.retry_plan == preparation.retry_plan
    assert request.dead_letter_record is None
    assert request.queue_message is None
    assert [draft.event_type for draft in request.event_drafts] == [
        "step_failed",
        "task_failed",
        "step_retry_scheduled",
    ]


def test_exhausted_failure_should_prepare_dead_letter_without_retry_plan(
) -> None:
    """
    验证次数耗尽后会准备死信，而且不再创建重试计划。

    参数：
        无。

    返回值：
        None：死信保留版本、执行身份和 Worker 身份时测试通过。
    """

    preparation = prepare_long_task_failure_commit(
        task=build_running_task(attempt_count=3, max_attempts=3),
        batch_result=build_failed_batch_result(),
        failure_category="transient",
        failure_commit_id="FAILURE-COMMIT-3",
        now=NOW,
        base_delay_seconds=10,
        max_delay_seconds=60,
        payload_reference="artifact://TASK-1/STEP-1/input",
    )

    assert preparation.decision.action == "dead_letter"
    assert preparation.retry_plan is None
    assert preparation.dead_letter_record is not None
    assert preparation.dead_letter_record.failure_task_version == 8
    assert preparation.dead_letter_record.failure_step_version == 4
    assert preparation.dead_letter_record.source_claim_id == "CLAIM-1"
    assert preparation.dead_letter_record.worker_id == "WORKER-1"


def test_business_failure_should_fail_task_without_plan_or_dead_letter(
) -> None:
    """
    验证业务失败只结束任务，不会创建重试计划或死信。

    参数：
        无。

    返回值：
        None：处理动作和两个可选记录符合业务失败规则时测试通过。
    """

    preparation = prepare_long_task_failure_commit(
        task=build_running_task(),
        batch_result=build_failed_batch_result(),
        failure_category="business",
        failure_commit_id="FAILURE-COMMIT-BUSINESS",
        now=NOW,
        base_delay_seconds=10,
        max_delay_seconds=60,
    )

    assert preparation.decision.action == "fail_task"
    assert preparation.failed_task.status == "failed"
    assert preparation.retry_plan is None
    assert preparation.dead_letter_record is None


def test_stale_claim_should_be_rejected_before_building_failure_snapshot(
) -> None:
    """
    验证旧 Worker 的过期领取令牌不能提交失败结果。

    参数：
        无。

    返回值：
        None：领取令牌不一致时服务拒绝继续组装数据即测试通过。
    """

    with pytest.raises(
        ValueError,
        match="失败结果的 claim_id 与当前领取令牌不一致",
    ):
        prepare_long_task_failure_commit(
            task=build_running_task(),
            batch_result=build_failed_batch_result(claim_id="CLAIM-OLD"),
            failure_category="transient",
            failure_commit_id="FAILURE-COMMIT-OLD",
            now=NOW,
            base_delay_seconds=10,
            max_delay_seconds=60,
        )
