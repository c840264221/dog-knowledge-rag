"""准备长任务失败结果及其后续治理记录。"""

from __future__ import annotations

from datetime import datetime, timezone

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskEventDraft,
    LongTaskQueueMessage,
    LongTaskStep,
)
from src.runtime.long_tasks.commit_service import (
    build_batch_result_fingerprint_view,
    build_long_task_fingerprint_view,
    build_queue_message_fingerprint_view,
    build_transition_event_drafts,
    calculate_request_fingerprint,
    calculate_submission_fingerprint,
    resolve_batch_actor,
)
from src.runtime.long_tasks.failure_contracts import (
    LongTaskDeadLetterRecord,
    LongTaskFailureAction,
    LongTaskFailureCategory,
    LongTaskFailureCommitPreparation,
    LongTaskFailureCommitRequest,
    LongTaskRetryPlan,
    LongTaskRetryCancellationRequest,
    LongTaskRetryDispatchRequest,
)
from src.runtime.long_tasks.failure_policy import decide_long_task_failure
from src.runtime.long_tasks.failure_records import (
    build_dead_letter_record,
    build_retry_plan,
)
from src.runtime.long_tasks.state_machine import (
    transition_long_task,
    transition_long_task_step,
)


def prepare_long_task_failure_commit(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
    failure_category: LongTaskFailureCategory,
    failure_commit_id: str,
    now: datetime,
    base_delay_seconds: float,
    max_delay_seconds: float,
    retry_deadline: datetime | None = None,
    payload_reference: str | None = None,
    worker_id: str | None = None,
) -> LongTaskFailureCommitPreparation:
    """
    准备一次步骤失败需要原子保存的完整数据。

    功能：
        校验失败结果属于当前运行步骤和当前领取令牌，把步骤及任务推进
        到 failed 新版本，再根据失败策略创建重试计划、死信记录，或
        仅按失败结束任务。函数只组装数据，不读取或写入 Redis。

    参数：
        task:
            提交失败结果前读取到的最新权威任务快照。
        batch_result:
            Worker 返回的批次结果；当前阶段只允许包含一个 failed 结果。
        failure_category:
            错误类别：临时错误、永久错误、业务错误或未知错误。
        failure_commit_id:
            本次逻辑失败使用的稳定提交编号。
        now:
            生成失败决定的当前时间，必须包含有效时区。
        base_delay_seconds/max_delay_seconds:
            计算自动重试时间使用的基础上限和总上限秒数。
        retry_deadline:
            可选的最晚重试时间，超过后改为进入死信。
        payload_reference:
            可选的原始任务输入引用，只在创建死信时使用。
        worker_id:
            可选的 Worker 身份；未提供时尝试使用批次提交者身份。

    返回值：
        LongTaskFailureCommitPreparation：
            包含旧版本、新失败快照、决定以及重试计划或死信记录的
            待提交数据包。

    异常：
        ValueError：
            任务状态、批次结果、步骤状态或领取令牌不符合提交条件时抛出。
    """

    failed_result = _require_single_failed_result(
        task=task,
        batch_result=batch_result,
    )
    current_step = _require_current_claimed_step(
        task=task,
        failed_result=failed_result,
    )
    failed_step = _build_failed_step(
        step=current_step,
        failed_result=failed_result,
        trace_id=batch_result.trace_id,
    )
    failed_task = _build_failed_task(
        task=task,
        failed_step=failed_step,
    )
    decision = decide_long_task_failure(
        task_id=task.task_id,
        step_id=current_step.step_id,
        failure_category=failure_category,
        error_code=failed_result.error_code or "LONG_TASK_STEP_FAILED",
        error_message=failed_result.error_message or "长任务步骤执行失败",
        attempt_count=current_step.attempt_count,
        max_attempts=current_step.max_attempts,
        now=now,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        retry_deadline=retry_deadline,
        source_trace_id=batch_result.trace_id,
        source_span_id=failed_result.span_id,
        source_claim_id=failed_result.claim_id,
    )

    retry_plan = None
    dead_letter_record = None
    if decision.action == "retry":
        retry_plan = build_retry_plan(
            decision=decision,
            failure_task_version=failed_task.version,
            failure_step_version=failed_step.version,
            failure_commit_id=failure_commit_id,
        )
    elif decision.action == "dead_letter":
        resolved_worker_id = worker_id
        if resolved_worker_id is None and batch_result.actor_type == "worker":
            resolved_worker_id = batch_result.actor_id
        dead_letter_record = build_dead_letter_record(
            decision=decision,
            failure_task_version=failed_task.version,
            failure_step_version=failed_step.version,
            failure_commit_id=failure_commit_id,
            payload_reference=payload_reference,
            worker_id=resolved_worker_id,
        )

    return LongTaskFailureCommitPreparation(
        failure_commit_id=failure_commit_id,
        expected_task_version=task.version,
        previous_task=task,
        batch_result=batch_result,
        failed_task=failed_task,
        decision=decision,
        retry_plan=retry_plan,
        dead_letter_record=dead_letter_record,
    )


def build_long_task_failure_commit_request(
    preparation: LongTaskFailureCommitPreparation,
) -> LongTaskFailureCommitRequest:
    """
    把失败提交材料包转换成 Redis 存储请求。

    功能：
        根据失败前后的任务变化生成不可变事件，并分别计算请求指纹和
        最终提交内容指纹。函数不会访问 Redis，也不会发布重试消息。

    参数：
        preparation:
            已经包含失败后快照、决定、计划或死信的待提交材料包。

    返回值：
        LongTaskFailureCommitRequest：
            可以交给 Store.commit_failure 原子保存的失败提交请求。
    """

    actor_type, actor_id = resolve_batch_actor(preparation.batch_result)
    event_drafts = build_transition_event_drafts(
        previous_task=preparation.previous_task,
        next_task=preparation.failed_task,
        actor_type=actor_type,
        actor_id=actor_id,
        correlation_id=preparation.batch_result.trace_id,
    )
    event_drafts.append(
        LongTaskEventDraft(
            event_type=_failure_action_event_type(
                preparation.decision.action
            ),
            actor_type=actor_type,
            actor_id=actor_id,
            step_id=preparation.decision.step_id,
            correlation_id=preparation.batch_result.trace_id,
            payload={
                "failure_action": preparation.decision.action,
                "failure_category": preparation.decision.failure_category,
                "error_code": preparation.decision.error_code,
                "attempt_count": preparation.decision.attempt_count,
                "max_attempts": preparation.decision.max_attempts,
                "retry_plan_id": (
                    preparation.retry_plan.retry_plan_id
                    if preparation.retry_plan is not None
                    else None
                ),
                "dead_letter_id": (
                    preparation.dead_letter_record.dead_letter_id
                    if preparation.dead_letter_record is not None
                    else None
                ),
            },
        )
    )

    request_payload = {
        "action": "handle_failure",
        "task_id": preparation.failed_task.task_id,
        "batch_result": build_batch_result_fingerprint_view(
            preparation.batch_result
        ),
        "failure_category": preparation.decision.failure_category,
    }
    submission_payload = {
        "operation": "update",
        "expected_version": preparation.expected_task_version,
        "task": build_long_task_fingerprint_view(
            preparation.failed_task
        ),
        "failure_action": preparation.decision.action,
        "event_drafts": [
            draft.model_dump(mode="json") for draft in event_drafts
        ],
        "retry_plan": _failure_record_fingerprint_view(
            preparation.retry_plan
        ),
        "dead_letter_record": _failure_record_fingerprint_view(
            preparation.dead_letter_record
        ),
    }
    return LongTaskFailureCommitRequest(
        operation="update",
        task=preparation.failed_task,
        expected_version=preparation.expected_task_version,
        commit_id=preparation.failure_commit_id,
        request_fingerprint=calculate_request_fingerprint(
            request_payload
        ),
        submission_fingerprint=calculate_submission_fingerprint(
            submission_payload
        ),
        event_drafts=event_drafts,
        failure_action=preparation.decision.action,
        retry_plan=preparation.retry_plan,
        dead_letter_record=preparation.dead_letter_record,
    )


def build_long_task_retry_dispatch_request(
    *,
    task: LongTask,
    claimed_plan: LongTaskRetryPlan,
    commit_id: str,
    now: datetime,
    correlation_id: str | None = None,
) -> LongTaskRetryDispatchRequest:
    """
    根据已领取计划构建重试消息的原子发布请求。

    功能：
        重新检查失败任务和步骤仍与计划版本一致，把步骤恢复为 ready、
        任务恢复为 queued，并创建供 Worker 后续领取的轻量队列消息。
        函数只构造请求，不写入 Redis。

    参数：
        task:
            发布前重新读取的当前权威失败任务快照。
        claimed_plan:
            已由调度器成功领取、状态为 dispatching 的重试计划。
        commit_id:
            本次消息发布提交使用的稳定编号。
        now:
            本次发布时间，必须包含有效时区且不能超过发布权租约。
        correlation_id:
            可选的业务关联编号，用于事件和队列消息关联。

    返回值：
        LongTaskRetryDispatchRequest：
            包含 queued 新快照、事件和重试消息的原子发布请求。

    异常：
        ValueError：计划、租约、任务或步骤已经不允许重试时抛出。
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now 必须包含时区")
    dispatched_at = now.astimezone(timezone.utc)
    if claimed_plan.status != "dispatching":
        raise ValueError("只有 dispatching 重试计划可以准备发布消息")
    if (
        claimed_plan.dispatcher_id is None
        or claimed_plan.dispatch_claim_id is None
        or claimed_plan.dispatch_lease_expires_at is None
    ):
        raise ValueError("已领取计划缺少完整发布权信息")
    if dispatched_at >= claimed_plan.dispatch_lease_expires_at:
        raise ValueError("重试计划发布权租约已经到期")
    if task.task_id != claimed_plan.task_id:
        raise ValueError("当前任务与重试计划的 task_id 不一致")
    if task.version != claimed_plan.failure_task_version:
        raise ValueError("当前任务版本已经不再符合重试计划")
    if task.status != "failed":
        raise ValueError("只有 failed 任务可以恢复为重试队列状态")

    failed_step = next(
        (
            step
            for step in task.steps
            if step.step_id == claimed_plan.step_id
        ),
        None,
    )
    if failed_step is None:
        raise ValueError("重试计划引用了当前任务中不存在的步骤")
    if failed_step.status != "failed":
        raise ValueError("重试计划对应的步骤已经不再是 failed 状态")
    if failed_step.version != claimed_plan.failure_step_version:
        raise ValueError("当前步骤版本已经不再符合重试计划")

    ready_step = transition_long_task_step(
        failed_step,
        target_status="ready",
    )
    updated_steps = [
        ready_step if step.step_id == ready_step.step_id else step
        for step in task.steps
    ]
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "steps": updated_steps,
            "active_step_ids": [ready_step.step_id],
        }
    )
    transition_base = LongTask.model_validate(task_data)
    queued_task = transition_long_task(
        transition_base,
        target_status="queued",
        active_step_ids=[ready_step.step_id],
    )
    queue_message = LongTaskQueueMessage(
        task_id=queued_task.task_id,
        task_version=queued_task.version,
        reason="retry",
        ready_step_ids=[ready_step.step_id],
        correlation_id=correlation_id,
        enqueued_at=dispatched_at,
    )
    event_drafts = build_transition_event_drafts(
        previous_task=task,
        next_task=queued_task,
        actor_type="system",
        actor_id=claimed_plan.dispatcher_id,
        correlation_id=correlation_id,
    )
    event_drafts.append(
        LongTaskEventDraft(
            event_type="retry_message_dispatched",
            actor_type="system",
            actor_id=claimed_plan.dispatcher_id,
            step_id=ready_step.step_id,
            correlation_id=correlation_id,
            payload={
                "retry_plan_id": claimed_plan.retry_plan_id,
                "next_attempt_count": claimed_plan.next_attempt_count,
            },
        )
    )

    request_payload = {
        "action": "dispatch_retry",
        "retry_plan_id": claimed_plan.retry_plan_id,
        "dispatch_claim_id": claimed_plan.dispatch_claim_id,
        "failure_task_version": claimed_plan.failure_task_version,
        "failure_step_version": claimed_plan.failure_step_version,
    }
    submission_payload = {
        "operation": "update",
        "expected_version": task.version,
        "task": build_long_task_fingerprint_view(queued_task),
        "event_drafts": [
            draft.model_dump(mode="json") for draft in event_drafts
        ],
        "queue_message": build_queue_message_fingerprint_view(
            queue_message
        ),
        "retry_plan_id": claimed_plan.retry_plan_id,
        "dispatch_claim_id": claimed_plan.dispatch_claim_id,
    }
    return LongTaskRetryDispatchRequest(
        operation="update",
        task=queued_task,
        expected_version=task.version,
        commit_id=commit_id,
        request_fingerprint=calculate_request_fingerprint(request_payload),
        submission_fingerprint=calculate_submission_fingerprint(
            submission_payload
        ),
        event_drafts=event_drafts,
        queue_message=queue_message,
        retry_plan_id=claimed_plan.retry_plan_id,
        step_id=claimed_plan.step_id,
        dispatcher_id=claimed_plan.dispatcher_id,
        dispatch_claim_id=claimed_plan.dispatch_claim_id,
        failure_task_version=claimed_plan.failure_task_version,
        failure_step_version=claimed_plan.failure_step_version,
        dispatched_at=dispatched_at,
    )


def find_long_task_retry_ineligibility_reason(
    *,
    task: LongTask,
    claimed_plan: LongTaskRetryPlan,
) -> str | None:
    """
    判断已领取计划是否因最新任务状态变化而永久失效。

    参数：
        task:
            发布前重新读取的当前权威任务快照。
        claimed_plan:
            当前调度器已经领取发布权的重试计划。

    返回值：
        str | None：
            确定不能继续发布时返回中文原因；仍符合原计划时返回 None。
    """

    if task.task_id != claimed_plan.task_id:
        return "当前任务与重试计划的 task_id 不一致"
    if task.version != claimed_plan.failure_task_version:
        return "当前任务版本已经不再符合重试计划"
    if task.status != "failed":
        return "当前任务已经不再处于 failed 状态"
    current_step = next(
        (step for step in task.steps if step.step_id == claimed_plan.step_id),
        None,
    )
    if current_step is None:
        return "重试计划引用的步骤已经不存在"
    if current_step.status != "failed":
        return "重试计划对应的步骤已经不再是 failed 状态"
    if current_step.version != claimed_plan.failure_step_version:
        return "当前步骤版本已经不再符合重试计划"
    return None


def build_long_task_retry_cancellation_request(
    *,
    claimed_plan: LongTaskRetryPlan,
    cancellation_reason: str,
    now: datetime,
) -> LongTaskRetryCancellationRequest:
    """
    根据已领取计划构造原子取消请求。

    参数：
        claimed_plan:
            当前调度器已经领取、状态为 dispatching 的重试计划。
        cancellation_reason:
            根据最新权威任务确认的计划失效原因。
        now:
            本次取消时间，必须包含有效时区。

    返回值：
        LongTaskRetryCancellationRequest：
            携带当前发布权身份、可交给 Store 执行的取消请求。
    """

    if claimed_plan.status != "dispatching":
        raise ValueError("只有 dispatching 重试计划可以准备取消")
    if claimed_plan.dispatcher_id is None:
        raise ValueError("已领取计划缺少 dispatcher_id")
    if claimed_plan.dispatch_claim_id is None:
        raise ValueError("已领取计划缺少 dispatch_claim_id")
    normalized_reason = str(cancellation_reason or "").strip()
    if not normalized_reason:
        raise ValueError("cancellation_reason 不能为空")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now 必须包含时区")
    return LongTaskRetryCancellationRequest(
        retry_plan_id=claimed_plan.retry_plan_id,
        task_id=claimed_plan.task_id,
        step_id=claimed_plan.step_id,
        dispatcher_id=claimed_plan.dispatcher_id,
        dispatch_claim_id=claimed_plan.dispatch_claim_id,
        cancellation_reason=normalized_reason,
        cancelled_at=now.astimezone(timezone.utc),
    )


def _failure_record_fingerprint_view(
    record: LongTaskRetryPlan | LongTaskDeadLetterRecord | None,
) -> dict[str, object] | None:
    """
    构建排除自动创建时间的重试计划或死信指纹视图。

    参数：
        record:
            可选的 Pydantic 重试计划或死信记录。

    返回值：
        object | None：不存在时返回 None；存在时返回稳定字段字典。
    """

    if record is None:
        return None
    record_view = record.model_dump(mode="json")
    record_view.pop("created_at", None)
    return record_view


def _failure_action_event_type(action: LongTaskFailureAction) -> str:
    """
    把失败处理动作转换成已经发生的业务事件名称。

    参数：
        action:
            retry、dead_letter 或 fail_task 处理动作。

    返回值：
        str：对应的过去式业务事件名称。
    """

    return {
        "retry": "step_retry_scheduled",
        "dead_letter": "step_dead_lettered",
        "fail_task": "task_failure_finalized",
    }[action]


def _require_single_failed_result(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
) -> LongTaskBatchStepResult:
    """
    读取并校验当前阶段唯一允许的失败步骤结果。

    参数：
        task:
            当前权威任务快照。
        batch_result:
            Worker 返回的批次结果。

    返回值：
        LongTaskBatchStepResult：唯一的 failed 步骤结果。

    异常：
        ValueError：任务状态、任务编号或失败结果数量不符合要求时抛出。
    """

    if task.status != "running":
        raise ValueError("只有 running 任务可以提交执行失败结果")
    if batch_result.task_id != task.task_id:
        raise ValueError("批次结果与当前任务的 task_id 不一致")
    if len(batch_result.step_results) != 1:
        raise ValueError("当前失败治理阶段只支持单步骤失败批次")
    failed_result = batch_result.step_results[0]
    if failed_result.status != "failed":
        raise ValueError("失败治理只能接收 failed 步骤结果")
    return failed_result


def _require_current_claimed_step(
    *,
    task: LongTask,
    failed_result: LongTaskBatchStepResult,
) -> LongTaskStep:
    """
    确认失败结果来自当前仍持有执行权的步骤。

    参数：
        task:
            当前权威任务快照。
        failed_result:
            准备提交的失败步骤结果。

    返回值：
        LongTaskStep：与失败结果匹配的当前 running 步骤。

    异常：
        ValueError：步骤不存在、未运行或领取令牌不一致时抛出。
    """

    current_step = next(
        (
            step
            for step in task.steps
            if step.step_id == failed_result.step_id
        ),
        None,
    )
    if current_step is None:
        raise ValueError("失败结果引用了当前任务中不存在的步骤")
    if current_step.status != "running":
        raise ValueError("只有 running 步骤可以提交执行失败结果")
    if current_step.claim_id != failed_result.claim_id:
        raise ValueError("失败结果的 claim_id 与当前领取令牌不一致")
    return current_step


def _build_failed_step(
    *,
    step: LongTaskStep,
    failed_result: LongTaskBatchStepResult,
    trace_id: str | None,
) -> LongTaskStep:
    """
    生成清除领取权并写入最后错误事实的失败步骤快照。

    参数：
        step:
            提交失败结果前的 running 步骤。
        failed_result:
            Worker 返回的失败事实。
        trace_id:
            本次执行批次对应的链路编号。

    返回值：
        LongTaskStep：状态为 failed、版本递增 1 的新步骤快照。
    """

    transitioned_step = transition_long_task_step(
        step,
        target_status="failed",
    )
    step_data = transitioned_step.model_dump(mode="python")
    step_data.update(
        {
            "output_summary": failed_result.output_summary,
            "output_ref": failed_result.output_ref,
            "last_error_code": (
                failed_result.error_code or "LONG_TASK_STEP_FAILED"
            ),
            "last_error_message": (
                failed_result.error_message or "长任务步骤执行失败"
            ),
            "last_trace_id": trace_id,
            "last_span_id": failed_result.span_id,
        }
    )
    return LongTaskStep.model_validate(step_data)


def _build_failed_task(
    *,
    task: LongTask,
    failed_step: LongTaskStep,
) -> LongTask:
    """
    用失败步骤生成任务级 failed 新快照。

    参数：
        task:
            提交前的权威运行中任务。
        failed_step:
            已经写入错误事实并递增版本的步骤。

    返回值：
        LongTask：状态为 failed、活动步骤已清空且版本递增 1 的新任务。
    """

    updated_steps = [
        failed_step if step.step_id == failed_step.step_id else step
        for step in task.steps
    ]
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "steps": updated_steps,
            "active_step_ids": [],
        }
    )
    transition_base = LongTask.model_validate(task_data)
    return transition_long_task(
        transition_base,
        target_status="failed",
        active_step_ids=[],
    )
