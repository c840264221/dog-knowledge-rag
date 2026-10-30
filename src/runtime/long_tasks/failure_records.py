"""根据长任务失败决定构建后续持久化记录。"""

from __future__ import annotations

import hashlib

from src.runtime.long_tasks.failure_contracts import (
    LongTaskDeadLetterRecord,
    LongTaskFailureDecision,
    LongTaskRetryPlan,
)


def build_retry_plan(
    *,
    decision: LongTaskFailureDecision,
    failure_task_version: int,
    failure_step_version: int,
    failure_commit_id: str,
) -> LongTaskRetryPlan:
    """
    根据重试决定构建待调度的重试计划。

    功能：
        只接受 action 为 retry 的失败决定，根据决定中的失败次数、
        下次执行时间和来源执行身份创建稳定重试计划。

    参数：
        decision:
            已经完成错误分类、次数和截止时间判断的失败决定。
        failure_task_version:
            失败结果正式提交后对应的任务版本。
        failure_step_version:
            失败结果正式提交后对应的步骤版本。
        failure_commit_id:
            准备原子保存失败结果和重试计划的稳定提交编号。

    返回值：
        LongTaskRetryPlan：
            状态为 scheduled、尚未发布消息的重试计划。

    异常：
        ValueError：
            决定动作不是 retry 或缺少下次执行时间时抛出。
    """

    if decision.action != "retry":
        raise ValueError("只有 retry 决定可以创建重试计划")
    if decision.next_retry_at is None:
        raise ValueError("重试决定缺少 next_retry_at")

    normalized_commit_id = _require_text(
        failure_commit_id,
        "failure_commit_id",
    )
    retry_plan_id = _build_record_id(
        prefix="retry_plan",
        task_id=decision.task_id,
        step_id=decision.step_id,
        failure_commit_id=normalized_commit_id,
    )

    return LongTaskRetryPlan(
        retry_plan_id=retry_plan_id,
        task_id=decision.task_id,
        step_id=decision.step_id,
        failure_task_version=failure_task_version,
        failure_step_version=failure_step_version,
        failed_attempt_count=decision.attempt_count,
        next_attempt_count=decision.attempt_count + 1,
        max_attempts=decision.max_attempts,
        scheduled_at=decision.next_retry_at,
        status="scheduled",
        error_code=decision.error_code,
        failure_commit_id=normalized_commit_id,
        source_trace_id=decision.source_trace_id,
        source_span_id=decision.source_span_id,
        source_claim_id=decision.source_claim_id,
        created_at=decision.decided_at,
    )


def build_dead_letter_record(
    *,
    decision: LongTaskFailureDecision,
    failure_task_version: int,
    failure_step_version: int,
    failure_commit_id: str,
    payload_reference: str | None = None,
    worker_id: str | None = None,
) -> LongTaskDeadLetterRecord:
    """
    根据死信决定构建不可修改的死信记录。

    功能：
        只接受 action 为 dead_letter 的失败决定，并复制最后失败的错误、
        次数、版本、Trace、Span 和 Claim，形成供人工处置引用的事实。

    参数：
        decision:
            已经确认需要进入死信的失败决定。
        failure_task_version:
            死信与失败结果正式提交后对应的任务版本。
        failure_step_version:
            死信与失败结果正式提交后对应的步骤版本。
        failure_commit_id:
            准备原子保存失败结果和死信的稳定提交编号。
        payload_reference:
            可选的大型原始输入引用。
        worker_id:
            可选的最后执行 Worker 身份。

    返回值：
        LongTaskDeadLetterRecord：
            保留最后失败事实、可以供后续人工处置引用的死信记录。

    异常：
        ValueError：
            决定动作不是 dead_letter 或提交编号为空时抛出。
    """

    if decision.action != "dead_letter":
        raise ValueError("只有 dead_letter 决定可以创建死信记录")

    normalized_commit_id = _require_text(
        failure_commit_id,
        "failure_commit_id",
    )
    dead_letter_id = _build_record_id(
        prefix="dead_letter",
        task_id=decision.task_id,
        step_id=decision.step_id,
        failure_commit_id=normalized_commit_id,
    )

    return LongTaskDeadLetterRecord(
        dead_letter_id=dead_letter_id,
        task_id=decision.task_id,
        step_id=decision.step_id,
        failure_task_version=failure_task_version,
        failure_step_version=failure_step_version,
        failed_attempt_count=decision.attempt_count,
        max_attempts=decision.max_attempts,
        failure_category=decision.failure_category,
        error_code=decision.error_code,
        error_message=decision.error_message,
        dead_letter_reason=decision.reason,
        failure_commit_id=normalized_commit_id,
        payload_reference=payload_reference,
        worker_id=worker_id,
        source_trace_id=decision.source_trace_id,
        source_span_id=decision.source_span_id,
        source_claim_id=decision.source_claim_id,
        created_at=decision.decided_at,
    )


def _build_record_id(
    *,
    prefix: str,
    task_id: str,
    step_id: str,
    failure_commit_id: str,
) -> str:
    """
    根据稳定业务身份生成失败治理记录编号。

    功能：
        使用记录类型、任务、步骤和失败提交编号生成 Hash，保证同一次
        逻辑失败重复构造时得到相同编号。

    参数：
        prefix:
            记录编号前缀，例如 retry_plan 或 dead_letter。
        task_id/step_id:
            当前失败所属的任务和步骤编号。
        failure_commit_id:
            本次失败的稳定逻辑提交编号。

    返回值：
        str：带记录类型前缀的稳定编号。
    """

    normalized_prefix = _require_text(prefix, "prefix")
    normalized_task_id = _require_text(task_id, "task_id")
    normalized_step_id = _require_text(step_id, "step_id")
    normalized_commit_id = _require_text(
        failure_commit_id,
        "failure_commit_id",
    )
    identity = ":".join(
        (
            normalized_prefix,
            normalized_task_id,
            normalized_step_id,
            normalized_commit_id,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{normalized_prefix}_{digest[:20]}"


def _require_text(value: str, field_name: str) -> str:
    """
    清理并检查失败记录构造所需的文本。

    参数：
        value:
            需要清理首尾空格的原始文本。
        field_name:
            数据不合法时显示的字段名称。

    返回值：
        str：清理后的非空文本。

    异常：
        ValueError：文本为空或只包含空格时抛出。
    """

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized
