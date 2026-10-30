"""长任务失败治理的数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskCommitReceipt,
    LongTaskCommitRequest,
    LongTaskContractModel,
    LongTaskFailureCategory,
    utc_now,
)


LongTaskFailureAction = Literal[
    "retry",
    "dead_letter",
    "fail_task",
]
LongTaskRetryPlanStatus = Literal[
    "scheduled",
    "dispatching",
    "dispatched",
    "cancelled",
]


class LongTaskFailureDecision(LongTaskContractModel):
    """
    保存一次步骤失败后的处理决定。

    功能：
        把错误分类、当前执行次数和失败处理动作整理成稳定数据，
        供 Runtime Driver 后续创建重试计划、死信记录或结束任务。

    字段说明：
        task_id:
            本次失败所属的长任务编号。
        step_id:
            本次失败所属的步骤编号。
        failure_category:
            错误类别。transient 表示临时错误；permanent 表示永久技术错误；
            business 表示业务条件不允许继续；unknown 表示暂时无法准确分类。
        action:
            本次决定采取的动作。retry 表示安排重试；dead_letter 表示进入
            死信；fail_task 表示不进入死信，直接按失败结束任务。
        attempt_count:
            刚刚失败的是第几次执行，从 1 开始。
        max_attempts:
            当前步骤允许的最大执行次数。
        error_code:
            供程序判断和统计使用的稳定错误代码。
        error_message:
            方便开发者和操作人员查看的错误说明。
        reason:
            为什么选择当前处理动作的中文说明。
        next_retry_at:
            下次允许执行的时间；只有 action 为 retry 时才能有值。
        source_trace_id/source_span_id:
            产生本次失败结果的执行链路和具体步骤编号。
        source_claim_id:
            产生本次失败结果时 Worker 持有的领取令牌。
        decided_at:
            本次失败决定生成的 UTC 时间。

    返回值：
        LongTaskFailureDecision：
            字段关系已经过校验、可以交给后续失败治理服务使用的决定。
    """

    task_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    failure_category: LongTaskFailureCategory
    action: LongTaskFailureAction
    attempt_count: int = Field(..., ge=1)
    max_attempts: int = Field(..., ge=1)
    error_code: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    next_retry_at: datetime | None = None
    source_trace_id: str | None = Field(default=None, min_length=1)
    source_span_id: str | None = Field(default=None, min_length=1)
    source_claim_id: str | None = Field(default=None, min_length=1)
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_failure_decision(self) -> Self:
        """
        检查执行次数、重试时间和 Trace 引用是否互相一致。

        功能：
            防止产生“已经达到最大次数却仍安排重试”、
            “安排重试但没有执行时间”或“只有 Span 没有 Trace”等
            无法继续处理的失败决定。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskFailureDecision：字段关系合法时返回当前模型。

        异常：
            ValueError：字段之间存在矛盾时抛出。
        """

        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count 不能超过 max_attempts")

        if self.action == "retry":
            if self.attempt_count >= self.max_attempts:
                raise ValueError("达到最大执行次数后不能继续安排重试")
            if self.next_retry_at is None:
                raise ValueError("重试决定必须提供 next_retry_at")
        elif self.next_retry_at is not None:
            raise ValueError("非重试决定不能提供 next_retry_at")

        for field_name, value in (
            ("next_retry_at", self.next_retry_at),
            ("decided_at", self.decided_at),
        ):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() is None
            ):
                raise ValueError(f"{field_name} 必须包含时区")

        if (
            self.next_retry_at is not None
            and self.next_retry_at < self.decided_at
        ):
            raise ValueError("next_retry_at 不能早于 decided_at")

        if self.source_span_id is not None and self.source_trace_id is None:
            raise ValueError("source_span_id 必须关联 source_trace_id")

        return self


class LongTaskRetryPlan(LongTaskContractModel):
    """
    保存一次已经确定的未来重试安排。

    功能：
        记录哪个步骤因哪次失败需要在什么时间再次执行，并保存计划从
        待调度到已发布或已取消的状态。计划只保存安排，不代表 Worker
        已经领取或开始执行。

    字段说明：
        retry_plan_id:
            重试计划的稳定唯一编号。
        task_id/step_id:
            计划所属的任务和步骤编号。
        failure_task_version/failure_step_version:
            失败结果正式保存后对应的任务和步骤版本，用于后续检查陈旧计划。
        failed_attempt_count:
            产生当前计划的失败执行次数。
        next_attempt_count:
            计划发布后准备进行的下一次执行次数。
        max_attempts:
            当前步骤允许的最大执行次数。
        scheduled_at:
            计划最早允许发布执行消息的 UTC 时间。
        status:
            scheduled 表示等待调度；dispatching 表示某个调度器正在发布；
            dispatched 表示消息已经发布；cancelled 表示计划已取消。
        error_code:
            触发本次重试计划的稳定错误代码。
        failure_commit_id:
            原子保存失败结果和当前计划时使用的提交编号。
        source_trace_id/source_span_id/source_claim_id:
            产生计划的失败执行链路、具体操作和领取令牌。
        queue_message_id/dispatched_at:
            计划成功发布后得到的消息编号和实际发布时间。
        dispatcher_id/dispatch_claim_id/dispatch_lease_expires_at:
            正在发布消息的调度器身份、发布权令牌和租约到期时间。
        cancellation_reason/cancelled_at:
            计划取消时记录的具体原因和 UTC 时间。cancelled_at 保持可选，
            用于兼容已经保存的旧计划；新的取消操作会写入该字段。
        created_at:
            当前计划的创建时间。

    返回值：
        LongTaskRetryPlan：
            字段关系合法、可以持久化和调度的重试计划。
    """

    retry_plan_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    failure_task_version: int = Field(..., ge=1)
    failure_step_version: int = Field(..., ge=1)
    failed_attempt_count: int = Field(..., ge=1)
    next_attempt_count: int = Field(..., ge=2)
    max_attempts: int = Field(..., ge=1)
    scheduled_at: datetime
    status: LongTaskRetryPlanStatus = "scheduled"
    error_code: str = Field(..., min_length=1)
    failure_commit_id: str = Field(..., min_length=1)
    source_trace_id: str | None = Field(default=None, min_length=1)
    source_span_id: str | None = Field(default=None, min_length=1)
    source_claim_id: str | None = Field(default=None, min_length=1)
    dispatcher_id: str | None = Field(default=None, min_length=1)
    dispatch_claim_id: str | None = Field(default=None, min_length=1)
    dispatch_lease_expires_at: datetime | None = None
    queue_message_id: str | None = Field(default=None, min_length=1)
    dispatched_at: datetime | None = None
    cancellation_reason: str | None = Field(default=None, min_length=1)
    cancelled_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_retry_plan(self) -> Self:
        """
        检查执行次数、调度时间和计划状态字段是否互相一致。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskRetryPlan：字段关系合法时返回当前计划。

        异常：
            ValueError：计划包含矛盾次数、时间、Trace 或状态时抛出。
        """

        if self.failed_attempt_count >= self.max_attempts:
            raise ValueError("执行次数耗尽后不能创建重试计划")
        if self.next_attempt_count != self.failed_attempt_count + 1:
            raise ValueError("next_attempt_count 必须是失败次数加 1")
        if self.next_attempt_count > self.max_attempts:
            raise ValueError("下一次执行不能超过最大执行次数")

        for field_name, value in (
            ("scheduled_at", self.scheduled_at),
            ("dispatch_lease_expires_at", self.dispatch_lease_expires_at),
            ("dispatched_at", self.dispatched_at),
            ("cancelled_at", self.cancelled_at),
            ("created_at", self.created_at),
        ):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() is None
            ):
                raise ValueError(f"{field_name} 必须包含时区")
        if self.scheduled_at < self.created_at:
            raise ValueError("scheduled_at 不能早于 created_at")

        if self.source_span_id is not None and self.source_trace_id is None:
            raise ValueError("source_span_id 必须关联 source_trace_id")

        dispatch_claim_values = (
            self.dispatcher_id,
            self.dispatch_claim_id,
            self.dispatch_lease_expires_at,
        )
        if any(value is not None for value in dispatch_claim_values) and not all(
            value is not None for value in dispatch_claim_values
        ):
            raise ValueError("重试计划发布权信息必须同时提供")
        if self.status == "dispatching":
            if not all(value is not None for value in dispatch_claim_values):
                raise ValueError("正在发布的计划必须持有发布权租约")
            if self.dispatch_lease_expires_at < self.scheduled_at:
                raise ValueError("发布权租约不能早于计划执行时间到期")
        elif any(value is not None for value in dispatch_claim_values):
            raise ValueError("只有 dispatching 计划可以持有发布权租约")

        if self.status == "scheduled":
            if (
                self.queue_message_id is not None
                or self.dispatched_at is not None
            ):
                raise ValueError("待调度计划不能包含消息编号或发布时间")
            if self.cancellation_reason is not None:
                raise ValueError("待调度计划不能包含取消原因")
            if self.cancelled_at is not None:
                raise ValueError("待调度计划不能包含取消时间")
        elif self.status == "dispatching":
            if (
                self.queue_message_id is not None
                or self.dispatched_at is not None
            ):
                raise ValueError("正在发布的计划不能提前包含消息结果")
            if self.cancellation_reason is not None:
                raise ValueError("正在发布的计划不能包含取消原因")
            if self.cancelled_at is not None:
                raise ValueError("正在发布的计划不能包含取消时间")
        elif self.status == "dispatched":
            if self.queue_message_id is None or self.dispatched_at is None:
                raise ValueError("已发布计划必须包含消息编号和发布时间")
            if self.dispatched_at < self.scheduled_at:
                raise ValueError("计划不能早于 scheduled_at 发布")
            if self.cancellation_reason is not None:
                raise ValueError("已发布计划不能包含取消原因")
            if self.cancelled_at is not None:
                raise ValueError("已发布计划不能包含取消时间")
        else:
            if self.cancellation_reason is None:
                raise ValueError("已取消计划必须提供 cancellation_reason")
            if (
                self.queue_message_id is not None
                or self.dispatched_at is not None
            ):
                raise ValueError("已取消计划不能包含消息编号或发布时间")
            if (
                self.cancelled_at is not None
                and self.cancelled_at < self.created_at
            ):
                raise ValueError("cancelled_at 不能早于 created_at")

        return self


class LongTaskDeadLetterRecord(LongTaskContractModel):
    """
    保存一个步骤进入死信时不可修改的失败事实。

    功能：
        记录步骤最后被接受的失败身份、错误、执行次数和当时版本，
        供人工排查和重新驱动使用。后续处置只能新增记录，不能覆盖它。

    字段说明：
        dead_letter_id:
            死信记录的稳定唯一编号。
        task_id/step_id:
            死信所属的任务和步骤编号。
        failure_task_version/failure_step_version:
            失败结果正式保存后对应的任务和步骤版本。
        failed_attempt_count/max_attempts:
            进入死信时的执行次数和最大允许执行次数。
        failure_category:
            当时确认的错误类别。
        error_code/error_message:
            稳定错误代码和供人查看的错误说明。
        dead_letter_reason:
            为什么不再自动执行并转入人工处理。
        failure_commit_id:
            原子保存失败结果和死信时使用的提交编号。
        payload_reference:
            可选的大型原始输入引用；不直接把大内容复制进死信。
        worker_id:
            产生最后失败结果的 Worker 身份。
        source_trace_id/source_span_id/source_claim_id:
            最后失败执行的链路、具体操作和领取令牌。
        created_at:
            死信记录创建时间。

    返回值：
        LongTaskDeadLetterRecord：
            可以持久化、审计和供重新驱动引用的原始死信事实。
    """

    dead_letter_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    failure_task_version: int = Field(..., ge=1)
    failure_step_version: int = Field(..., ge=1)
    failed_attempt_count: int = Field(..., ge=1)
    max_attempts: int = Field(..., ge=1)
    failure_category: LongTaskFailureCategory
    error_code: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)
    dead_letter_reason: str = Field(..., min_length=1)
    failure_commit_id: str = Field(..., min_length=1)
    payload_reference: str | None = Field(default=None, min_length=1)
    worker_id: str | None = Field(default=None, min_length=1)
    source_trace_id: str | None = Field(default=None, min_length=1)
    source_span_id: str | None = Field(default=None, min_length=1)
    source_claim_id: str | None = Field(default=None, min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_dead_letter_record(self) -> Self:
        """
        检查死信中的执行次数、时间和 Trace 引用是否有效。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskDeadLetterRecord：字段关系合法时返回当前死信。

        异常：
            ValueError：执行次数、时间或 Trace 引用不合法时抛出。
        """

        if self.failed_attempt_count > self.max_attempts:
            raise ValueError("failed_attempt_count 不能超过 max_attempts")
        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("created_at 必须包含时区")
        if self.source_span_id is not None and self.source_trace_id is None:
            raise ValueError("source_span_id 必须关联 source_trace_id")
        return self


class LongTaskFailureCommitPreparation(LongTaskContractModel):
    """
    保存一次失败结果准备原子提交时需要的完整数据。

    功能：
        把提交前版本、原始失败结果、失败后的任务快照、处理决定以及
        可选的重试计划或死信记录放在同一个对象中。这个对象只表示
        “已经准备好提交”，不表示数据已经写入 Redis。

    字段说明：
        failure_commit_id:
            本次失败提交的稳定编号，重复处理同一逻辑失败时应保持一致。
        expected_task_version:
            提交前读取到的权威任务版本，后续原子保存用它检查版本冲突。
        previous_task:
            Worker 提交失败结果前读取到的真实权威任务快照。
        batch_result:
            Worker 返回的原始批次执行结果，本阶段只支持一个失败步骤。
        failed_task:
            已写入错误事实、清除旧领取权并递增版本的新任务快照。
        decision:
            根据错误类别、执行次数和截止时间得到的处理决定。
        retry_plan:
            决定重试时生成的未来执行计划，其他动作必须为空。
        dead_letter_record:
            决定进入死信时生成的原始失败记录，其他动作必须为空。

    返回值：
        LongTaskFailureCommitPreparation：
            字段关系一致、可以交给后续原子存储层的失败提交准备结果。
    """

    failure_commit_id: str = Field(..., min_length=1)
    expected_task_version: int = Field(..., ge=1)
    previous_task: LongTask
    batch_result: LongTaskBatchResult
    failed_task: LongTask
    decision: LongTaskFailureDecision
    retry_plan: LongTaskRetryPlan | None = None
    dead_letter_record: LongTaskDeadLetterRecord | None = None

    @model_validator(mode="after")
    def validate_failure_commit_preparation(self) -> Self:
        """
        检查失败快照、决定和后续记录是否属于同一次失败提交。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskFailureCommitPreparation：字段关系合法时返回当前对象。

        异常：
            ValueError：任务编号、版本或处理动作之间存在矛盾时抛出。
        """

        if self.failed_task.version != self.expected_task_version + 1:
            raise ValueError("失败后任务版本必须是提交前版本加 1")
        if self.previous_task.version != self.expected_task_version:
            raise ValueError("提交前任务版本与 expected_task_version 不一致")
        if self.previous_task.task_id != self.failed_task.task_id:
            raise ValueError("提交前后任务的 task_id 不一致")
        if self.batch_result.task_id != self.failed_task.task_id:
            raise ValueError("批次结果与失败后任务的 task_id 不一致")
        if self.decision.task_id != self.failed_task.task_id:
            raise ValueError("失败决定与失败后任务的 task_id 不一致")
        if self.decision.step_id not in {
            result.step_id for result in self.batch_result.step_results
        }:
            raise ValueError("失败决定没有对应的批次步骤结果")

        failed_steps = {
            step.step_id: step
            for step in self.failed_task.steps
            if step.status == "failed"
        }
        failed_step = failed_steps.get(self.decision.step_id)
        if failed_step is None:
            raise ValueError("失败决定对应的步骤没有进入 failed 状态")

        if self.decision.action == "retry":
            if self.retry_plan is None:
                raise ValueError("retry 决定必须包含重试计划")
            if self.dead_letter_record is not None:
                raise ValueError("retry 决定不能包含死信记录")
        elif self.decision.action == "dead_letter":
            if self.dead_letter_record is None:
                raise ValueError("dead_letter 决定必须包含死信记录")
            if self.retry_plan is not None:
                raise ValueError("dead_letter 决定不能包含重试计划")
        elif self.retry_plan is not None or self.dead_letter_record is not None:
            raise ValueError("fail_task 决定不能包含重试计划或死信记录")

        if self.retry_plan is not None:
            if self.retry_plan.failure_commit_id != self.failure_commit_id:
                raise ValueError("重试计划的 failure_commit_id 不一致")
            if (
                self.retry_plan.failure_task_version
                != self.failed_task.version
                or self.retry_plan.failure_step_version
                != failed_step.version
            ):
                raise ValueError("重试计划没有引用失败后的最新版本")

        if self.dead_letter_record is not None:
            if (
                self.dead_letter_record.failure_commit_id
                != self.failure_commit_id
            ):
                raise ValueError("死信记录的 failure_commit_id 不一致")
            if (
                self.dead_letter_record.failure_task_version
                != self.failed_task.version
                or self.dead_letter_record.failure_step_version
                != failed_step.version
            ):
                raise ValueError("死信记录没有引用失败后的最新版本")

        return self


class LongTaskFailureCommitRequest(LongTaskCommitRequest):
    """
    描述一次需要原子落库的长任务失败提交。

    功能：
        在普通可靠提交的任务快照、事件和幂等指纹基础上，增加失败动作、
        重试计划或死信记录，使存储层能够在同一次 Redis 操作中保存。

    字段说明：
        failure_action:
            本次失败最终选择的 retry、dead_letter 或 fail_task 动作。
        retry_plan:
            retry 动作对应的计划；其他动作必须为空。
        dead_letter_record:
            dead_letter 动作对应的死信记录；其他动作必须为空。

    返回值：
        LongTaskFailureCommitRequest：
            可以交给失败提交 Store 原子执行的完整请求。
    """

    failure_action: LongTaskFailureAction
    retry_plan: LongTaskRetryPlan | None = None
    dead_letter_record: LongTaskDeadLetterRecord | None = None

    @model_validator(mode="after")
    def validate_failure_commit_request(self) -> Self:
        """
        检查失败动作、版本和可选记录属于同一次可靠提交。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskFailureCommitRequest：字段关系合法时返回当前请求。

        异常：
            ValueError：提交类型、消息或失败记录之间存在矛盾时抛出。
        """

        if self.operation != "update":
            raise ValueError("失败提交只能更新已经存在的任务")
        if self.queue_message is not None:
            raise ValueError("失败提交不能提前发布重试消息")

        if self.failure_action == "retry":
            if self.retry_plan is None:
                raise ValueError("retry 失败提交必须包含重试计划")
            if self.dead_letter_record is not None:
                raise ValueError("retry 失败提交不能包含死信记录")
        elif self.failure_action == "dead_letter":
            if self.dead_letter_record is None:
                raise ValueError("dead_letter 失败提交必须包含死信记录")
            if self.retry_plan is not None:
                raise ValueError("dead_letter 失败提交不能包含重试计划")
        elif self.retry_plan is not None or self.dead_letter_record is not None:
            raise ValueError("fail_task 失败提交不能包含计划或死信")

        failed_steps = {
            step.step_id: step
            for step in self.task.steps
            if step.status == "failed"
        }
        record = self.retry_plan or self.dead_letter_record
        if record is not None:
            failed_step = failed_steps.get(record.step_id)
            if failed_step is None:
                raise ValueError("失败记录没有对应的 failed 步骤")
            if record.task_id != self.task.task_id:
                raise ValueError("失败记录与提交任务的 task_id 不一致")
            if record.failure_commit_id != self.commit_id:
                raise ValueError("失败记录与提交请求的 commit_id 不一致")
            if (
                record.failure_task_version != self.task.version
                or record.failure_step_version != failed_step.version
            ):
                raise ValueError("失败记录没有引用本次提交后的最新版本")
        return self


class LongTaskRetryDispatchRequest(LongTaskCommitRequest):
    """
    描述一次领取成功后需要原子完成的重试消息发布提交。

    功能：
        保存恢复为 queued/ready 的任务快照，并要求存储层使用发布权令牌
        校验当前调度器、写入重试消息、完成计划和清理时间索引。

    字段说明：
        retry_plan_id/step_id:
            本次准备完成发布的重试计划和目标步骤编号。
        dispatcher_id/dispatch_claim_id:
            当前持有发布权的调度器身份和令牌。
        failure_task_version/failure_step_version:
            计划创建时对应的失败任务和失败步骤版本。
        dispatched_at:
            本次发布提交使用的 UTC 时间。

    返回值：
        LongTaskRetryDispatchRequest：
            可以交给 Store 原子完成重试消息发布的请求。
    """

    retry_plan_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    dispatcher_id: str = Field(..., min_length=1)
    dispatch_claim_id: str = Field(..., min_length=1)
    failure_task_version: int = Field(..., ge=1)
    failure_step_version: int = Field(..., ge=1)
    dispatched_at: datetime

    @model_validator(mode="after")
    def validate_retry_dispatch_request(self) -> Self:
        """
        检查状态版本、目标步骤和重试消息属于同一次发布提交。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskRetryDispatchRequest：字段关系合法时返回当前请求。

        异常：
            ValueError：任务、步骤、消息或时间不符合发布条件时抛出。
        """

        if self.operation != "update":
            raise ValueError("重试发布只能更新已经存在的任务")
        if self.expected_version != self.failure_task_version:
            raise ValueError("重试发布必须基于计划记录的失败任务版本")
        if self.task.status != "queued":
            raise ValueError("重试发布后的任务必须处于 queued 状态")
        target_step = next(
            (step for step in self.task.steps if step.step_id == self.step_id),
            None,
        )
        if target_step is None:
            raise ValueError("重试发布引用了不存在的步骤")
        if target_step.status != "ready":
            raise ValueError("重试发布后的目标步骤必须处于 ready 状态")
        if target_step.version != self.failure_step_version + 1:
            raise ValueError("重试发布后的步骤版本必须是失败版本加 1")

        message = self.queue_message
        if message is None:
            raise ValueError("重试发布必须包含队列消息")
        if message.reason != "retry":
            raise ValueError("重试发布消息的 reason 必须为 retry")
        if message.ready_step_ids != [self.step_id]:
            raise ValueError("重试发布消息必须只提示当前目标步骤")
        if (
            self.dispatched_at.tzinfo is None
            or self.dispatched_at.utcoffset() is None
        ):
            raise ValueError("dispatched_at 必须包含时区")
        return self


class LongTaskRetryCancellationRequest(LongTaskContractModel):
    """
    描述一次已经领取发布权的失效重试计划取消请求。

    功能：
        携带计划、调度器和发布权令牌，让存储层只允许当前发布权持有者
        原子取消计划，同时移除时间索引并释放短期发布锁。

    字段说明：
        retry_plan_id/task_id/step_id:
            需要取消的计划及其所属任务、步骤编号。
        dispatcher_id/dispatch_claim_id:
            当前持有计划发布权的调度器身份和令牌。
        cancellation_reason:
            根据最新任务状态确认计划不能继续发布的具体原因。
        cancelled_at:
            调度器作出取消决定的 UTC 时间。

    返回值：
        LongTaskRetryCancellationRequest：
            可以交给 Store 原子取消重试计划的完整请求。
    """

    retry_plan_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    dispatcher_id: str = Field(..., min_length=1)
    dispatch_claim_id: str = Field(..., min_length=1)
    cancellation_reason: str = Field(..., min_length=1)
    cancelled_at: datetime

    @model_validator(mode="after")
    def validate_retry_cancellation_request(self) -> Self:
        """
        检查取消时间包含有效时区。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskRetryCancellationRequest：字段合法时返回当前请求。
        """

        if (
            self.cancelled_at.tzinfo is None
            or self.cancelled_at.utcoffset() is None
        ):
            raise ValueError("cancelled_at 必须包含时区")
        return self


class LongTaskRetryDispatchCycleResult(LongTaskContractModel):
    """
    汇总重试调度器执行一轮后的处理结果。

    字段说明：
        due_plan_ids:
            本轮从时间索引读取到的全部到期计划编号。
        claimed_plan_ids:
            本轮成功获得发布权的计划编号。
        dispatched_plan_ids:
            本轮已经成功发布消息并完成计划收尾的编号。
        cancelled_plan_ids/cancellation_reasons:
            本轮确认已经失效并完成取消的计划编号及具体原因。
        skipped_plan_ids:
            因未抢到发布权或计划状态变化而跳过的编号。
        failed_reasons:
            本轮处理失败的计划编号及简短原因。
        receipts:
            每个成功发布计划对应的可靠提交回执。

    返回值：
        LongTaskRetryDispatchCycleResult：一次调度执行的结构化汇总。
    """

    due_plan_ids: list[str] = Field(default_factory=list)
    claimed_plan_ids: list[str] = Field(default_factory=list)
    dispatched_plan_ids: list[str] = Field(default_factory=list)
    cancelled_plan_ids: list[str] = Field(default_factory=list)
    cancellation_reasons: dict[str, str] = Field(default_factory=dict)
    skipped_plan_ids: list[str] = Field(default_factory=list)
    failed_reasons: dict[str, str] = Field(default_factory=dict)
    receipts: list[LongTaskCommitReceipt] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retry_dispatch_cycle_result(self) -> Self:
        """
        检查各类计划编号都来自本轮到期集合且没有重复。

        参数：
            无。Pydantic 会把当前模型作为 self 传入。

        返回值：
            LongTaskRetryDispatchCycleResult：字段关系合法时返回当前结果。
        """

        due_ids = set(self.due_plan_ids)
        for field_name, values in (
            ("due_plan_ids", self.due_plan_ids),
            ("claimed_plan_ids", self.claimed_plan_ids),
            ("dispatched_plan_ids", self.dispatched_plan_ids),
            ("cancelled_plan_ids", self.cancelled_plan_ids),
            ("skipped_plan_ids", self.skipped_plan_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} 不能包含重复计划编号")
        if not set(self.claimed_plan_ids).issubset(due_ids):
            raise ValueError("claimed_plan_ids 必须来自 due_plan_ids")
        if not set(self.dispatched_plan_ids).issubset(
            set(self.claimed_plan_ids)
        ):
            raise ValueError("dispatched_plan_ids 必须已经领取成功")
        if not set(self.cancelled_plan_ids).issubset(
            set(self.claimed_plan_ids)
        ):
            raise ValueError("cancelled_plan_ids 必须已经领取成功")
        if set(self.cancellation_reasons) != set(self.cancelled_plan_ids):
            raise ValueError("cancellation_reasons 必须对应全部已取消计划")
        if not set(self.skipped_plan_ids).issubset(due_ids):
            raise ValueError("skipped_plan_ids 必须来自 due_plan_ids")
        if not set(self.failed_reasons).issubset(due_ids):
            raise ValueError("failed_reasons 必须来自 due_plan_ids")
        if set(self.skipped_plan_ids) & set(self.claimed_plan_ids):
            raise ValueError("已领取计划不能同时标记为 skipped")
        if set(self.dispatched_plan_ids) & set(self.cancelled_plan_ids):
            raise ValueError("同一计划不能同时标记为 dispatched 和 cancelled")
        return self
