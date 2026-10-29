from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class TraceReference(BaseModel):
    """保存可以跨组件传播的 Trace 引用。

    字段：
        trace_id：整条逻辑调用链共享的追踪 ID。
        span_id：当前处理阶段自己的追踪片段 ID。
        parent_span_id：父级处理阶段的 Span ID；根阶段可以为 None。

    说明：
        本模型只保存关联 ID，不保存日志、耗时或完整 Span 数据。
    """

    trace_id: str = Field(..., min_length=1)
    span_id: str = Field(..., min_length=1)
    parent_span_id: str | None = Field(default=None, min_length=1)

def propagate_api_trace_to_task(
    api_trace_reference: TraceReference,
    task_span_id: str,
) -> TraceReference:
    """把 API 阶段的 Trace 引用传播到新建任务。

    参数：
        api_trace_reference：API 请求当前持有的 Trace 引用。
        task_span_id：为任务阶段准备的新 Span ID。
    返回：
        返回任务阶段的 TraceReference。
        trace_id 与 API 保持一致；
        span_id 使用 task_span_id；
        parent_span_id 指向 API 的 span_id。
    说明：
        本函数只建立追踪引用关系，不修改 API 引用，
        不保存完整 Trace，也不在内部随机生成 ID。
    """
    return create_child_trace_reference(api_trace_reference, task_span_id)


def propagate_task_trace_to_stream_message(
    task_trace_reference: TraceReference,
    stream_span_id: str,
) -> TraceReference:
    """把任务阶段的 Trace 引用传播到 Stream 消息。

    参数：
        task_trace_reference：任务阶段当前持有的 Trace 引用。
        stream_span_id：发布 Stream 消息时使用的新 Span ID。
    返回：
        返回随 Stream 消息传递的 TraceReference。
        trace_id 与任务保持一致；
        span_id 使用 stream_span_id；
        parent_span_id 指向任务阶段的 span_id。
    说明：
        发布 Stream 消息是一个新的重要操作，因此使用新的 Span ID。
        本函数只构造需要随消息传递的引用，不真正发布消息。
    """
    return create_child_trace_reference(task_trace_reference,stream_span_id)


def propagate_stream_trace_to_worker(
    stream_trace_reference: TraceReference,
    worker_span_id: str,
) -> TraceReference:
    """把 Stream 消息中的 Trace 引用传播到 Worker 执行阶段。

    参数：
        stream_trace_reference：Stream 消息携带的 Trace 引用。
        worker_span_id：Worker 本次处理消息使用的新 Span ID。
    返回：
        返回 Worker 执行阶段的 TraceReference。
        trace_id 与 Stream 消息保持一致；
        span_id 使用 worker_span_id；
        parent_span_id 指向 Stream 阶段的 span_id。
    说明：
        Worker 开始处理消息是一个新的重要操作，因此使用新的 Span ID。
        worker_span_id 表示本次处理过程，不表示 Worker 的固定身份。
    """
    return create_child_trace_reference(stream_trace_reference,worker_span_id)

class EventTraceReference(BaseModel):
    """保存业务事件与 Worker 执行过程的关联信息。

    字段：
        event_id：当前业务事件自己的唯一 ID。
        trace_id：当前事件所属的完整调用链 ID。
        source_span_id：产生当前事件的 Worker 执行 Span ID。

    说明：
        event_id 用来区分不同事件；
        source_span_id 用来说明事件产生于哪一次 Worker 操作。
        本模型只保存关联 ID，不保存完整事件内容。
    """

    event_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    source_span_id: str = Field(..., min_length=1)

def propagate_worker_trace_to_event(
    worker_trace_reference: TraceReference,
    event_id: str,
) -> EventTraceReference:
    """把 Worker 执行阶段的追踪信息关联到业务事件。

    参数：
        worker_trace_reference：Worker 本次执行使用的 Trace 引用。
        event_id：本次产生的业务事件 ID。
    返回：
        返回事件使用的 EventTraceReference。
        trace_id 继续使用 Worker 的 trace_id；
        source_span_id 指向 Worker 本次执行的 span_id；
        event_id 使用传入的事件 ID。
    说明：
        本函数只建立事件与 Worker 执行过程的关联，
        不保存事件，也不创建新的 Span。
    """
    event_trace_reference = EventTraceReference(
        event_id=event_id,
        trace_id=worker_trace_reference.trace_id,
        source_span_id=worker_trace_reference.span_id,
    )
    return event_trace_reference

def create_child_trace_reference(
    parent_trace_reference: TraceReference,
    child_span_id: str,
) -> TraceReference:
    """根据上一步的追踪信息创建下一步的追踪引用。

    参数：
        parent_trace_reference：上一个操作使用的 Trace 引用。
        child_span_id：下一个操作准备使用的新 Span ID。
    返回：
        返回下一个操作使用的 TraceReference。
        trace_id 与上一步保持一致；
        span_id 使用 child_span_id；
        parent_span_id 指向上一步的 span_id。
    异常：
        child_span_id 为空字符串或全空白时抛出 ValueError；
        child_span_id 与上一步 span_id 相同时抛出 ValueError。
    说明：
        本函数只负责创建上下级关系，不生成随机 ID，
        也不保存完整的追踪数据。
    """
    if not child_span_id.strip():
        raise ValueError("新的 span ID 不能为空白")
    if child_span_id.strip() == parent_trace_reference.span_id.strip():
        raise ValueError("新的 span ID 不能与父级 span ID 相同")
    child_trace_reference = TraceReference(
        trace_id=parent_trace_reference.trace_id,
        span_id=child_span_id,
        parent_span_id=parent_trace_reference.span_id,
    )
    return child_trace_reference

def test_trace引用从API一直传播到Event() -> None:
    """验证同一 Trace 引用能够从 API 传播到最终业务事件。

    参数：无。
    返回：
        无，通过断言检查所有阶段使用相同的 trace_id，
        每个重要操作使用不同的 span_id，
        每一步正确指向上一步，事件正确引用 Worker 的执行记录。
    """
    api_trace_reference = TraceReference(
        trace_id="TRACE-1",
        span_id="API-SPAN-1",
        parent_span_id=None,
    )
    task_trace_reference = propagate_api_trace_to_task(
        api_trace_reference=api_trace_reference,
        task_span_id="TASK-SPAN-1",
    )
    stream_trace_reference = propagate_task_trace_to_stream_message(
        task_trace_reference=task_trace_reference,
        stream_span_id="STREAM-SPAN-1",
    )
    worker_trace_reference = propagate_stream_trace_to_worker(
        stream_trace_reference=stream_trace_reference,
        worker_span_id="WORKER-SPAN-1",
    )
    event_trace_reference = propagate_worker_trace_to_event(
        worker_trace_reference=worker_trace_reference,
        event_id="EVENT-1",
    )
    assert {
               api_trace_reference.trace_id,
               task_trace_reference.trace_id,
               stream_trace_reference.trace_id,
               worker_trace_reference.trace_id,
               event_trace_reference.trace_id,
           } == {"TRACE-1"}
    assert len({
        api_trace_reference.span_id,
        task_trace_reference.span_id,
        stream_trace_reference.span_id,
        worker_trace_reference.span_id,
    }) == 4

    assert task_trace_reference.parent_span_id == api_trace_reference.span_id
    assert stream_trace_reference.parent_span_id == task_trace_reference.span_id
    assert worker_trace_reference.parent_span_id == stream_trace_reference.span_id
    assert event_trace_reference.event_id == "EVENT-1"
    assert event_trace_reference.source_span_id == worker_trace_reference.span_id

class RetryExecutionReference(BaseModel):
    task_id: str
    correlation_id: str
    trace_id: str
    claim_id: str
    previous_trace_id: str
    attempt: int = Field(ge=1)

class RetryScheduleReference(BaseModel):
    task_id: str
    correlation_id: str
    previous_trace_id: str
    next_attempt: int
    scheduled_at: datetime


def create_retry_schedule_reference(
    failed_execution_reference: RetryExecutionReference,
    next_attempt: int,
    scheduled_at: datetime,
) -> RetryScheduleReference:
    """
    根据失败的执行记录创建自动重试计划。

    参数：
        failed_execution_reference:
            本次失败执行的引用信息，用于取得任务编号、
            业务关联编号和本次失败的追踪编号。
        next_attempt:
            下一次执行的次数，例如本次是第 1 次执行，
            下一次应为第 2 次执行。
        scheduled_at:
            下一次允许开始执行的时间。

    返回：
        自动重试计划引用。
        其中保留原任务编号和业务关联编号，
        previous_trace_id 记录本次失败执行的 trace_id。
        此时不创建下一次执行的 trace_id 和 claim_id。
        next_attempt 必须等于失败执行的 attempt 加 1，
        防止执行次数重复或跳号。
    """
    if next_attempt <= 0:
        raise ValueError("next_attempt必须大于0")
    expected_next_attempt = failed_execution_reference.attempt + 1

    if next_attempt != expected_next_attempt:
        raise ValueError("下一次执行次数必须在本次执行次数的基础上增加1")

    previous_trace_id = failed_execution_reference.trace_id
    retry_schedule_reference = RetryScheduleReference(
        task_id=failed_execution_reference.task_id,
        correlation_id=failed_execution_reference.correlation_id,
        next_attempt=next_attempt,
        scheduled_at=scheduled_at,
        previous_trace_id=previous_trace_id,
    )
    return retry_schedule_reference

class RetryDispatchMessage(BaseModel):
    task_id: str
    correlation_id: str
    trace_id: str
    previous_trace_id: str
    attempt: int
    source_span_id: str


def create_retry_dispatch_message(
    retry_schedule_reference: RetryScheduleReference,
    new_trace_id: str,
    dispatch_span_id: str,
) -> RetryDispatchMessage:
    """
    根据已经到期的重试计划创建待发布的重试消息。

    参数：
        retry_schedule_reference:
            已经到达执行时间的重试计划，包含任务编号、
            业务关联编号、上一次执行的追踪编号和执行次数。
        new_trace_id:
            本次重试执行新生成的追踪编号。
        dispatch_span_id:
            调度器发布这条重试消息时使用的操作编号。

    返回：
        待发布的重试消息。
        trace_id 表示本次重试执行，
        source_span_id 表示发布消息的操作，
        previous_trace_id 指向上一次失败执行。
        此时 Worker 尚未领取消息，因此不包含 claim_id。
    """
    if not new_trace_id.strip():
        raise ValueError("新trace_id不能为空")
    source_span_id = dispatch_span_id.strip()
    if not source_span_id:
        raise ValueError("dispatch_span_id不能为空")

    previous_trace_id = retry_schedule_reference.previous_trace_id.strip()
    if new_trace_id.strip() == previous_trace_id:
        raise ValueError("新旧trace_id不能相同")

    retry_dispatch_message = RetryDispatchMessage(
        task_id=retry_schedule_reference.task_id,
        correlation_id=retry_schedule_reference.correlation_id,
        trace_id=new_trace_id,
        previous_trace_id = previous_trace_id,
        attempt=retry_schedule_reference.next_attempt,
        source_span_id=source_span_id,
    )
    return retry_dispatch_message

def claim_retry_dispatch_message(
    retry_dispatch_message: RetryDispatchMessage,
    new_claim_id: str,
) -> RetryExecutionReference:
    """
    根据重试消息创建 Worker 已领取的执行引用。

    参数：
        retry_dispatch_message:
            调度器发布的重试消息，包含本次执行的追踪编号、
            上一次执行的追踪编号、任务编号和业务关联编号。
        new_claim_id:
            Worker 获得执行权后生成的新领取令牌。

    返回：
        Worker 已经获得执行权的重试执行引用。
        task_id、correlation_id、trace_id 和 previous_trace_id
        都从重试消息中继承，claim_id 使用本次领取的新编号。
        attempt 继承重试消息中的执行次数。
    """
    if not new_claim_id.strip():
        raise ValueError("新令牌不能为空")

    retry_execution_reference = RetryExecutionReference(
        task_id=retry_dispatch_message.task_id,
        correlation_id=retry_dispatch_message.correlation_id,
        trace_id=retry_dispatch_message.trace_id,
        claim_id=new_claim_id,
        previous_trace_id=retry_dispatch_message.previous_trace_id,
        attempt=retry_dispatch_message.attempt,
    )
    return retry_execution_reference

def create_worker_trace_from_retry_message(
    retry_dispatch_message: RetryDispatchMessage,
    worker_span_id: str,
) -> TraceReference:
    """
    根据重试消息创建 Worker 本次处理使用的追踪引用。

    参数：
        retry_dispatch_message:
            调度器发布的重试消息，其中包含本次重试执行的
            trace_id，以及发布消息时使用的 source_span_id。
        worker_span_id:
            Worker 开始处理本次重试任务时生成的新操作编号。

    返回：
        Worker 本次处理使用的追踪引用。
        trace_id 继承重试消息中的值，
        span_id 使用 Worker 新生成的编号，
        parent_span_id 使用消息中的 source_span_id。
    """
    if not worker_span_id.strip():
        raise ValueError("worker_span_id不能为空")
    if worker_span_id.strip() == retry_dispatch_message.source_span_id.strip():
        raise ValueError("worker_span_id与消息中的source_span_id相同")
    worker_trace_reference = TraceReference(
        trace_id=retry_dispatch_message.trace_id,
        span_id=worker_span_id,
        parent_span_id=retry_dispatch_message.source_span_id,
    )
    return worker_trace_reference

def test_自动重试从失败记录传播到Worker执行():
    """
    验证一次失败执行经过重试计划、消息发布和 Worker 领取后，
    任务关联信息保持不变，新执行使用新的 trace_id、claim_id
    和连续增加的 attempt，并且 Worker Span 正确连接发布消息的 Span。
    """
    failed_execution_reference = RetryExecutionReference(
        task_id="TASK-1",
        correlation_id="CORRELATION-1",
        trace_id="TRACE-ATTEMPT-1",
        claim_id="CLAIM-1",
        previous_trace_id="TRACE-INITIAL",
        attempt=1,
    )

    retry_schedule_reference = create_retry_schedule_reference(
        failed_execution_reference=failed_execution_reference,
        next_attempt=2,
        scheduled_at=datetime(2026, 9, 7, 12, 0, 0),
    )
    retry_dispatch_message = create_retry_dispatch_message(
        retry_schedule_reference=retry_schedule_reference,
        new_trace_id="TRACE-ATTEMPT-2",
        dispatch_span_id="DISPATCH-SPAN-2",
    )
    worker_trace_reference = create_worker_trace_from_retry_message(
        retry_dispatch_message=retry_dispatch_message,
        worker_span_id="WORKER-SPAN-2",
    )
    claimed_execution_reference = claim_retry_dispatch_message(
        retry_dispatch_message=retry_dispatch_message,
        new_claim_id="CLAIM-2",
    )

    assert claimed_execution_reference.task_id == failed_execution_reference.task_id
    assert (
        claimed_execution_reference.correlation_id
        == failed_execution_reference.correlation_id
    )
    assert retry_dispatch_message.trace_id != failed_execution_reference.trace_id
    assert worker_trace_reference.trace_id == retry_dispatch_message.trace_id
    assert claimed_execution_reference.trace_id == retry_dispatch_message.trace_id
    assert (
        worker_trace_reference.parent_span_id
        == retry_dispatch_message.source_span_id
    )
    assert claimed_execution_reference.attempt == 2
    assert claimed_execution_reference.claim_id == "CLAIM-2"


class RetryExecutionResult(BaseModel):
    task_id: str
    correlation_id: str
    trace_id: str
    claim_id: str
    attempt: int = Field(ge=1)
    status: Literal["succeeded", "failed"]
    error_code: str | None = None
    error_message: str | None = None
    finished_at: datetime

def create_retry_execution_result(
    execution_reference: RetryExecutionReference,
    status: Literal["succeeded", "failed"],
    finished_at: datetime,
    error_code: str | None = None,
    error_message: str | None = None,
) -> RetryExecutionResult:
    """
    根据已经完成的重试执行创建执行结果。

    参数：
        execution_reference:
            Worker 获得执行权后创建的执行身份记录，
            包含任务编号、追踪编号、领取令牌和执行次数。
        status:
            本次执行结果，只能是 succeeded（成功）
            或 failed（失败）。
        finished_at:
            本次执行完成的时间。
        error_code:
            执行失败时的错误编号；执行成功时应为 None。
        error_message:
            执行失败时的错误原因；执行成功时应为 None。

    返回：
        本次重试的执行结果。
        任务编号、业务关联编号、追踪编号、领取令牌和执行次数
        都从 execution_reference 中继承。
    """
    normalized_error_code = error_code.strip() if error_code else None
    normalized_error_message = error_message.strip() if error_message else None
    if status == "succeeded":
        if normalized_error_code is not None or normalized_error_message is not None:
            raise ValueError("状态是成功时，错误代码和错误信息必须是None")
    else:
        if normalized_error_code is None or normalized_error_message is None:
            raise ValueError("状态是失败时，错误代码和错误信息必须有值")

    retry_execution_result = RetryExecutionResult(
        task_id=execution_reference.task_id,
        correlation_id=execution_reference.correlation_id,
        trace_id=execution_reference.trace_id,
        claim_id=execution_reference.claim_id,
        attempt=execution_reference.attempt,
        status=status,
        error_code=normalized_error_code,
        error_message=normalized_error_message,
        finished_at=finished_at,
    )
    return retry_execution_result

def create_retry_result_event_trace(
    execution_result: RetryExecutionResult,
    worker_trace_reference: TraceReference,
    event_id: str,
) -> EventTraceReference:
    """
    为重试执行的结果事件创建追踪引用。

    参数：
        execution_result:
            已经保存的重试执行结果，包含本次执行使用的 trace_id。
        worker_trace_reference:
            Worker 执行本次任务时使用的追踪引用。
        event_id:
            本次结果事件的新编号。

    返回：
        结果事件的追踪引用。
        trace_id 继承 Worker 的 trace_id，
        source_span_id 使用 Worker 的 span_id，
        表示该事件由本次 Worker 操作产生。
    """

    if not event_id.strip():
        raise ValueError("事件ID不能为空")
    result_trace_id = execution_result.trace_id
    worker_trace_id = worker_trace_reference.trace_id
    if result_trace_id != worker_trace_id:
        raise ValueError("执行结果中的trace_id必须与追踪链路中的trace_id相等")

    event_trace_reference = propagate_worker_trace_to_event(
        worker_trace_reference=worker_trace_reference,
        event_id=event_id,
    )
    return event_trace_reference
