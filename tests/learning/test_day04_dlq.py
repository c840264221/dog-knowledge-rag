from datetime import datetime, timezone, timedelta

from pydantic import BaseModel, model_validator, field_validator, ValidationError, Field, ValidationInfo
from enum import Enum
import pytest
from torch.fx.experimental.unification.utils import raises


class DispositionAction(str, Enum):
    REDRIVE = "redrive"
    REJECTED = "rejected"
    IGNORE = "ignore"
    CLOSE = "close"

class DeadLetterDisposition(BaseModel):
    """
    死信处置记录契约
    """
    disposition_id: str
    dlq_id : str
    redrive_request_id: str

    action: DispositionAction
    actor_id : str
    reason: str

    created_retry_plan_id : str | None= None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls,v:datetime) -> datetime:
        """
        校验处置创建时间，并统一转换为 UTC。
        参数：cls 为当前模型类；v 为待校验的创建时间。
        返回：转换为 UTC 的带时区 datetime 对象。
        """
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("死信处置记录中的创建时间必须携带有效时区")
        return v.astimezone(timezone.utc)


    @model_validator(mode='after')
    def validate_retry_schedule_consistency(self):
        """校验处置动作与待执行重试安排 ID 是否保持一致。"""
        if self.action == DispositionAction.REDRIVE and (self.created_retry_plan_id is None or not self.created_retry_plan_id.strip()):
            raise ValueError("动作是批准重新驱动时，待执行重试安排id必须有值")

        if self.action != DispositionAction.REDRIVE and self.created_retry_plan_id is not None:
            raise ValueError("动作不是批准重新驱动时，待执行重试安排id必须为None")
        return self

class DeadLetterRecord(BaseModel):
    """记录某次执行进入死信时的原始失败事实"""
    dlq_id: str

    task_id : str
    task_version: int = Field(ge=1)
    step_id : str


    source_trace_id : str
    source_span_id: str
    event_id: str | None = None
    claim_id : str
    payload_reference : str

    attempt_count : int = Field(ge=1)

    error_code : str
    error_reason: str

    worker_id: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: datetime) -> datetime:
        """校验死信创建时间携带有效时区，并统一转换为UTC。"""
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("死信创建时间必须携带有效时区")
        return v.astimezone(timezone.utc)

    @field_validator("dlq_id")
    @classmethod
    def validate_dlq_id(cls, v: str) -> str:
        """校验死信记录 ID 不是空字符串或全空白字符串。

        参数：cls 为当前模型类；v 为待校验的死信记录 ID。
        返回：通过校验的原始 ID 字符串。
        """
        if not v.strip():
            raise ValueError("死信记录 ID 不能为空")
        return v


def test_死信处置记录合法数据应该可以通过契约():
    """
    测试死信记录合法数据应该可以通过契约
    """
    dlq_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="test_redrive_request_id_1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="死信记录合法数据应该可以通过契约",
        created_retry_plan_id="test_created_retry_plan_id_1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )

    assert dlq_disposition.action == DispositionAction.REDRIVE
    assert dlq_disposition.created_retry_plan_id == "test_created_retry_plan_id_1"

@pytest.mark.parametrize("plan_id",[None, "", "  "])
def test_死信处置记录动作为批准重试但没有重试计划时应该抛出ValidationError(plan_id):
    """验证批准重新驱动时，不合法的重试计划 ID 会被拒绝。

    参数：plan_id 为待验证的计划 ID，包含 None、空字符串和全空白。
    返回：无，通过断言检查预期的数据校验异常。
    """
    with pytest.raises(ValidationError, match="动作是批准重新驱动时，待执行重试安排id必须有值"):
        DeadLetterDisposition(
            disposition_id="test_disposition_id_1",
            dlq_id="test_dlq_id_1",
            redrive_request_id="test_redrive_request_id_1",
            action=DispositionAction.REDRIVE,
            actor_id="test_actor_id_1",
            reason="死信记录合法数据应该可以通过契约",
            created_retry_plan_id=plan_id,
            created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_死信处置记录动作不是批准重试且有重试计划应该抛出ValidationError():
    """
    测试死信记录接收了不合法数据应该抛出ValidationError
    :return:
    """
    with pytest.raises(ValidationError, match="动作不是批准重新驱动时，待执行重试安排id必须为None"):
        DeadLetterDisposition(
            disposition_id="test_disposition_id_1",
            dlq_id="test_dlq_id_1",
            redrive_request_id="test_redrive_request_id_1",
            action=DispositionAction.REJECTED,
            actor_id="test_actor_id_1",
            reason="死信记录合法数据应该可以通过契约",
            created_retry_plan_id="test_created_retry_plan_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_死信记录接收合法数据后是否可以正常创建():
    """验证合法的原始死信数据能够通过契约并保留关键字段。"""
    dlq_record = DeadLetterRecord(
        dlq_id="test_dlq_id_1",
        task_id="test_task_id_1",
        task_version=1,
        step_id="test_step_id_1",
        source_trace_id="test_source_trace_id_1",
        source_span_id="test_source_span_id_1",
        event_id="test_event_id_1",
        claim_id="test_claim_id_1",
        payload_reference="artifact://task-001/step-002/input/version-18",
        attempt_count=1,
        error_code="test_error_code_1",
        error_reason="test_error_reason_1",
        worker_id="test_worker_id_1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    assert dlq_record.dlq_id == "test_dlq_id_1"
    assert dlq_record.task_id == "test_task_id_1"
    assert dlq_record.task_version == 1
    assert dlq_record.step_id == "test_step_id_1"
    assert dlq_record.source_trace_id == "test_source_trace_id_1"
    assert dlq_record.source_span_id == "test_source_span_id_1"
    assert dlq_record.event_id == "test_event_id_1"
    assert dlq_record.claim_id == "test_claim_id_1"
    assert dlq_record.payload_reference == "artifact://task-001/step-002/input/version-18"
    assert dlq_record.attempt_count == 1
    assert dlq_record.error_code == "test_error_code_1"
    assert dlq_record.error_reason == "test_error_reason_1"
    assert dlq_record.worker_id == "test_worker_id_1"

def test_死信记录的执行次数小于1时应该抛出ValidationError():
    """验证执行次数为0的死信记录不能通过契约。"""
    with pytest.raises(ValidationError, match="attempt_count"):
        DeadLetterRecord(
            dlq_id="test_dlq_id_1",
            task_id="test_task_id_1",
            task_version=1,
            step_id="test_step_id_1",
            source_trace_id="test_source_trace_id_1",
            source_span_id="test_source_span_id_1",
            event_id="test_event_id_1",
            claim_id="test_claim_id_1",
            payload_reference="artifact://task-001/step-002/input/version-18",
            attempt_count=0,
            error_code="test_error_code_1",
            error_reason="test_error_reason_1",
            worker_id="test_worker_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_死信记录的任务版本小于1时应该抛出ValidationError():
    """验证任务版本为0的死信记录不能通过契约。"""
    with pytest.raises(ValidationError, match="task_version"):
        DeadLetterRecord(
            dlq_id="test_dlq_id_1",
            task_id="test_task_id_1",
            task_version=0,
            step_id="test_step_id_1",
            source_trace_id="test_source_trace_id_1",
            source_span_id="test_source_span_id_1",
            event_id="test_event_id_1",
            claim_id="test_claim_id_1",
            payload_reference="artifact://task-001/step-002/input/version-18",
            attempt_count=1,
            error_code="test_error_code_1",
            error_reason="test_error_reason_1",
            worker_id="test_worker_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_死信记录的时间参数如果没有携带有效时区应该抛出ValidationError():
    """验证时间参数不携带有效时区的死信记录不能通过契约。"""
    with pytest.raises(ValidationError, match="死信创建时间必须携带有效时区"):
        DeadLetterRecord(
            dlq_id="test_dlq_id_1",
            task_id="test_task_id_1",
            task_version=1,
            step_id="test_step_id_1",
            source_trace_id="test_source_trace_id_1",
            source_span_id="test_source_span_id_1",
            event_id="test_event_id_1",
            claim_id="test_claim_id_1",
            payload_reference="artifact://task-001/step-002/input/version-18",
            attempt_count=1,
            error_code="test_error_code_1",
            error_reason="test_error_reason_1",
            worker_id="test_worker_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0),
        )

def test_死信记录的携带有效时区的时间参数会转换成UTC时区():
    """验证死信记录的携带有效时区的时间参数会转换成UTC时区。"""
    created_at = datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    dlq_record = DeadLetterRecord(
        dlq_id="test_dlq_id_1",
        task_id="test_task_id_1",
        task_version=1,
        step_id="test_step_id_1",
        source_trace_id="test_source_trace_id_1",
        source_span_id="test_source_span_id_1",
        event_id="test_event_id_1",
        claim_id="test_claim_id_1",
        payload_reference="artifact://task-001/step-002/input/version-18",
        attempt_count=1,
        error_code="test_error_code_1",
        error_reason="test_error_reason_1",
        worker_id="test_worker_id_1",
        created_at=created_at
    )

    assert dlq_record.created_at == datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)
    assert dlq_record.created_at.tzinfo is timezone.utc

def test_死信处置记录的携带有效时区的时间参数会转换成UTC时区():
    """验证死信处置记录的携带有效时区的时间参数会转换成UTC时区"""
    created_at = datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    dlq_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="test_redrive_request_id_1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="死信处置记录的携带有效时区的时间参数会转换成UTC时区",
        created_retry_plan_id="test_created_retry_plan_id_1",
        created_at=created_at
    )

    assert dlq_disposition.created_at == datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)
    assert dlq_disposition.created_at.tzinfo is timezone.utc

def test_死信处置记录的时间参数如果没有携带有效时区时应抛出ValidationError():
    """死信处置记录的时间参数如果没有携带有效时区时应抛出ValidationError"""
    with pytest.raises(ValidationError, match="死信处置记录中的创建时间必须携带有效时区"):
        DeadLetterDisposition(
            disposition_id="test_disposition_id_1",
            dlq_id="test_dlq_id_1",
            redrive_request_id="test_redrive_request_id_1",
            action=DispositionAction.REDRIVE,
            actor_id="test_actor_id_1",
            reason="死信处置记录的携带有效时区的时间参数会转换成UTC时区",
            created_retry_plan_id="test_created_retry_plan_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0)
        )

@pytest.mark.parametrize("dlq_id", ["", "  "])
def test_死信记录的id为空白字符串时会抛出ValidationError(dlq_id):
    """验证空字符串和全空白 ID 均被拒绝。

    参数：dlq_id 为本次测试的不合法死信 ID。
    返回：无，通过断言检查预期异常。
    """
    with pytest.raises(ValidationError, match="死信记录 ID 不能为空"):
        DeadLetterRecord(
            dlq_id=dlq_id,
            task_id="test_task_id_1",
            task_version=1,
            step_id="test_step_id_1",
            source_trace_id="test_source_trace_id_1",
            source_span_id="test_source_span_id_1",
            event_id="test_event_id_1",
            claim_id="test_claim_id_1",
            payload_reference="artifact://task-001/step-002/input/version-18",
            attempt_count=1,
            error_code="test_error_code_1",
            error_reason="test_error_reason_1",
            worker_id="test_worker_id_1",
            created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_死信记录的id经过校验后会原样返回():
    """验证合法死信 ID 经过校验后保留原值。

    参数：无。
    返回：无，通过断言检查模型保存的 ID 与输入一致。
    """
    dlq_record = DeadLetterRecord(
        dlq_id=" test_dlq_id_1 ",
        task_id="test_task_id_1",
        task_version=1,
        step_id="test_step_id_1",
        source_trace_id="test_source_trace_id_1",
        source_span_id="test_source_span_id_1",
        event_id="test_event_id_1",
        claim_id="test_claim_id_1",
        payload_reference="artifact://task-001/step-002/input/version-18",
        attempt_count=1,
        error_code="test_error_code_1",
        error_reason="test_error_reason_1",
        worker_id="test_worker_id_1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    assert dlq_record.dlq_id == " test_dlq_id_1 "

def test_死信处置记录拒绝重新驱动时没有重试计划是合法的():
    """验证死信处置记录拒绝重新驱动时没有重试计划是合法的"""
    dlq_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="test_redrive_request_id_1",
        action=DispositionAction.REJECTED,
        actor_id="test_actor_id_1",
        reason="死信处置记录的携带有效时区的时间参数会转换成UTC时区",
        created_retry_plan_id=None,
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    assert dlq_disposition.action == DispositionAction.REJECTED
    assert dlq_disposition.created_retry_plan_id is None

def test_执行次数等于上限时应该判定为耗尽():
    """验证已执行次数等于上限时返回 True。

    参数：无。
    返回：无，通过断言检查次数耗尽判断。
    """
    result = is_attempts_exhausted(3,3)
    assert result is True

def test_执行次数小于上限时应该判定为未耗尽():
    """验证已执行次数小于上限时返回 False。

    参数：无。
    返回：无，通过断言检查次数未耗尽。
    """
    result = is_attempts_exhausted(2, 3)
    assert result is False

def test_执行次数大于上限时应该判定为耗尽():
    """验证已执行次数大于上限时返回 True。

    参数：无。
    返回：无，通过断言检查次数耗尽判断。
    """
    result = is_attempts_exhausted(4,3)
    assert result is True

def test_执行次数为0时应该抛出ValueError():
    """验证执行次数为0时应该抛出ValueError。

    参数：无。
    返回：无，通过断言检查次数耗尽判断。
    """
    with pytest.raises(ValueError, match="执行次数和最大执行次数均必须大于0"):
        is_attempts_exhausted(0, 3)

def test_最大执行次数为0时应该抛出ValueError() -> None:
    """验证最大执行次数为0时拒绝进行次数耗尽判断。

    参数：无。
    返回：无，通过断言检查预期的值错误及错误提示。
    """
    with pytest.raises(ValueError, match="执行次数和最大执行次数均必须大于0"):
        is_attempts_exhausted(2, 0)


def is_attempts_exhausted(
    attempt_count: int,
    max_attempts: int,
) -> bool:
    """判断是否已达到最大执行次数。

    参数：attempt_count 为包含本次执行的已执行次数；
          max_attempts 为包含首次执行的最大允许次数。
    返回：次数已耗尽返回 True，否则返回 False。
    """
    if attempt_count < 1 or max_attempts < 1:
        raise ValueError("执行次数和最大执行次数均必须大于0")

    if attempt_count >= max_attempts:
        return True
    else:
        return False

def should_dead_letter_after_failure(
    execution_succeeded: bool,
    attempt_count: int,
    max_attempts: int,
) -> bool:
    """判断是否因本次执行失败且次数耗尽而需要进入死信。

    参数：
        execution_succeeded：本次执行是否成功。
        attempt_count：包含本次执行的已执行次数。
        max_attempts：包含首次执行的最大允许次数。
    返回：
        本次失败且次数耗尽时返回 True，否则返回 False。
        本函数只判断条件，不修改状态或保存记录。
    """
    if execution_succeeded:
        return False
    else:
        if is_attempts_exhausted(attempt_count, max_attempts):
            return True
        else:
            return False

def test_本次执行成功即使次数耗尽也不应进入死信() -> None:
    """验证本次执行成功时，不会因次数耗尽而判定进入死信。

    参数：无。
    返回：无，通过断言检查死信条件判断返回 False。
    """
    should_enter_dlq = should_dead_letter_after_failure(True, 3, 3)
    assert should_enter_dlq is False

def test_本次执行失败且次数耗尽时应判定进入死信() -> None:
    """验证本次执行失败且次数达到上限时，满足死信条件。

    参数：无。
    返回：无，通过断言检查死信条件判断返回 True。
    """
    should_enter_dlq = should_dead_letter_after_failure(False, 3, 3)
    assert should_enter_dlq is True

def test_本次执行失败但次数未耗尽时不满足耗尽死信条件() -> None:
    """验证本次失败但次数未达上限时，不满足次数耗尽死信条件。

    参数：无。
    返回：无，通过断言检查死信条件判断返回 False。
    """
    # 建议变量名：should_enter_dlq，表示是否满足当前死信条件
    # 请在这里编写调用和断言逻辑
    should_enter_dlq = should_dead_letter_after_failure(False, 2, 3)
    assert should_enter_dlq is False

def test_失败且次数耗尽后构造死信记录并保留失败原因() -> None:
    """验证先判断死信条件，再构造保留原始失败信息的记录。

    参数：无。
    返回：无，通过断言检查死信条件、执行次数和原始错误码。
        本测试只构造内存对象，不保存数据库，也不确认消息。
    """
    # should_enter_dlq：是否满足次数耗尽死信条件
    # dlq_record：判断通过后构造的死信记录
    # 请在这里编写逻辑
    should_enter_dlq = should_dead_letter_after_failure(False, 3, 3)
    assert should_enter_dlq is True
    dlq_record = DeadLetterRecord(
        dlq_id = "test_dlq_id_1",
        task_id = "test_task_id_1",
        task_version = 1,
        step_id = "test_step_id_1",
        source_trace_id = "test_source_trace_id_1",
        source_span_id = "test_source_span_id_1",
        event_id = "test_event_id_1",
        claim_id = "test_claim_id_1",
        payload_reference = "test_payload_reference",

        attempt_count = 3,

        error_code = "UPSTREAM_TIMEOUT",
        error_reason = "调用上游服务超时",

        worker_id = "test_worker_id",
        created_at = datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )

    assert dlq_record.error_code == "UPSTREAM_TIMEOUT"
    assert dlq_record.error_reason == "调用上游服务超时"
    assert dlq_record.attempt_count == 3


class RedriveRequest(BaseModel):
    """表示一次人工重新驱动请求。

    字段：
        redrive_request_id：请求幂等标识，重发同一请求时保持不变。
        dlq_id：希望处理的死信记录 ID。
        actor_id：发起操作的人员标识。
        reason：申请重新驱动的原因。
    """
    redrive_request_id : str
    dlq_id: str
    actor_id: str
    reason: str

    @field_validator("redrive_request_id", "dlq_id", "actor_id", "reason")
    @classmethod
    def validate_non_blank_text(cls, v: str, info:ValidationInfo) -> str:
        """逐个检查请求中的必填字符串不是空字符串或全空白。
        info 为校验上下文，info.field_name 为当前字段名。
        参数：cls 为当前模型类；v 为当前正在校验的字段值。
        返回：通过校验的原始字符串，不修改首尾空白。
        """
        # 判断 v.strip() 是否为空
        # 为空时抛出 ValueError，提示“请求字段不能为空白”
        # 否则返回原始 v
        if v.strip() == "":
            raise ValueError(f"请求字段不能为空白：{info.field_name}")
        return v
    

def test_合法人工重新驱动请求可以创建并保留字段() -> None:
    """验证合法的人工重新驱动请求能够创建并保留输入字段。

    参数：无。
    返回：无，通过断言检查请求标识、死信标识、操作人和申请原因。
    """
    # 建议变量名：redrive_request，表示人工重新驱动请求对象
    # 请在这里编写模型构造和四个字段的断言
    redrive_request = RedriveRequest(
        redrive_request_id = "test_redrive_request_id_1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证合法的人工重新驱动请求能够创建并保留输入字段"
    )
    assert redrive_request.redrive_request_id == "test_redrive_request_id_1"
    assert redrive_request.dlq_id == "test_dlq_id_1"
    assert redrive_request.actor_id == "test_actor_id_1"
    assert redrive_request.reason == "验证合法的人工重新驱动请求能够创建并保留输入字段"

def test_人工重新驱动请求的申请原因为全空白时应被拒绝() -> None:
    """验证申请原因为全空白时，人工重新驱动请求不能通过校验。

    参数：无。
    返回：无，通过断言检查数据校验异常及错误提示。
    """
    # 在预期异常范围内构造请求，其他字段参考合法请求测试
    # 本次构造预期失败，不需要接收返回值的变量
    with pytest.raises(ValidationError, match="请求字段不能为空白：reason"):
        RedriveRequest(
            redrive_request_id = "test_redrive_request_id_2",
            dlq_id = "test_dlq_id_2",
            actor_id = "test_actor_id_2",
            reason = " "
        )

@pytest.mark.parametrize(
    "field_name",
    ["redrive_request_id", "dlq_id", "actor_id", "reason"],
)
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_人工重新驱动请求的必填字段为空白时应被拒绝(
    field_name: str,
    invalid_value: str,
) -> None:
    """验证请求的四个必填字段均拒绝空字符串和全空白。

    参数：
        field_name：本次要验证的字段名。
        invalid_value：本次使用的空字符串或全空白字符串。
    返回：无，通过断言检查数据校验异常及对应字段的错误提示。
    """
    # request_data：每次测试独立创建的合法请求数据字典
    # 修改目标字段：request_data[field_name] = invalid_value
    # 预期异常：ValidationError
    # 匹配提示：f"请求字段不能为空白：{field_name}"
    # 请编写数据准备、字段替换和异常断言
    request_data = {
        "redrive_request_id" : "test_redrive_request_id_1",
        "dlq_id" : "test_dlq_id_1",
        "actor_id" : "test_actor_id_1",
        "reason" : "验证人工重新驱动请求的必填字段为空白时应被拒绝",
    }

    request_data[field_name] = invalid_value
    with pytest.raises(ValidationError, match=f"请求字段不能为空白：{field_name}"):
        RedriveRequest(**request_data)

def passes_no_active_retry_plan_check(
    has_active_retry_plan: bool,
) -> bool:
    """判断是否通过无重复重试计划检查。

    参数：
        has_active_retry_plan：当前权威状态中，
            同一任务步骤是否已有有效重试计划。
    返回：
        已有有效计划时返回 False，没有有效计划时返回 True。
        仅检查重复计划，不代表最终批准，也不创建或保存计划。
    """
    # 请在这里编写判断和返回逻辑
    if has_active_retry_plan:
        return False
    else:
        return True

@pytest.mark.parametrize(
    "has_active_retry_plan, expected",
    [(True, False), (False, True)],
)
def test_无重复计划检查根据当前有效计划返回结果(
    has_active_retry_plan: bool,
    expected: bool,
) -> None:
    """验证已有有效计划时检查不通过，没有时检查通过。

    参数：
        has_active_retry_plan：当前步骤是否已有有效重试计划。
        expected：本次预期的布尔判断结果。
    返回：无，通过断言检查计划检查结果。
    """
    # passes_plan_check：调用判断函数得到的结果
    # 请编写函数调用，并断言 passes_plan_check is expected
    passes_plan_check = passes_no_active_retry_plan_check(has_active_retry_plan)
    assert passes_plan_check is expected

def passes_failed_step_status_check(
    current_step_status: str,
) -> bool:
    """判断当前权威步骤是否处于失败状态。

    参数：
        current_step_status：从最新 LongTask 中取得的目标步骤状态。
    返回：
        状态为 failed 时返回 True，否则返回 False。
        仅检查步骤状态，不代表最终批准，也不读取或修改数据库。
    """
    # 请在这里编写判断和返回逻辑
    if current_step_status == "failed":
        return True
    else:
        return False

@pytest.mark.parametrize(
    "current_step_status, expected",
    [
        ("failed", True),
        ("pending", False),
        ("ready", False),
        ("running", False),
        ("awaiting_input", False),
        ("completed", False),
        ("skipped", False),
        ("cancelled", False),
    ],
)
def test_重新驱动的步骤状态检查仅允许失败状态通过(
    current_step_status: str,
    expected: bool,
) -> None:
    """验证现有步骤状态中，只有失败状态通过重新驱动状态检查。

    参数：
        current_step_status：本次模拟的当前权威步骤状态。
        expected：预期是否通过这项状态检查。
    返回：无，通过断言检查函数返回值是否符合预期。
    """
    # passes_status_check：步骤状态检查结果
    # 请编写函数调用，并断言 passes_status_check is expected
    passes_status_check = passes_failed_step_status_check(current_step_status)
    assert passes_status_check is expected

def passes_task_terminal_state_check(
    current_task_status: str,
) -> bool:
    """判断当前任务是否通过终态限制检查。

    参数：
        current_task_status：从最新 LongTask 读取的任务整体状态。
    返回：
        completed 或 cancelled 时返回 False，其他状态返回 True。
        假设输入为合法任务状态；通过不代表最终批准重新驱动。
        不读取数据库，不修改任务，也不创建重试计划。
    """
    # 请在这里编写判断和返回逻辑
    if current_task_status in ["completed", "cancelled"]:
        return False
    else:
        return True

@pytest.mark.parametrize(
    "current_task_status, expected",
    [
        ("created", True),
        ("queued", True),
        ("running", True),
        ("awaiting_input", True),
        ("failed", True),
        ("completed", False),
        ("cancelled", False),
    ],
)
def test_任务终态检查拒绝已完成和已取消的任务(
    current_task_status: str,
    expected: bool,
) -> None:
    """验证任务终态检查拒绝已完成和已取消的任务。

    参数：
        current_task_status：当前任务整体状态。
        expected：预期是否通过终态限制检查。
    返回：无，通过断言检查函数返回值。
    """
    # passes_task_check：任务终态检查结果
    # 请编写函数调用，并断言 passes_task_check is expected
    passes_task_check = passes_task_terminal_state_check(current_task_status)
    assert passes_task_check is expected

def passes_redrive_basic_checks(
    current_task_status: str,
    current_step_status: str,
    has_active_retry_plan: bool,
) -> bool:
    """组合重新驱动的三项基础检查。

    参数：
        current_task_status：当前权威任务的整体状态。
        current_step_status：目标步骤的当前状态。
        has_active_retry_plan：同一步骤是否已有有效重试计划。
    返回：
        三项检查全部通过时返回 True，否则返回 False。
        假设状态输入合法；不代表最终批准，不读取或保存数据。
    """
    # passes_task_check：调用任务终态检查
    # passes_status_check：调用失败步骤状态检查
    # passes_plan_check：调用无重复计划检查
    # 请复用三个已有函数，并组合返回结果
    passes_task_check = passes_task_terminal_state_check(current_task_status)
    passes_status_check = passes_failed_step_status_check(current_step_status)
    passes_plan_check = passes_no_active_retry_plan_check(has_active_retry_plan)

    if all([passes_task_check, passes_status_check, passes_plan_check]):
        return True
    else:
        return False

@pytest.mark.parametrize(
    "task_status, step_status, has_plan, expected",
    [
        ("failed", "failed", False, True),
        ("completed", "failed", False, False),
        ("failed", "completed", False, False),
        ("failed", "failed", True, False),
    ],
)
def test_重新驱动基础检查要求三项条件全部通过(
    task_status: str,
    step_status: str,
    has_plan: bool,
    expected: bool,
) -> None:
    """验证三项基础条件全部通过才返回 True。

    参数：
        task_status：当前任务状态。
        step_status：当前步骤状态。
        has_plan：是否已有有效重试计划。
        expected：预期的组合判断结果。
    返回：无，通过断言检查基础条件组合结果。
    """
    # passes_basic_checks：调用组合检查函数得到的结果
    # 请编写函数调用，并断言 passes_basic_checks is expected
    passes_basic_checks = passes_redrive_basic_checks(task_status, step_status, has_plan)
    assert passes_basic_checks is expected

class RedriveEligibilityDecision(BaseModel):
    """表示人工重新驱动的资格检查结果。

    字段：
        eligible：是否通过当前资格检查，类型为 bool。
        reason：通过或拒绝的具体原因，类型为 str。
    本模型不创建重试计划，不保存处置记录，也不代表已经执行。
    """
    # 请将 eligible 和 reason 声明为必填字段
    eligible: bool
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """校验资格检查原因不是空字符串或全空白。

        参数：cls 为当前模型类；value 为待校验的原因。
        返回：通过校验的原始原因字符串。
        """
        if value.strip() == "":
            raise ValueError("原因不能为空字符串")
        return value

def evaluate_redrive_basic_eligibility(
    current_task_status: str,
    current_step_status: str,
    has_active_retry_plan: bool,
) -> RedriveEligibilityDecision:
    """依次检查重新驱动基础资格，并返回首个拒绝原因。

    参数：
        current_task_status：当前权威任务整体状态。
        current_step_status：目标步骤当前状态。
        has_active_retry_plan：同一步骤是否已有有效重试计划。
    返回：
        包含 eligible 和 reason 的资格检查结果。
        假设状态输入合法；只检查三项基础条件，不代表提交已批准。
        不读取数据库，不创建计划，也不保存处置记录。
    """
    # 复用三个已有的单项检查函数
    # 每项不通过时，返回 eligible=False 和对应原因
    # 全部通过后，返回 eligible=True 和通过原因
    passes_task_check = passes_task_terminal_state_check(current_task_status)
    if not passes_task_check:
        return RedriveEligibilityDecision(
            eligible=False,
            reason="任务已进入终态，不能重新驱动",
        )
    passes_status_check = passes_failed_step_status_check(current_step_status)
    if not passes_status_check:
        return RedriveEligibilityDecision(
            eligible=False,
            reason="步骤状态不是failed，不支持重新驱动",
        )
    passes_plan_check = passes_no_active_retry_plan_check(has_active_retry_plan)
    if not passes_plan_check:
        return RedriveEligibilityDecision(
            eligible=False,
            reason="当前步骤已有有效重试计划",
        )

    return RedriveEligibilityDecision(
        eligible=True,
        reason="基础资格检查通过",
    )

@pytest.mark.parametrize(
    "task_status, step_status, has_plan, expected_eligible, expected_reason",
    [
        ("failed", "failed", False, True, "基础资格检查通过"),
        ("failed", "running", False, False, "步骤状态不是failed，不支持重新驱动"),
        ("failed", "failed", True, False, "当前步骤已有有效重试计划"),
        ("cancelled", "running", True, False, "任务已进入终态，不能重新驱动"),
    ],
)
def test_重新驱动基础资格返回判断结果和首个拒绝原因(
    task_status: str,
    step_status: str,
    has_plan: bool,
    expected_eligible: bool,
    expected_reason: str,
) -> None:
    """验证基础资格判断及多项不通过时的原因优先级。

    参数：
        task_status：当前任务状态。
        step_status：当前步骤状态。
        has_plan：是否已有有效重试计划。
        expected_eligible：预期是否通过基础检查。
        expected_reason：预期的检查原因。
    返回：无，通过断言检查判断结果和原因。
    """
    # eligibility_decision：调用 evaluate_redrive_basic_eligibility 的结果
    # 请检查 eligible is expected_eligible
    # 并检查 reason == expected_reason
    eligibility_decision = evaluate_redrive_basic_eligibility(task_status, step_status, has_plan)
    assert eligibility_decision.eligible is expected_eligible
    assert eligibility_decision.reason == expected_reason

def find_existing_redrive_disposition(
    redrive_request_id: str,
    dispositions_by_request_id: dict[str, DeadLetterDisposition],
) -> DeadLetterDisposition | None:
    """根据重新驱动请求 ID 查询已保存的处置。

    参数：
        redrive_request_id：本次请求的幂等标识。
        dispositions_by_request_id：以请求 ID 为键、处置对象为值的字典。
    返回：
        找到时返回原处置对象，未找到时返回 None。
        不修改字典，不新增处置，也不进行资格检查。
        仅模拟查询，尚未校验请求内容指纹或实现原子提交。
    """
    # existing_disposition：查询到的原处置
    # 请使用字典查询并返回结果
    existing_disposition = dispositions_by_request_id.get(redrive_request_id, None)
    return existing_disposition

def test_查询已处理请求时返回原处置对象() -> None:
    """验证按已有请求 ID 查询时返回原处置对象。

    参数：无。
    返回：无，通过对象身份断言检查查询结果，并检查字典条数未增加。
    """
    # original_disposition：参考已有合法测试构造的批准处置
    # dispositions_by_request_id：以请求 ID 为键的字典
    # existing_disposition：调用查询函数得到的结果
    # 断言 existing_disposition is original_disposition
    # 断言 len(dispositions_by_request_id) == 1
    original_disposition = DeadLetterDisposition(
        disposition_id = "test_disposition_id_1",
        dlq_id = "test_dlq_id_1",
        redrive_request_id = "test_redrive_request_id_1",
        action = DispositionAction.REDRIVE,
        actor_id = "test_actor_id_1",
        reason = "验证按已有请求 ID 查询时返回原处置对象",

        created_retry_plan_id = "test_retry_plan_id_1",
        created_at=datetime(2026, 9, 6, 8, 0, 0, tzinfo=timezone.utc)
    )
    dispositions_by_request_id = {"test_redrive_request_id_1": original_disposition}
    existing_disposition = find_existing_redrive_disposition("test_redrive_request_id_1", dispositions_by_request_id)

    assert existing_disposition is original_disposition
    assert len(dispositions_by_request_id) == 1

def test_查询未处理请求时返回None且不新增处置() -> None:
    """验证查询不存在的请求时返回 None，且不修改存储字典。

    参数：无。
    返回：无，通过断言检查查询结果为空、字典仍为空。
    """
    # dispositions_by_request_id：空字典
    # existing_disposition：查询一个请求 ID 后的返回值
    # 请断言 existing_disposition is None
    # 并断言 len(dispositions_by_request_id) == 0
    dispositions_by_request_id = {}
    existing_disposition = find_existing_redrive_disposition("test_redrive_request_id_1", dispositions_by_request_id)

    assert existing_disposition is None
    assert len(dispositions_by_request_id) == 0

def has_same_redrive_request_content(
    original_request: RedriveRequest,
    incoming_request: RedriveRequest,
) -> bool:
    """判断同一请求 ID 下，两次重新驱动请求的内容是否一致。

    参数：
        original_request：此前保存的原始请求。
        incoming_request：本次收到的请求。
    返回：
        dlq_id、actor_id、reason 全部相同时返回 True，否则返回 False。
        前提是调用方已确认请求 ID 相同。
        不修改请求，不查询数据库，也不产生处置记录。
    """
    # same_dlq：死信 ID 是否相同
    # same_actor：操作人是否相同
    # same_reason：申请原因是否相同
    # 请比较三个字段，并组合返回结果

    same_dlq = original_request.dlq_id == incoming_request.dlq_id
    same_actor = original_request.actor_id == incoming_request.actor_id
    same_reason = original_request.reason == incoming_request.reason

    return all([same_dlq, same_actor, same_reason])

def test_相同请求ID且内容一致时判定为相同请求内容() -> None:
    """验证同一请求 ID 下，内容一致的两个请求对象比较结果为 True。

    参数：无。
    返回：无，通过断言检查请求内容一致，不要求对象身份相同。
    """
    # original_request：构造原始请求
    # incoming_request：另行构造请求，四个字段与原请求完全相同
    # same_request_content：调用 has_same_redrive_request_content 的结果
    # 请断言 same_request_content is True
    original_request = RedriveRequest(
        redrive_request_id = "test_redrive_request_id_1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求 ID 下，内容一致的两个请求对象比较结果为 True"
    )

    incoming_request = RedriveRequest(
        redrive_request_id = "test_redrive_request_id_1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求 ID 下，内容一致的两个请求对象比较结果为 True"
    )
    same_request_content = has_same_redrive_request_content(original_request, incoming_request)
    assert same_request_content is True

@pytest.mark.parametrize(
    "field_name, changed_value",
    [
        ("dlq_id", "another_dlq"),
        ("actor_id", "another_actor"),
        ("reason", "修改后的申请原因"),
    ],
)
def test_相同请求ID下任一业务字段变化时内容不一致(
    field_name: str,
    changed_value: str,
) -> None:
    """验证同一请求 ID 下，任一业务字段变化都会被识别。

    参数：
        field_name：本次修改的业务字段名。
        changed_value：该字段修改后的合法值。
    返回：无，通过断言检查请求内容比较结果为 False。
    """
    # original_request：合法原请求
    # request_data：使用 original_request.model_dump() 得到字段字典
    # 修改 request_data[field_name]，保持请求 ID 不变
    # incoming_request：用 RedriveRequest(**request_data) 创建新对象
    # same_request_content：比较结果，断言 is False

    original_request = RedriveRequest(
        redrive_request_id = "test_redrive_request_id_1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求 ID 下，内容一致的两个请求对象比较结果为 True"
    )
    request_data = original_request.model_dump()
    request_data[field_name] = changed_value
    incoming_request  = RedriveRequest(
        **request_data,
    )
    same_request_content = has_same_redrive_request_content(original_request, incoming_request )
    assert same_request_content is False

class ProcessedRedriveResult(BaseModel):
    """保存一次已处理的重新驱动请求及其处置结果。

    字段：
        original_request：原始请求，类型为 RedriveRequest。
        disposition：该请求对应的处置，类型为 DeadLetterDisposition。
    本模型用于内存练习，不自动持久化，也不等同于正式提交回执。
    """
    # 请声明上述两个必填字段
    original_request: RedriveRequest
    disposition: DeadLetterDisposition

    @model_validator(mode="after")
    def validate_request_disposition_consistency(self):
        """校验原请求与处置属于同一个重新驱动请求。

        参数：self 为已完成字段校验的当前结果模型。
        返回：校验通过后返回当前模型自身。
        异常：两个请求 ID 不一致时抛出 ValueError。
        """
        # 比较 original_request.redrive_request_id
        # 与 disposition.redrive_request_id
        # 不一致时提示：“原请求与处置的请求 ID 不一致”
        # 一致时返回 self
        if self.original_request.redrive_request_id != self.disposition.redrive_request_id:
            raise ValueError("原请求与处置的请求 ID 不一致")
        return self

def test_原请求与处置的请求ID不一致时应拒绝组合() -> None:
    """验证不能把不同请求的原始数据和处置组合保存。

    参数：无。
    返回：无，通过断言检查数据校验异常及请求 ID 不一致提示。
    """
    # original_request：合法请求，redrive_request_id 使用 "R1"
    # disposition：合法拒绝处置，redrive_request_id 使用 "R2"
    # 两者的 dlq_id、actor_id 保持一致
    # 拒绝处置的 created_retry_plan_id 为 None，时间使用带时区时间
    # 先在异常断言外构造上述两个对象
    # 再用 pytest.raises(ValidationError, match="原请求与处置的请求 ID 不一致")
    # 包住 ProcessedRedriveResult(...) 的构造
    original_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="test_dlq_id_1",
        actor_id="test_actor_id_1",
        reason="验证不能把不同请求的原始数据和处置组合保存"
    )
    disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="R2",
        action=DispositionAction.REJECTED,
        actor_id="test_actor_id_1",
        reason="验证不能把不同请求的原始数据和处置组合保存",
        created_retry_plan_id=None,
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValidationError, match="原请求与处置的请求 ID 不一致"):
        ProcessedRedriveResult(original_request=original_request, disposition=disposition)

def test_原请求与处置的请求ID一致时可以组合() -> None:
    """验证同一请求的原始数据与处置能够组合。

    参数：无。
    返回：无，通过断言检查组合模型保留请求和处置数据。
    """
    # original_request：合法请求，请求 ID 为 "R1"
    # disposition：合法拒绝处置，请求 ID 也为 "R1"
    # 两者的 dlq_id、actor_id 保持一致
    # processed_result：构造 ProcessedRedriveResult 得到的对象
    # 断言 processed_result.original_request == original_request
    # 断言 processed_result.disposition == disposition
    original_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="test_dlq_id_1",
        actor_id="test_actor_id_1",
        reason="验证不能把不同请求的原始数据和处置组合保存"
    )
    disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="R1",
        action=DispositionAction.REJECTED,
        actor_id="test_actor_id_1",
        reason="验证不能把不同请求的原始数据和处置组合保存",
        created_retry_plan_id=None,
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=disposition)
    assert processed_result.original_request == original_request
    assert processed_result.disposition == disposition

def resolve_existing_redrive_result(
    incoming_request: RedriveRequest,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
) -> DeadLetterDisposition | None:
    """查询已处理请求，校验内容后复用原处置。

    参数：
        incoming_request：本次收到的重新驱动请求。
        processed_results_by_request_id：以请求 ID 为键的已处理结果字典。
    返回：
        没有历史时返回 None；请求内容一致时返回原处置。
    异常：
        同一请求 ID 对应不同内容时抛出 ValueError。
        假设字典键与保存的请求 ID 一致。
        仅用于内存练习，不修改字典，不重新判断资格，不实现原子提交。
    """
    # processed_result：按 incoming_request.redrive_request_id 查询的结果
    # same_request_content：复用 has_same_redrive_request_content 的结果
    # 冲突提示：“同一请求 ID 对应的请求内容不一致”
    # 请按上述三个分支编写逻辑

    processed_result = processed_results_by_request_id.get(incoming_request.redrive_request_id, None)
    if processed_result is None:
        return None
    same_request_content = has_same_redrive_request_content(processed_result.original_request, incoming_request)

    if not same_request_content:
        raise ValueError("同一请求 ID 对应的请求内容不一致")
    else:
        return processed_result.disposition

def test_重复请求内容一致时复用原批准处置() -> None:
    """验证同一请求重发时复用原批准结果，不新增或改写处置。

    参数：无。
    返回：无，通过断言检查原处置、批准动作、计划 ID 和字典条数。
    """
    # original_request：合法请求，请求 ID 为 "R1"
    # disposition：对应的批准处置，计划 ID 为 "P1"
    # processed_result：将原请求和处置组合
    # processed_results_by_request_id：{"R1": processed_result}
    # incoming_request：另行创建内容完全相同的请求
    # existing_disposition：调用 resolve_existing_redrive_result 的结果
    # 断言返回值 is processed_result.disposition
    # 断言动作仍为 DispositionAction.REDRIVE
    # 断言计划 ID 仍为 "P1"，字典条数仍为 1

    original_request = RedriveRequest(
        redrive_request_id = "R1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求重发时复用原批准结果，不新增或改写处置"
    )
    disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="R1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="验证同一请求重发时复用原批准结果，不新增或改写处置",
        created_retry_plan_id="P1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=disposition)
    processed_results_by_request_id = {"R1": processed_result}
    incoming_request = RedriveRequest(
        redrive_request_id = "R1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求重发时复用原批准结果，不新增或改写处置"
    )
    existing_disposition = resolve_existing_redrive_result(incoming_request, processed_results_by_request_id)

    assert existing_disposition is processed_result.disposition
    assert existing_disposition.action == DispositionAction.REDRIVE
    assert existing_disposition.created_retry_plan_id == "P1"
    assert len(processed_results_by_request_id) == 1

def test_同一请求ID更换死信时应拒绝复用原结果() -> None:
    """验证同一请求 ID 被用于另一条死信时报告内容冲突。

    参数：无。
    返回：无，通过断言检查冲突异常，并确认原保存数据未改变。
    """
    # original_request：原请求，ID 为 "R1"，dlq_id 为 "DLQ-1"
    # disposition：对应批准处置，请求 ID 为 "R1"，dlq_id 为 "DLQ-1"
    # processed_result：原请求和处置的组合
    # processed_results_by_request_id：{"R1": processed_result}
    # incoming_request：仍为 "R1"，仅将 dlq_id 改为 "DLQ-2"
    # expected_snapshot：调用前，对 processed_result.model_dump() 的结果留存
    # 用 pytest.raises 包住 resolve_existing_redrive_result(...) 调用
    # 匹配提示：“同一请求 ID 对应的请求内容不一致”
    # 调用后检查字典仍只有键 "R1"，且该值的 model_dump() 等于快照
    original_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="DLQ-1",
        actor_id="test_actor_id_1",
        reason="验证同一请求重发时复用原批准结果，不新增或改写处置"
    )
    disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="DLQ-1",
        redrive_request_id="R1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="验证同一请求重发时复用原批准结果，不新增或改写处置",
        created_retry_plan_id="P1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=disposition)
    processed_results_by_request_id = {"R1": processed_result}
    incoming_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="DLQ-2",
        actor_id="test_actor_id_1",
        reason="验证同一请求重发时复用原批准结果，不新增或改写处置"
    )
    expected_snapshot = processed_result.model_dump()
    with pytest.raises(ValueError, match="同一请求 ID 对应的请求内容不一致"):
        resolve_existing_redrive_result(incoming_request, processed_results_by_request_id)
    assert processed_results_by_request_id["R1"].model_dump() == expected_snapshot
    assert set(processed_results_by_request_id) == {"R1"}

def test_新请求没有历史结果时返回None且不新增记录() -> None:
    """验证新请求未命中历史时交给后续流程，不自动生成处置。

    参数：无。
    返回：无，通过断言检查返回 None，且已处理结果字典仍为空。
    """
    # incoming_request：合法请求，请求 ID 使用 "R2"
    # processed_results_by_request_id：空字典
    # existing_disposition：调用 resolve_existing_redrive_result 的结果
    # 请断言 existing_disposition is None
    # 并断言 processed_results_by_request_id == {}

    incoming_request = RedriveRequest(
        redrive_request_id="R2",
        dlq_id="DLQ-2",
        actor_id="test_actor_id_1",
        reason="验证同一请求重发时复用原批准结果，不新增或改写处置"
    )
    processed_results_by_request_id = {}
    existing_disposition = resolve_existing_redrive_result(incoming_request, processed_results_by_request_id)
    assert existing_disposition is None
    assert processed_results_by_request_id == {}

def test_重复请求内容一致时复用原拒绝处置() -> None:
    """验证已被拒绝的同一请求再次提交时，复用原拒绝结果。

    参数：无。
    返回：无，通过断言检查返回原处置对象、拒绝动作、
          无重试计划，并确认已处理结果字典没有变化。
    """
    original_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="DLQ-1",
        actor_id="test_actor_id_1",
        reason="验证已被拒绝的同一请求再次提交时，复用原拒绝结果"
    )
    rejected_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="DLQ-1",
        redrive_request_id="R1",
        action=DispositionAction.REJECTED,
        actor_id="test_actor_id_1",
        reason="<UNK>",
        created_retry_plan_id=None,
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=rejected_disposition)
    processed_results_by_request_id = {"R1": processed_result}
    incoming_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="DLQ-1",
        actor_id="test_actor_id_1",
        reason="验证已被拒绝的同一请求再次提交时，复用原拒绝结果"
    )
    existing_disposition = resolve_existing_redrive_result(incoming_request, processed_results_by_request_id)
    assert existing_disposition is rejected_disposition
    assert existing_disposition.action == DispositionAction.REJECTED
    assert existing_disposition.created_retry_plan_id is None
    assert len(processed_results_by_request_id) == 1

def resolve_redrive_request_entry(
    incoming_request: RedriveRequest,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
    task_status: str,
    step_status: str,
    has_active_retry_plan: bool,
) -> DeadLetterDisposition | RedriveEligibilityDecision:
    """解析重新驱动请求的入口结果。

    参数：
        incoming_request：本次收到的重新驱动请求。
        processed_results_by_request_id：以请求 ID 为键的已处理结果字典。
        task_status：当前任务的权威状态。
        step_status：当前步骤的权威状态。
        has_active_retry_plan：当前步骤是否已有有效重试计划。
    返回：
        请求已有一致历史时，返回原 DeadLetterDisposition 处置；
        请求没有历史时，返回 RedriveEligibilityDecision 基础资格判断结果。
    异常：
        同一请求 ID 对应的请求内容不一致时，沿用历史解析函数抛出 ValueError。
    说明：
        本函数只负责编排历史查询和基础资格检查；
        不保存处置、不创建计划，也不代表完成正式授权。
    """
    existing_disposition = resolve_existing_redrive_result(incoming_request, processed_results_by_request_id)
    if existing_disposition is not None:
        return existing_disposition
    else:
        eligibility_decision = evaluate_redrive_basic_eligibility(task_status, step_status, has_active_retry_plan)
        return eligibility_decision

@pytest.mark.parametrize(
    "task_status, step_status, has_active_retry_plan, expected_eligible, expected_reason",
    [
        ("failed", "failed", False, True, "基础资格检查通过"),
        ("failed", "running", False, False, "步骤状态不是failed，不支持重新驱动"),
    ]
)
def test_新请求没有历史时进入基础资格判断(
    task_status: str,
    step_status: str,
    has_active_retry_plan: bool,
    expected_eligible: bool,
    expected_reason: str,
) -> None:
    """验证没有处理历史的新请求会进入基础资格判断。

    参数：
        task_status：当前任务状态。
        step_status：当前步骤状态。
        has_active_retry_plan：当前步骤是否已有有效重试计划。
        expected_eligible：预期是否通过基础资格检查。
        expected_reason：预期返回原因。
    返回：
        无，通过断言检查入口函数返回资格判断结果及对应原因。
    """
    incoming_request = RedriveRequest(
        redrive_request_id = "R2",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证同一请求 ID 下，内容一致的两个请求对象比较结果为 True"
    )
    processed_results_by_request_id = {}
    entry_result = resolve_redrive_request_entry(incoming_request, processed_results_by_request_id, task_status, step_status, has_active_retry_plan)
    assert isinstance(entry_result, RedriveEligibilityDecision)
    assert entry_result.eligible is expected_eligible
    assert entry_result.reason == expected_reason
    assert processed_results_by_request_id == {}

def test_重复请求命中历史时不重新进行资格判断() -> None:
    """验证内容一致的重复请求优先复用历史处置。

    参数：无。
    返回：
        无，通过断言检查即使当前基础条件不通过，
        入口函数仍返回原来保存的批准处置。
    """
    original_request = RedriveRequest(
        redrive_request_id = "R1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证内容一致的重复请求优先复用历史处置"
    )
    approved_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="R1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="验证内容一致的重复请求优先复用历史处置",
        created_retry_plan_id="P1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=approved_disposition)
    processed_results_by_request_id = {"R1": processed_result}
    incoming_request = RedriveRequest(
        redrive_request_id = "R1",
        dlq_id = "test_dlq_id_1",
        actor_id = "test_actor_id_1",
        reason = "验证内容一致的重复请求优先复用历史处置"
    )
    entry_result =resolve_redrive_request_entry(incoming_request, processed_results_by_request_id, "completed", "running", True)
    assert entry_result is approved_disposition
    assert entry_result.action == DispositionAction.REDRIVE
    assert entry_result.created_retry_plan_id == "P1"
    assert len(processed_results_by_request_id) == 1

def test_同一请求ID内容冲突时不进入资格判断() -> None:
    """验证请求 ID 已存在但内容不同时，入口函数优先报告冲突。

    参数：无。
    返回：
        无，通过断言检查入口函数抛出内容冲突异常，
        并确认原来的已处理结果没有被修改。
    """
    original_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="test_dlq_id_1",
        actor_id="test_actor_id_1",
        reason="验证请求 ID 已存在但内容不同时，入口函数优先报告冲突"
    )
    approved_disposition = DeadLetterDisposition(
        disposition_id="test_disposition_id_1",
        dlq_id="test_dlq_id_1",
        redrive_request_id="R1",
        action=DispositionAction.REDRIVE,
        actor_id="test_actor_id_1",
        reason="验证请求 ID 已存在但内容不同时，入口函数优先报告冲突",
        created_retry_plan_id="P1",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    processed_result = ProcessedRedriveResult(original_request=original_request, disposition=approved_disposition)
    processed_results_by_request_id = {"R1": processed_result}
    processed_snapshot = processed_result.model_dump()
    incoming_request = RedriveRequest(
        redrive_request_id="R1",
        dlq_id="test_dlq_id_2",
        actor_id="test_actor_id_1",
        reason="验证请求 ID 已存在但内容不同时，入口函数优先报告冲突"
    )
    with pytest.raises(ValueError, match="同一请求 ID 对应的请求内容不一致"):
        resolve_redrive_request_entry(incoming_request, processed_results_by_request_id, "failed", "failed", False)
    assert processed_results_by_request_id["R1"].model_dump() == processed_snapshot
    assert set(processed_results_by_request_id.keys()) == {"R1"}

def has_matching_redrive_target(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    current_task_id: str,
    current_step_id: str,
) -> bool:
    """判断重新驱动请求、死信记录与当前任务步骤是否一致。

    参数：
        incoming_request：本次收到的重新驱动请求。
        dead_letter_record：根据请求目标查询到的原始死信记录。
        current_task_id：当前权威任务的 ID。
        current_step_id：当前权威步骤的 ID。
    返回：
        请求的死信 ID、死信所属任务 ID、死信所属步骤 ID
        全部与当前目标一致时返回 True，否则返回 False。
    说明：
        本函数只比较关联标识，不修改任何对象，不检查状态、
        权限、有效计划，也不代表重新驱动已经获得批准。
    """
    same_dlq = incoming_request.dlq_id == dead_letter_record.dlq_id
    same_task = current_task_id == dead_letter_record.task_id
    same_step = current_step_id == dead_letter_record.step_id
    return all([same_dlq, same_task, same_step])

@pytest.mark.parametrize(
    "request_dlq_id, current_task_id, current_step_id, expected_matching",
    [
        ("DLQ-1", "TASK-1", "STEP-1", True),
        ("DLQ-2", "TASK-1", "STEP-1", False),
        ("DLQ-1", "TASK-2", "STEP-1", False),
        ("DLQ-1", "TASK-1", "STEP-2", False),
    ]
)
def test_重新驱动目标关联必须全部一致(
    request_dlq_id: str,
    current_task_id: str,
    current_step_id: str,
    expected_matching: bool,
) -> None:
    """验证死信、任务和步骤关联必须全部一致。

    参数：
        request_dlq_id：重新驱动请求携带的死信 ID。
        current_task_id：当前权威任务 ID。
        current_step_id：当前权威步骤 ID。
        expected_matching：预期的关联判断结果。
    返回：
        无，通过参数化断言检查全部一致及各类单字段不一致。
    """
    dead_letter_record = DeadLetterRecord(
        dlq_id= "DLQ-1",
        task_id="TASK-1",
        task_version=1,
        step_id="STEP-1",
        source_trace_id="test_source_trace_id_1",
        source_span_id="test_source_span_id_1",
        event_id="test_event_id_1",
        claim_id="test_claim_id_1",
        payload_reference="test_payload_reference",

        attempt_count=3,

        error_code="UPSTREAM_TIMEOUT",
        error_reason="调用上游服务超时",

        worker_id="test_worker_id",
        created_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
    )
    incoming_request = RedriveRequest(
        redrive_request_id= "R1",
        dlq_id=request_dlq_id,
        actor_id="test_actor_id_1",
        reason="验证死信、任务和步骤关联必须全部一致"
    )
    is_matching_target = has_matching_redrive_target(
        incoming_request=incoming_request,
        dead_letter_record=dead_letter_record,
        current_task_id=current_task_id,
        current_step_id=current_step_id,
    )
    assert is_matching_target is expected_matching

def evaluate_redrive_target_consistency(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    current_task_id: str,
    current_step_id: str,
) -> RedriveEligibilityDecision:
    """检查重新驱动目标关联，并返回明确的判断原因。

    参数：
        incoming_request：本次收到的重新驱动请求。
        dead_letter_record：根据目标查询到的原始死信记录。
        current_task_id：当前权威任务 ID。
        current_step_id：当前权威步骤 ID。
    返回：
        返回包含 eligible 和 reason 的关联检查结果。
        请求死信、当前任务和当前步骤全部匹配时 eligible 为 True；
        任一关联不一致时 eligible 为 False，并返回首个失败原因。
    说明：
        本函数只检查关联关系，不检查状态、权限和有效计划，
        不创建处置或计划，也不代表重新驱动已经获得批准。
    """
    same_dlq = incoming_request.dlq_id == dead_letter_record.dlq_id
    same_task = current_task_id == dead_letter_record.task_id
    same_step = current_step_id == dead_letter_record.step_id
    if not same_dlq:
        return RedriveEligibilityDecision(
            eligible = False,
            reason = "请求指向的死信记录不一致"
        )
    if not same_task:
        return RedriveEligibilityDecision(
            eligible=False,
            reason="死信记录不属于当前任务"
        )
    if not same_step:
        return RedriveEligibilityDecision(
            eligible=False,
            reason="死信记录不属于当前步骤"
        )
    return RedriveEligibilityDecision(
        eligible=True,
        reason="重新驱动目标关联检查通过"
    )
def evaluate_new_redrive_request_eligibility(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    current_task_id: str,
    current_step_id: str,
    task_status: str,
    step_status: str,
    has_active_retry_plan: bool,
) -> RedriveEligibilityDecision:
    """评估一个无处理历史的新重新驱动请求。

    参数：
        incoming_request：本次收到的新请求。
        dead_letter_record：请求目标对应的原始死信记录。
        current_task_id：当前权威任务 ID。
        current_step_id：当前权威步骤 ID。
        task_status：当前权威任务状态。
        step_status：当前权威步骤状态。
        has_active_retry_plan：当前步骤是否已有有效重试计划。
    返回：
        目标关联不通过时返回对应关联失败结果；
        目标关联通过后，返回基础资格检查结果。
    说明：
        本函数只进行资格评估，不创建处置、计划或保存记录，
        资格通过也不等于正式批准已经提交。
    """
    target_decision = evaluate_redrive_target_consistency(
        incoming_request=incoming_request,
        dead_letter_record=dead_letter_record,
        current_task_id=current_task_id,
        current_step_id=current_step_id,
    )
    if not target_decision.eligible:
        return RedriveEligibilityDecision(
            eligible = False,
            reason = target_decision.reason
        )
    basic_decision = evaluate_redrive_basic_eligibility(
        current_task_status = task_status,
        current_step_status = step_status,
        has_active_retry_plan = has_active_retry_plan,
    )

    if not basic_decision.eligible:
        return RedriveEligibilityDecision(
            eligible = False,
            reason = basic_decision.reason
        )
    return RedriveEligibilityDecision(
        eligible = True,
        reason = "目标关联检查与基础资格判断均通过"
    )

def create_rejected_redrive_disposition(
    incoming_request: RedriveRequest,
    eligibility_decision: RedriveEligibilityDecision,
    disposition_id: str,
    reviewer_id: str,
    created_at: datetime,
) -> DeadLetterDisposition:
    """根据未通过的资格判断创建重新驱动拒绝处置。

    参数：
        incoming_request：本次收到的重新驱动请求。
        eligibility_decision：此前得到的资格判断结果。
        disposition_id：本次人工处置记录 ID。
        reviewer_id：作出处置决定的审核人 ID。
        created_at：处置创建时间，必须携带有效时区。
    返回：
        返回动作是 REJECTED、且不包含重试计划 ID 的死信处置记录。
        处置原因来自资格判断原因，不覆盖请求本身的申请原因。
    异常：
        当资格判断已经通过时抛出 ValueError，因为通过结果不能创建拒绝处置。
    说明：
        本函数只构造处置对象，不保存数据，也不创建重试计划。
    """
    if eligibility_decision.eligible:
        raise ValueError("资格检查通过时不能创建拒绝处置")
    rejected_disposition = DeadLetterDisposition(
        disposition_id=disposition_id,
        dlq_id=incoming_request.dlq_id,
        redrive_request_id=incoming_request.redrive_request_id,
        action=DispositionAction.REJECTED,
        actor_id=reviewer_id,
        reason=eligibility_decision.reason,
        created_retry_plan_id=None,
        created_at=created_at,
    )
    return rejected_disposition

def create_approved_redrive_disposition(
    incoming_request: RedriveRequest,
    eligibility_decision: RedriveEligibilityDecision,
    disposition_id: str,
    reviewer_id: str,
    approval_reason: str,
    retry_plan_id: str,
    created_at: datetime,
) -> DeadLetterDisposition:
    """根据通过的资格判断创建重新驱动批准处置。

    参数：
        incoming_request：本次收到的重新驱动请求。
        eligibility_decision：此前得到的资格判断结果。
        disposition_id：本次人工处置记录 ID。
        reviewer_id：作出批准决定的审核人 ID。
        approval_reason：审核人给出的批准原因。
        retry_plan_id：本次批准所关联的重试计划 ID。
        created_at：处置创建时间，必须携带有效时区。
    返回：
        返回动作是 REDRIVE、并关联重试计划 ID 的死信处置记录。
    异常：
        资格判断未通过时抛出 ValueError；
        重试计划 ID 为空时由 DeadLetterDisposition 契约拒绝。
    说明：
        请求原因、资格判断原因和批准原因是三个不同概念。
        本函数只构造批准处置，不保存记录，也不执行重试计划。
    """
    if not eligibility_decision.eligible:
        raise ValueError("资格检查未通过时不能创建批准处置")
    approved_disposition = DeadLetterDisposition(
        disposition_id=disposition_id,
        dlq_id=incoming_request.dlq_id,
        redrive_request_id=incoming_request.redrive_request_id,
        action=DispositionAction.REDRIVE,
        actor_id=reviewer_id,
        reason=approval_reason,
        created_retry_plan_id=retry_plan_id,
        created_at=created_at,
    )
    return approved_disposition

class RedriveRetryPlan(BaseModel):
    """描述一次获批后等待调度的重新驱动计划。

    字段：
        retry_plan_id：重试计划 ID。
        redrive_request_id：来源请求 ID。
        dlq_id：来源死信记录 ID。
        task_id：目标任务 ID。
        step_id：目标步骤 ID。
        scheduled_at：计划执行时间。
        created_at：计划创建时间。

    说明：
        计划尚未被 worker 领取，因此不包含新的 claim_id。
        本模型只用于第四天内存练习，不代表正式持久化契约。
    """

    retry_plan_id: str
    redrive_request_id: str
    dlq_id: str
    task_id: str
    step_id: str
    scheduled_at: datetime
    created_at: datetime

def create_redrive_retry_plan(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    retry_plan_id: str,
    scheduled_at: datetime,
    created_at: datetime,
) -> RedriveRetryPlan:
    """根据已通过资格检查的请求构造重新驱动计划。

    参数：
        incoming_request：已通过资格检查的重新驱动请求。
        dead_letter_record：请求对应的原始死信记录。
        retry_plan_id：新创建的重试计划 ID。
        scheduled_at：计划调度执行时间。
        created_at：计划创建时间。
    返回：
        返回关联请求、死信、任务和步骤的新重试计划。
    异常：
        请求的死信 ID 与死信记录不一致时抛出 ValueError。
    说明：
        本函数只构造计划，不保存、不发布，也不生成 claim_id。
    """
    if incoming_request.dlq_id != dead_letter_record.dlq_id:
        raise ValueError("重新驱动请求与死信记录不一致")
    redrive_plan = RedriveRetryPlan(
        retry_plan_id=retry_plan_id,
        redrive_request_id=incoming_request.redrive_request_id,
        dlq_id=dead_letter_record.dlq_id,
        task_id=dead_letter_record.task_id,
        step_id=dead_letter_record.step_id,
        scheduled_at=scheduled_at,
        created_at=created_at,
    )
    return redrive_plan

def create_approved_redrive_artifacts(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    eligibility_decision: RedriveEligibilityDecision,
    retry_plan_id: str,
    disposition_id: str,
    reviewer_id: str,
    approval_reason: str,
    scheduled_at: datetime,
    created_at: datetime,
) -> tuple[RedriveRetryPlan, DeadLetterDisposition]:
    """构造一次批准重新驱动所需的计划和处置记录。

    参数：
        incoming_request：已通过资格检查的重新驱动请求。
        dead_letter_record：请求对应的原始死信记录。
        eligibility_decision：此前得到的资格判断结果。
        retry_plan_id：新重试计划 ID。
        disposition_id：新处置记录 ID。
        reviewer_id：作出批准决定的审核人 ID。
        approval_reason：审核人给出的批准原因。
        scheduled_at：计划调度执行时间。
        created_at：计划及处置的创建时间。
    返回：
        二元组，第一个元素是重试计划，第二个元素是批准处置。
    说明：
        本函数复用已有构造函数，保证批准处置引用同一个计划 ID。
        只构造内存对象，不保存、不发布，也不执行重试。
    """
    redrive_plan = create_redrive_retry_plan(
        incoming_request=incoming_request,
        dead_letter_record=dead_letter_record,
        retry_plan_id=retry_plan_id,
        scheduled_at=scheduled_at,
        created_at=created_at,
    )
    approved_disposition = create_approved_redrive_disposition(
        incoming_request=incoming_request,
        eligibility_decision=eligibility_decision,
        disposition_id=disposition_id,
        reviewer_id=reviewer_id,
        approval_reason=approval_reason,
        retry_plan_id=redrive_plan.retry_plan_id,
        created_at=created_at,
    )
    return redrive_plan, approved_disposition

def save_approved_redrive_in_memory(
    original_request: RedriveRequest,
    redrive_plan: RedriveRetryPlan,
    approved_disposition: DeadLetterDisposition,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
    retry_plans_by_id: dict[str, RedriveRetryPlan],
) -> ProcessedRedriveResult:
    """校验并保存一次批准重新驱动的内存结果。

    参数：
        original_request：本次获批的原始重新驱动请求。
        redrive_plan：本次创建的重试计划。
        approved_disposition：引用该计划的批准处置。
        processed_results_by_request_id：按请求 ID 保存的处理结果字典。
        retry_plans_by_id：按计划 ID 保存的重试计划字典。
    返回：
        返回已保存的 ProcessedRedriveResult。
    异常：
        请求、死信或计划关联不一致时抛出 ValueError；
        请求 ID 或计划 ID 已存在时抛出 ValueError。
    说明：
        所有关联检查必须在修改字典之前完成。
        本函数只模拟单进程内存提交，不提供并发、进程崩溃或数据库事务原子性。
    """
    if approved_disposition.action != DispositionAction.REDRIVE:
        raise ValueError("保存对象不是批准重新驱动处置")
    if any([original_request.redrive_request_id != redrive_plan.redrive_request_id,
            original_request.redrive_request_id != approved_disposition.redrive_request_id,
            redrive_plan.redrive_request_id != approved_disposition.redrive_request_id
            ]):
        raise ValueError("批准产物的请求 ID 不一致")


    if any([original_request.dlq_id != redrive_plan.dlq_id,
            original_request.dlq_id != approved_disposition.dlq_id,
            redrive_plan.dlq_id != approved_disposition.dlq_id
            ]):
        raise ValueError("批准产物的死信 ID 不一致")
    if redrive_plan.retry_plan_id != approved_disposition.created_retry_plan_id:
        raise ValueError("批准处置与重试计划 ID 不一致")

    if processed_results_by_request_id.get(approved_disposition.redrive_request_id, None) is not None:
        raise ValueError("重新驱动请求已存在处理结果")
    if retry_plans_by_id.get(redrive_plan.retry_plan_id, None) is not None:
        raise ValueError("重试计划 ID 已存在")
    processed_result = ProcessedRedriveResult(
        original_request = original_request,
        disposition = approved_disposition,
    )
    processed_results_by_request_id[
        approved_disposition.redrive_request_id
    ] = processed_result
    retry_plans_by_id[redrive_plan.retry_plan_id] = redrive_plan
    return processed_result

def save_rejected_redrive_in_memory(
    original_request: RedriveRequest,
    rejected_disposition: DeadLetterDisposition,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
) -> ProcessedRedriveResult:
    """校验并保存一次被拒绝的重新驱动结果。

    参数：
        original_request：本次被拒绝的原始重新驱动请求。
        rejected_disposition：该请求对应的拒绝处置。
        processed_results_by_request_id：按请求 ID 保存的处理结果字典。
    返回：
        返回已保存的 ProcessedRedriveResult。
    异常：
        处置不是 REJECTED、关联 ID 不一致或请求已经存在时抛出 ValueError。
    说明：
        拒绝结果同样需要保存，以便相同请求再次到达时复用原决定。
        本函数不创建重试计划，只模拟单进程内存保存。
    """
    if rejected_disposition.action != DispositionAction.REJECTED:
        raise ValueError("保存对象不是拒绝重新驱动处置")
    if original_request.redrive_request_id != rejected_disposition.redrive_request_id:
        raise ValueError("拒绝结果的请求 ID 不一致")

    if original_request.dlq_id != rejected_disposition.dlq_id:
        raise ValueError("拒绝结果的死信 ID 不一致")

    if processed_results_by_request_id.get(rejected_disposition.redrive_request_id, None) is not None:
        raise ValueError("重新驱动请求已存在处理结果")
    processed_result = ProcessedRedriveResult(
        original_request = original_request,
        disposition = rejected_disposition,
    )
    processed_results_by_request_id[
        rejected_disposition.redrive_request_id
    ] = processed_result
    return processed_result

class RedriveCommitReceipt(BaseModel):
    """记录重新驱动决定已经提交的回执。

    字段：
        receipt_id：提交回执 ID。
        redrive_request_id：对应的重新驱动请求 ID。
        disposition_id：对应的处置记录 ID。
        action：最终提交的处置动作。
        retry_plan_id：批准时关联的计划 ID，拒绝时为 None。
        created_at：回执创建时间。

    说明：
        回执只证明处置结果已经提交，不代表重试任务执行成功。
    """

    receipt_id: str
    redrive_request_id: str
    disposition_id: str
    action: DispositionAction
    retry_plan_id: str | None = None
    created_at: datetime

def create_redrive_commit_receipt(
    disposition: DeadLetterDisposition,
    receipt_id: str,
    created_at: datetime,
) -> RedriveCommitReceipt:
    """根据已提交的处置构造重新驱动提交回执。

    参数：
        disposition：已经确定并准备提交的死信处置记录。
        receipt_id：本次提交回执 ID。
        created_at：回执创建时间。
    返回：
        返回关联请求、处置动作和可选计划 ID 的提交回执。
    说明：
        批准处置会携带计划 ID，拒绝处置的计划 ID 为 None。
        本函数只构造回执对象，不保存数据，也不代表业务执行成功。
    """
    commit_receipt = RedriveCommitReceipt(
        receipt_id = receipt_id,
        redrive_request_id = disposition.redrive_request_id,
        disposition_id = disposition.disposition_id,
        action = disposition.action,
        retry_plan_id = disposition.created_retry_plan_id,
        created_at = created_at,
    )
    return commit_receipt

def save_redrive_commit_bundle_in_memory(
    original_request: RedriveRequest,
    disposition: DeadLetterDisposition,
    receipt: RedriveCommitReceipt,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
    receipts_by_id: dict[str, RedriveCommitReceipt],
    retry_plans_by_id: dict[str, RedriveRetryPlan],
    redrive_plan: RedriveRetryPlan | None = None,
) -> ProcessedRedriveResult:
    """统一校验并保存重新驱动请求、处置、可选计划和提交回执。

    参数：
        original_request：本次处理的原始重新驱动请求。
        disposition：本次人工处置决定。
        receipt：本次决定对应的提交回执。
        processed_results_by_request_id：按请求 ID 保存的处理结果字典。
        receipts_by_id：按回执 ID 保存的提交回执字典。
        retry_plans_by_id：按计划 ID 保存的重试计划字典。
        redrive_plan：批准时必须提供的重试计划；拒绝时必须为 None。
    返回：
        返回保存后的 ProcessedRedriveResult。
    异常：
        请求、处置、回执或计划关联不一致时抛出 ValueError；
        任一需要保存的 ID 已存在时抛出 ValueError。
    说明：
        所有检查及组合对象构造必须发生在首次写入字典之前。
        本函数只是内存事务边界练习，不具备数据库事务和并发原子性。
    """
    if original_request.redrive_request_id != disposition.redrive_request_id:
        raise ValueError("请求与处置记录的请求ID不一致")
    if original_request.dlq_id != disposition.dlq_id:
        raise ValueError("请求与处置记录的死信ID不一致")
    if receipt.redrive_request_id != disposition.redrive_request_id:
        raise ValueError("回执与处置记录的请求ID不一致")
    if receipt.disposition_id != disposition.disposition_id:
        raise ValueError("回执与处置记录的处置ID不一致")
    if receipt.action != disposition.action:
        raise ValueError("回执与处置记录的动作不一致")
    if receipt.retry_plan_id != disposition.created_retry_plan_id:
        raise ValueError("回执与处置记录的计划ID不一致")
    if disposition.action == DispositionAction.REDRIVE:
        if redrive_plan is None:
            raise ValueError("动作为重试时计划为空")
        if not (
                redrive_plan.retry_plan_id == disposition.created_retry_plan_id
                == receipt.retry_plan_id
        ):
            raise ValueError("记录和回执中的计划ID与计划中的计划ID不一致")
        if redrive_plan.redrive_request_id != original_request.redrive_request_id:
            raise ValueError("计划中的请求ID与请求的请求ID不一致")
        if redrive_plan.dlq_id != original_request.dlq_id:
            raise ValueError("计划中的死信ID与请求中的死信ID不一致")
    else:
        if redrive_plan is not None:
            raise ValueError("动作不是批准时重试计划不为None")

    same_request_id = processed_results_by_request_id.get(original_request.redrive_request_id, None)
    same_receipt_id = receipts_by_id.get(receipt.receipt_id, None)
    if redrive_plan is not None:
        same_plan_id = retry_plans_by_id.get(redrive_plan.retry_plan_id, None)
        if same_plan_id is not None:
            raise ValueError("计划已存在")
    if same_request_id is not None:
        raise ValueError("该请求的处理结果已存在")
    if same_receipt_id is not None:
        raise ValueError("回执已存在")

    processed_result = ProcessedRedriveResult(
        original_request = original_request,
        disposition = disposition,
    )
    processed_results_by_request_id[original_request.redrive_request_id] = processed_result
    receipts_by_id[receipt.receipt_id] = receipt
    if redrive_plan is not None:
        retry_plans_by_id[redrive_plan.retry_plan_id] = redrive_plan
    return processed_result

def process_redrive_request_in_memory(
    incoming_request: RedriveRequest,
    dead_letter_record: DeadLetterRecord,
    current_task_id: str,
    current_step_id: str,
    task_status: str,
    step_status: str,
    has_active_retry_plan: bool,
    disposition_id: str,
    receipt_id: str,
    reviewer_id: str,
    approval_reason: str,
    created_at: datetime,
    processed_results_by_request_id: dict[str, ProcessedRedriveResult],
    receipts_by_id: dict[str, RedriveCommitReceipt],
    retry_plans_by_id: dict[str, RedriveRetryPlan],
    retry_plan_id: str | None = None,
    scheduled_at: datetime | None = None,
) -> DeadLetterDisposition:
    """处理一次重新驱动申请并返回最终处置。

    参数：
        incoming_request：本次收到的重新驱动请求。
        dead_letter_record：请求目标对应的原始死信记录。
        current_task_id：当前权威任务 ID。
        current_step_id：当前权威步骤 ID。
        task_status：当前权威任务状态。
        step_status：当前权威步骤状态。
        has_active_retry_plan：当前步骤是否已有有效重试计划。
        disposition_id：准备创建的处置记录 ID。
        receipt_id：准备创建的提交回执 ID。
        reviewer_id：本次审核人 ID。
        approval_reason：资格通过后的批准原因。
        created_at：处置、计划和回执的创建时间。
        processed_results_by_request_id：按请求 ID 保存的处理结果。
        receipts_by_id：按回执 ID 保存的提交回执。
        retry_plans_by_id：按计划 ID 保存的重试计划。
        retry_plan_id：批准时使用的计划 ID；拒绝路径可以为 None。
        scheduled_at：批准计划的调度时间；拒绝路径可以为 None。
    返回：
        命中一致历史时返回原处置；
        新请求资格失败时返回并保存拒绝处置；
        新请求资格通过时返回并保存批准处置。
    异常：
        请求内容冲突时抛出 ValueError；
        资格通过但缺少计划 ID 或调度时间时抛出 ValueError；
        保存对象关联错误或 ID 重复时抛出 ValueError。
    说明：
        本练习假设调用方已经完成审核人权限校验。
        资格通过表示当前审核操作进入批准路径。
        本函数只模拟内存编排，不提供生产级事务原子性。
    """
    existing_disposition = resolve_existing_redrive_result(
        incoming_request,
        processed_results_by_request_id
    )
    if existing_disposition is not None:
        return existing_disposition
    eligibility_decision = evaluate_new_redrive_request_eligibility(
        incoming_request,
        dead_letter_record,
        current_task_id,
        current_step_id,
        task_status,
        step_status,
        has_active_retry_plan,
    )
    if not eligibility_decision.eligible:
        rejected_disposition = create_rejected_redrive_disposition(
            incoming_request=incoming_request,
            eligibility_decision=eligibility_decision,
            disposition_id=disposition_id,
            reviewer_id=reviewer_id,
            created_at=created_at,
        )
        commit_receipt = RedriveCommitReceipt(
            receipt_id=receipt_id,
            redrive_request_id=incoming_request.redrive_request_id,
            disposition_id=disposition_id,
            action=DispositionAction.REJECTED,
            retry_plan_id=None,
            created_at=created_at
        )
        _ = save_redrive_commit_bundle_in_memory(
            original_request=incoming_request,
            disposition=rejected_disposition,
            receipt=commit_receipt,
            processed_results_by_request_id=processed_results_by_request_id,
            receipts_by_id=receipts_by_id,
            retry_plans_by_id=retry_plans_by_id,
            redrive_plan=None
        )
        return rejected_disposition

    if retry_plan_id is None or scheduled_at is None:
        raise ValueError("资格检查通过后必须提供计划 ID 和调度时间")

    redrive_plan, approved_disposition = create_approved_redrive_artifacts(
        incoming_request=incoming_request,
        dead_letter_record=dead_letter_record,
        eligibility_decision=eligibility_decision,
        retry_plan_id=retry_plan_id,
        disposition_id=disposition_id,
        reviewer_id=reviewer_id,
        approval_reason=approval_reason,
        scheduled_at=scheduled_at,
        created_at=created_at,
    )
    commit_receipt = RedriveCommitReceipt(
        receipt_id=receipt_id,
        redrive_request_id=incoming_request.redrive_request_id,
        disposition_id=disposition_id,
        action=DispositionAction.REDRIVE,
        retry_plan_id=redrive_plan.retry_plan_id,
        created_at=created_at
    )
    _ = save_redrive_commit_bundle_in_memory(
        original_request=incoming_request,
        disposition=approved_disposition,
        receipt=commit_receipt,
        processed_results_by_request_id=processed_results_by_request_id,
        receipts_by_id=receipts_by_id,
        retry_plans_by_id=retry_plans_by_id,
        redrive_plan=redrive_plan
    )
    return approved_disposition
