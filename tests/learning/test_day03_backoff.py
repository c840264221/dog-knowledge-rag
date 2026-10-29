import pytest
from datetime import datetime, timedelta, timezone
from enum import Enum
from pydantic import BaseModel, model_validator, ValidationError


class RetryAction(str, Enum):
    """定义重试策略允许返回的处理动作。"""

    STOP = "stop"
    RETRY = "retry"
    SEND_TO_DLQ = "send_to_dlq"

class RetryDecision(BaseModel):
    """保存重试策略的处理决定。"""
    action: RetryAction
    reason: str
    next_retry_at: datetime | None = None

    @model_validator(mode='after')
    def validate_retry_schedule(self):
        """校验动作与重试时间的一致性；self为当前对象，返回校验后的对象。"""
        if self.action == RetryAction.RETRY and self.next_retry_at is None:
            raise ValueError("动作为重试时，下次执行时间不能为空")
        if self.action != RetryAction.RETRY and self.next_retry_at is not None:
            raise ValueError("动作不是重试时，下次执行时间必须为空")
        return self

def calculate_backoff_seconds(
    attempt_number: int,
    base_delay_seconds: int,
    max_delay_seconds: int,
) -> int:
    """
    计算指数级退避时间
    :param attempt_number: 执行次数，是刚失败的第几次执行，从1开始
    :param base_delay_seconds: 基础等待时间，单位是秒
    :param max_delay_seconds: 最大等待时间，单位是秒
    :return: 最终等待时间的值  类型是int
    :raises ValueError:当执行次数不大于0、基础等待时间不大于0、最大等待时间不大于0、最大等待时间小于基础等待时间时抛出ValueError异常
    """
    if attempt_number <= 0 :
        raise ValueError("执行次数必须大于0")
    if base_delay_seconds <= 0 :
        raise ValueError("基础等待时间必须大于0")
    if max_delay_seconds <= 0 or max_delay_seconds < base_delay_seconds:
        raise ValueError("最大等待时间必须大于0且不能小于基础等待时间")
    waiting_time = min(max_delay_seconds,base_delay_seconds * (2 ** (attempt_number-1)))
    return waiting_time

import random
def 计算指数退避后加入抖动范围的等待时间(
        upper_bound: int,
)-> float:
    """在0到退避上限之间随机生成等待秒数。"""
    if upper_bound <= 0 :
        raise ValueError("最大时间取值范围必须大于0")
    waiting_time = random.uniform(0, upper_bound)
    return waiting_time

def 计算实际退避时间(
        attempt_number: int,
        base_delay_seconds: int,
        max_delay_seconds: int,

) -> float:
    """
    使用指数退避来计算时间抖动上限，然后使用抖动方式得到最终等待时间
    :param attempt_number:刚失败的执行次数，从1开始
    :param base_delay_seconds:基础等待时间，单位是秒，应大于0
    :param max_delay_seconds:最大等待时间，单位是秒，应大于0且不小于基础等待时间
    :return:返回最终等待时间，单位是秒，类型为浮点数
    :raises ValueError:以上参数不符合校验时抛出ValueError
    """
    upper_bound = calculate_backoff_seconds(attempt_number, base_delay_seconds, max_delay_seconds)
    waiting_time = 计算指数退避后加入抖动范围的等待时间(upper_bound)
    return waiting_time

def calculate_next_retry_at(
    now: datetime,
    waiting_seconds: float,
) -> datetime:
    """
    接收带时区的时间，统一返回 UTC 时间
    :param now: 带有效时区的当前时间，不要求输入为UTC。
    :param waiting_seconds: 最终等待时间，单位为秒
    :return: 下次执行时间，为utc时区
    """
    if waiting_seconds < 0 :
        raise ValueError("最终等待时间不能小于0")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("接受的时间参数必须携带有效时区")
    utc_now = now.astimezone(timezone.utc)
    next_retry_at = utc_now + timedelta(seconds=waiting_seconds)
    return next_retry_at

def 重试策略PDP(
    error_retryable: bool,
    attempt_count: int,
    max_attempts: int,
    base_delay_seconds: int,
    max_delay_seconds: int,
    now: datetime,
    deadline: datetime
) -> RetryDecision:
    """
    重试的决定策略，根据输入参数判断是否重试、重试时间
    :param error_retryable: 是否允许重试
    :param attempt_count: 刚失败的执行次数，最小值为1
    :param max_attempts: 最大执行次数，最小值为1
    :param base_delay_seconds: 基础等待时间，单位为秒
    :param max_delay_seconds: 最大等待时间，单位为秒，且不小于基础等待时间
    :param now: 携带有效时区的时间参数
    :param deadline: 携带有效时区的截止时间，函数内部统一转换为UTC
    :return: 重试决定策略，包含下次的动作、原因，如果可以重试的话还会包含下次执行的时间
    """
    if attempt_count < 1 :
        raise ValueError("执行次数必须是大于1的整数")
    if max_attempts < 1 :
        raise ValueError("最大执行次数必须大于1")

    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("截止时间必须携带有效时区")
    deadline = deadline.astimezone(timezone.utc)

    if not error_retryable:
        return RetryDecision(
            action=RetryAction.STOP,
            reason="不允许重试",
            next_retry_at=None,
        )
    else:
        if attempt_count >= max_attempts :
            return RetryDecision(
                action=RetryAction.SEND_TO_DLQ,
                reason="达到执行次数上限，转为死信队列",
                next_retry_at=None,
            )
        else:
            waiting_time = 计算实际退避时间(attempt_count, base_delay_seconds, max_delay_seconds)
            next_retry_at = calculate_next_retry_at(now,waiting_time)
            if next_retry_at < deadline :
                return RetryDecision(
                    action=RetryAction.RETRY,
                    reason="条件符合重试，并计算好了下次执行时间",
                    next_retry_at=next_retry_at
                )
            else:
                return RetryDecision(
                    action=RetryAction.SEND_TO_DLQ,
                    reason="下次执行时间超过了截止时间，放入死信队列",
                    next_retry_at=None
                )

def test_backoff_should_respect_max_delay():
    """
    测试指数级退避函数可以在等待时间大于最大等待时间时返回最大等待时间
    """
    waiting_time = calculate_backoff_seconds(4,3,20)
    assert waiting_time == 20

def test_backoff_should_return_waiting_time():
    """
    测试等待时间不大于最大等待时间时可以正常返回等待时间
    """
    waiting_time = calculate_backoff_seconds(3,3,20)
    assert waiting_time == 12

def test_退避函数应该在执行次数不大于0时抛出值异常():
    """
    测试参数中执行次数不大于0时是否正常抛出ValueError
    """
    with pytest.raises(ValueError):
        calculate_backoff_seconds(0,3,20)

def test_退避函数应该在最大等待时间小于基础等待时间或不大于0时抛出值异常():
    """
    退避函数应该在最大等待时间小于基础等待时间时是否抛出值异常
    """
    with pytest.raises(ValueError):
        calculate_backoff_seconds(4,3,1)
    with pytest.raises(ValueError):
        calculate_backoff_seconds(4, 3, 0)

def test_退避函数应该在基础等待时间为0时抛出值异常():
    """测试退避函数是否在基础等待时间为0的时候抛出ValueError"""
    with pytest.raises(ValueError):
        calculate_backoff_seconds(4,0,20)

def test_计算实际退避时间的取值范围在0到最大抖动范围之间():
    """测试计算实际退避时间的取值范围在0到最大抖动范围之间"""
    waiting_time = 计算实际退避时间(4,3,20)
    assert 0 <= waiting_time <= 20

def test_抖动函数允许返回零(monkeypatch):
    """验证抖动函数正确返回随机函数给出的值。"""
    def 随机取值函数的替身(min_num, max_num):
        """作为random.uniform的替身，让他返回固定值0，而不是随机范围内取值"""
        assert min_num == 0
        assert max_num == 20
        return 0
    monkeypatch.setattr(random,"uniform", 随机取值函数的替身)

    抖动时间 = 计算指数退避后加入抖动范围的等待时间(20)

    assert 抖动时间 == 0


def test_最终执行时间等于UTC时区的当前时间加上最终等待时间():
    """测试最终执行时间等于UTC时区的当前时间加上最终等待时间"""
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime(2026, 9, 3, 16, 0, tzinfo=beijing_tz)
    next_retry_at = calculate_next_retry_at(now,7.5)

    assert next_retry_at.tzinfo == timezone.utc
    assert next_retry_at == datetime(2026, 9, 3, 8, 0, 7, 500000, tzinfo=timezone.utc)


def test_计算下次执行时间函数应该对没有携带时区的事件参数抛出ValueError():
    """测试计算下次执行时间函数是否对没有携带时区的时间参数抛出ValueError"""
    now = datetime(2026, 9, 3, 16, 0)
    with pytest.raises(ValueError):
        calculate_next_retry_at(now,7.5)

def test_重试决策中重试动作却没有下次执行时间应该抛出ValidationError():
    """测试重试决策中重试动作却没有下次执行时间是否抛出ValidationError"""
    with pytest.raises(ValidationError,match="动作为重试时"):
        RetryDecision(
            action=RetryAction.RETRY,
            reason="测试重试策略契约中传入的参数不符合校验是否会返回ValidationError",
            next_retry_at=None,
        )

def test_重试决策中停止动作时却有下次执行时间应该抛出ValidationError():
    """测试重试决策中停止动作时却有下次执行时间是否抛出ValidationError"""
    with pytest.raises(ValidationError,match="动作不是重试时"):
        RetryDecision(
            action=RetryAction.STOP,
            reason="测试重试决策中停止动作时却有下次执行时间是否抛出ValidationError",
            next_retry_at=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        )

def test_重试策略PDP不可重试时即使有执行次数也不能安排继续执行():
    """测试重试策略PDP在不可重试时即使有执行次数也不能继续安排执行"""
    retry_decision = 重试策略PDP(
        error_retryable=False,
        attempt_count=1,
        max_attempts=3,
        base_delay_seconds=3,
        max_delay_seconds=20,
        now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        deadline=datetime(2026, 9, 3, 9, 0, 0, tzinfo=timezone.utc),
    )

    assert retry_decision.action == RetryAction.STOP
    assert retry_decision.next_retry_at is None

def test_重试策略PDP执行次数耗尽时不能安排继续执行():
    """重试策略PDP执行次数耗尽时不能安排继续执行"""
    retry_decision = 重试策略PDP(
        error_retryable=True,
        attempt_count=3,
        max_attempts=3,
        base_delay_seconds=3,
        max_delay_seconds=20,
        now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        deadline=datetime(2026, 9, 3, 9, 0, 0, tzinfo=timezone.utc),
    )

    assert retry_decision.action == RetryAction.SEND_TO_DLQ
    assert retry_decision.next_retry_at is None

def test_重试策略PDP在允许重试时返回正确的决策(monkeypatch):
    """测试重试策略PDP在允许重试时返回正确的决策"""
    def 随机函数的替身(min_num, max_num):
        assert min_num == 0
        assert max_num == 3
        return 2.5
    monkeypatch.setattr(random, "uniform", 随机函数的替身)
    retry_decision = 重试策略PDP(
        error_retryable=True,
        attempt_count=1,
        max_attempts=3,
        base_delay_seconds=3,
        max_delay_seconds=20,
        now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        deadline=datetime(2026, 9, 3, 9, 0, 0, tzinfo=timezone.utc)
    )
    assert retry_decision.action == RetryAction.RETRY
    assert retry_decision.next_retry_at == datetime(2026, 9, 3, 8, 0, 2, 500000, tzinfo=timezone.utc)

def test_重试策略PDP在其他参数合法时max_attempts为0应该抛出ValueError():
    """测试重试策略PDP在其他参数合法时max_attempts为0应该抛出ValueError"""
    with pytest.raises(ValueError,match="最大执行次数"):
        重试策略PDP(
            error_retryable=True,
            attempt_count=1,
            max_attempts=0,
            base_delay_seconds=3,
            max_delay_seconds=20,
            now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
            deadline=datetime(2026, 9, 3, 9, 0, 0, tzinfo=timezone.utc)
        )

def test_重试策略PDP在下次执行时间刚好等于截止时间的情况应该进入死信队列(monkeypatch):
    """测试重试策略PDP在下次执行时间刚好等于截止时间的情况是否进入死信队列"""
    def 随机函数的替身(min_num, max_num):
        assert min_num == 0
        assert max_num == 3
        return 2.5
    monkeypatch.setattr(random, "uniform", 随机函数的替身)
    retry_decision = 重试策略PDP(
        error_retryable=True,
        attempt_count=1,
        max_attempts=3,
        base_delay_seconds=3,
        max_delay_seconds=20,
        now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
        deadline=datetime(2026, 9, 3, 8, 0, 2,500000, tzinfo=timezone.utc)
    )
    assert retry_decision.action == RetryAction.SEND_TO_DLQ
    assert retry_decision.next_retry_at is None


def test_重试策略PDP在其他参数合法的情况下传入不带时区的截止时间应该抛出ValueError():
    """测试重试策略PDP在其他参数合法的情况下传入不带时区的截止时间是否抛出ValueError"""

    with pytest.raises(ValueError,match="截止时间必须携带有效时区"):
        重试策略PDP(
            error_retryable=False,
            attempt_count=1,
            max_attempts=3,
            base_delay_seconds=3,
            max_delay_seconds=20,
            now=datetime(2026, 9, 3, 8, 0, 0, tzinfo=timezone.utc),
            deadline=datetime(2026, 9, 3, 9, 0,0)
        )

