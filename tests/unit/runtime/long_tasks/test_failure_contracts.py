"""长任务失败治理数据契约单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.runtime.long_tasks.failure_contracts import (
    LongTaskDeadLetterRecord,
    LongTaskFailureDecision,
    LongTaskRetryPlan,
)
from src.runtime.long_tasks.failure_policy import (
    decide_long_task_failure,
)


def build_retry_decision_data() -> dict[str, object]:
    """
    构建失败决定测试使用的合法重试数据。

    参数：
        无。

    返回值：
        dict[str, object]：
            可以创建 LongTaskFailureDecision 的基础字段。
    """

    decided_at = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)
    return {
        "task_id": "task_001",
        "step_id": "step_001",
        "failure_category": "transient",
        "action": "retry",
        "attempt_count": 1,
        "max_attempts": 3,
        "error_code": "UPSTREAM_TIMEOUT",
        "error_message": "调用上游服务超时",
        "reason": "临时错误且仍有剩余执行次数",
        "next_retry_at": decided_at + timedelta(seconds=30),
        "source_trace_id": "trace_001",
        "source_span_id": "span_001",
        "source_claim_id": "claim_001",
        "decided_at": decided_at,
    }


def test_failure_decision_should_accept_valid_retry() -> None:
    """
    验证合法重试决定会保留动作、时间和执行链路。

    参数：
        无。

    返回值：
        None：断言全部成立时测试通过。
    """

    decision = LongTaskFailureDecision.model_validate(
        build_retry_decision_data()
    )

    assert decision.action == "retry"
    assert decision.attempt_count == 1
    assert decision.next_retry_at is not None
    assert decision.source_trace_id == "trace_001"
    assert decision.source_span_id == "span_001"
    assert decision.source_claim_id == "claim_001"


@pytest.mark.parametrize(
    ("changed_fields", "expected_message"),
    [
        (
            {"next_retry_at": None},
            "重试决定必须提供 next_retry_at",
        ),
        (
            {"action": "dead_letter"},
            "非重试决定不能提供 next_retry_at",
        ),
        (
            {"attempt_count": 3},
            "达到最大执行次数后不能继续安排重试",
        ),
        (
            {"source_trace_id": None},
            "source_span_id 必须关联 source_trace_id",
        ),
    ],
)
def test_failure_decision_should_reject_conflicting_fields(
    changed_fields: dict[str, object],
    expected_message: str,
) -> None:
    """
    验证失败决定中的互相矛盾字段会被拒绝。

    参数：
        changed_fields:
            用来覆盖合法基础数据并制造矛盾的字段。
        expected_message:
            当前矛盾应该产生的错误说明。

    返回值：
        None：模型按预期拒绝矛盾数据时测试通过。
    """

    decision_data = build_retry_decision_data()
    decision_data.update(changed_fields)

    with pytest.raises(ValidationError, match=expected_message):
        LongTaskFailureDecision.model_validate(decision_data)


def test_transient_failure_should_create_stable_retry_decision() -> None:
    """
    验证相同临时失败会得到稳定且不超过退避上限的重试时间。

    参数：
        无。

    返回值：
        None：两次决定一致且满足退避范围时测试通过。
    """

    now = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)
    decision_fields = {
        "task_id": "task_001",
        "step_id": "step_001",
        "failure_category": "transient",
        "error_code": "UPSTREAM_TIMEOUT",
        "error_message": "调用上游服务超时",
        "attempt_count": 2,
        "max_attempts": 3,
        "now": now,
        "base_delay_seconds": 10.0,
        "max_delay_seconds": 60.0,
        "source_trace_id": "trace_002",
        "source_span_id": "span_002",
        "source_claim_id": "claim_002",
    }

    first_decision = decide_long_task_failure(**decision_fields)
    repeated_decision = decide_long_task_failure(**decision_fields)

    assert first_decision.action == "retry"
    assert first_decision.next_retry_at is not None
    assert now <= first_decision.next_retry_at <= now + timedelta(seconds=20)
    assert repeated_decision.next_retry_at == first_decision.next_retry_at


@pytest.mark.parametrize(
    ("failure_category", "attempt_count", "deadline_seconds", "action"),
    [
        ("business", 1, None, "fail_task"),
        ("permanent", 1, None, "dead_letter"),
        ("unknown", 1, None, "dead_letter"),
        ("transient", 3, None, "dead_letter"),
        ("transient", 1, 0, "dead_letter"),
    ],
)
def test_failure_policy_should_route_non_retry_scenarios(
    failure_category: str,
    attempt_count: int,
    deadline_seconds: int | None,
    action: str,
) -> None:
    """
    验证业务、永久、未知、次数耗尽和截止时间场景不会继续重试。

    参数：
        failure_category:
            当前测试使用的错误类别。
        attempt_count:
            刚刚失败的执行次数。
        deadline_seconds:
            相对于当前时间的截止秒数；None 表示没有截止时间。
        action:
            当前场景预期得到的处理动作。

    返回值：
        None：失败策略返回预期动作时测试通过。
    """

    now = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)
    retry_deadline = (
        now + timedelta(seconds=deadline_seconds)
        if deadline_seconds is not None
        else None
    )

    decision = decide_long_task_failure(
        task_id="task_001",
        step_id="step_001",
        failure_category=failure_category,  # type: ignore[arg-type]
        error_code="TEST_FAILURE",
        error_message="测试失败",
        attempt_count=attempt_count,
        max_attempts=3,
        now=now,
        base_delay_seconds=10.0,
        max_delay_seconds=60.0,
        retry_deadline=retry_deadline,
    )

    assert decision.action == action
    assert decision.next_retry_at is None


def test_retry_plan_should_keep_schedule_separate_from_execution_claim() -> None:
    """
    验证待调度计划保存未来时间，但不会提前生成执行消息和领取令牌。

    参数：
        无。

    返回值：
        None：计划只保存调度事实时测试通过。
    """

    created_at = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)
    plan = LongTaskRetryPlan(
        retry_plan_id="retry_plan_001",
        task_id="task_001",
        step_id="step_001",
        failure_task_version=3,
        failure_step_version=2,
        failed_attempt_count=1,
        next_attempt_count=2,
        max_attempts=3,
        scheduled_at=created_at + timedelta(seconds=30),
        error_code="UPSTREAM_TIMEOUT",
        failure_commit_id="batch:task_001:batch_001",
        source_trace_id="trace_001",
        source_span_id="span_001",
        source_claim_id="claim_001",
        created_at=created_at,
    )

    assert plan.status == "scheduled"
    assert plan.next_attempt_count == 2
    assert plan.queue_message_id is None
    assert plan.dispatched_at is None
    assert plan.source_claim_id == "claim_001"

    with pytest.raises(
        ValidationError,
        match="已取消计划不能包含消息编号或发布时间",
    ):
        LongTaskRetryPlan(
            **{
                **plan.model_dump(mode="python"),
                "status": "cancelled",
                "cancellation_reason": "当前步骤已经完成",
                "queue_message_id": "message_001",
                "dispatched_at": plan.scheduled_at,
            }
        )


def test_dead_letter_should_preserve_original_failure_fact() -> None:
    """
    验证死信记录会保留最后失败的版本、错误和执行身份。

    参数：
        无。

    返回值：
        None：死信完整保留原失败事实时测试通过。
    """

    dead_letter = LongTaskDeadLetterRecord(
        dead_letter_id="dead_letter_001",
        task_id="task_001",
        step_id="step_001",
        failure_task_version=5,
        failure_step_version=4,
        failed_attempt_count=3,
        max_attempts=3,
        failure_category="transient",
        error_code="UPSTREAM_TIMEOUT",
        error_message="第三次调用上游服务仍然超时",
        dead_letter_reason="临时错误已达到最大执行次数",
        failure_commit_id="batch:task_001:batch_003",
        payload_reference="artifact://task_001/step_001/input",
        worker_id="worker_002",
        source_trace_id="trace_003",
        source_span_id="span_003",
        source_claim_id="claim_003",
        created_at=datetime(2026, 9, 11, 8, 5, tzinfo=timezone.utc),
    )

    assert dead_letter.failed_attempt_count == 3
    assert dead_letter.error_code == "UPSTREAM_TIMEOUT"
    assert dead_letter.source_trace_id == "trace_003"
    assert dead_letter.source_span_id == "span_003"
    assert dead_letter.source_claim_id == "claim_003"
