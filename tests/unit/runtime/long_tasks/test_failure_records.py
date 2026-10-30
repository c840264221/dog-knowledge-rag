"""长任务失败决定到持久化记录的构造测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.runtime.long_tasks.failure_contracts import (
    LongTaskFailureDecision,
)
from src.runtime.long_tasks.failure_records import (
    build_dead_letter_record,
    build_retry_plan,
)


def build_failure_decision(
    *,
    action: str,
) -> LongTaskFailureDecision:
    """
    构建失败记录测试使用的决定。

    参数：
        action:
            当前决定动作，测试中使用 retry 或 dead_letter。

    返回值：
        LongTaskFailureDecision：
            可以传给失败记录构造函数的决定。
    """

    decided_at = datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc)
    return LongTaskFailureDecision(
        task_id="task_001",
        step_id="step_001",
        failure_category="transient",
        action=action,  # type: ignore[arg-type]
        attempt_count=1,
        max_attempts=3,
        error_code="UPSTREAM_TIMEOUT",
        error_message="调用上游服务超时",
        reason=(
            "临时错误且仍允许重试"
            if action == "retry"
            else "临时错误已达到当前治理边界"
        ),
        next_retry_at=(
            decided_at + timedelta(seconds=30)
            if action == "retry"
            else None
        ),
        source_trace_id="trace_001",
        source_span_id="span_001",
        source_claim_id="claim_001",
        decided_at=decided_at,
    )


def test_failure_builders_should_map_decision_to_stable_records() -> None:
    """
    验证重试计划和死信会复制决定事实并生成稳定编号。

    参数：
        无。

    返回值：
        None：字段映射正确且重复构造编号一致时测试通过。
    """

    retry_decision = build_failure_decision(action="retry")
    retry_plan = build_retry_plan(
        decision=retry_decision,
        failure_task_version=4,
        failure_step_version=3,
        failure_commit_id="batch:task_001:batch_001",
    )
    repeated_plan = build_retry_plan(
        decision=retry_decision,
        failure_task_version=4,
        failure_step_version=3,
        failure_commit_id="batch:task_001:batch_001",
    )

    assert retry_plan.retry_plan_id == repeated_plan.retry_plan_id
    assert retry_plan.next_attempt_count == 2
    assert retry_plan.scheduled_at == retry_decision.next_retry_at
    assert retry_plan.source_claim_id == "claim_001"

    dead_letter_decision = build_failure_decision(action="dead_letter")
    dead_letter = build_dead_letter_record(
        decision=dead_letter_decision,
        failure_task_version=5,
        failure_step_version=4,
        failure_commit_id="batch:task_001:batch_002",
        payload_reference="artifact://task_001/step_001/input",
        worker_id="worker_001",
    )
    repeated_dead_letter = build_dead_letter_record(
        decision=dead_letter_decision,
        failure_task_version=5,
        failure_step_version=4,
        failure_commit_id="batch:task_001:batch_002",
        payload_reference="artifact://task_001/step_001/input",
        worker_id="worker_001",
    )

    assert dead_letter.dead_letter_id == repeated_dead_letter.dead_letter_id
    assert dead_letter.error_code == dead_letter_decision.error_code
    assert dead_letter.dead_letter_reason == dead_letter_decision.reason
    assert dead_letter.source_claim_id == "claim_001"


@pytest.mark.parametrize(
    ("builder_name", "decision_action", "expected_message"),
    [
        (
            "retry",
            "dead_letter",
            "只有 retry 决定可以创建重试计划",
        ),
        (
            "dead_letter",
            "retry",
            "只有 dead_letter 决定可以创建死信记录",
        ),
    ],
)
def test_failure_builders_should_reject_mismatched_action(
    builder_name: str,
    decision_action: str,
    expected_message: str,
) -> None:
    """
    验证失败动作与目标记录不一致时不能错误创建记录。

    参数：
        builder_name:
            当前测试调用的构造函数名称。
        decision_action:
            用来制造动作不匹配的失败决定。
        expected_message:
            当前错误应该返回的说明。

    返回值：
        None：构造函数拒绝错误动作时测试通过。
    """

    decision = build_failure_decision(action=decision_action)

    with pytest.raises(ValueError, match=expected_message):
        if builder_name == "retry":
            build_retry_plan(
                decision=decision,
                failure_task_version=4,
                failure_step_version=3,
                failure_commit_id="batch:task_001:batch_001",
            )
        else:
            build_dead_letter_record(
                decision=decision,
                failure_task_version=4,
                failure_step_version=3,
                failure_commit_id="batch:task_001:batch_001",
            )
