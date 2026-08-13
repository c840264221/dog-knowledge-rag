"""LLM 调用软预算检查单元测试。"""

from __future__ import annotations

from src.runtime.observability.llm_call_budget import (
    LLMCallBudgetLimits,
    evaluate_llm_call_budgets,
    render_llm_budget_warning,
)
from src.runtime.observability.llm_call_records import (
    LLMCallPurpose,
    LLMCallMetadata,
    LLMCallRecord,
)


def _build_call(
    *,
    call_id: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> LLMCallRecord:
    """
    构建一条预算测试调用记录。

    参数含义：
        call_id：逻辑调用编号。
        purpose：调用目的。
        input_tokens：输入 Token 数。
        output_tokens：输出 Token 数。
        latency_ms：逻辑调用耗时。

    返回值含义：
        LLMCallRecord：可以交给软预算检查器的标准调用记录。
    """

    return LLMCallRecord(
        call_id=call_id,
        trace_id="trace-budget",
        metadata=LLMCallMetadata(
            call_purpose=purpose,
            component="generate_node",
            agent_name="dog_knowledge_agent",
            step_id="step-training",
        ),
        requested_model="main-model",
        final_model="main-model",
        attempt_count=1,
        status="completed",
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def test_empty_budget_configuration_should_only_collect_metrics() -> None:
    """
    验证空预算配置只统计实际调用，不产生超限结论。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluation = evaluate_llm_call_budgets(
        calls=[
            _build_call(
                call_id="call-1",
                purpose="answer_generation",
                input_tokens=1000,
                output_tokens=1000,
                latency_ms=5000,
            )
        ],
        budgets_by_purpose={},
    )

    assert evaluation.status == "not_configured"
    assert evaluation.violations == []


def test_budget_should_be_evaluated_independently_by_purpose() -> None:
    """
    验证不同调用目的使用各自阈值，未配置的用途不参与检查。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluation = evaluate_llm_call_budgets(
        calls=[
            _build_call(
                call_id="routing-call",
                purpose="routing_decision",
                input_tokens=90,
                output_tokens=20,
                latency_ms=100,
            ),
            _build_call(
                call_id="answer-call",
                purpose="answer_generation",
                input_tokens=900,
                output_tokens=600,
                latency_ms=1000,
            ),
        ],
        budgets_by_purpose={
            "routing_decision": {
                "max_total_tokens_per_call": 100,
            },
        },
    )

    assert evaluation.status == "exceeded"
    assert evaluation.evaluated_call_count == 1
    assert len(evaluation.violations) == 1
    violation = evaluation.violations[0]
    assert violation.call_purpose == "routing_decision"
    assert violation.call_id == "routing-call"
    assert violation.actual == 110
    assert violation.limit == 100


def test_budget_should_check_per_call_and_request_purpose_totals() -> None:
    """
    验证预算同时检查单次消耗和同一目的的请求级累计消耗。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls = [
        _build_call(
            call_id="answer-1",
            purpose="answer_generation",
            input_tokens=80,
            output_tokens=30,
            latency_ms=400,
        ),
        _build_call(
            call_id="answer-2",
            purpose="answer_generation",
            input_tokens=70,
            output_tokens=40,
            latency_ms=500,
        ),
    ]
    evaluation = evaluate_llm_call_budgets(
        calls=calls,
        budgets_by_purpose={
            "answer_generation": LLMCallBudgetLimits(
                max_output_tokens_per_call=35,
                max_logical_calls_per_request=1,
                max_total_tokens_per_request=200,
                max_latency_ms_per_request=800,
            )
        },
    )

    assert evaluation.status == "exceeded"
    assert {
        (violation.scope, violation.metric)
        for violation in evaluation.violations
    } == {
        ("single_call", "output_tokens_per_call"),
        ("request_purpose", "logical_calls_per_request"),
        ("request_purpose", "total_tokens_per_request"),
        ("request_purpose", "latency_ms_per_request"),
    }
    assert "answer_generation" in render_llm_budget_warning(evaluation)


def test_configured_budget_without_violation_should_pass() -> None:
    """
    验证存在阈值但实际消耗未超限时返回预算内状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluation = evaluate_llm_call_budgets(
        calls=[
            _build_call(
                call_id="call-ok",
                purpose="routing_decision",
                input_tokens=10,
                output_tokens=5,
                latency_ms=50,
            )
        ],
        budgets_by_purpose={
            "routing_decision": {
                "max_total_tokens_per_call": 100,
            }
        },
    )

    assert evaluation.status == "within_budget"
    assert evaluation.violations == []


def test_budget_should_accept_enum_purpose_from_settings() -> None:
    """
    验证 Settings 校验后的枚举键可以被预算检查器正常读取。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluation = evaluate_llm_call_budgets(
        calls=[
            _build_call(
                call_id="enum-purpose",
                purpose="routing_decision",
                input_tokens=10,
                output_tokens=5,
                latency_ms=50,
            )
        ],
        budgets_by_purpose={
            LLMCallPurpose.ROUTING_DECISION: LLMCallBudgetLimits(
                max_total_tokens_per_call=100,
            )
        },
    )

    assert evaluation.status == "within_budget"
    assert "routing_decision" in evaluation.configured_budgets
