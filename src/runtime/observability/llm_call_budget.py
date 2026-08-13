"""LLM 调用软预算配置契约与超限检查。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.runtime.observability.llm_call_records import (
    LLMCallPurpose,
    LLMCallRecord,
)


BudgetEvaluationStatus = Literal[
    "not_configured",
    "within_budget",
    "exceeded",
]
BudgetViolationScope = Literal["single_call", "request_purpose"]


class LLMCallBudgetLimits(BaseModel):
    """
    定义某一种 LLM 调用目的允许使用的软预算。

    功能：
        分别限制单次调用和一次用户请求内同一调用目的的累计消耗。所有字段
        默认都是 None，表示该指标暂时只统计、不设置阈值。

    参数含义：
        max_logical_calls_per_request：同一目的在一次请求中的逻辑调用数上限。
        max_input_tokens_per_call：单次调用输入 Token 上限。
        max_output_tokens_per_call：单次调用输出 Token 上限。
        max_total_tokens_per_call：单次调用总 Token 上限。
        max_total_tokens_per_request：同一目的在一次请求中的累计 Token 上限。
        max_latency_ms_per_call：单次调用耗时上限，单位毫秒。
        max_latency_ms_per_request：同一目的累计调用耗时上限，单位毫秒。

    返回值含义：
        LLMCallBudgetLimits：经过 Pydantic 校验的单用途软预算配置。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_logical_calls_per_request: int | None = Field(default=None, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=0)
    max_output_tokens_per_call: int | None = Field(default=None, ge=0)
    max_total_tokens_per_call: int | None = Field(default=None, ge=0)
    max_total_tokens_per_request: int | None = Field(default=None, ge=0)
    max_latency_ms_per_call: float | None = Field(default=None, ge=0)
    max_latency_ms_per_request: float | None = Field(default=None, ge=0)

    def has_configured_limit(self) -> bool:
        """
        判断当前用途是否至少配置了一个有效阈值。

        参数含义：
            无。

        返回值含义：
            bool：存在至少一个非 None 阈值时返回 True。
        """

        return any(
            value is not None
            for value in self.model_dump(mode="python").values()
        )


class LLMBudgetViolation(BaseModel):
    """
    描述一条 LLM 软预算超限证据。

    功能：
        保存超限指标、实际值、阈值和调用身份，供日志、JSON 与 Markdown
        报告共同使用。它只表示告警，不会中断 LLM 调用。

    参数含义：
        call_purpose：发生超限的受控调用目的。
        metric：超限指标名称。
        scope：单次调用超限或请求内同一目的累计超限。
        actual：实际消耗值。
        limit：配置阈值。
        call_id：单次调用超限时对应的调用编号；累计超限时为空。
        agent_name：单次调用所属 Agent。
        step_id：单次调用所属多智能体步骤。

    返回值含义：
        LLMBudgetViolation：可审计的软预算超限记录。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_purpose: str
    metric: str
    scope: BudgetViolationScope
    actual: float = Field(ge=0)
    limit: float = Field(ge=0)
    call_id: str = ""
    agent_name: str = ""
    step_id: str = ""


class LLMBudgetEvaluation(BaseModel):
    """
    保存一次请求的 LLM 软预算检查结论。

    功能：
        区分未配置阈值、预算内和已超限三种状态，并保留本次实际使用的
        配置快照与全部超限证据。

    参数含义：
        status：预算检查状态。
        configured_budgets：按调用目的保存的有效阈值快照。
        evaluated_call_count：参与预算检查的逻辑调用数量。
        violations：全部软预算超限证据。

    返回值含义：
        LLMBudgetEvaluation：可以直接写入 LLM 调用报告的预算检查结果。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: BudgetEvaluationStatus
    configured_budgets: dict[str, LLMCallBudgetLimits] = Field(
        default_factory=dict
    )
    evaluated_call_count: int = Field(default=0, ge=0)
    violations: list[LLMBudgetViolation] = Field(default_factory=list)


def evaluate_llm_call_budgets(
    *,
    calls: Sequence[LLMCallRecord],
    budgets_by_purpose: Mapping[
        LLMCallPurpose | str,
        LLMCallBudgetLimits | Mapping[str, int | float | None],
    ] | None,
) -> LLMBudgetEvaluation:
    """
    按调用目的检查本次请求的 LLM 软预算。

    功能：
        先校验并过滤没有任何阈值的配置，再分别执行单次调用检查和同一目的
        的请求级累计检查。函数只返回结论，不抛出预算异常、不停止调用。

    参数含义：
        calls：本次请求中通过契约校验的 LLM 逻辑调用明细。
        budgets_by_purpose：调用目的到软预算阈值的映射；空映射表示未启用阈值。

    返回值含义：
        LLMBudgetEvaluation：未配置、预算内或已超限的结构化检查结果。
    """

    configured_budgets = _normalize_budget_configuration(
        budgets_by_purpose
    )
    if not configured_budgets:
        return LLMBudgetEvaluation(status="not_configured")

    calls_by_purpose: dict[str, list[LLMCallRecord]] = defaultdict(list)
    for call in calls:
        purpose = str(call.metadata.call_purpose or "").strip()
        if purpose in configured_budgets:
            calls_by_purpose[purpose].append(call)

    violations: list[LLMBudgetViolation] = []
    for purpose, purpose_calls in sorted(calls_by_purpose.items()):
        limits = configured_budgets[purpose]
        for call in purpose_calls:
            violations.extend(
                _evaluate_single_call(
                    call=call,
                    purpose=purpose,
                    limits=limits,
                )
            )
        violations.extend(
            _evaluate_request_purpose_totals(
                calls=purpose_calls,
                purpose=purpose,
                limits=limits,
            )
        )

    return LLMBudgetEvaluation(
        status="exceeded" if violations else "within_budget",
        configured_budgets=configured_budgets,
        evaluated_call_count=sum(len(items) for items in calls_by_purpose.values()),
        violations=violations,
    )


def render_llm_budget_warning(evaluation: LLMBudgetEvaluation) -> str:
    """
    将预算超限结论压缩成不含 Prompt 和回答内容的日志警告。

    参数含义：
        evaluation：已经完成的 LLM 软预算检查结果。

    返回值含义：
        str：可以直接交给 logger.warning 的单行警告文本。
    """

    purposes = sorted(
        {violation.call_purpose for violation in evaluation.violations}
    )
    return (
        "LLM 软预算超限: "
        f"violations={len(evaluation.violations)}, "
        f"purposes={purposes}"
    )


def _normalize_budget_configuration(
    raw_budgets: Mapping[
        LLMCallPurpose | str,
        LLMCallBudgetLimits | Mapping[str, int | float | None],
    ] | None,
) -> dict[str, LLMCallBudgetLimits]:
    """
    校验预算配置并移除没有设置任何阈值的用途。

    参数含义：
        raw_budgets：Settings 或测试传入的原始用途预算映射。

    返回值含义：
        dict[str, LLMCallBudgetLimits]：以标准用途字符串为键的有效预算配置。
    """

    normalized: dict[str, LLMCallBudgetLimits] = {}
    for raw_purpose, raw_limits in (raw_budgets or {}).items():
        purpose = (
            raw_purpose.value
            if isinstance(raw_purpose, LLMCallPurpose)
            else LLMCallPurpose(str(raw_purpose)).value
        )
        limits = (
            raw_limits
            if isinstance(raw_limits, LLMCallBudgetLimits)
            else LLMCallBudgetLimits.model_validate(raw_limits)
        )
        if limits.has_configured_limit():
            normalized[purpose] = limits
    return normalized


def _evaluate_single_call(
    *,
    call: LLMCallRecord,
    purpose: str,
    limits: LLMCallBudgetLimits,
) -> list[LLMBudgetViolation]:
    """
    检查一条逻辑调用的 Token 和耗时是否超限。

    参数含义：
        call：当前逻辑调用明细。
        purpose：标准调用目的字符串。
        limits：该调用目的的预算阈值。

    返回值含义：
        list[LLMBudgetViolation]：当前调用产生的全部超限证据。
    """

    checks = (
        ("input_tokens_per_call", call.input_tokens, limits.max_input_tokens_per_call),
        ("output_tokens_per_call", call.output_tokens, limits.max_output_tokens_per_call),
        ("total_tokens_per_call", call.total_tokens, limits.max_total_tokens_per_call),
        ("latency_ms_per_call", call.latency_ms, limits.max_latency_ms_per_call),
    )
    violations: list[LLMBudgetViolation] = []
    for metric, actual, limit in checks:
        if limit is None or actual <= limit:
            continue
        violations.append(
            LLMBudgetViolation(
                call_purpose=purpose,
                metric=metric,
                scope="single_call",
                actual=actual,
                limit=limit,
                call_id=call.call_id,
                agent_name=call.metadata.agent_name,
                step_id=call.metadata.step_id,
            )
        )
    return violations


def _evaluate_request_purpose_totals(
    *,
    calls: Sequence[LLMCallRecord],
    purpose: str,
    limits: LLMCallBudgetLimits,
) -> list[LLMBudgetViolation]:
    """
    检查同一调用目的在一次请求中的累计消耗。

    参数含义：
        calls：当前调用目的对应的全部逻辑调用。
        purpose：标准调用目的字符串。
        limits：该调用目的的预算阈值。

    返回值含义：
        list[LLMBudgetViolation]：请求级累计超限证据。
    """

    checks = (
        ("logical_calls_per_request", len(calls), limits.max_logical_calls_per_request),
        (
            "total_tokens_per_request",
            sum(call.total_tokens for call in calls),
            limits.max_total_tokens_per_request,
        ),
        (
            "latency_ms_per_request",
            sum(call.latency_ms for call in calls),
            limits.max_latency_ms_per_request,
        ),
    )
    violations: list[LLMBudgetViolation] = []
    for metric, actual, limit in checks:
        if limit is None or actual <= limit:
            continue
        violations.append(
            LLMBudgetViolation(
                call_purpose=purpose,
                metric=metric,
                scope="request_purpose",
                actual=actual,
                limit=limit,
            )
        )
    return violations
