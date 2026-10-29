"""长任务失败后的纯决策策略。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from src.runtime.long_tasks.failure_contracts import (
    LongTaskFailureCategory,
    LongTaskFailureDecision,
)


def calculate_retry_delay_seconds(
    *,
    task_id: str,
    step_id: str,
    error_code: str,
    attempt_count: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> float:
    """
    计算带稳定抖动的指数退避秒数。

    功能：
        先根据当前执行次数计算指数增长的等待上限，
        再根据任务、步骤、错误代码和执行次数生成稳定抖动。
        相同逻辑失败重复计算时会得到相同结果，不会破坏提交幂等。

    参数：
        task_id:
            当前失败所属的任务编号。
        step_id:
            当前失败所属的步骤编号。
        error_code:
            当前失败的稳定错误代码。
        attempt_count:
            刚刚失败的是第几次执行，从 1 开始。
        base_delay_seconds:
            第一次失败对应的基础退避上限秒数。
        max_delay_seconds:
            所有执行次数都不能超过的退避上限秒数。

    返回值：
        float：
            从 0 到当前退避上限之间的稳定抖动秒数。

    异常：
        ValueError：
            文本参数为空、执行次数不合法或退避配置不合法时抛出。
    """

    normalized_task_id = str(task_id or "").strip()
    normalized_step_id = str(step_id or "").strip()
    normalized_error_code = str(error_code or "").strip()
    if not normalized_task_id:
        raise ValueError("task_id 不能为空")
    if not normalized_step_id:
        raise ValueError("step_id 不能为空")
    if not normalized_error_code:
        raise ValueError("error_code 不能为空")
    if attempt_count < 1:
        raise ValueError("attempt_count 必须大于等于 1")
    if base_delay_seconds <= 0:
        raise ValueError("base_delay_seconds 必须大于 0")
    if max_delay_seconds < base_delay_seconds:
        raise ValueError(
            "max_delay_seconds 不能小于 base_delay_seconds"
        )

    delay_upper_bound = min(
        max_delay_seconds,
        base_delay_seconds * (2 ** (attempt_count - 1)),
    )
    jitter_key = ":".join(
        (
            normalized_task_id,
            normalized_step_id,
            normalized_error_code,
            str(attempt_count),
        )
    )
    digest = hashlib.sha256(jitter_key.encode("utf-8")).digest()
    jitter_ratio = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return delay_upper_bound * jitter_ratio


def decide_long_task_failure(
    *,
    task_id: str,
    step_id: str,
    failure_category: LongTaskFailureCategory,
    error_code: str,
    error_message: str,
    attempt_count: int,
    max_attempts: int,
    now: datetime,
    base_delay_seconds: float,
    max_delay_seconds: float,
    retry_deadline: datetime | None = None,
    source_trace_id: str | None = None,
    source_span_id: str | None = None,
    source_claim_id: str | None = None,
) -> LongTaskFailureDecision:
    """
    根据一次步骤失败生成后续处理决定。

    功能：
        业务错误直接结束任务；永久错误和未知错误进入死信；
        临时错误在次数和截止时间允许时安排重试，否则进入死信。

    参数：
        task_id/step_id:
            本次失败所属的任务和步骤编号。
        failure_category:
            已经完成的错误分类。
        error_code/error_message:
            程序使用的稳定错误代码和供人查看的错误说明。
        attempt_count/max_attempts:
            刚失败的执行次数和最大允许执行次数。
        now:
            生成本次决定的当前时间，必须携带有效时区。
        base_delay_seconds/max_delay_seconds:
            指数退避的基础上限和总上限秒数。
        retry_deadline:
            可选重试截止时间；计算出的重试时间不能晚于它。
        source_trace_id/source_span_id/source_claim_id:
            产生本次失败的执行链路、具体操作和领取令牌。

    返回值：
        LongTaskFailureDecision：
            动作为 retry、dead_letter 或 fail_task 的失败决定。

    异常：
        ValueError：
            时间或退避配置不合法时抛出。
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now 必须包含时区")
    decided_at = now.astimezone(timezone.utc)

    normalized_deadline = None
    if retry_deadline is not None:
        if (
            retry_deadline.tzinfo is None
            or retry_deadline.utcoffset() is None
        ):
            raise ValueError("retry_deadline 必须包含时区")
        normalized_deadline = retry_deadline.astimezone(timezone.utc)

    common_fields = {
        "task_id": task_id,
        "step_id": step_id,
        "failure_category": failure_category,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "error_code": error_code,
        "error_message": error_message,
        "source_trace_id": source_trace_id,
        "source_span_id": source_span_id,
        "source_claim_id": source_claim_id,
        "decided_at": decided_at,
    }

    if failure_category == "business":
        return LongTaskFailureDecision(
            **common_fields,
            action="fail_task",
            reason="业务条件不允许继续执行，不进行自动重试",
        )

    if failure_category in {"permanent", "unknown"}:
        return LongTaskFailureDecision(
            **common_fields,
            action="dead_letter",
            reason=(
                "永久错误需要人工处理"
                if failure_category == "permanent"
                else "错误类型无法确认，进入死信等待人工判断"
            ),
        )

    if attempt_count >= max_attempts:
        return LongTaskFailureDecision(
            **common_fields,
            action="dead_letter",
            reason="临时错误已达到最大执行次数",
        )

    retry_delay_seconds = calculate_retry_delay_seconds(
        task_id=task_id,
        step_id=step_id,
        error_code=error_code,
        attempt_count=attempt_count,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )
    next_retry_at = decided_at + timedelta(seconds=retry_delay_seconds)

    if (
        normalized_deadline is not None
        and next_retry_at > normalized_deadline
    ):
        return LongTaskFailureDecision(
            **common_fields,
            action="dead_letter",
            reason="下一次执行时间超过重试截止时间",
        )

    return LongTaskFailureDecision(
        **common_fields,
        action="retry",
        reason="临时错误且执行次数和截止时间仍允许重试",
        next_retry_at=next_retry_at,
    )
