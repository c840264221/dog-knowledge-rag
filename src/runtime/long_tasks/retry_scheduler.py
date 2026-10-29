"""把到期重试计划推进到消息发布完成的一轮调度服务。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from src.runtime.long_tasks.contracts import LongTaskCommitReceipt, utc_now
from src.runtime.long_tasks.failure_contracts import (
    LongTaskRetryDispatchCycleResult,
)
from src.runtime.long_tasks.failure_service import (
    build_long_task_retry_cancellation_request,
    build_long_task_retry_dispatch_request,
    find_long_task_retry_ineligibility_reason,
)
from src.runtime.long_tasks.store import (
    LongTaskReliableCommitStore,
    LongTaskStoreError,
)


IdFactory = Callable[[], str]
Clock = Callable[[], datetime]


def _default_dispatch_claim_id_factory() -> str:
    """
    生成一次重试计划发布权领取使用的唯一令牌。

    参数：
        无。

    返回值：
        str：带 dispatch_claim 前缀的随机编号。
    """

    return f"dispatch_claim_{uuid4().hex}"


class LongTaskRetryScheduler:
    """
    执行一轮到期重试计划的领取和消息发布。

    功能：
        查询到期计划，逐个领取发布权，重新读取当前任务，构建发布请求，
        再通过 Store 原子发布消息。当前类不创建常驻循环，由外层运行时
        决定多久调用一次。

    参数：
        store:
            提供任务读取、到期查询、计划领取和原子发布能力的 Store。
        dispatcher_id:
            当前调度器进程或实例的稳定身份。
        dispatch_claim_id_factory:
            可选发布权令牌生成函数，默认使用随机 UUID。
        clock:
            可选当前时间函数，默认返回 UTC 时间。

    返回值：
        LongTaskRetryScheduler：可以按需执行一轮调度的服务对象。
    """

    def __init__(
        self,
        store: LongTaskReliableCommitStore,
        *,
        dispatcher_id: str,
        dispatch_claim_id_factory: IdFactory | None = None,
        clock: Clock | None = None,
    ) -> None:
        normalized_dispatcher_id = str(dispatcher_id or "").strip()
        if not normalized_dispatcher_id:
            raise ValueError("dispatcher_id 不能为空")
        self._store = store
        self._dispatcher_id = normalized_dispatcher_id
        self._dispatch_claim_id_factory = (
            dispatch_claim_id_factory
            or _default_dispatch_claim_id_factory
        )
        self._clock = clock or utc_now

    async def dispatch_due_retries(
        self,
        *,
        due_at: datetime | None = None,
        limit: int = 100,
        lease_seconds: float = 30.0,
    ) -> LongTaskRetryDispatchCycleResult:
        """
        查询并处理一轮已经到期的重试计划。

        参数：
            due_at:
                可选查询截止时间；未提供时使用当前 UTC 时间。
            limit:
                本轮最多读取的到期计划数量。
            lease_seconds:
                每个计划发布权的租约秒数。

        返回值：
            LongTaskRetryDispatchCycleResult：
                记录到期、领取、发布、跳过和失败计划的本轮汇总。
        """

        cycle_now = due_at or self._clock()
        if cycle_now.tzinfo is None or cycle_now.utcoffset() is None:
            raise ValueError("due_at 必须包含时区")

        due_plan_ids = await self._store.load_due_retry_plan_ids(
            due_at=cycle_now,
            limit=limit,
        )
        claimed_plan_ids: list[str] = []
        dispatched_plan_ids: list[str] = []
        cancelled_plan_ids: list[str] = []
        cancellation_reasons: dict[str, str] = {}
        skipped_plan_ids: list[str] = []
        failed_reasons: dict[str, str] = {}
        receipts: list[LongTaskCommitReceipt] = []

        for retry_plan_id in due_plan_ids:
            dispatch_claim_id = str(
                self._dispatch_claim_id_factory() or ""
            ).strip()
            if not dispatch_claim_id:
                failed_reasons[retry_plan_id] = (
                    "发布权令牌生成器返回了空值"
                )
                continue
            try:
                claimed_plan = await self._store.claim_due_retry_plan(
                    retry_plan_id=retry_plan_id,
                    dispatcher_id=self._dispatcher_id,
                    dispatch_claim_id=dispatch_claim_id,
                    now=cycle_now,
                    lease_seconds=lease_seconds,
                )
                if claimed_plan is None:
                    skipped_plan_ids.append(retry_plan_id)
                    continue
                claimed_plan_ids.append(retry_plan_id)

                current_task = await self._store.load(
                    claimed_plan.task_id
                )
                cancellation_reason = (
                    "重试计划所属任务不存在"
                    if current_task is None
                    else find_long_task_retry_ineligibility_reason(
                        task=current_task,
                        claimed_plan=claimed_plan,
                    )
                )
                if cancellation_reason is not None:
                    cancellation_request = (
                        build_long_task_retry_cancellation_request(
                            claimed_plan=claimed_plan,
                            cancellation_reason=cancellation_reason,
                            now=cycle_now,
                        )
                    )
                    await self._store.cancel_claimed_retry_plan(
                        cancellation_request
                    )
                    cancelled_plan_ids.append(retry_plan_id)
                    cancellation_reasons[retry_plan_id] = (
                        cancellation_reason
                    )
                    continue

                dispatch_request = build_long_task_retry_dispatch_request(
                    task=current_task,
                    claimed_plan=claimed_plan,
                    commit_id=(
                        f"retry_dispatch:{retry_plan_id}:"
                        f"{dispatch_claim_id}"
                    ),
                    now=cycle_now,
                )
                receipt = await self._store.commit_retry_dispatch(
                    dispatch_request
                )
            except (LongTaskStoreError, ValueError) as exc:
                failed_reasons[retry_plan_id] = str(exc)
                continue

            dispatched_plan_ids.append(retry_plan_id)
            receipts.append(receipt)

        return LongTaskRetryDispatchCycleResult(
            due_plan_ids=due_plan_ids,
            claimed_plan_ids=claimed_plan_ids,
            dispatched_plan_ids=dispatched_plan_ids,
            cancelled_plan_ids=cancelled_plan_ids,
            cancellation_reasons=cancellation_reasons,
            skipped_plan_ids=skipped_plan_ids,
            failed_reasons=failed_reasons,
            receipts=receipts,
        )
