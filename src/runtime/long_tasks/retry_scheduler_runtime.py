"""长任务重试调度器的后台循环运行入口。"""

from __future__ import annotations

import asyncio
import logging

from src.runtime.long_tasks.retry_scheduler import LongTaskRetryScheduler
from src.runtime.long_tasks.store import LongTaskReliableCommitStore


logger = logging.getLogger(__name__)


async def run_long_task_retry_scheduler(
    *,
    store: LongTaskReliableCommitStore,
    dispatcher_id: str,
    stop_event: asyncio.Event,
    poll_interval_seconds: float = 1.0,
    batch_limit: int = 100,
    lease_seconds: float = 30.0,
    scheduler: LongTaskRetryScheduler | None = None,
) -> None:
    """
    按固定间隔持续查询并处理到期重试计划。

    功能：
        反复调用调度器的单轮处理方法；整轮查询或处理意外失败时记录
        异常并在下个间隔继续，收到停止信号后及时退出。单个计划的失败
        仍由 LongTaskRetryScheduler 记录在本轮结果中，不会中断循环。

    参数含义：
        store:
            提供到期查询、发布权领取、计划取消和消息原子发布的 Store。
        dispatcher_id:
            当前调度器进程或实例的稳定身份。
        stop_event:
            外层要求后台调度器停止时设置的异步事件。
        poll_interval_seconds:
            两轮到期计划查询之间的等待秒数。
        batch_limit:
            每轮最多读取和处理的到期计划数量。
        lease_seconds:
            每个计划发布权的租约秒数。
        scheduler:
            可选的单轮调度器；未提供时根据 Store 和身份创建，测试可以
            注入可控替身。

    返回值含义：
        None：收到停止信号后正常退出，不返回业务结果。
    """

    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds 必须大于 0")
    if batch_limit < 1:
        raise ValueError("batch_limit 必须大于等于 1")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds 必须大于 0")
    normalized_dispatcher_id = str(dispatcher_id or "").strip()
    if not normalized_dispatcher_id:
        raise ValueError("dispatcher_id 不能为空")

    active_scheduler = scheduler or LongTaskRetryScheduler(
        store,
        dispatcher_id=normalized_dispatcher_id,
    )
    while not stop_event.is_set():
        try:
            cycle_result = await active_scheduler.dispatch_due_retries(
                limit=batch_limit,
                lease_seconds=lease_seconds,
            )
            if cycle_result.due_plan_ids:
                logger.info(
                    "长任务重试调度完成一轮：到期=%s，发布=%s，取消=%s，"
                    "跳过=%s，失败=%s",
                    len(cycle_result.due_plan_ids),
                    len(cycle_result.dispatched_plan_ids),
                    len(cycle_result.cancelled_plan_ids),
                    len(cycle_result.skipped_plan_ids),
                    len(cycle_result.failed_reasons),
                )
        except Exception:
            logger.exception(
                "长任务重试调度整轮执行失败，等待下个间隔继续：%s",
                normalized_dispatcher_id,
            )

        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=poll_interval_seconds,
            )
        except TimeoutError:
            pass
