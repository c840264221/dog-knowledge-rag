"""长任务重试调度器后台循环入口测试。"""

from __future__ import annotations

import asyncio

import pytest

from src.runtime.long_tasks.failure_contracts import (
    LongTaskRetryDispatchCycleResult,
)
from src.runtime.long_tasks.retry_scheduler_runtime import (
    run_long_task_retry_scheduler,
)


class FailingOnceRetryScheduler:
    """第一轮失败、第二轮请求停止的可控调度器替身。"""

    def __init__(self, stop_event: asyncio.Event) -> None:
        """
        保存停止事件并初始化调用记录。

        参数含义：
            stop_event:
                第二轮执行时需要设置的后台停止信号。

        返回值含义：
            None：初始化结果保存在实例字段中。
        """

        self.stop_event = stop_event
        self.calls: list[tuple[int, float]] = []

    async def dispatch_due_retries(
        self,
        *,
        limit: int,
        lease_seconds: float,
    ) -> LongTaskRetryDispatchCycleResult:
        """
        首次模拟 Redis 故障，第二次返回空结果并请求停止。

        参数含义：
            limit/lease_seconds:
                Runtime 传入的单轮计划上限和发布权租约秒数。

        返回值含义：
            LongTaskRetryDispatchCycleResult：第二轮返回的空调度结果。
        """

        self.calls.append((limit, lease_seconds))
        if len(self.calls) == 1:
            raise ConnectionError("模拟 Redis 临时不可用")
        self.stop_event.set()
        return LongTaskRetryDispatchCycleResult()


@pytest.mark.asyncio
async def test_runtime_should_retry_failed_cycle_and_stop_cooperatively(
) -> None:
    """
    验证整轮失败不会结束后台服务，下一轮成功后可按停止信号退出。

    参数含义：
        无。

    返回值含义：
        None：调度器执行两轮并收到参数和停止信号时测试通过。
    """

    stop_event = asyncio.Event()
    scheduler = FailingOnceRetryScheduler(stop_event)

    await run_long_task_retry_scheduler(
        store=object(),  # type: ignore[arg-type]
        dispatcher_id="DISPATCHER-RUNTIME-1",
        stop_event=stop_event,
        poll_interval_seconds=0.001,
        batch_limit=25,
        lease_seconds=15,
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    assert scheduler.calls == [(25, 15), (25, 15)]
    assert stop_event.is_set() is True
