"""长任务后台 Worker 的最小组装与持续运行入口。"""

from __future__ import annotations

import asyncio

from src.runtime.long_tasks.store import LongTaskStore
from src.runtime.long_tasks.stream import (
    LongTaskStreamWorker,
    RedisLongTaskStream,
)
from src.runtime.long_tasks.worker_handler import (
    LongTaskQueueMessageHandler,
    LongTaskStepExecutor,
)


async def run_long_task_worker(
    *,
    stream: RedisLongTaskStream,
    store: LongTaskStore,
    step_executor: LongTaskStepExecutor,
    worker_name: str,
    stop_event: asyncio.Event,
    block_ms: int = 1000,
    min_idle_time_ms: int = 30_000,
    recovery_interval_ms: int = 30_000,
    lease_duration_ms: int = 30_000,
) -> None:
    """
    组装长任务 Handler 与 Stream Worker，并持续处理后台消息。

    功能：
        使用同一个 worker_name 标识 Redis Consumer 和 Step Claim，避免
        消息消费者身份与业务执行身份不一致；循环停止与进程重启仍交给
        外层 Runtime、Docker 或进程管理器负责。

    参数含义：
        stream:
            提供消费者组读取、超时消息接管和 ACK 的 Redis Stream 网关。
        store:
            提供最新权威 LongTask 加载与乐观锁保存能力的 Store。
        step_executor:
            真正执行一个已领取 Step 并返回结构化结果的异步适配器。
        worker_name:
            当前后台 Worker 的稳定名称，同时用于 Consumer 和 claimed_by。
        stop_event:
            外层关闭 Worker 时设置的异步停止信号。
        block_ms:
            没有新消息时单次阻塞读取的最长毫秒数。
        min_idle_time_ms:
            Pending 消息至少空闲多少毫秒后允许被接管。
        recovery_interval_ms:
            两次 Pending 恢复扫描之间的最小毫秒间隔。
        lease_duration_ms:
            Step 被领取后获得的租约有效毫秒数。

    返回值含义：
        None：收到 stop_event 后正常退出；处理异常继续交给外层进程管理。
    """

    handler = LongTaskQueueMessageHandler(
        store=store,
        step_executor=step_executor,
        worker_name=worker_name,
        lease_duration_ms=lease_duration_ms,
        queue_publisher=stream,
    )
    worker = LongTaskStreamWorker(
        stream=stream,
        handler=handler,
    )
    await worker.run_forever(
        stop_event=stop_event,
        consumer_name=worker_name,
        block_ms=block_ms,
        min_idle_time_ms=min_idle_time_ms,
        recovery_interval_ms=recovery_interval_ms,
    )
