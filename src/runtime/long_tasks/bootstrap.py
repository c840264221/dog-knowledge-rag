"""使用 RuntimeContainer 组装并运行长任务后台进程。"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.collaboration.adapters import LongTaskStepExecutorAdapter
from src.runtime.long_tasks.store import RedisLongTaskStore
from src.runtime.long_tasks.stream import RedisLongTaskStream
from src.runtime.long_tasks.retry_scheduler_runtime import (
    run_long_task_retry_scheduler,
)
from src.runtime.long_tasks.worker_runtime import run_long_task_worker


async def run_container_long_task_worker(
    *,
    runtime_container: Any,
    worker_name: str,
    stop_event: asyncio.Event,
    block_ms: int = 1000,
    min_idle_time_ms: int = 30_000,
    recovery_interval_ms: int = 30_000,
    lease_duration_ms: int = 30_000,
) -> None:
    """
    从 RuntimeContainer 获取真实依赖并持续运行长任务 Worker。

    功能：
        启动 Container，复用 RedisProvider 客户端与 GraphRuntimeService 已创建
        的协作 Worker，组装 Redis Store、Stream 和 Step Executor；Worker
        正常停止或异常退出后按生命周期关闭 Container。

    参数含义：
        runtime_container:
            已注册 redis 与 graph_runtime 服务、支持 startup/get/shutdown 的
            RuntimeContainer 或测试替身。
        worker_name:
            当前后台进程的稳定名称，同时作为 Redis Consumer 与 Step
            claimed_by 身份。
        stop_event:
            外层收到关闭信号时设置的异步停止事件。
        block_ms:
            Redis 没有新消息时单次阻塞读取的最长毫秒数。
        min_idle_time_ms:
            Pending 消息空闲多久后允许当前 Worker 接管。
        recovery_interval_ms:
            两次 Pending 恢复扫描之间的最小毫秒间隔。
        lease_duration_ms:
            Worker 成功领取 Step 后持有执行租约的毫秒数。

    返回值含义：
        None：收到停止信号后正常退出；启动或运行错误继续向外抛出。
    """

    await runtime_container.startup()
    try:
        redis_provider = runtime_container.get("redis")
        if not redis_provider.enabled:
            raise RuntimeError("长任务 Worker 要求启用 Redis")
        graph_runtime = runtime_container.get("graph_runtime")
        step_executor = LongTaskStepExecutorAdapter(
            graph_runtime.collaboration_workers
        )
        redis_client = redis_provider.client
        await run_long_task_worker(
            stream=RedisLongTaskStream(redis_client),
            store=RedisLongTaskStore(redis_client),
            step_executor=step_executor,
            worker_name=worker_name,
            stop_event=stop_event,
            block_ms=block_ms,
            min_idle_time_ms=min_idle_time_ms,
            recovery_interval_ms=recovery_interval_ms,
            lease_duration_ms=lease_duration_ms,
        )
    finally:
        await runtime_container.shutdown()


async def run_container_long_task_retry_scheduler(
    *,
    runtime_container: Any,
    dispatcher_id: str,
    stop_event: asyncio.Event,
    poll_interval_seconds: float = 1.0,
    batch_limit: int = 100,
    lease_seconds: float = 30.0,
) -> None:
    """
    从 RuntimeContainer 获取 Redis 并持续运行重试计划调度器。

    功能：
        启动 Container，取得项目统一管理的 Redis 客户端，创建长任务
        Store，并把它交给重试调度循环。调度器停止或异常退出后，始终
        关闭 Container。

    参数含义：
        runtime_container:
            已注册 redis 服务、支持 startup/get/shutdown 的
            RuntimeContainer 或测试替身。
        dispatcher_id:
            当前调度器进程的身份，用于领取重试计划的发布权。
        stop_event:
            外层收到关闭信号时设置的异步停止事件。
        poll_interval_seconds:
            两轮到期计划扫描之间等待的秒数。
        batch_limit:
            每轮最多查询和处理的到期重试计划数量。
        lease_seconds:
            调度器领取一个计划后持有发布权的秒数。

    返回值含义：
        None：收到停止信号后正常退出；启动或运行错误继续向外抛出。
    """

    await runtime_container.startup()
    try:
        redis_provider = runtime_container.get("redis")
        if not redis_provider.enabled:
            raise RuntimeError("长任务重试调度器要求启用 Redis")
        await run_long_task_retry_scheduler(
            store=RedisLongTaskStore(redis_provider.client),
            dispatcher_id=dispatcher_id,
            stop_event=stop_event,
            poll_interval_seconds=poll_interval_seconds,
            batch_limit=batch_limit,
            lease_seconds=lease_seconds,
        )
    finally:
        await runtime_container.shutdown()
