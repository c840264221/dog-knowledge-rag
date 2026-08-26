"""验证独立长任务 Worker 的真实 Redis 与 Agent 执行闭环。"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from src.runtime.container.providers.redis_provider import RedisProvider
from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskGoal,
    LongTaskQueueMessage,
    LongTaskStep,
)
from src.runtime.long_tasks.store import (
    DEFAULT_LONG_TASK_KEY_PREFIX,
    LongTaskStore,
    RedisLongTaskStore,
)
from src.runtime.long_tasks.stream import (
    DEFAULT_LONG_TASK_CONSUMER_GROUP,
    DEFAULT_LONG_TASK_STREAM_KEY,
    RedisLongTaskStream,
)
from src.settings import settings


SleepCallable = Callable[[float], Awaitable[None]]
ClockCallable = Callable[[], float]


def positive_float(value: str) -> float:
    """
    把命令行文本转换为大于零的浮点数。

    参数含义：
        value:
            argparse 传入的原始命令行文本。

    返回值含义：
        float:
            校验通过的大于零浮点数。
    """

    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("参数必须大于零")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建长任务 Worker Smoke Test 命令行解析器。

    返回值含义：
        argparse.ArgumentParser:
            可以解析超时、轮询间隔和数据保留开关的参数解析器。
    """

    parser = argparse.ArgumentParser(
        description="执行一次真实长任务 Worker 闭环验证。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=120.0,
        help="等待任务完成和消息 ACK 的最长秒数，默认 120。",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_float,
        default=1.0,
        help="两次 Redis 状态查询之间的等待秒数，默认 1。",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="测试完成后保留本次 Task Key 和 Stream Entry 供人工检查。",
    )
    return parser


def build_smoke_task(task_id: str) -> LongTask:
    """
    构建一个交给 GeneralAgent 执行的单步骤持久化长任务。

    参数含义：
        task_id:
            本次 Smoke Test 随机生成的唯一任务编号。

    返回值含义：
        LongTask:
            版本为 1、Task 为 running、Step 为 ready 的合法后台任务。
    """

    return LongTask(
        task_id=task_id,
        user_id="smoke-user",
        thread_id=f"smoke-thread-{task_id}",
        goal=LongTaskGoal(
            original_request="只回答长任务实际执行成功",
            objective="返回长任务实际执行成功",
        ),
        steps=[
            LongTaskStep(
                step_id="smoke-step",
                task_id=task_id,
                title="执行最小通用问答",
                description="只输出指定的测试文本",
                assigned_agent="general_agent",
                input_data={
                    "question": (
                        "只回答：长任务实际执行成功。"
                        "不要调用工具，不要补充其他内容。"
                    ),
                },
                status="ready",
                max_attempts=1,
            )
        ],
        status="running",
        execution_mode="durable",
    )


async def run_long_task_worker_smoke(
    *,
    store: LongTaskStore,
    stream: RedisLongTaskStream,
    redis_client: Any,
    timeout_seconds: float,
    poll_interval_seconds: float,
    keep_data: bool = False,
    task_id: str | None = None,
    sleep: SleepCallable = asyncio.sleep,
    clock: ClockCallable | None = None,
) -> dict[str, Any]:
    """
    发布一条真实长任务，等待 Worker 完成并验证消息已经 ACK。

    参数含义：
        store:
            创建和轮询权威 LongTask 快照的正式 Store。
        stream:
            发布轻量任务通知的正式 Redis Stream 网关。
        redis_client:
            用于精确检查 PEL 并清理本次测试数据的异步 Redis 客户端。
        timeout_seconds:
            等待任务结果和消息 ACK 的总超时秒数。
        poll_interval_seconds:
            两次状态查询之间的异步等待秒数。
        keep_data:
            为 True 时即使成功也保留测试 Task Key 和 Stream Entry。
        task_id:
            可选固定任务编号；生产调用为空时自动生成随机编号。
        sleep:
            异步等待函数，测试可注入无等待替身。
        clock:
            单调时钟函数，测试可注入确定性时间。

    返回值含义：
        dict[str, Any]:
            包含成功标记、最终状态、版本、摘要、ACK 和清理情况的报告。
    """

    if timeout_seconds <= 0:
        raise ValueError("Smoke Test 超时时间必须大于零")
    if poll_interval_seconds <= 0:
        raise ValueError("Smoke Test 轮询间隔必须大于零")

    resolved_task_id = str(
        task_id or f"smoke-exec-{uuid4().hex}"
    ).strip()
    task = build_smoke_task(resolved_task_id)
    message = LongTaskQueueMessage(
        task_id=task.task_id,
        task_version=task.version,
        reason="submitted",
        ready_step_ids=[task.steps[0].step_id],
    )
    resolved_clock = clock or asyncio.get_running_loop().time
    deadline = resolved_clock() + timeout_seconds
    stream_id: str | None = None
    final_task: LongTask | None = None
    acknowledged = False

    await store.create(task)
    try:
        stream_id = await stream.publish(message)
        while resolved_clock() < deadline:
            current_task = await store.load(task.task_id)
            if current_task is None:
                raise RuntimeError("Smoke Test 的 LongTask 快照意外消失")
            if current_task.status in {
                "completed",
                "failed",
                "cancelled",
                "awaiting_input",
            }:
                final_task = current_task
                break
            await sleep(poll_interval_seconds)

        if final_task is not None and stream_id is not None:
            while resolved_clock() < deadline:
                pending_entries = await redis_client.xpending_range(
                    DEFAULT_LONG_TASK_STREAM_KEY,
                    DEFAULT_LONG_TASK_CONSUMER_GROUP,
                    min=stream_id,
                    max=stream_id,
                    count=1,
                )
                if not pending_entries:
                    acknowledged = True
                    break
                await sleep(poll_interval_seconds)

        step = final_task.steps[0] if final_task is not None else None
        last_batch_result = (
            step.metadata.get("last_batch_result", {})
            if step is not None
            else {}
        )
        error_message = (
            last_batch_result.get("error_message")
            if isinstance(last_batch_result, dict)
            else None
        )
        success = bool(
            final_task is not None
            and final_task.status == "completed"
            and step is not None
            and step.status == "completed"
            and acknowledged
        )
        return {
            "success": success,
            "task_id": task.task_id,
            "stream_id": stream_id,
            "task_status": (
                final_task.status if final_task is not None else "timeout"
            ),
            "task_version": (
                final_task.version if final_task is not None else None
            ),
            "step_status": step.status if step is not None else None,
            "step_version": step.version if step is not None else None,
            "attempt_count": (
                step.attempt_count if step is not None else None
            ),
            "output_summary": (
                step.output_summary if step is not None else ""
            ),
            "error_message": (
                error_message
            ),
            "acknowledged": acknowledged,
            "cleaned": bool(
                not keep_data and final_task is not None and acknowledged
            ),
        }
    finally:
        should_clean = bool(
            not keep_data
            and final_task is not None
            and acknowledged
            and stream_id is not None
        )
        if should_clean:
            await redis_client.xdel(
                DEFAULT_LONG_TASK_STREAM_KEY,
                stream_id,
            )
            await redis_client.delete(
                f"{DEFAULT_LONG_TASK_KEY_PREFIX}:task:{task.task_id}"
            )
        elif stream_id is None:
            await redis_client.delete(
                f"{DEFAULT_LONG_TASK_KEY_PREFIX}:task:{task.task_id}"
            )


async def run_from_settings(args: argparse.Namespace) -> dict[str, Any]:
    """
    使用环境配置创建 RedisProvider 并执行 Smoke Test。

    参数含义：
        args:
            已由 argparse 校验的超时、轮询和数据保留参数。

    返回值含义：
        dict[str, Any]:
            `run_long_task_worker_smoke` 生成的结构化测试报告。
    """

    provider = RedisProvider(settings.redis)
    await provider.startup()
    try:
        client = provider.client
        return await run_long_task_worker_smoke(
            store=RedisLongTaskStore(client),
            stream=RedisLongTaskStream(client),
            redis_client=client,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            keep_data=args.keep_data,
        )
    finally:
        await provider.shutdown()


def main() -> int:
    """
    执行长任务 Worker Smoke Test 并输出 JSON 报告。

    返回值含义：
        int:
            完整闭环成功返回 0；超时、失败或等待用户输入时返回 1。
    """

    args = build_argument_parser().parse_args()
    report = asyncio.run(run_from_settings(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
