"""启动 Dog Agent 长任务重试计划调度器。"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
from collections.abc import Callable
from typing import Any

from src.runtime.long_tasks.bootstrap import (
    run_container_long_task_retry_scheduler,
)


def positive_integer(value: str) -> int:
    """
    把命令行文本转换为大于零的整数。

    参数含义：
        value：argparse 传入的原始命令行文本。

    返回值含义：
        int：校验通过的大于零整数。
    """

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("参数必须是大于零的整数")
    return parsed_value


def positive_float(value: str) -> float:
    """
    把命令行文本转换为大于零的浮点数。

    参数含义：
        value：argparse 传入的原始命令行文本。

    返回值含义：
        float：校验通过的大于零浮点数。
    """

    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("参数必须是大于零的数字")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建重试调度器的命令行参数解析器。

    返回值含义：
        argparse.ArgumentParser：可解析调度器身份和扫描参数的解析器。
    """

    parser = argparse.ArgumentParser(
        description="启动 Dog Agent 长任务重试计划调度器。",
    )
    parser.add_argument(
        "--dispatcher-id",
        default=None,
        help=(
            "领取重试计划发布权时使用的调度器身份；未提供时读取 "
            "LONG_TASK_RETRY_DISPATCHER_ID，仍为空则自动生成。"
        ),
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_float,
        default=1.0,
        help="两轮到期计划扫描之间等待的秒数，默认 1。",
    )
    parser.add_argument(
        "--batch-limit",
        type=positive_integer,
        default=100,
        help="每轮最多处理的到期计划数量，默认 100。",
    )
    parser.add_argument(
        "--lease-seconds",
        type=positive_float,
        default=30.0,
        help="领取一个计划后持有发布权的秒数，默认 30。",
    )
    return parser


def resolve_dispatcher_id(explicit_id: str | None) -> str:
    """
    解析当前进程领取重试计划发布权时使用的身份。

    参数含义：
        explicit_id：命令行显式传入的身份；为空时读取环境变量或自动生成。

    返回值含义：
        str：去除首尾空白且长度不超过 200 的调度器身份。
    """

    dispatcher_id = str(
        explicit_id
        or os.getenv("LONG_TASK_RETRY_DISPATCHER_ID")
        or f"{socket.gethostname()}-{os.getpid()}-retry-dispatcher"
    ).strip()
    if not dispatcher_id:
        raise ValueError("长任务重试调度器身份不能为空")
    if len(dispatcher_id) > 200:
        raise ValueError("长任务重试调度器身份不能超过 200 个字符")
    return dispatcher_id


def install_stop_signal_handlers(
    stop_event: asyncio.Event,
) -> Callable[[], None]:
    """
    安装退出信号处理器，并返回恢复原处理器的函数。

    参数含义：
        stop_event：收到 Ctrl+C 或进程停止信号后需要设置的停止事件。

    返回值含义：
        Callable[[], None]：调度器退出后恢复操作系统信号处理器的函数。
    """

    loop = asyncio.get_running_loop()
    loop_signals: list[signal.Signals] = []
    fallback_handlers: dict[signal.Signals, Any] = {}

    def request_stop() -> None:
        """通知调度循环在当前一轮结束后停止。"""

        stop_event.set()

    def fallback_handler(_signum: int, _frame: Any) -> None:
        """
        在 Windows 等不支持事件循环信号处理的平台中请求停止。

        参数含义：
            _signum：操作系统传入的信号编号，本函数不需要读取。
            _frame：Python 传入的当前栈帧，本函数不需要读取。

        返回值含义：
            None：只把停止请求交给事件循环，不返回业务数据。
        """

        loop.call_soon_threadsafe(stop_event.set)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
            loop_signals.append(signum)
        except NotImplementedError:
            fallback_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, fallback_handler)

    def restore_handlers() -> None:
        """移除本进程安装的处理器并恢复原来的信号处理方式。"""

        for signum in loop_signals:
            loop.remove_signal_handler(signum)
        for signum, previous_handler in fallback_handlers.items():
            signal.signal(signum, previous_handler)

    return restore_handlers


async def run_scheduler_process(
    args: argparse.Namespace,
    *,
    runtime_container: Any,
    scheduler_runner: Callable[..., Any] = (
        run_container_long_task_retry_scheduler
    ),
    signal_installer: Callable[
        [asyncio.Event],
        Callable[[], None],
    ] = install_stop_signal_handlers,
) -> None:
    """
    创建停止事件并把命令行参数交给重试调度器 Bootstrap。

    参数含义：
        args：已经由 argparse 校验的命令行参数。
        runtime_container：已注册 Redis 服务的 RuntimeContainer。
        scheduler_runner：实际运行调度器的异步函数，测试时可替换。
        signal_installer：安装退出信号并返回恢复函数的函数。

    返回值含义：
        None：调度器收到退出信号并完成协作式停止后返回。
    """

    stop_event = asyncio.Event()
    restore_handlers = signal_installer(stop_event)
    try:
        await scheduler_runner(
            runtime_container=runtime_container,
            dispatcher_id=resolve_dispatcher_id(args.dispatcher_id),
            stop_event=stop_event,
            poll_interval_seconds=args.poll_interval_seconds,
            batch_limit=args.batch_limit,
            lease_seconds=args.lease_seconds,
        )
    finally:
        restore_handlers()


def main() -> int:
    """
    解析参数、加载全局 Container 并运行重试调度器事件循环。

    返回值含义：
        int：调度器正常关闭时返回 0；异常由进程返回非零状态。
    """

    from src.runtime.container.init import container

    args = build_argument_parser().parse_args()
    asyncio.run(
        run_scheduler_process(
            args,
            runtime_container=container,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
