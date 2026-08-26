"""启动 Dog Agent 持久化长任务后台 Worker。"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
from collections.abc import Callable
from typing import Any

from src.runtime.long_tasks.bootstrap import (
    run_container_long_task_worker,
)


def positive_integer(value: str) -> int:
    """
    把命令行文本转换为大于零的整数。

    参数含义：
        value:
            argparse 传入的原始命令行文本。

    返回值含义：
        int:
            校验通过的大于零整数。
    """

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("参数必须是大于零的整数")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建长任务 Worker 命令行参数解析器。

    返回值含义：
        argparse.ArgumentParser:
            可以解析 Worker 身份和运行间隔参数的命令行解析器。
    """

    parser = argparse.ArgumentParser(
        description="启动 Dog Agent Redis 长任务后台 Worker。",
    )
    parser.add_argument(
        "--worker-name",
        default=None,
        help=(
            "Redis Consumer 和 Step claimed_by 使用的 Worker 名称；"
            "未提供时读取 LONG_TASK_WORKER_NAME，仍为空则自动生成。"
        ),
    )
    parser.add_argument(
        "--block-ms",
        type=positive_integer,
        default=1000,
        help="没有新消息时单次 Redis 阻塞读取毫秒数，默认 1000。",
    )
    parser.add_argument(
        "--min-idle-time-ms",
        type=positive_integer,
        default=30_000,
        help="Pending 消息允许被重新认领前的最小空闲毫秒数。",
    )
    parser.add_argument(
        "--recovery-interval-ms",
        type=positive_integer,
        default=30_000,
        help="两次 Pending 恢复扫描之间的最小毫秒数。",
    )
    parser.add_argument(
        "--lease-duration-ms",
        type=positive_integer,
        default=30_000,
        help="Worker 领取一个 Step 后持有执行租约的毫秒数。",
    )
    return parser


def resolve_worker_name(explicit_name: str | None) -> str:
    """
    解析当前进程用于 Redis Consumer 和 Step Claim 的唯一名称。

    参数含义：
        explicit_name:
            命令行显式传入的 Worker 名称；为空时读取环境变量或自动生成。

    返回值含义：
        str:
            去除首尾空白且长度不超过 200 的 Worker 名称。
    """

    worker_name = str(
        explicit_name
        or os.getenv("LONG_TASK_WORKER_NAME")
        or f"{socket.gethostname()}-{os.getpid()}"
    ).strip()
    if not worker_name:
        raise ValueError("长任务 Worker 名称不能为空")
    if len(worker_name) > 200:
        raise ValueError("长任务 Worker 名称不能超过 200 个字符")
    return worker_name


def install_stop_signal_handlers(
    stop_event: asyncio.Event,
) -> Callable[[], None]:
    """
    安装 SIGINT 和 SIGTERM 处理器并返回恢复函数。

    参数含义：
        stop_event:
            收到操作系统退出信号后需要设置的异步停止事件。

    返回值含义：
        Callable[[], None]:
            Worker 退出后移除新处理器并恢复旧处理器的清理函数。
    """

    loop = asyncio.get_running_loop()
    loop_signals: list[signal.Signals] = []
    fallback_handlers: dict[signal.Signals, Any] = {}

    def request_stop() -> None:
        """把操作系统退出请求转换成 Worker 的协作式停止事件。"""

        stop_event.set()

    def fallback_handler(
        _signum: int,
        _frame: Any,
    ) -> None:
        """
        在不支持事件循环信号处理的系统中线程安全地请求停止。

        参数含义：
            _signum:
                当前收到的操作系统信号编号，本函数不需要区分具体编号。
            _frame:
                Python 标准库传入的当前栈帧，本函数不读取该对象。

        返回值含义：
            None:
                只向事件循环提交 stop_event.set，不返回业务数据。
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
        """移除本入口安装的处理器并恢复标准库原处理器。"""

        for signum in loop_signals:
            loop.remove_signal_handler(signum)
        for signum, previous_handler in fallback_handlers.items():
            signal.signal(signum, previous_handler)

    return restore_handlers


async def run_worker_process(
    args: argparse.Namespace,
    *,
    runtime_container: Any,
    worker_runner: Callable[..., Any] = run_container_long_task_worker,
    signal_installer: Callable[
        [asyncio.Event],
        Callable[[], None],
    ] = install_stop_signal_handlers,
) -> None:
    """
    创建停止事件并把命令行参数交给现有 Container Worker Bootstrap。

    参数含义：
        args:
            已由 argparse 校验的 Worker 命令行参数。
        runtime_container:
            已注册 Redis 和 GraphRuntime 服务的 RuntimeContainer。
        worker_runner:
            实际组装并运行后台 Worker 的异步函数，测试可注入替身。
        signal_installer:
            安装退出信号并返回清理函数的函数，测试可注入替身。

    返回值含义：
        None:
            Worker 收到退出信号并完成协作式停止后返回。
    """

    stop_event = asyncio.Event()
    restore_handlers = signal_installer(stop_event)
    try:
        await worker_runner(
            runtime_container=runtime_container,
            worker_name=resolve_worker_name(args.worker_name),
            stop_event=stop_event,
            block_ms=args.block_ms,
            min_idle_time_ms=args.min_idle_time_ms,
            recovery_interval_ms=args.recovery_interval_ms,
            lease_duration_ms=args.lease_duration_ms,
        )
    finally:
        restore_handlers()


def main() -> int:
    """
    解析参数、加载全局 Container 并运行长任务 Worker 事件循环。

    返回值含义：
        int:
            Worker 正常收到停止信号并关闭时返回 0；异常由进程返回非零状态。
    """

    from src.runtime.container.init import container

    args = build_argument_parser().parse_args()
    asyncio.run(
        run_worker_process(
            args,
            runtime_container=container,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
