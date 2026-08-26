"""长任务独立 Worker 进程入口测试。"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

import scripts.run_long_task_worker as worker_entrypoint


def build_args(**overrides: Any) -> argparse.Namespace:
    """
    构建独立 Worker 入口测试使用的命令行参数。

    参数含义：
        overrides:
            需要覆盖的单个命令行参数值。

    返回值含义：
        argparse.Namespace:
            包含完整 Worker 运行参数的测试对象。
    """

    values = {
        "worker_name": "worker-entrypoint-1",
        "block_ms": 10,
        "min_idle_time_ms": 20,
        "recovery_interval_ms": 30,
        "lease_duration_ms": 40,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolve_worker_name_should_follow_source_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式参数优先于环境变量，缺省时可以自动生成名称。"""

    monkeypatch.setenv("LONG_TASK_WORKER_NAME", "worker-from-env")
    assert (
        worker_entrypoint.resolve_worker_name("worker-from-cli")
        == "worker-from-cli"
    )
    assert (
        worker_entrypoint.resolve_worker_name(None)
        == "worker-from-env"
    )

    monkeypatch.delenv("LONG_TASK_WORKER_NAME")
    monkeypatch.setattr(worker_entrypoint.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(worker_entrypoint.os, "getpid", lambda: 123)
    assert worker_entrypoint.resolve_worker_name(None) == "host-123"


def test_positive_integer_should_reject_non_positive_value() -> None:
    """验证 Worker 时间参数不能使用零或负数。"""

    assert worker_entrypoint.positive_integer("15") == 15
    with pytest.raises(argparse.ArgumentTypeError):
        worker_entrypoint.positive_integer("0")


@pytest.mark.asyncio
async def test_run_worker_process_should_delegate_validated_arguments() -> None:
    """验证进程入口会安装信号并把参数交给现有 Bootstrap。"""

    runtime_container = object()
    captured: dict[str, Any] = {}
    lifecycle: list[str] = []

    def fake_signal_installer(
        stop_event: asyncio.Event,
    ) -> Any:
        """记录停止事件并返回测试清理函数。"""

        captured["installed_stop_event"] = stop_event

        def restore() -> None:
            """记录入口已恢复信号处理器。"""

            lifecycle.append("signals_restored")

        return restore

    async def fake_worker_runner(**kwargs: Any) -> None:
        """记录进程入口交给 Bootstrap 的参数。"""

        captured.update(kwargs)
        lifecycle.append("worker_finished")

    await worker_entrypoint.run_worker_process(
        build_args(),
        runtime_container=runtime_container,
        worker_runner=fake_worker_runner,
        signal_installer=fake_signal_installer,
    )

    assert captured["runtime_container"] is runtime_container
    assert captured["worker_name"] == "worker-entrypoint-1"
    assert captured["stop_event"] is captured["installed_stop_event"]
    assert captured["block_ms"] == 10
    assert captured["min_idle_time_ms"] == 20
    assert captured["recovery_interval_ms"] == 30
    assert captured["lease_duration_ms"] == 40
    assert lifecycle == ["worker_finished", "signals_restored"]


@pytest.mark.asyncio
async def test_run_worker_process_should_restore_signals_after_failure() -> None:
    """验证 Bootstrap 抛出异常后仍会恢复进程原有信号处理器。"""

    lifecycle: list[str] = []

    def fake_signal_installer(
        _stop_event: asyncio.Event,
    ) -> Any:
        """返回记录清理动作的信号处理替身。"""

        return lambda: lifecycle.append("signals_restored")

    async def failing_worker_runner(**_kwargs: Any) -> None:
        """模拟后台 Worker 异常退出。"""

        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        await worker_entrypoint.run_worker_process(
            build_args(),
            runtime_container=object(),
            worker_runner=failing_worker_runner,
            signal_installer=fake_signal_installer,
        )

    assert lifecycle == ["signals_restored"]
