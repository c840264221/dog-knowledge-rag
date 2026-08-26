"""RuntimeContainer 长任务 Worker 组装入口测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration.adapters import LongTaskStepExecutorAdapter
from src.agents.collaboration.contracts import (
    AgentTaskResult,
    AgentTaskStep,
)
import src.runtime.long_tasks.bootstrap as bootstrap
from src.runtime.long_tasks.store import RedisLongTaskStore
from src.runtime.long_tasks.stream import RedisLongTaskStream


async def fake_agent_worker(
    step: AgentTaskStep,
    _: Mapping[str, AgentTaskResult],
) -> AgentTaskResult:
    """
    返回测试 Bootstrap 注册表使用的标准 Worker 结果。

    参数含义：
        step：当前标准协作步骤。
        _：当前测试不使用的依赖结果。

    返回值含义：
        AgentTaskResult：与输入 Step 归属一致的完成结果。
    """

    return AgentTaskResult(
        step_id=step.step_id,
        assigned_agent=step.assigned_agent,
        status="completed",
    )


class FakeRedisProvider:
    """提供已启用的测试 Redis 客户端占位对象。"""

    enabled = True

    def __init__(self) -> None:
        self.client = object()


class FakeGraphRuntime:
    """提供 GraphRuntime 已创建的协作 Worker 映射。"""

    collaboration_workers = {
        "general_agent": fake_agent_worker,
    }


class FakeRuntimeContainer:
    """记录 Bootstrap 生命周期和服务读取顺序的测试 Container。"""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.redis_provider = FakeRedisProvider()
        self.graph_runtime = FakeGraphRuntime()

    async def startup(self) -> None:
        """记录 Container 启动。"""

        self.events.append("startup")

    def get(self, name: str) -> Any:
        """按照服务名称返回测试 Provider 或 Runtime。"""

        self.events.append(f"get:{name}")
        if name == "redis":
            return self.redis_provider
        if name == "graph_runtime":
            return self.graph_runtime
        raise ValueError(f"未知测试服务: {name}")

    async def shutdown(self) -> None:
        """记录 Container 关闭。"""

        self.events.append("shutdown")


@pytest.mark.asyncio
async def test_bootstrap_should_assemble_real_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Bootstrap 会组装 Redis 网关、Store 和真实 Worker 映射。"""

    container = FakeRuntimeContainer()
    captured: dict[str, Any] = {}

    async def fake_run_long_task_worker(**kwargs: Any) -> None:
        """记录 Bootstrap 最终交给后台循环的全部依赖。"""

        captured.update(kwargs)

    monkeypatch.setattr(
        bootstrap,
        "run_long_task_worker",
        fake_run_long_task_worker,
    )
    stop_event = asyncio.Event()
    await bootstrap.run_container_long_task_worker(
        runtime_container=container,
        worker_name="worker-runtime-1",
        stop_event=stop_event,
        block_ms=10,
        min_idle_time_ms=20,
        recovery_interval_ms=30,
        lease_duration_ms=40,
    )

    assert container.events == [
        "startup",
        "get:redis",
        "get:graph_runtime",
        "shutdown",
    ]
    assert isinstance(captured["stream"], RedisLongTaskStream)
    assert isinstance(captured["store"], RedisLongTaskStore)
    assert isinstance(
        captured["step_executor"],
        LongTaskStepExecutorAdapter,
    )
    assert captured["worker_name"] == "worker-runtime-1"
    assert captured["stop_event"] is stop_event
    assert captured["block_ms"] == 10
    assert captured["min_idle_time_ms"] == 20
    assert captured["recovery_interval_ms"] == 30
    assert captured["lease_duration_ms"] == 40


@pytest.mark.asyncio
async def test_bootstrap_should_shutdown_container_after_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证后台循环异常退出时仍会关闭 Container。"""

    container = FakeRuntimeContainer()

    async def failing_run_long_task_worker(**_: Any) -> None:
        """模拟后台 Worker 处理异常。"""

        raise RuntimeError("worker failed")

    monkeypatch.setattr(
        bootstrap,
        "run_long_task_worker",
        failing_run_long_task_worker,
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        await bootstrap.run_container_long_task_worker(
            runtime_container=container,
            worker_name="worker-runtime-1",
            stop_event=asyncio.Event(),
        )

    assert container.events[-1] == "shutdown"
