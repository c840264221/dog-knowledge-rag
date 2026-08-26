"""一键长任务 Worker Smoke Test 脚本单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.smoke_long_task_worker import (
    build_smoke_task,
    run_long_task_worker_smoke,
)
from src.runtime.long_tasks.contracts import LongTask


class FakeSmokeStore:
    """按顺序返回测试任务快照的内存 Store。"""

    def __init__(self, snapshots: list[LongTask]) -> None:
        """
        初始化按顺序返回快照的内存 Store。

        参数含义：
            snapshots:
                每次 load 应当返回的任务快照列表。

        返回值含义：
            None:
                初始化方法不返回业务数据。
        """

        self.snapshots = list(snapshots)
        self.created_task: LongTask | None = None

    async def create(self, task: LongTask) -> LongTask:
        """
        记录脚本创建的初始任务。

        参数含义：
            task:
                Smoke Test 创建的初始长任务。

        返回值含义：
            LongTask:
                原样返回已记录的任务。
        """

        self.created_task = task
        return task

    async def load(self, _task_id: str) -> LongTask | None:
        """
        返回下一份快照，耗尽后持续返回最后一份。

        参数含义：
            _task_id:
                待查询的任务编号；测试替身无需使用该值。

        返回值含义：
            LongTask | None:
                当前轮询应当看到的任务快照。
        """

        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        满足 Store Protocol；Smoke 脚本不会直接保存更新。

        参数含义：
            task:
                未使用的任务快照。
            expected_version:
                未使用的旧版本。

        返回值含义：
            LongTask:
                原样返回输入；本测试不会调用该方法。
        """

        _ = expected_version
        return task


class FakeSmokeStream:
    """返回固定消息编号的测试 Stream。"""

    def __init__(self) -> None:
        """
        初始化消息记录位置。

        返回值含义：
            None:
                初始化方法不返回业务数据。
        """

        self.message: Any = None

    async def publish(self, message: Any) -> str:
        """
        记录消息并返回固定 Stream ID。

        参数含义：
            message:
                Smoke Test 发布的轻量队列消息。

        返回值含义：
            str:
                固定的 Redis Stream 消息编号。
        """

        self.message = message
        return "1000-0"


class FakeSmokeRedis:
    """记录 ACK 检查和精确清理调用的测试 Redis。"""

    def __init__(self, pending_pages: list[list[Any]]) -> None:
        """
        初始化 PEL 查询结果和删除调用记录。

        参数含义：
            pending_pages:
                每次 PEL 查询应当返回的数据列表。

        返回值含义：
            None:
                初始化方法不返回业务数据。
        """

        self.pending_pages = list(pending_pages)
        self.deleted_keys: list[str] = []
        self.deleted_entries: list[tuple[str, str]] = []

    async def xpending_range(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        """
        按顺序返回指定 Stream Entry 的 PEL 查询结果。

        参数含义：
            *_args:
                Redis 方法的位置参数；测试替身无需解析。
            **_kwargs:
                Redis 方法的命名参数；测试替身无需解析。

        返回值含义：
            list[Any]:
                当前查询应当看到的 Pending Entry 列表。
        """

        if len(self.pending_pages) > 1:
            return self.pending_pages.pop(0)
        return self.pending_pages[0]

    async def xdel(self, stream_key: str, stream_id: str) -> int:
        """
        记录精确删除的 Stream Entry。

        参数含义：
            stream_key:
                Redis Stream 的 Key。
            stream_id:
                本次测试消息的 Stream ID。

        返回值含义：
            int:
                模拟 Redis 成功删除一条消息。
        """

        self.deleted_entries.append((stream_key, stream_id))
        return 1

    async def delete(self, key: str) -> int:
        """
        记录精确删除的 LongTask Key。

        参数含义：
            key:
                本次测试任务对应的 Redis Key。

        返回值含义：
            int:
                模拟 Redis 成功删除一个 Key。
        """

        self.deleted_keys.append(key)
        return 1


async def no_wait(_seconds: float) -> None:
    """
    替代测试中的真实异步等待。

    参数含义：
        _seconds:
            生产代码计划等待的秒数；测试中不会真正等待。

    返回值含义：
        None:
            该替身不返回业务数据。
    """


@pytest.mark.asyncio
async def test_smoke_should_verify_completed_task_ack_and_cleanup() -> None:
    """验证脚本会等待完成、确认 ACK 并精确清理测试数据。"""

    initial_task = build_smoke_task("smoke-test-1")
    running_task = initial_task.model_copy(
        update={
            "version": 2,
            "steps": [
                initial_task.steps[0].model_copy(
                    update={
                        "status": "running",
                        "version": 2,
                        "attempt_count": 1,
                    }
                )
            ],
        }
    )
    completed_task = running_task.model_copy(
        update={
            "status": "completed",
            "version": 3,
            "steps": [
                running_task.steps[0].model_copy(
                    update={
                        "status": "completed",
                        "version": 3,
                        "output_summary": "长任务实际执行成功。",
                    }
                )
            ],
        }
    )
    store = FakeSmokeStore([running_task, completed_task])
    stream = FakeSmokeStream()
    redis = FakeSmokeRedis([[{"message_id": "1000-0"}], []])

    report = await run_long_task_worker_smoke(
        store=store,
        stream=stream,
        redis_client=redis,
        timeout_seconds=10,
        poll_interval_seconds=1,
        task_id="smoke-test-1",
        sleep=no_wait,
        clock=lambda: 0,
    )

    assert report["success"] is True
    assert report["task_status"] == "completed"
    assert report["task_version"] == 3
    assert report["step_status"] == "completed"
    assert report["acknowledged"] is True
    assert report["cleaned"] is True
    assert stream.message.task_id == "smoke-test-1"
    assert redis.deleted_entries == [
        ("dog-agent:long-task:v1:stream", "1000-0")
    ]
    assert redis.deleted_keys == [
        "dog-agent:long-task:v1:task:smoke-test-1"
    ]


@pytest.mark.asyncio
async def test_smoke_should_retain_running_task_after_timeout() -> None:
    """验证超时时不会删除仍可能由 Worker 执行的任务和消息。"""

    task = build_smoke_task("smoke-timeout-1")
    clock_values = iter([0.0, 0.0, 2.0])
    redis = FakeSmokeRedis([[]])

    report = await run_long_task_worker_smoke(
        store=FakeSmokeStore([task]),
        stream=FakeSmokeStream(),
        redis_client=redis,
        timeout_seconds=1,
        poll_interval_seconds=1,
        task_id="smoke-timeout-1",
        sleep=no_wait,
        clock=lambda: next(clock_values),
    )

    assert report["success"] is False
    assert report["task_status"] == "timeout"
    assert report["acknowledged"] is False
    assert report["cleaned"] is False
    assert redis.deleted_entries == []
    assert redis.deleted_keys == []
