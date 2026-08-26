"""Redis 长任务快照 Store 单元测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.runtime.long_tasks import (
    CorruptLongTaskSnapshotError,
    LongTask,
    LongTaskAlreadyExistsError,
    LongTaskGoal,
    LongTaskNotFoundError,
    LongTaskStep,
    LongTaskVersionConflictError,
    RedisLongTaskStore,
    transition_long_task,
)


class FakeRedis:
    """为 Store 测试模拟 Redis String、SET NX 和原子版本比较。"""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.eval_calls: list[tuple[str, int, tuple[Any, ...]]] = []

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
    ) -> bool:
        """
        模拟 Redis SET，并在 nx=True 时拒绝覆盖已存在的 Key。

        参数含义：
            key:
                Redis Key。
            value:
                准备保存的 JSON 字符串。
            nx:
                是否只允许 Key 不存在时写入。

        返回值含义：
            bool:
                写入成功返回 True，NX 条件不满足时返回 False。
        """

        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key: str) -> str | None:
        """
        读取测试字典中的 Redis String。

        参数含义：
            key:
                Redis Key。

        返回值含义：
            str | None:
                Key 存在时返回字符串，否则返回 None。
        """

        return self.values.get(key)

    async def eval(
        self,
        script: str,
        numkeys: int,
        *args: Any,
    ) -> list[int]:
        """
        模拟 Store Lua 脚本的原子版本检查与覆盖行为。

        参数含义：
            script:
                Store 传入的 Lua 脚本文本。
            numkeys:
                参数中属于 Redis Key 的数量。
            args:
                Key、期望版本、新 JSON 和新版本。

        返回值含义：
            list[int]:
                与真实脚本一致的结果码和当前版本。
        """

        self.eval_calls.append((script, numkeys, args))
        key = str(args[0])
        expected_version = int(args[1])
        next_json = str(args[2])
        next_version = int(args[3])
        current_json = self.values.get(key)
        if current_json is None:
            return [-1, -1]
        try:
            current_version = int(json.loads(current_json)["version"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return [-2, -1]
        if current_version != expected_version:
            return [0, current_version]
        self.values[key] = next_json
        return [1, next_version]


def build_task(*, task_id: str = "task_001") -> LongTask:
    """
    构建 Store 测试使用的初始长任务。

    参数含义：
        task_id:
            可选长任务唯一编号。

    返回值含义：
        LongTask:
            版本为 1、状态为 created 的合法任务快照。
    """

    return LongTask(
        task_id=task_id,
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="生成狗狗健康计划",
            objective="生成狗狗健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id=task_id,
                title="读取档案",
                assigned_agent="profile_agent",
                status="ready",
            )
        ],
    )


@pytest.mark.asyncio
async def test_create_and_load_should_round_trip_task_json() -> None:
    """验证任务能够按命名 Key 创建，并从 JSON 恢复类型。"""

    redis = FakeRedis()
    store = RedisLongTaskStore(redis)
    task = build_task()

    created_task = await store.create(task)
    loaded_task = await store.load(task.task_id)

    assert created_task == task
    assert loaded_task == task
    assert list(redis.values) == [
        "dog-agent:long-task:v1:task:task_001"
    ]


@pytest.mark.asyncio
async def test_create_should_reject_duplicate_task_id() -> None:
    """验证 SET NX 防止同一任务被重复创建。"""

    store = RedisLongTaskStore(FakeRedis())
    task = build_task()
    await store.create(task)

    with pytest.raises(LongTaskAlreadyExistsError, match="已经存在"):
        await store.create(task)


@pytest.mark.asyncio
async def test_save_should_compare_version_and_replace_snapshot() -> None:
    """验证版本一致时原子保存版本递增后的新快照。"""

    redis = FakeRedis()
    store = RedisLongTaskStore(redis)
    original_task = build_task()
    await store.create(original_task)
    running_task = transition_long_task(
        original_task,
        target_status="running",
        active_step_ids=["step_1"],
    )

    saved_task = await store.save(
        running_task,
        expected_version=original_task.version,
    )

    assert saved_task.version == 2
    assert await store.load(original_task.task_id) == running_task
    assert len(redis.eval_calls) == 1


@pytest.mark.asyncio
async def test_save_should_reject_stale_expected_version() -> None:
    """验证两个调用方同时修改时，后保存的旧快照发生版本冲突。"""

    store = RedisLongTaskStore(FakeRedis())
    original_task = build_task()
    await store.create(original_task)
    first_update = transition_long_task(
        original_task,
        target_status="running",
        active_step_ids=["step_1"],
    )
    stale_update = transition_long_task(
        original_task,
        target_status="queued",
    )
    await store.save(first_update, expected_version=1)

    with pytest.raises(
        LongTaskVersionConflictError,
        match="expected=1, actual=2",
    ):
        await store.save(stale_update, expected_version=1)

    assert await store.load(original_task.task_id) == first_update


@pytest.mark.asyncio
async def test_save_should_require_existing_task() -> None:
    """验证更新不存在的任务时返回明确异常，而不是静默创建。"""

    original_task = build_task()
    running_task = transition_long_task(
        original_task,
        target_status="running",
        active_step_ids=["step_1"],
    )

    with pytest.raises(LongTaskNotFoundError, match="不存在"):
        await RedisLongTaskStore(FakeRedis()).save(
            running_task,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_save_should_require_exact_next_version() -> None:
    """验证调用方不能跳过版本号或重复保存相同版本。"""

    task = build_task().model_copy(update={"version": 3})

    with pytest.raises(ValueError, match=r"expected_version \+ 1"):
        await RedisLongTaskStore(FakeRedis()).save(
            task,
            expected_version=1,
        )


@pytest.mark.asyncio
async def test_load_should_reject_corrupt_snapshot() -> None:
    """验证损坏 JSON 不会伪装成可执行任务进入运行时。"""

    redis = FakeRedis()
    redis.values[
        "dog-agent:long-task:v1:task:task_001"
    ] = "not-json"

    with pytest.raises(CorruptLongTaskSnapshotError, match="无法恢复"):
        await RedisLongTaskStore(redis).load("task_001")


@pytest.mark.asyncio
async def test_load_missing_task_should_return_none() -> None:
    """验证普通查询不存在的任务时返回 None。"""

    loaded_task = await RedisLongTaskStore(FakeRedis()).load(
        "task_missing"
    )

    assert loaded_task is None
