"""使用真实 Redis 验证长任务可靠提交 Lua 边界。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.runtime.long_tasks import (
    LongTask,
    LongTaskCommitContentConflictError,
    LongTaskCommitFingerprintConflictError,
    LongTaskGoal,
    LongTaskQueueMessage,
    LongTaskStep,
    LongTaskVersionConflictError,
    RedisLongTaskStore,
)
from src.runtime.long_tasks.commit_service import build_commit_request


@pytest_asyncio.fixture
async def real_redis_store() -> AsyncIterator[tuple[Redis, RedisLongTaskStore]]:
    """
    为每个测试创建独立 Redis 命名空间，并在结束后精确清理。

    返回值含义：
        AsyncIterator[tuple[Redis, RedisLongTaskStore]]：真实异步 Redis 客户端
        与使用随机 Key 前缀的长任务 Store。
    """

    redis_url = os.getenv(
        "LONG_TASK_REDIS_TEST_URL",
        "redis://localhost:6379/15",
    )
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except RedisError as exc:
        await client.aclose()
        pytest.skip(f"真实 Redis 未启动或不可连接: {exc}")

    namespace = f"dog-agent:test:reliable:{uuid4().hex}"
    store = RedisLongTaskStore(
        client,
        key_prefix=namespace,
        stream_key=f"{namespace}:stream",
    )
    try:
        yield client, store
    finally:
        keys = [
            key
            async for key in client.scan_iter(match=f"{namespace}:*")
        ]
        if keys:
            await client.delete(*keys)
        await client.aclose()


def _build_initial_task() -> LongTask:
    """
    构建第一步运行中、第二步等待依赖的版本 1 后台任务。

    返回值含义：
        LongTask：可用于创建提交和后续版本竞争的初始权威快照。
    """

    return LongTask(
        task_id="task_real_redis",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="读取档案后生成建议",
            objective="按依赖完成健康建议",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_real_redis",
                title="读取档案",
                assigned_agent="profile_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_real_redis",
                title="生成建议",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="pending",
            ),
        ],
        status="running",
        execution_mode="durable",
        active_step_ids=["step_1"],
    )


def _build_advanced_task(
    task: LongTask,
    *,
    output_summary: str = "档案读取完成",
) -> LongTask:
    """
    构建步骤一完成、步骤二 Ready 的版本 2 候选快照。

    参数含义：
        task：版本 1 的旧任务快照。
        output_summary：用于区分并发候选内容的步骤结果摘要。

    返回值含义：
        LongTask：版本只增加 1、包含两个步骤状态变化的候选快照。
    """

    return LongTask.model_validate(
        {
            **task.model_dump(mode="python"),
            "version": 2,
            "active_step_ids": [],
            "steps": [
                {
                    **task.steps[0].model_dump(mode="python"),
                    "status": "completed",
                    "version": 2,
                    "output_summary": output_summary,
                    "output_ref": "artifact-profile@v1",
                },
                {
                    **task.steps[1].model_dump(mode="python"),
                    "status": "ready",
                    "version": 2,
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_real_redis_commit_should_retry_without_duplicate_writes(
    real_redis_store: tuple[Redis, RedisLongTaskStore],
) -> None:
    """
    验证真实 Lua 首次提交和相同内容重试只产生一套正式结果。

    参数含义：
        real_redis_store：隔离命名空间中的真实 Redis 客户端和 Store。

    返回值含义：
        None：快照、事件、消息和回执满足幂等断言时测试通过。
    """

    client, store = real_redis_store
    task = _build_initial_task()
    queue_message = LongTaskQueueMessage(
        task_id=task.task_id,
        task_version=task.version,
        reason="submitted",
        ready_step_ids=[],
        correlation_id="handoff-real-001",
    )
    request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id="handoff:task_real_redis",
        fingerprint_payload={
            "action": "handoff",
            "task_id": task.task_id,
        },
        actor_type="system",
        actor_id="integration_test",
        correlation_id="handoff-real-001",
        queue_message=queue_message,
    )

    first_receipt = await store.commit(request)
    second_receipt = await store.commit(request)

    assert second_receipt == first_receipt
    assert await store.load(task.task_id) == task
    assert len(await store.load_events(task.task_id)) == 1
    assert await client.xlen(store._stream_key) == 1
    assert first_receipt.queue_message_id is not None
    assert first_receipt.request_fingerprint == request.request_fingerprint
    assert (
        first_receipt.submission_fingerprint
        == request.submission_fingerprint
    )

    conflicting_content_request = request.model_copy(
        update={"submission_fingerprint": "sha256:different"}
    )
    with pytest.raises(LongTaskCommitContentConflictError):
        await store.commit(conflicting_content_request)
    conflicting_action_request = request.model_copy(
        update={"request_fingerprint": "sha256:different"}
    )
    with pytest.raises(LongTaskCommitFingerprintConflictError):
        await store.commit(conflicting_action_request)
    assert len(await store.load_events(task.task_id)) == 1
    assert await client.xlen(store._stream_key) == 1


@pytest.mark.asyncio
async def test_real_redis_update_should_allocate_ordered_events(
    real_redis_store: tuple[Redis, RedisLongTaskStore],
) -> None:
    """
    验证真实 Redis 为同一任务版本连续分配多条事件序号。

    参数含义：
        real_redis_store：隔离命名空间中的真实 Redis 客户端和 Store。

    返回值含义：
        None：步骤完成和下游 Ready 事件顺序稳定时测试通过。
    """

    _, store = real_redis_store
    current_task = _build_initial_task()
    await store.commit(
        build_commit_request(
            previous_task=None,
            next_task=current_task,
            commit_id="create:task_real_redis",
            fingerprint_payload={"action": "create"},
            actor_type="system",
            actor_id="integration_test",
            correlation_id=None,
        )
    )
    next_task = _build_advanced_task(current_task)
    receipt = await store.commit(
        build_commit_request(
            previous_task=current_task,
            next_task=next_task,
            commit_id="batch:task_real_redis:batch-1",
            fingerprint_payload={
                "action": "batch",
                "batch_id": "batch-1",
            },
            actor_type="worker",
            actor_id="worker-real-1",
            correlation_id="batch-1",
        )
    )

    events = await store.load_events(current_task.task_id)
    assert receipt.event_sequence_start == 2
    assert receipt.event_sequence_end == 3
    assert [event.sequence for event in events] == [1, 2, 3]
    assert [event.event_type for event in events[1:]] == [
        "step_completed",
        "step_became_ready",
    ]
    assert all(event.task_version == 2 for event in events[1:])


@pytest.mark.asyncio
async def test_real_redis_concurrent_updates_should_have_one_winner(
    real_redis_store: tuple[Redis, RedisLongTaskStore],
) -> None:
    """
    验证两个同版本并发提交只有一个能写入事件、消息和回执。

    参数含义：
        real_redis_store：隔离命名空间中的真实 Redis 客户端和 Store。

    返回值含义：
        None：一个提交成功、另一个版本冲突且无半套副作用时测试通过。
    """

    client, store = real_redis_store
    current_task = _build_initial_task()
    await store.commit(
        build_commit_request(
            previous_task=None,
            next_task=current_task,
            commit_id="create:task_real_redis",
            fingerprint_payload={"action": "create"},
            actor_type="system",
            actor_id="integration_test",
            correlation_id=None,
        )
    )

    requests = []
    for suffix, summary in (("a", "候选 A"), ("b", "候选 B")):
        next_task = _build_advanced_task(
            current_task,
            output_summary=summary,
        )
        requests.append(
            build_commit_request(
                previous_task=current_task,
                next_task=next_task,
                commit_id=f"batch:task_real_redis:{suffix}",
                fingerprint_payload={
                    "action": "batch",
                    "candidate": suffix,
                },
                actor_type="worker",
                actor_id=f"worker-{suffix}",
                correlation_id=f"trace-{suffix}",
                queue_message=LongTaskQueueMessage(
                    task_id=next_task.task_id,
                    task_version=next_task.version,
                    reason="continued",
                    ready_step_ids=["step_2"],
                    correlation_id=f"trace-{suffix}",
                ),
            )
        )

    results = await asyncio.gather(
        *(store.commit(request) for request in requests),
        return_exceptions=True,
    )

    receipts = [result for result in results if not isinstance(result, Exception)]
    conflicts = [
        result
        for result in results
        if isinstance(result, LongTaskVersionConflictError)
    ]
    assert len(receipts) == 1
    assert len(conflicts) == 1
    assert len(await store.load_events(current_task.task_id)) == 3
    assert await client.xlen(store._stream_key) == 1
    stored_receipts = [
        await store.load_receipt(
            task_id=current_task.task_id,
            commit_id=request.commit_id,
        )
        for request in requests
    ]
    assert sum(receipt is not None for receipt in stored_receipts) == 1


@pytest.mark.asyncio
async def test_real_redis_same_action_with_different_content_should_conflict(
    real_redis_store: tuple[Redis, RedisLongTaskStore],
) -> None:
    """
    验证相同业务动作并发生成不同候选内容时不会被当成普通重试。

    参数含义：
        real_redis_store：隔离命名空间中的真实 Redis 客户端和 Store。

    返回值含义：
        None：一个候选成功、另一个内容冲突且只产生一套副作用时通过。
    """

    client, store = real_redis_store
    current_task = _build_initial_task()
    await store.commit(
        build_commit_request(
            previous_task=None,
            next_task=current_task,
            commit_id="create:task_real_redis",
            fingerprint_payload={"action": "create"},
            actor_type="system",
            actor_id="integration_test",
            correlation_id=None,
        )
    )

    requests = []
    for suffix, summary in (("a", "候选 A"), ("b", "候选 B")):
        next_task = _build_advanced_task(
            current_task,
            output_summary=summary,
        )
        requests.append(
            build_commit_request(
                previous_task=current_task,
                next_task=next_task,
                commit_id="batch:task_real_redis:same-batch",
                fingerprint_payload={
                    "action": "batch",
                    "batch_id": "same-batch",
                },
                actor_type="worker",
                actor_id=f"worker-{suffix}",
                correlation_id=f"trace-{suffix}",
                queue_message=LongTaskQueueMessage(
                    task_id=next_task.task_id,
                    task_version=next_task.version,
                    reason="continued",
                    ready_step_ids=["step_2"],
                    correlation_id=f"trace-{suffix}",
                ),
            )
        )

    assert requests[0].request_fingerprint == requests[1].request_fingerprint
    assert (
        requests[0].submission_fingerprint
        != requests[1].submission_fingerprint
    )
    results = await asyncio.gather(
        *(store.commit(request) for request in requests),
        return_exceptions=True,
    )

    receipts = [result for result in results if not isinstance(result, Exception)]
    content_conflicts = [
        result
        for result in results
        if isinstance(result, LongTaskCommitContentConflictError)
    ]
    assert len(receipts) == 1
    assert len(content_conflicts) == 1
    assert len(await store.load_events(current_task.task_id)) == 3
    assert await client.xlen(store._stream_key) == 1
