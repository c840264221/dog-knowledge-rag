"""Redis 长任务可靠业务提交 Store 单元测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskApplicationService,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskCommitContentConflictError,
    LongTaskCommitFingerprintConflictError,
    LongTaskExecutionContext,
    LongTaskGoal,
    LongTaskQueueMessage,
    LongTaskRuntimeDriver,
    LongTaskStep,
    LongTaskVersionConflictError,
    RedisLongTaskStore,
)
from src.runtime.long_tasks.commit_service import (
    build_batch_result_fingerprint_view,
    build_commit_request,
    calculate_request_fingerprint,
)
from src.runtime.long_tasks.interaction_service import (
    LongTaskInteractionService,
)


class FakeReliableRedis:
    """模拟可靠提交脚本需要的 Redis String、List 和 Stream。"""

    def __init__(self) -> None:
        """初始化空 Redis 数据和脚本调用记录。"""

        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.eval_count = 0

    async def get(self, key: str) -> str | None:
        """读取模拟 Redis String。"""

        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
    ) -> bool:
        """模拟普通 SET 和 SET NX。"""

        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def lrange(
        self,
        key: str,
        start: int,
        end: int,
    ) -> list[str]:
        """按 Redis LRANGE 的闭区间语义读取事件列表。"""

        values = self.lists.get(key, [])
        return values[start : end + 1]

    async def eval(
        self,
        script: str,
        numkeys: int,
        *args: Any,
    ) -> list[Any]:
        """
        模拟可靠提交 Lua 的幂等、版本、事件和消息写入语义。

        参数含义：
            script/numkeys/args：Store 传入的 Lua 和键值参数。

        返回值含义：
            list[Any]：与可靠提交脚本一致的结果码、版本和回执 JSON。
        """

        del script
        assert numkeys == 5
        self.eval_count += 1
        task_key, sequence_key, events_key, receipt_key, stream_key = (
            str(value) for value in args[:5]
        )
        (
            operation,
            raw_expected_version,
            task_json,
            raw_task_version,
            task_id,
            request_fingerprint,
            submission_fingerprint,
            event_blueprints_json,
            receipt_template_json,
            queue_payload,
        ) = args[5:]

        existing_receipt = self.values.get(receipt_key)
        if existing_receipt is not None:
            receipt = json.loads(existing_receipt)
            if receipt["request_fingerprint"] != request_fingerprint:
                return [-3, receipt["task_version"], existing_receipt]
            if (
                receipt["submission_fingerprint"]
                != submission_fingerprint
            ):
                return [-5, receipt["task_version"], existing_receipt]
            return [2, receipt["task_version"], existing_receipt]

        current_json = self.values.get(task_key)
        if operation == "create":
            if current_json is not None:
                return [-4, -1, ""]
        else:
            if current_json is None:
                return [-1, -1, ""]
            current_version = int(json.loads(current_json)["version"])
            if current_version != int(raw_expected_version):
                return [0, current_version, ""]

        task_version = int(raw_task_version)
        next_task = json.loads(str(task_json))
        assert next_task["task_id"] == task_id
        assert int(next_task["version"]) == task_version
        current_sequence = int(self.values.get(sequence_key, "0"))
        blueprints = json.loads(str(event_blueprints_json))
        event_ids: list[str] = []
        event_jsons: list[str] = []
        for index, event in enumerate(blueprints, start=1):
            event["sequence"] = current_sequence + index
            event_ids.append(event["event_id"])
            event_jsons.append(
                json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            )

        sequence_start = current_sequence + 1
        sequence_end = current_sequence + len(event_jsons)
        self.values[task_key] = str(task_json)
        self.values[sequence_key] = str(sequence_end)
        self.lists.setdefault(events_key, []).extend(event_jsons)

        queue_message_id = None
        if queue_payload:
            entries = self.streams.setdefault(stream_key, [])
            queue_message_id = f"1000-{len(entries)}"
            entries.append(
                (queue_message_id, {"payload": str(queue_payload)})
            )

        receipt = json.loads(str(receipt_template_json))
        receipt.update(
            {
                "event_ids": event_ids,
                "event_sequence_start": sequence_start,
                "event_sequence_end": sequence_end,
                "queue_message_id": queue_message_id,
            }
        )
        receipt_json = json.dumps(
            receipt,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.values[receipt_key] = receipt_json
        return [1, task_version, receipt_json]


class ReceiptReadBarrierStore:
    """让两个可靠提交在首次回执查询后同时继续的 Store 包装器。"""

    def __init__(self, store: RedisLongTaskStore) -> None:
        """保存真实 Store，并初始化两方回执读取屏障。"""

        self._store = store
        self._empty_receipt_reads = 0
        self._receipt_barrier = asyncio.Event()
        self._initial_task_reads = 0
        self._task_barrier = asyncio.Event()

    async def load(self, task_id: str) -> LongTask | None:
        """把权威任务读取委托给内部 Redis Store。"""

        task = await self._store.load(task_id)
        if (
            self._receipt_barrier.is_set()
            and self._initial_task_reads < 2
        ):
            self._initial_task_reads += 1
            if self._initial_task_reads >= 2:
                self._task_barrier.set()
            await self._task_barrier.wait()
        return task

    async def load_receipt(
        self,
        *,
        task_id: str,
        commit_id: str,
    ) -> Any:
        """确保前两个调用都先看到空回执，再允许它们并发构建提交。"""

        receipt = await self._store.load_receipt(
            task_id=task_id,
            commit_id=commit_id,
        )
        if receipt is not None:
            return receipt
        self._empty_receipt_reads += 1
        if self._empty_receipt_reads >= 2:
            self._receipt_barrier.set()
        await self._receipt_barrier.wait()
        return None

    async def commit(self, request: Any) -> Any:
        """把可靠提交委托给内部 Redis Store。"""

        return await self._store.commit(request)


def build_initial_task() -> LongTask:
    """构建第一步运行中、第二步等待依赖的版本 1 后台任务。"""

    return LongTask(
        task_id="task_reliable",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="读取档案后生成建议",
            objective="按依赖完成健康建议",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_reliable",
                title="读取档案",
                assigned_agent="profile_agent",
                status="running",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_reliable",
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


def build_advanced_task(task: LongTask) -> LongTask:
    """构建步骤一完成且步骤二 Ready 的版本 2 快照。"""

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
                    "output_summary": "档案读取完成",
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


def test_batch_result_fingerprint_view_should_ignore_diagnostics_and_order(
) -> None:
    """验证诊断信息和多智能体结果到达顺序不影响请求指纹。"""

    first_result = LongTaskBatchResult(
        batch_id="batch-stable-1",
        task_id="task_reliable",
        step_results=[
            LongTaskBatchStepResult(
                step_id="step_2",
                status="completed",
                output_summary="建议生成完成",
                output_ref="artifact-advice@v1",
                claim_id="claim-2",
                metadata={"duration_ms": 200},
            ),
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                output_summary="档案读取完成",
                output_ref="artifact-profile@v1",
                claim_id="claim-1",
                metadata={"trace_id": "trace-first"},
            ),
        ],
        metadata={
            "worker_name": "worker-1",
            "finished_at": "2026-09-01T10:00:00Z",
        },
    )
    retried_result = LongTaskBatchResult(
        batch_id=first_result.batch_id,
        task_id=first_result.task_id,
        actor_type="worker",
        actor_id="worker-1",
        trace_id="trace-retry",
        step_results=[
            first_result.step_results[1].model_copy(
                update={
                    "span_id": "span-step-1-retry",
                    "metadata": {"trace_id": "trace-retry"},
                }
            ),
            first_result.step_results[0].model_copy(
                update={
                    "span_id": "span-step-2-retry",
                    "metadata": {"duration_ms": 999},
                }
            ),
        ],
        metadata={
            "finished_at": "2026-09-01T10:00:05Z",
        },
    )

    first_view = build_batch_result_fingerprint_view(first_result)
    retried_view = build_batch_result_fingerprint_view(retried_result)

    assert first_view == retried_view
    assert [
        result["step_id"] for result in first_view["step_results"]
    ] == ["step_1", "step_2"]
    assert calculate_request_fingerprint(first_view) == (
        calculate_request_fingerprint(retried_view)
    )


def test_batch_result_business_change_should_change_fingerprint() -> None:
    """验证产物引用等稳定业务事实变化时请求指纹随之改变。"""

    first_result = LongTaskBatchResult(
        batch_id="batch-stable-1",
        task_id="task_reliable",
        step_results=[
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                output_summary="档案读取完成",
                output_ref="artifact-profile@v1",
                claim_id="claim-1",
            )
        ],
    )
    changed_result = first_result.model_copy(
        update={
            "step_results": [
                first_result.step_results[0].model_copy(
                    update={"output_ref": "artifact-profile@v2"}
                )
            ]
        }
    )

    first_view = build_batch_result_fingerprint_view(first_result)
    changed_view = build_batch_result_fingerprint_view(changed_result)

    assert calculate_request_fingerprint(first_view) != (
        calculate_request_fingerprint(changed_view)
    )


@pytest.mark.asyncio
async def test_create_commit_should_persist_event_queue_and_receipt() -> None:
    """验证首次提交把快照、事件、消息和回执作为一个结果保存。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    task = build_initial_task()
    queue_message = LongTaskQueueMessage(
        task_id=task.task_id,
        task_version=task.version,
        reason="submitted",
        ready_step_ids=[],
        correlation_id="handoff-001",
    )
    request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id="handoff:task_reliable",
        fingerprint_payload={"action": "handoff", "task_id": task.task_id},
        actor_type="system",
        actor_id="collaboration_handoff",
        correlation_id="handoff-001",
        queue_message=queue_message,
    )

    receipt = await store.commit(request)

    assert await store.load(task.task_id) == task
    assert receipt.task_version == 1
    assert receipt.event_sequence_start == 1
    assert receipt.event_sequence_end == 1
    assert receipt.queue_message_id == "1000-0"
    assert receipt.request_fingerprint == request.request_fingerprint
    assert receipt.submission_fingerprint == request.submission_fingerprint
    events = await store.load_events(task.task_id)
    assert [event.event_type for event in events] == ["task_submitted"]
    assert events[0].commit_id == request.commit_id
    assert len(redis.streams["dog-agent:long-task:v1:stream"]) == 1


@pytest.mark.asyncio
async def test_same_commit_should_return_receipt_without_duplicate_writes() -> None:
    """验证相同 commit_id 和指纹重试不会重复追加事件或消息。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    task = build_initial_task()
    request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id="create:task_reliable",
        fingerprint_payload={"action": "create", "task_id": task.task_id},
        actor_type="system",
        actor_id="test",
        correlation_id="trace-001",
    )

    first_receipt = await store.commit(request)
    second_receipt = await store.commit(request)

    assert second_receipt == first_receipt
    assert len(await store.load_events(task.task_id)) == 1
    assert redis.eval_count == 2


@pytest.mark.asyncio
async def test_same_commit_with_different_request_fingerprint_should_fail() -> None:
    """验证同一提交编号不能静默接受不同业务动作。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    task = build_initial_task()
    first_request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id="create:task_reliable",
        fingerprint_payload={"action": "create", "value": 1},
        actor_type="system",
        actor_id="test",
        correlation_id=None,
    )
    conflicting_request = first_request.model_copy(
        update={"request_fingerprint": "sha256:different"}
    )
    await store.commit(first_request)

    with pytest.raises(
        LongTaskCommitFingerprintConflictError,
        match="不同业务动作",
    ):
        await store.commit(conflicting_request)

    assert len(await store.load_events(task.task_id)) == 1


@pytest.mark.asyncio
async def test_same_request_with_different_submission_should_fail() -> None:
    """验证同一业务动作不能静默接受不同的最终持久化内容。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    task = build_initial_task()
    first_request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id="create:task_reliable",
        fingerprint_payload={"action": "create", "task_id": task.task_id},
        actor_type="system",
        actor_id="test",
        correlation_id=None,
    )
    conflicting_request = build_commit_request(
        previous_task=None,
        next_task=task,
        commit_id=first_request.commit_id,
        fingerprint_payload={"action": "create", "task_id": task.task_id},
        actor_type="system",
        actor_id="test",
        correlation_id=None,
        queue_message=LongTaskQueueMessage(
            task_id=task.task_id,
            task_version=task.version,
            reason="submitted",
            ready_step_ids=[],
        ),
    )

    assert (
        conflicting_request.request_fingerprint
        == first_request.request_fingerprint
    )
    assert (
        conflicting_request.submission_fingerprint
        != first_request.submission_fingerprint
    )
    await store.commit(first_request)

    with pytest.raises(
        LongTaskCommitContentConflictError,
        match="不同提交内容",
    ):
        await store.commit(conflicting_request)

    assert len(await store.load_events(task.task_id)) == 1
    assert redis.streams == {}


@pytest.mark.asyncio
async def test_technical_times_should_not_change_commit_fingerprints() -> None:
    """验证框架自动时间变化不会把同一语义提交误判为内容冲突。"""

    task = build_claimed_runtime_task()
    later_time = datetime.now(timezone.utc) + timedelta(minutes=5)
    retried_task = LongTask.model_validate(
        {
            **task.model_dump(mode="python"),
            "created_at": later_time,
            "updated_at": later_time,
            "steps": [
                {
                    **step.model_dump(mode="python"),
                    "created_at": later_time,
                    "updated_at": later_time,
                    "last_trace_id": "trace-retry",
                    "last_span_id": f"span-{step.step_id}-retry",
                    "lease_expires_at": (
                        later_time if step.claim_id is not None else None
                    ),
                }
                for step in task.steps
            ],
        }
    )
    first_message = LongTaskQueueMessage(
        task_id=task.task_id,
        task_version=task.version,
        reason="continued",
        ready_step_ids=[],
        enqueued_at=datetime.now(timezone.utc),
    )
    retried_message = first_message.model_copy(
        update={"enqueued_at": later_time}
    )
    common_arguments = {
        "previous_task": None,
        "commit_id": "create:task_reliable",
        "fingerprint_payload": {
            "action": "create",
            "task_id": task.task_id,
        },
        "actor_type": "system",
        "actor_id": "test",
        "correlation_id": None,
    }

    first_request = build_commit_request(
        next_task=task,
        queue_message=first_message,
        **common_arguments,
    )
    retried_request = build_commit_request(
        next_task=retried_task,
        queue_message=retried_message,
        **common_arguments,
    )

    assert task.model_dump(mode="json") != retried_task.model_dump(
        mode="json"
    )
    assert first_message.enqueued_at != retried_message.enqueued_at
    assert first_request.request_fingerprint == (
        retried_request.request_fingerprint
    )
    assert first_request.submission_fingerprint == (
        retried_request.submission_fingerprint
    )


@pytest.mark.asyncio
async def test_create_retry_with_new_times_should_return_authoritative_task(
) -> None:
    """验证创建重试只改变技术时间时返回第一次保存的权威快照。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    service = LongTaskApplicationService(store)
    first_task = build_initial_task()
    later_time = datetime.now(timezone.utc) + timedelta(minutes=5)
    retried_task = LongTask.model_validate(
        {
            **first_task.model_dump(mode="python"),
            "created_at": later_time,
            "updated_at": later_time,
            "steps": [
                {
                    **step.model_dump(mode="python"),
                    "created_at": later_time,
                    "updated_at": later_time,
                }
                for step in first_task.steps
            ],
        }
    )

    first_result = await service.create_task(first_task)
    retried_result = await service.create_task(retried_task)

    assert retried_result == first_result
    assert retried_result != retried_task
    assert await store.load(first_task.task_id) == first_result
    assert len(await store.load_events(first_task.task_id)) == 1


@pytest.mark.asyncio
async def test_update_commit_should_append_multiple_ordered_events() -> None:
    """验证一个任务版本可以连续追加步骤完成和下游 Ready 两条事件。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    current_task = build_initial_task()
    await store.commit(
        build_commit_request(
            previous_task=None,
            next_task=current_task,
            commit_id="create:task_reliable",
            fingerprint_payload={"action": "create"},
            actor_type="system",
            actor_id="test",
            correlation_id=None,
        )
    )
    next_task = build_advanced_task(current_task)
    request = build_commit_request(
        previous_task=current_task,
        next_task=next_task,
        commit_id="batch:task_reliable:batch-1",
        fingerprint_payload={"action": "batch", "batch_id": "batch-1"},
        actor_type="worker",
        actor_id="worker-1",
        correlation_id="batch-1",
    )

    receipt = await store.commit(request)

    assert receipt.event_sequence_start == 2
    assert receipt.event_sequence_end == 3
    events = await store.load_events(current_task.task_id)
    assert [event.sequence for event in events] == [1, 2, 3]
    assert [event.event_type for event in events[1:]] == [
        "step_completed",
        "step_became_ready",
    ]
    assert all(event.task_version == 2 for event in events[1:])


@pytest.mark.asyncio
async def test_update_commit_version_conflict_should_write_nothing() -> None:
    """验证 expected_version 过期时不追加事件、消息或回执。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    current_task = build_initial_task()
    await store.commit(
        build_commit_request(
            previous_task=None,
            next_task=current_task,
            commit_id="create:task_reliable",
            fingerprint_payload={"action": "create"},
            actor_type="system",
            actor_id="test",
            correlation_id=None,
        )
    )
    next_task = build_advanced_task(current_task)
    stale_request = build_commit_request(
        previous_task=current_task,
        next_task=next_task,
        commit_id="batch:stale",
        fingerprint_payload={"action": "stale"},
        actor_type="worker",
        actor_id="worker-old",
        correlation_id=None,
    )
    redis.values["dog-agent:long-task:v1:task:task_reliable"] = (
        next_task.model_dump_json()
    )

    with pytest.raises(LongTaskVersionConflictError):
        await store.commit(stale_request)

    assert len(await store.load_events(current_task.task_id)) == 1
    assert (
        await store.load_receipt(
            task_id=current_task.task_id,
            commit_id="batch:stale",
        )
        is None
    )


class RejectingPublisher:
    """确保可靠提交路径不会再次调用独立 Stream Publisher。"""

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """任何独立发布调用都表示原子提交接入发生回退。"""

        raise AssertionError(f"不应独立发布消息: {message.reason}")


def build_handoff_task() -> LongTask:
    """构建包含请求内历史结果和后台 Ready Step 的交接任务。"""

    task = build_initial_task()
    return LongTask.model_validate(
        {
            **task.model_dump(mode="python"),
            "active_step_ids": [],
            "steps": [
                {
                    **task.steps[0].model_dump(mode="python"),
                    "status": "completed",
                    "metadata": {
                        "migrated_from_inline_result": True,
                    },
                },
                {
                    **task.steps[1].model_dump(mode="python"),
                    "status": "ready",
                },
            ],
            "metadata": {
                "source_type": "collaboration_budget_handoff",
                "source_collaboration_id": "collaboration-001",
            },
        }
    )


@pytest.mark.asyncio
async def test_handoff_should_atomically_create_and_enqueue() -> None:
    """验证首次后台交接不会再调用保存后的独立发布器。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    service = LongTaskApplicationService(
        store,
        queue_publisher=RejectingPublisher(),
    )

    saved_task = await service.handoff_paused_collaboration_task(
        build_handoff_task()
    )

    assert saved_task.version == 1
    assert await store.load(saved_task.task_id) == saved_task
    stream_entries = redis.streams["dog-agent:long-task:v1:stream"]
    assert len(stream_entries) == 1
    message = LongTaskQueueMessage.model_validate_json(
        stream_entries[0][1]["payload"]
    )
    assert message.reason == "submitted"
    assert message.ready_step_ids == ["step_2"]


def build_claimed_runtime_task() -> LongTask:
    """构建可由 RuntimeDriver 接受结果的版本 1 已领取任务。"""

    task = build_initial_task()
    return LongTask.model_validate(
        {
            **task.model_dump(mode="python"),
            "steps": [
                {
                    **task.steps[0].model_dump(mode="python"),
                    "claimed_by": "worker-1",
                    "claim_id": "claim-1",
                    "lease_expires_at": (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=30)
                    ),
                },
                task.steps[1],
            ],
        }
    )


@pytest.mark.asyncio
async def test_driver_should_atomically_commit_events_and_continuation() -> None:
    """验证 Worker 结果接受与 continued 消息处于同一可靠提交。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    current_task = build_claimed_runtime_task()
    redis.values[
        "dog-agent:long-task:v1:task:task_reliable"
    ] = current_task.model_dump_json()
    batch_result = LongTaskBatchResult(
        batch_id="batch-runtime-1",
        task_id=current_task.task_id,
        step_results=[
            LongTaskBatchStepResult(
                step_id="step_1",
                status="completed",
                output_summary="档案读取完成",
                output_ref="artifact-profile@v1",
                claim_id="claim-1",
            )
        ],
        metadata={"worker_name": "worker-1"},
    )
    driver = LongTaskRuntimeDriver(store)

    saved_task = await driver.handle_batch_result(
        task_id=current_task.task_id,
        batch_result=batch_result,
        execution_context=LongTaskExecutionContext(
            elapsed_ms=10,
            inline_budget_ms=1,
        ),
        publish_ready_message=True,
        correlation_id="trace-runtime-1",
    )

    assert saved_task.version == 2
    assert [step.status for step in saved_task.steps] == [
        "completed",
        "ready",
    ]
    events = await store.load_events(saved_task.task_id)
    assert [event.event_type for event in events] == [
        "step_completed",
        "step_became_ready",
    ]
    stream_entries = redis.streams["dog-agent:long-task:v1:stream"]
    assert len(stream_entries) == 1
    message = LongTaskQueueMessage.model_validate_json(
        stream_entries[0][1]["payload"]
    )
    assert message.reason == "continued"
    assert message.ready_step_ids == ["step_2"]

    # 模拟提交已成功但调用方没有收到响应，使用同一批次再次提交。
    retried_batch_result = batch_result.model_copy(
        update={
            "step_results": [
                batch_result.step_results[0].model_copy(
                    update={"metadata": {"duration_ms": 999}}
                )
            ],
            "metadata": {
                "worker_name": "worker-1",
                "finished_at": "2026-09-01T10:00:05Z",
            },
        }
    )
    retried_task = await driver.handle_batch_result(
        task_id=current_task.task_id,
        batch_result=retried_batch_result,
        execution_context=LongTaskExecutionContext(
            elapsed_ms=999,
            inline_budget_ms=1,
        ),
        publish_ready_message=True,
        correlation_id="trace-runtime-1",
    )

    assert retried_task == saved_task
    assert len(await store.load_events(saved_task.task_id)) == 2
    assert len(redis.streams["dog-agent:long-task:v1:stream"]) == 1


@pytest.mark.asyncio
async def test_concurrent_same_batch_should_return_one_authoritative_task(
) -> None:
    """验证并发重复批次都返回 Redis 中同一份权威任务。"""

    redis = FakeReliableRedis()
    redis_store = RedisLongTaskStore(redis)
    current_task = build_claimed_runtime_task()
    redis.values[
        "dog-agent:long-task:v1:task:task_reliable"
    ] = current_task.model_dump_json()
    barrier_store = ReceiptReadBarrierStore(redis_store)
    driver = LongTaskRuntimeDriver(barrier_store)

    def build_result(trace_suffix: str) -> LongTaskBatchResult:
        """构建业务事实相同、诊断引用不同的同批次结果。"""

        return LongTaskBatchResult(
            batch_id="batch-concurrent-same",
            task_id=current_task.task_id,
            actor_type="worker",
            actor_id="worker-1",
            trace_id=f"trace-{trace_suffix}",
            step_results=[
                LongTaskBatchStepResult(
                    step_id="step_1",
                    status="completed",
                    output_summary="档案读取完成",
                    output_ref="artifact-profile@v1",
                    claim_id="claim-1",
                    span_id=f"span-{trace_suffix}",
                    metadata={"duration_ms": len(trace_suffix)},
                )
            ],
        )

    first_task, second_task = await asyncio.gather(
        driver.handle_batch_result(
            task_id=current_task.task_id,
            batch_result=build_result("first"),
            execution_context=LongTaskExecutionContext(
                elapsed_ms=10,
                inline_budget_ms=1,
            ),
            publish_ready_message=True,
            correlation_id="trace-business-stable",
        ),
        driver.handle_batch_result(
            task_id=current_task.task_id,
            batch_result=build_result("second"),
            execution_context=LongTaskExecutionContext(
                elapsed_ms=20,
                inline_budget_ms=1,
            ),
            publish_ready_message=True,
            correlation_id="trace-business-stable",
        ),
    )

    authoritative_task = await redis_store.load(current_task.task_id)
    assert authoritative_task is not None
    assert first_task == authoritative_task
    assert second_task == authoritative_task
    assert authoritative_task.steps[0].last_trace_id in {
        "trace-first",
        "trace-second",
    }
    assert len(await redis_store.load_events(current_task.task_id)) == 2
    assert len(redis.streams["dog-agent:long-task:v1:stream"]) == 1


@pytest.mark.asyncio
async def test_user_input_retry_should_return_original_commit() -> None:
    """验证用户输入提交响应丢失后不会重复恢复或重复发布消息。"""

    redis = FakeReliableRedis()
    store = RedisLongTaskStore(redis)
    interaction_service = LongTaskInteractionService(
        interaction_id_factory=lambda: "interaction-001"
    )
    task = build_handoff_task()
    waiting_step = LongTaskStep.model_validate(
        {
            **task.steps[1].model_dump(mode="python"),
            "status": "awaiting_input",
            "waiting_reason": "missing_input",
        }
    )
    waiting_task = interaction_service.wait_for_step_input(
        LongTask.model_validate(
            {
                **task.model_dump(mode="python"),
                "steps": [task.steps[0], waiting_step],
            }
        ),
        waiting_step_ids=["step_2"],
        interaction_type="missing_input",
        prompt="请补充狗狗年龄。",
    )
    redis.values[
        "dog-agent:long-task:v1:task:task_reliable"
    ] = waiting_task.model_dump_json()
    service = LongTaskApplicationService(
        store,
        interaction_service=interaction_service,
        queue_publisher=RejectingPublisher(),
    )
    input_args = {
        "task_id": waiting_task.task_id,
        "user_id": waiting_task.user_id,
        "interaction_id": "interaction-001",
        "action": "submit_input",
        "answers": {"step_2": "6岁"},
    }

    first_result = await service.respond_to_missing_input(**input_args)
    second_result = await service.respond_to_missing_input(**input_args)

    assert second_result == first_result
    assert second_result.status == "running"
    assert second_result.steps[1].status == "ready"
    assert len(await store.load_events(waiting_task.task_id)) == 2
    assert len(redis.streams["dog-agent:long-task:v1:stream"]) == 1
