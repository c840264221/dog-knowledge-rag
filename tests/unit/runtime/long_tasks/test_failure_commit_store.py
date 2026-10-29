"""长任务失败原子提交 Store 单元测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskGoal,
    LongTaskStep,
)
from src.runtime.long_tasks.failure_service import (
    build_long_task_failure_commit_request,
    build_long_task_retry_cancellation_request,
    build_long_task_retry_dispatch_request,
    prepare_long_task_failure_commit,
)
from src.runtime.long_tasks.failure_contracts import (
    LongTaskFailureCommitRequest,
)
from src.runtime.long_tasks.store import RedisLongTaskStore


NOW = datetime(2026, 9, 11, 11, 0, tzinfo=timezone.utc)


class FakeFailureCommitRedis:
    """模拟失败提交所需的 Redis String 和 List。"""

    def __init__(self) -> None:
        """初始化空存储和脚本执行次数。"""

        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.eval_count = 0

    async def get(self, key: str) -> str | None:
        """读取一个模拟 Redis String。"""

        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
    ) -> bool:
        """模拟 SET 和 SET NX。"""

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
        """按闭区间读取模拟事件列表。"""

        return self.lists.get(key, [])[start : end + 1]

    async def zrangebyscore(
        self,
        key: str,
        minimum: str,
        maximum: float,
        *,
        start: int,
        num: int,
    ) -> list[str]:
        """按分数和数量限制读取模拟有序集合成员。"""

        assert minimum == "-inf"
        members = [
            member
            for member, score in sorted(
                self.sorted_sets.get(key, {}).items(),
                key=lambda item: (item[1], item[0]),
            )
            if score <= float(maximum)
        ]
        return members[start : start + num]

    async def eval(
        self,
        script: str,
        numkeys: int,
        *args: Any,
    ) -> list[Any]:
        """
        模拟失败 Lua 的版本检查、原子写入和幂等回执。

        参数：
            script/numkeys/args:
                Store 传给 Redis EVAL 的脚本、键数量和全部参数。

        返回值：
            list[Any]：结果码、任务版本和回执 JSON。
        """

        self.eval_count += 1
        if numkeys == 8:
            return self._commit_retry_dispatch(args)
        if numkeys == 3:
            if 'plan["status"] = "cancelled"' in script:
                return self._cancel_retry_plan(args)
            return self._claim_retry_plan(args)
        assert numkeys == 7
        (
            task_key,
            sequence_key,
            events_key,
            receipt_key,
            retry_plan_key,
            dead_letter_key,
            retry_schedule_key,
        ) = (str(value) for value in args[:7])
        (
            expected_version,
            task_json,
            task_version,
            task_id,
            request_fingerprint,
            submission_fingerprint,
            event_blueprints_json,
            receipt_template_json,
            retry_plan_json,
            dead_letter_json,
            retry_schedule_score,
        ) = args[7:]

        existing_receipt = self.values.get(receipt_key)
        if existing_receipt is not None:
            receipt = json.loads(existing_receipt)
            if receipt["request_fingerprint"] != request_fingerprint:
                return [-3, receipt["task_version"], existing_receipt]
            if receipt["submission_fingerprint"] != submission_fingerprint:
                return [-5, receipt["task_version"], existing_receipt]
            return [2, receipt["task_version"], existing_receipt]

        current_json = self.values.get(task_key)
        if current_json is None:
            return [-1, -1, ""]
        current_version = int(json.loads(current_json)["version"])
        if current_version != int(expected_version):
            return [0, current_version, ""]

        next_task = json.loads(str(task_json))
        assert next_task["task_id"] == task_id
        assert next_task["version"] == int(task_version)
        current_sequence = int(self.values.get(sequence_key, "0"))
        event_blueprints = json.loads(str(event_blueprints_json))
        event_ids: list[str] = []
        event_jsons: list[str] = []
        for index, event in enumerate(event_blueprints, start=1):
            event["sequence"] = current_sequence + index
            event_ids.append(event["event_id"])
            event_jsons.append(json.dumps(event, ensure_ascii=False))

        self.values[task_key] = str(task_json)
        self.values[sequence_key] = str(
            current_sequence + len(event_jsons)
        )
        self.lists.setdefault(events_key, []).extend(event_jsons)
        if retry_plan_json:
            self.values[retry_plan_key] = str(retry_plan_json)
            retry_plan = json.loads(str(retry_plan_json))
            self.sorted_sets.setdefault(retry_schedule_key, {})[
                retry_plan["retry_plan_id"]
            ] = float(retry_schedule_score)
        if dead_letter_json:
            self.values[dead_letter_key] = str(dead_letter_json)

        receipt = json.loads(str(receipt_template_json))
        receipt.update(
            {
                "event_ids": event_ids,
                "event_sequence_start": current_sequence + 1,
                "event_sequence_end": current_sequence + len(event_ids),
            }
        )
        receipt_json = json.dumps(receipt, ensure_ascii=False)
        self.values[receipt_key] = receipt_json
        return [1, int(task_version), receipt_json]

    def _claim_retry_plan(self, args: tuple[Any, ...]) -> list[Any]:
        """
        模拟到期重试计划的原子发布权领取。

        参数：
            args:
                领取脚本收到的三个 Key 和六个普通参数。

        返回值：
            list[Any]：领取结果码和领取后的计划 JSON。
        """

        plan_key, schedule_key, lock_key = (
            str(value) for value in args[:3]
        )
        (
            retry_plan_id,
            now_score,
            dispatch_claim_id,
            dispatcher_id,
            lease_milliseconds,
            lease_expires_at,
        ) = args[3:]
        del lease_milliseconds

        plan_json = self.values.get(plan_key)
        if plan_json is None:
            return [-1, ""]
        plan = json.loads(plan_json)
        if plan["status"] not in {"scheduled", "dispatching"}:
            return [0, ""]
        scheduled_score = self.sorted_sets.get(schedule_key, {}).get(
            str(retry_plan_id)
        )
        if scheduled_score is None:
            return [-3, ""]
        if scheduled_score > float(now_score):
            return [0, ""]

        current_claim_id = self.values.get(lock_key)
        if current_claim_id is not None:
            if (
                current_claim_id == dispatch_claim_id
                and plan.get("dispatch_claim_id") == dispatch_claim_id
            ):
                return [2, plan_json]
            return [0, ""]

        self.values[lock_key] = str(dispatch_claim_id)
        plan.update(
            {
                "status": "dispatching",
                "dispatcher_id": dispatcher_id,
                "dispatch_claim_id": dispatch_claim_id,
                "dispatch_lease_expires_at": lease_expires_at,
            }
        )
        claimed_plan_json = json.dumps(plan, ensure_ascii=False)
        self.values[plan_key] = claimed_plan_json
        return [1, claimed_plan_json]

    def _cancel_retry_plan(self, args: tuple[Any, ...]) -> list[Any]:
        """
        模拟失效重试计划的原子取消和发布权清理。

        参数：
            args:
                取消脚本收到的三个 Key 和六个普通参数。

        返回值：
            list[Any]：取消结果码和取消后的计划 JSON。
        """

        plan_key, schedule_key, lock_key = (
            str(value) for value in args[:3]
        )
        (
            retry_plan_id,
            task_id,
            step_id,
            dispatcher_id,
            dispatch_claim_id,
            cancellation_reason,
            cancelled_at,
        ) = args[3:]
        plan_json = self.values.get(plan_key)
        if plan_json is None:
            return [-1, ""]
        plan = json.loads(plan_json)
        if (
            plan["retry_plan_id"] != retry_plan_id
            or plan["task_id"] != task_id
            or plan["step_id"] != step_id
        ):
            return [-2, ""]
        if (
            plan["status"] != "dispatching"
            or plan["dispatcher_id"] != dispatcher_id
            or plan["dispatch_claim_id"] != dispatch_claim_id
            or self.values.get(lock_key) != dispatch_claim_id
        ):
            return [0, ""]
        if retry_plan_id not in self.sorted_sets.get(schedule_key, {}):
            return [0, ""]

        plan.update(
            {
                "status": "cancelled",
                "dispatcher_id": None,
                "dispatch_claim_id": None,
                "dispatch_lease_expires_at": None,
                "cancellation_reason": cancellation_reason,
                "cancelled_at": cancelled_at,
            }
        )
        cancelled_plan_json = json.dumps(plan, ensure_ascii=False)
        self.values[plan_key] = cancelled_plan_json
        self.sorted_sets[schedule_key].pop(str(retry_plan_id))
        self.values.pop(lock_key, None)
        return [1, cancelled_plan_json]

    def _commit_retry_dispatch(self, args: tuple[Any, ...]) -> list[Any]:
        """
        模拟重试发布的令牌检查、消息写入和计划收尾。

        参数：
            args:
                发布脚本收到的八个 Key 和十三个普通参数。

        返回值：
            list[Any]：提交结果码、任务版本和提交回执 JSON。
        """

        (
            task_key,
            plan_key,
            schedule_key,
            lock_key,
            sequence_key,
            events_key,
            receipt_key,
            stream_key,
        ) = (str(value) for value in args[:8])
        (
            expected_version,
            task_json,
            task_version,
            task_id,
            retry_plan_id,
            dispatcher_id,
            dispatch_claim_id,
            request_fingerprint,
            submission_fingerprint,
            event_blueprints_json,
            receipt_template_json,
            queue_message_json,
            dispatched_at,
        ) = args[8:]

        existing_receipt = self.values.get(receipt_key)
        if existing_receipt is not None:
            receipt = json.loads(existing_receipt)
            if receipt["request_fingerprint"] != request_fingerprint:
                return [-3, receipt["task_version"], existing_receipt]
            if receipt["submission_fingerprint"] != submission_fingerprint:
                return [-5, receipt["task_version"], existing_receipt]
            return [2, receipt["task_version"], existing_receipt]

        current_task_json = self.values.get(task_key)
        plan_json = self.values.get(plan_key)
        if current_task_json is None or plan_json is None:
            return [-1, -1, ""]
        current_version = int(json.loads(current_task_json)["version"])
        if current_version != int(expected_version):
            return [0, current_version, ""]

        plan = json.loads(plan_json)
        if (
            plan["status"] != "dispatching"
            or plan["dispatcher_id"] != dispatcher_id
            or plan["dispatch_claim_id"] != dispatch_claim_id
            or self.values.get(lock_key) != dispatch_claim_id
        ):
            return [-9, current_version, ""]

        event_blueprints = json.loads(str(event_blueprints_json))
        current_sequence = int(self.values.get(sequence_key, "0"))
        event_ids: list[str] = []
        event_jsons: list[str] = []
        for index, event in enumerate(event_blueprints, start=1):
            event["sequence"] = current_sequence + index
            event_ids.append(event["event_id"])
            event_jsons.append(json.dumps(event, ensure_ascii=False))

        stream_entries = self.streams.setdefault(stream_key, [])
        queue_message_id = f"2000-{len(stream_entries)}"
        stream_entries.append(
            (queue_message_id, {"payload": str(queue_message_json)})
        )
        plan.update(
            {
                "status": "dispatched",
                "dispatcher_id": None,
                "dispatch_claim_id": None,
                "dispatch_lease_expires_at": None,
                "queue_message_id": queue_message_id,
                "dispatched_at": dispatched_at,
            }
        )
        self.values[task_key] = str(task_json)
        self.values[plan_key] = json.dumps(plan, ensure_ascii=False)
        self.values[sequence_key] = str(
            current_sequence + len(event_jsons)
        )
        self.lists.setdefault(events_key, []).extend(event_jsons)
        self.sorted_sets.get(schedule_key, {}).pop(
            str(retry_plan_id),
            None,
        )
        self.values.pop(lock_key, None)

        receipt = json.loads(str(receipt_template_json))
        receipt.update(
            {
                "event_ids": event_ids,
                "event_sequence_start": current_sequence + 1,
                "event_sequence_end": current_sequence + len(event_ids),
                "queue_message_id": queue_message_id,
            }
        )
        receipt_json = json.dumps(receipt, ensure_ascii=False)
        self.values[receipt_key] = receipt_json
        return [1, int(task_version), receipt_json]


def build_failure_request(
) -> tuple[LongTask, LongTaskFailureCommitRequest]:
    """
    构建失败 Store 测试使用的旧任务和重试提交请求。

    参数：
        无。

    返回值：
        tuple[LongTask, LongTaskFailureCommitRequest]：
            提交前任务和失败提交请求。
    """

    step = LongTaskStep(
        step_id="STEP-1",
        task_id="TASK-1",
        title="读取狗狗资料",
        assigned_agent="dog_knowledge_agent",
        status="running",
        attempt_count=1,
        max_attempts=3,
        version=3,
        claimed_by="WORKER-1",
        claim_id="CLAIM-1",
        lease_expires_at=NOW + timedelta(minutes=5),
    )
    task = LongTask(
        task_id="TASK-1",
        user_id="USER-1",
        thread_id="THREAD-1",
        goal=LongTaskGoal(
            original_request="查询边境牧羊犬资料",
            objective="整理边境牧羊犬资料",
        ),
        steps=[step],
        status="running",
        execution_mode="durable",
        active_step_ids=["STEP-1"],
        version=7,
    )
    batch_result = LongTaskBatchResult(
        batch_id="BATCH-1",
        task_id="TASK-1",
        actor_type="worker",
        actor_id="WORKER-1",
        trace_id="TRACE-1",
        step_results=[
            LongTaskBatchStepResult(
                step_id="STEP-1",
                status="failed",
                error_code="UPSTREAM_TIMEOUT",
                error_message="调用知识服务超时",
                claim_id="CLAIM-1",
                span_id="SPAN-1",
            )
        ],
    )
    preparation = prepare_long_task_failure_commit(
        task=task,
        batch_result=batch_result,
        failure_category="transient",
        failure_commit_id="FAILURE-COMMIT-1",
        now=NOW,
        base_delay_seconds=10,
        max_delay_seconds=60,
    )
    return task, build_long_task_failure_commit_request(preparation)


@pytest.mark.asyncio
async def test_failure_commit_should_atomically_persist_retry_and_receipt(
) -> None:
    """
    验证失败提交会保存新快照、事件、计划和可重复读取的回执。

    参数：
        无。

    返回值：
        None：首次提交和幂等重放均满足预期时测试通过。
    """

    redis = FakeFailureCommitRedis()
    store = RedisLongTaskStore(redis)
    previous_task, request = build_failure_request()
    # 预置已运行到版本7的历史快照；首次创建接口只接收版本1。
    await redis.set(
        store._task_key(previous_task.task_id), previous_task.model_dump_json()
    )

    first_receipt = await store.commit_failure(request)
    repeated_receipt = await store.commit_failure(request)

    assert request.retry_plan is not None
    assert first_receipt.retry_plan_id is not None
    saved_task = await store.load(previous_task.task_id)
    saved_plan = await store.load_retry_plan(
        first_receipt.retry_plan_id
    )
    events = await store.load_events(previous_task.task_id)
    before_due_ids = await store.load_due_retry_plan_ids(
        due_at=NOW,
    )
    due_ids = await store.load_due_retry_plan_ids(
        due_at=request.retry_plan.scheduled_at,
    )
    claimed_by_a = await store.claim_due_retry_plan(
        retry_plan_id=request.retry_plan.retry_plan_id,
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id="DISPATCH-CLAIM-A",
        now=request.retry_plan.scheduled_at,
        lease_seconds=30,
    )
    repeated_by_a = await store.claim_due_retry_plan(
        retry_plan_id=request.retry_plan.retry_plan_id,
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id="DISPATCH-CLAIM-A",
        now=request.retry_plan.scheduled_at,
        lease_seconds=30,
    )
    claimed_by_b = await store.claim_due_retry_plan(
        retry_plan_id=request.retry_plan.retry_plan_id,
        dispatcher_id="DISPATCHER-B",
        dispatch_claim_id="DISPATCH-CLAIM-B",
        now=request.retry_plan.scheduled_at,
        lease_seconds=30,
    )
    assert claimed_by_a is not None
    dispatch_request = build_long_task_retry_dispatch_request(
        task=request.task,
        claimed_plan=claimed_by_a,
        commit_id="RETRY-DISPATCH-COMMIT-1",
        now=request.retry_plan.scheduled_at + timedelta(seconds=1),
        correlation_id="CORRELATION-1",
    )
    dispatch_receipt = await store.commit_retry_dispatch(
        dispatch_request
    )
    repeated_dispatch_receipt = await store.commit_retry_dispatch(
        dispatch_request
    )
    dispatched_task = await store.load(previous_task.task_id)
    dispatched_plan = await store.load_retry_plan(
        request.retry_plan.retry_plan_id
    )
    remaining_due_ids = await store.load_due_retry_plan_ids(
        due_at=request.retry_plan.scheduled_at + timedelta(minutes=1),
    )
    all_events = await store.load_events(previous_task.task_id)
    assert saved_task is not None
    assert saved_task.version == 8
    assert saved_task.steps[0].status == "failed"
    assert saved_plan == request.retry_plan
    assert first_receipt == repeated_receipt
    assert first_receipt.retry_plan_id == request.retry_plan.retry_plan_id
    assert first_receipt.dead_letter_id is None
    assert before_due_ids == []
    assert due_ids == [request.retry_plan.retry_plan_id]
    assert claimed_by_a.status == "dispatching"
    assert claimed_by_a.dispatcher_id == "DISPATCHER-A"
    assert claimed_by_a.dispatch_claim_id == "DISPATCH-CLAIM-A"
    assert repeated_by_a == claimed_by_a
    assert claimed_by_b is None
    assert dispatched_task is not None
    assert dispatched_task.version == 9
    assert dispatched_task.status == "queued"
    assert dispatched_task.steps[0].status == "ready"
    assert dispatched_plan is not None
    assert dispatched_plan.status == "dispatched"
    assert dispatched_plan.dispatch_claim_id is None
    assert dispatched_plan.queue_message_id == "2000-0"
    assert dispatch_receipt == repeated_dispatch_receipt
    assert dispatch_receipt.queue_message_id == "2000-0"
    assert remaining_due_ids == []
    assert [event.event_type for event in all_events] == [
        "step_failed",
        "task_failed",
        "step_retry_scheduled",
        "step_became_ready",
        "task_queued",
        "retry_message_dispatched",
    ]
    assert len(redis.streams[store._stream_key]) == 1
    assert [event.event_type for event in events] == [
        "step_failed",
        "task_failed",
        "step_retry_scheduled",
    ]
    assert len(events) == 3


@pytest.mark.asyncio
async def test_cancel_claimed_retry_plan_should_clear_schedule_and_lock(
) -> None:
    """
    验证失效计划取消后会离开时间表并释放当前发布权。

    参数：
        无。

    返回值：
        None：计划状态、时间索引和发布锁都完成清理时测试通过。
    """

    redis = FakeFailureCommitRedis()
    store = RedisLongTaskStore(redis)
    previous_task, failure_request = build_failure_request()
    # 本用例从已有任务恢复，不通过首次创建接口伪造历史版本。
    await redis.set(
        store._task_key(previous_task.task_id), previous_task.model_dump_json()
    )
    await store.commit_failure(failure_request)
    assert failure_request.retry_plan is not None
    retry_plan = failure_request.retry_plan
    claimed_plan = await store.claim_due_retry_plan(
        retry_plan_id=retry_plan.retry_plan_id,
        dispatcher_id="DISPATCHER-A",
        dispatch_claim_id="DISPATCH-CLAIM-A",
        now=retry_plan.scheduled_at,
        lease_seconds=30,
    )
    assert claimed_plan is not None
    cancellation_request = build_long_task_retry_cancellation_request(
        claimed_plan=claimed_plan,
        cancellation_reason="当前任务版本已经不再符合重试计划",
        now=retry_plan.scheduled_at + timedelta(seconds=1),
    )

    cancelled_plan = await store.cancel_claimed_retry_plan(
        cancellation_request
    )
    due_plan_ids = await store.load_due_retry_plan_ids(
        due_at=retry_plan.scheduled_at + timedelta(minutes=1),
    )

    assert cancelled_plan.status == "cancelled"
    assert cancelled_plan.cancellation_reason == (
        "当前任务版本已经不再符合重试计划"
    )
    assert cancelled_plan.dispatcher_id is None
    assert cancelled_plan.dispatch_claim_id is None
    assert cancelled_plan.dispatch_lease_expires_at is None
    assert cancelled_plan.cancelled_at == (
        retry_plan.scheduled_at + timedelta(seconds=1)
    )
    assert due_plan_ids == []
    assert store._retry_dispatch_lock_key(retry_plan.retry_plan_id) not in (
        redis.values
    )
