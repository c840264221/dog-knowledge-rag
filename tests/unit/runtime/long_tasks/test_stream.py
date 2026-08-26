"""Redis Stream 长任务队列与 Worker 单元测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from redis.exceptions import ResponseError

from src.runtime.long_tasks import (
    CorruptLongTaskStreamMessageError,
    LongTaskQueueMessage,
    LongTaskStreamWorker,
    RedisLongTaskStream,
)


class FakeRedisStreamClient:
    """模拟 Runtime Stream 当前使用的异步 Redis 命令。"""

    def __init__(self) -> None:
        self.group_exists = False
        self.group_create_calls = 0
        self.xadd_calls: list[tuple[str, dict[str, str]]] = []
        self.read_responses: list[Any] = []
        self.xreadgroup_calls: list[dict[str, Any]] = []
        self.claim_responses: list[Any] = []
        self.xautoclaim_calls: list[dict[str, Any]] = []
        self.acknowledged_ids: list[str] = []

    async def xgroup_create(self, **kwargs: Any) -> bool:
        """创建一次消费者组，重复创建时模拟 BUSYGROUP。"""

        self.group_create_calls += 1
        if self.group_exists:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.group_exists = True
        return True

    async def xadd(
        self,
        name: str,
        fields: dict[str, str],
    ) -> str:
        """记录 XADD 参数并返回固定递增消息编号。"""

        self.xadd_calls.append((name, fields))
        return f"{len(self.xadd_calls)}-0"

    async def xreadgroup(self, **kwargs: Any) -> Any:
        """返回预先放入的 XREADGROUP 测试响应。"""

        self.xreadgroup_calls.append(kwargs)
        if not self.read_responses:
            return []
        return self.read_responses.pop(0)

    async def xack(
        self,
        name: str,
        group: str,
        stream_id: str,
    ) -> int:
        """记录成功确认的 Stream 消息编号。"""

        self.acknowledged_ids.append(stream_id)
        return 1

    async def xautoclaim(self, **kwargs: Any) -> Any:
        """记录 XAUTOCLAIM 参数并返回预先设置的认领结果。"""

        self.xautoclaim_calls.append(kwargs)
        if not self.claim_responses:
            return ["0-0", [], []]
        return self.claim_responses.pop(0)


def build_queue_message() -> LongTaskQueueMessage:
    """
    构建 Stream 测试使用的轻量长任务通知。

    返回值含义：
        LongTaskQueueMessage:
            指向 task_001 版本 3 和下一 Ready Step 的合法消息。
    """

    return LongTaskQueueMessage(
        task_id="task_001",
        task_version=3,
        reason="continued",
        ready_step_ids=["step_2"],
        correlation_id="trace_001",
    )


@pytest.mark.asyncio
async def test_stream_should_publish_and_restore_typed_message() -> None:
    """验证 XADD 只写轻量 JSON，并能从 bytes 响应恢复类型化消息。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    message = build_queue_message()

    stream_id = await stream.publish(message)
    payload = redis.xadd_calls[0][1]["payload"]
    redis.read_responses.append(
        [
            (
                b"dog-agent:long-task:v1:stream",
                [(b"1-0", {b"payload": payload.encode("utf-8")})],
            )
        ]
    )
    entries = await stream.read_new(
        consumer_name="worker-1",
        block_ms=10,
    )

    assert stream_id == "1-0"
    assert entries[0].stream_id == "1-0"
    assert entries[0].message == message
    assert "steps" not in payload


@pytest.mark.asyncio
async def test_consumer_group_creation_should_be_idempotent() -> None:
    """验证重复启动 Worker 时 BUSYGROUP 不会导致启动失败。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)

    await stream.ensure_consumer_group()
    await stream.ensure_consumer_group()

    assert redis.group_create_calls == 1


@pytest.mark.asyncio
async def test_worker_should_ack_only_after_handler_succeeds() -> None:
    """验证业务处理成功后才把消息从 Pending 列表确认移除。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    message = build_queue_message()
    redis.read_responses.append(
        [
            (
                "dog-agent:long-task:v1:stream",
                [("10-0", {"payload": message.model_dump_json()})],
            )
        ]
    )
    handled_messages: list[LongTaskQueueMessage] = []

    async def handler(received: LongTaskQueueMessage) -> None:
        """记录 Worker 已交给业务层处理的测试消息。"""

        handled_messages.append(received)

    processed = await LongTaskStreamWorker(
        stream=stream,
        handler=handler,
    ).process_once(
        consumer_name="worker-1",
        block_ms=10,
    )

    assert processed is True
    assert handled_messages == [message]
    assert redis.acknowledged_ids == ["10-0"]


@pytest.mark.asyncio
async def test_worker_should_not_ack_when_handler_fails() -> None:
    """验证业务异常继续抛出，消息保持 Pending 供后续恢复。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    redis.read_responses.append(
        [
            (
                "dog-agent:long-task:v1:stream",
                [
                    (
                        "11-0",
                        {"payload": build_queue_message().model_dump_json()},
                    )
                ],
            )
        ]
    )

    async def failing_handler(_: LongTaskQueueMessage) -> None:
        """模拟加载任务或执行下一批时发生业务异常。"""

        raise RuntimeError("模拟 Worker 失败")

    with pytest.raises(RuntimeError, match="模拟 Worker 失败"):
        await LongTaskStreamWorker(
            stream=stream,
            handler=failing_handler,
        ).process_once(
            consumer_name="worker-1",
            block_ms=10,
        )

    assert redis.acknowledged_ids == []


@pytest.mark.asyncio
async def test_stream_should_reject_corrupt_payload() -> None:
    """验证缺少合法契约的消息不会交给业务 Handler。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    redis.read_responses.append(
        [
            (
                "dog-agent:long-task:v1:stream",
                [("12-0", {"payload": "{not-json}"})],
            )
        ]
    )

    with pytest.raises(
        CorruptLongTaskStreamMessageError,
        match="无法恢复",
    ):
        await stream.read_new(
            consumer_name="worker-1",
            block_ms=10,
        )


@pytest.mark.asyncio
async def test_stream_should_scan_until_stale_message_is_claimed() -> None:
    """验证当前页没有超时消息时会继续使用返回游标扫描。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    message = build_queue_message()
    redis.claim_responses.extend(
        [
            [b"20-0", [], []],
            [
                b"0-0",
                [(b"21-0", {b"payload": message.model_dump_json().encode()})],
                [],
            ],
        ]
    )

    entries = await stream.claim_stale(
        consumer_name="worker-2",
        min_idle_time_ms=30_000,
    )

    assert entries[0].stream_id == "21-0"
    assert entries[0].message == message
    assert [
        call["start_id"] for call in redis.xautoclaim_calls
    ] == ["0-0", "20-0"]
    assert redis.xautoclaim_calls[0]["consumername"] == "worker-2"


@pytest.mark.asyncio
async def test_worker_should_recover_and_ack_stale_message() -> None:
    """验证接管超时消息并处理成功后才执行 ACK。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    message = build_queue_message()
    redis.claim_responses.append(
        ["0-0", [("22-0", {"payload": message.model_dump_json()})], []]
    )
    handled_messages: list[LongTaskQueueMessage] = []

    async def handler(received: LongTaskQueueMessage) -> None:
        """记录恢复链路交给业务层的消息。"""

        handled_messages.append(received)

    processed = await LongTaskStreamWorker(
        stream=stream,
        handler=handler,
    ).recover_stale_once(
        consumer_name="worker-2",
        min_idle_time_ms=30_000,
    )

    assert processed is True
    assert handled_messages == [message]
    assert redis.acknowledged_ids == ["22-0"]


@pytest.mark.asyncio
async def test_worker_should_return_false_when_no_stale_message_exists() -> None:
    """验证 PEL 中没有超时消息时恢复轮次正常结束。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)

    async def unused_handler(_: LongTaskQueueMessage) -> None:
        """提供合法异步 Handler；没有认领到消息时不应被调用。"""

    processed = await LongTaskStreamWorker(
        stream=stream,
        handler=unused_handler,
    ).recover_stale_once(
        consumer_name="worker-2",
        min_idle_time_ms=30_000,
    )

    assert processed is False
    assert redis.acknowledged_ids == []


@pytest.mark.asyncio
async def test_worker_should_not_ack_when_stale_handler_fails() -> None:
    """验证接管后的业务处理失败时消息仍留在 PEL。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    redis.claim_responses.append(
        [
            "0-0",
            [("23-0", {"payload": build_queue_message().model_dump_json()})],
            [],
        ]
    )

    async def failing_handler(_: LongTaskQueueMessage) -> None:
        """模拟接管消息后的业务处理失败。"""

        raise RuntimeError("模拟恢复处理失败")

    with pytest.raises(RuntimeError, match="模拟恢复处理失败"):
        await LongTaskStreamWorker(
            stream=stream,
            handler=failing_handler,
        ).recover_stale_once(
            consumer_name="worker-2",
            min_idle_time_ms=30_000,
        )

    assert redis.acknowledged_ids == []


@pytest.mark.asyncio
async def test_worker_runner_should_automatically_process_new_message() -> None:
    """验证后台循环自动携带固定 Worker 名称领取并处理新消息。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    stop_event = asyncio.Event()
    message = build_queue_message()
    redis.read_responses.append(
        [
            (
                "dog-agent:long-task:v1:stream",
                [("30-0", {"payload": message.model_dump_json()})],
            )
        ]
    )

    async def handler(_: LongTaskQueueMessage) -> None:
        """处理一条消息后通知后台循环退出。"""

        stop_event.set()

    await LongTaskStreamWorker(
        stream=stream,
        handler=handler,
    ).run_forever(
        stop_event=stop_event,
        consumer_name="worker-auto-1",
        block_ms=10,
        recovery_interval_ms=30_000,
    )

    assert redis.xautoclaim_calls[0]["consumername"] == "worker-auto-1"
    assert redis.xreadgroup_calls[0]["consumername"] == "worker-auto-1"
    assert redis.acknowledged_ids == ["30-0"]
    assert redis.group_create_calls == 1


@pytest.mark.asyncio
async def test_worker_runner_should_recover_stale_message_before_new() -> None:
    """验证后台循环启动后会优先执行一次超时消息恢复检查。"""

    redis = FakeRedisStreamClient()
    stream = RedisLongTaskStream(redis)
    stop_event = asyncio.Event()
    message = build_queue_message()
    redis.claim_responses.append(
        ["0-0", [("31-0", {"payload": message.model_dump_json()})], []]
    )

    async def handler(_: LongTaskQueueMessage) -> None:
        """恢复一条消息后通知后台循环退出。"""

        stop_event.set()

    await LongTaskStreamWorker(
        stream=stream,
        handler=handler,
    ).run_forever(
        stop_event=stop_event,
        consumer_name="worker-auto-2",
        block_ms=10,
        recovery_interval_ms=30_000,
    )

    assert redis.acknowledged_ids == ["31-0"]
    assert redis.xreadgroup_calls == []


@pytest.mark.asyncio
async def test_worker_runner_should_not_read_after_stop_requested() -> None:
    """验证启动前已经收到停止信号时不会访问 Redis。"""

    redis = FakeRedisStreamClient()
    stop_event = asyncio.Event()
    stop_event.set()

    async def unused_handler(_: LongTaskQueueMessage) -> None:
        """停止状态下不应被调用。"""

    await LongTaskStreamWorker(
        stream=RedisLongTaskStream(redis),
        handler=unused_handler,
    ).run_forever(
        stop_event=stop_event,
        consumer_name="worker-auto-3",
        block_ms=10,
    )

    assert redis.xautoclaim_calls == []
    assert redis.xreadgroup_calls == []
    assert redis.acknowledged_ids == []
