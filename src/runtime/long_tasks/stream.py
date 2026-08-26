"""Redis Stream 长任务通知队列与最小消费者 Worker。"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from redis.exceptions import ResponseError

from src.runtime.long_tasks.contracts import LongTaskQueueMessage


DEFAULT_LONG_TASK_STREAM_KEY = "dog-agent:long-task:v1:stream"
DEFAULT_LONG_TASK_CONSUMER_GROUP = "long-task-workers"


class CorruptLongTaskStreamMessageError(ValueError):
    """表示 Stream Entry 缺少 payload 或无法恢复成队列消息契约。"""


@dataclass(frozen=True, slots=True)
class LongTaskStreamEntry:
    """
    保存 Redis Stream 消息编号和类型化长任务通知。

    参数含义：
        stream_id:
            Redis 为当前 Entry 分配的唯一消息编号，用于 ACK。
        message:
            已通过 LongTaskQueueMessage 契约校验的轻量任务通知。

    返回值含义：
        LongTaskStreamEntry:
            同时保留运输层消息编号和业务消息内容的不可变对象。
    """

    stream_id: str
    message: LongTaskQueueMessage


class RedisLongTaskStream:
    """
    使用 Redis Stream 发布、消费和确认长任务轻量通知。

    参数含义：
        redis_client:
            RedisProvider 已启动并完成健康检查的异步客户端。
        stream_key:
            当前长任务通知使用的 Redis Stream Key。
        consumer_group:
            多个 Worker 共享的 Consumer Group 名称。

    返回值含义：
        RedisLongTaskStream:
            不执行具体 Agent、只负责可靠消息运输的 Stream 网关。
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        stream_key: str = DEFAULT_LONG_TASK_STREAM_KEY,
        consumer_group: str = DEFAULT_LONG_TASK_CONSUMER_GROUP,
    ) -> None:
        self._redis = redis_client
        self._stream_key = _require_non_empty(
            stream_key,
            field_name="stream_key",
        )
        self._consumer_group = _require_non_empty(
            consumer_group,
            field_name="consumer_group",
        )
        self._consumer_group_ready = False

    async def ensure_consumer_group(self) -> None:
        """
        幂等创建消费者组和不存在的 Stream。

        返回值含义：
            None。消费者组已经存在时保持成功，其他 Redis 错误继续抛出。
        """

        if self._consumer_group_ready:
            return

        try:
            await self._redis.xgroup_create(
                name=self._stream_key,
                groupname=self._consumer_group,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc).upper():
                raise
        self._consumer_group_ready = True

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        将轻量任务通知追加到 Redis Stream。

        参数含义：
            message:
                包含 task_id、task_version 和 Ready Step 提示的队列消息。

        返回值含义：
            str:
                Redis 为新 Entry 生成的 Stream 消息编号。
        """

        stream_id = await self._redis.xadd(
            self._stream_key,
            {"payload": message.model_dump_json()},
        )
        return _decode_text(stream_id)

    async def read_new(
        self,
        *,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 1000,
    ) -> list[LongTaskStreamEntry]:
        """
        使用消费者组读取尚未分配给其他消费者的新消息。

        参数含义：
            consumer_name:
                当前 Worker 实例在消费者组内的稳定名称。
            count:
                本次最多读取的消息数量。
            block_ms:
                没有新消息时允许阻塞等待的毫秒数；0 表示持续等待。

        返回值含义：
            list[LongTaskStreamEntry]:
                已解析的 Stream Entry；超时没有消息时返回空列表。
        """

        normalized_consumer = _require_non_empty(
            consumer_name,
            field_name="consumer_name",
        )
        if count < 1:
            raise ValueError("count 必须大于等于 1")
        if block_ms < 0:
            raise ValueError("block_ms 不能小于 0")

        raw_streams = await self._redis.xreadgroup(
            groupname=self._consumer_group,
            consumername=normalized_consumer,
            streams={self._stream_key: ">"},
            count=count,
            block=block_ms,
        )
        entries: list[LongTaskStreamEntry] = []
        for _, raw_entries in raw_streams or []:
            for raw_stream_id, raw_fields in raw_entries:
                entries.append(
                    _parse_stream_entry(
                        stream_id=raw_stream_id,
                        fields=raw_fields,
                    )
                )
        return entries

    async def claim_stale(
        self,
        *,
        consumer_name: str,
        min_idle_time_ms: int,
        count: int = 1,
    ) -> list[LongTaskStreamEntry]:
        """
        将消费者组内超时未确认的消息转交给当前 Worker。

        参数含义：
            consumer_name:
                准备接管超时消息的当前 Worker 稳定名称。
            min_idle_time_ms:
                消息距离上次投递至少空闲多少毫秒后才允许接管。
            count:
                本次最多认领的消息数量。

        返回值含义：
            list[LongTaskStreamEntry]:
                已转交给当前 Worker 的类型化消息；没有符合条件的消息时
                返回空列表。Redis 的扫描游标由本方法内部继续推进。
        """

        normalized_consumer = _require_non_empty(
            consumer_name,
            field_name="consumer_name",
        )
        if min_idle_time_ms < 1:
            raise ValueError("min_idle_time_ms 必须大于等于 1")
        if count < 1:
            raise ValueError("count 必须大于等于 1")

        start_id = "0-0"
        while True:
            raw_claim = await self._redis.xautoclaim(
                name=self._stream_key,
                groupname=self._consumer_group,
                consumername=normalized_consumer,
                min_idle_time=min_idle_time_ms,
                start_id=start_id,
                count=count,
            )
            next_start_id, raw_entries = _parse_xautoclaim_page(raw_claim)
            entries = [
                _parse_stream_entry(
                    stream_id=raw_stream_id,
                    fields=raw_fields,
                )
                for raw_stream_id, raw_fields in raw_entries
            ]
            if entries:
                return entries
            if next_start_id == "0-0" or next_start_id == start_id:
                return []
            start_id = next_start_id

    async def acknowledge(self, stream_id: str) -> int:
        """
        确认一条消息已经成功完成业务处理。

        参数含义：
            stream_id:
                准备从 Pending Entries List 中移除的 Redis 消息编号。

        返回值含义：
            int:
                Redis 实际确认的消息数量，正常为 0 或 1。
        """

        normalized_stream_id = _require_non_empty(
            stream_id,
            field_name="stream_id",
        )
        return int(
            await self._redis.xack(
                self._stream_key,
                self._consumer_group,
                normalized_stream_id,
            )
        )


LongTaskMessageHandler = Callable[
    [LongTaskQueueMessage],
    Awaitable[None],
]


class LongTaskStreamWorker:
    """
    从消费者组读取一条通知，业务处理成功后再执行 ACK。

    参数含义：
        stream:
            提供消费者组读取与 ACK 的 Redis Stream 网关。
        handler:
            根据消息加载权威 LongTask 并执行下一批的异步业务处理器。

    返回值含义：
        LongTaskStreamWorker:
            保证 handler 抛出异常时不确认消息的最小 Worker。
    """

    def __init__(
        self,
        *,
        stream: RedisLongTaskStream,
        handler: LongTaskMessageHandler,
    ) -> None:
        self._stream = stream
        self._handler = handler

    async def process_once(
        self,
        *,
        consumer_name: str,
        block_ms: int = 1000,
    ) -> bool:
        """
        最多读取并处理一条新消息，成功后确认消费。

        参数含义：
            consumer_name:
                当前在消费者组内的稳定名称。
            block_ms:
                没有新消息时的最长等待毫秒数。

        返回值含义：
            bool:
                成功处理一条消息时为 True，超时没有消息时为 False。
                handler 失败时异常继续向上抛出且不会 ACK。
        """

        await self._stream.ensure_consumer_group()
        entries = await self._stream.read_new(
            consumer_name=consumer_name,
            count=1,
            block_ms=block_ms,
        )
        if not entries:
            return False

        entry = entries[0]
        await self._handler(entry.message)
        await self._stream.acknowledge(entry.stream_id)
        return True

    async def recover_stale_once(
        self,
        *,
        consumer_name: str,
        min_idle_time_ms: int,
    ) -> bool:
        """
        最多接管并处理一条超时未确认消息，成功后执行 ACK。

        参数含义：
            consumer_name:
                当前准备接管 Pending 消息的 Worker 稳定名称。
            min_idle_time_ms:
                消息进入 Pending 后至少空闲多少毫秒才允许接管。

        返回值含义：
            bool:
                成功恢复一条消息时为 True，没有超时消息时为 False。
                handler 失败时异常继续向上抛出且不会 ACK。
        """

        await self._stream.ensure_consumer_group()
        entries = await self._stream.claim_stale(
            consumer_name=consumer_name,
            min_idle_time_ms=min_idle_time_ms,
            count=1,
        )
        if not entries:
            return False

        entry = entries[0]
        await self._handler(entry.message)
        await self._stream.acknowledge(entry.stream_id)
        return True

    async def run_forever(
        self,
        *,
        stop_event: asyncio.Event,
        consumer_name: str | None = None,
        block_ms: int = 1000,
        min_idle_time_ms: int = 30_000,
        recovery_interval_ms: int = 30_000,
    ) -> None:
        """
        持续处理新消息，并按固定间隔恢复超时 Pending 消息。

        参数含义：
            stop_event:
                后台服务关闭时由外层设置的异步停止信号。
            consumer_name:
                当前 Worker 的稳定名称；省略时按机器名和进程号生成一次。
            block_ms:
                每次等待新消息的最长毫秒数，也决定空闲时的停止响应上限。
            min_idle_time_ms:
                Pending 消息至少空闲多少毫秒后才允许被当前 Worker 接管。
            recovery_interval_ms:
                两次超时消息恢复检查之间至少间隔多少毫秒。

        返回值含义：
            None:
                收到 stop_event 后正常退出；消息处理异常继续向上抛出，
                交给进程管理器记录并按部署策略重启。
        """

        if block_ms < 1:
            raise ValueError("run_forever 的 block_ms 必须大于等于 1")
        if min_idle_time_ms < 1:
            raise ValueError("min_idle_time_ms 必须大于等于 1")
        if recovery_interval_ms < 1:
            raise ValueError("recovery_interval_ms 必须大于等于 1")

        normalized_consumer = _require_non_empty(
            consumer_name or _build_default_consumer_name(),
            field_name="consumer_name",
        )
        next_recovery_at = 0.0

        while not stop_event.is_set():
            current_time = time.monotonic()
            if current_time >= next_recovery_at:
                recovered = await self.recover_stale_once(
                    consumer_name=normalized_consumer,
                    min_idle_time_ms=min_idle_time_ms,
                )
                next_recovery_at = (
                    time.monotonic() + recovery_interval_ms / 1000
                )
                if recovered:
                    continue

            await self.process_once(
                consumer_name=normalized_consumer,
                block_ms=block_ms,
            )


def _parse_stream_entry(
    *,
    stream_id: Any,
    fields: Any,
) -> LongTaskStreamEntry:
    """
    将 Redis 原始字段恢复成带消息编号的类型化通知。

    参数含义：
        stream_id:
            Redis 返回的字符串或 bytes 消息编号。
        fields:
            Redis Entry 字段映射，必须包含 payload。

    返回值含义：
        LongTaskStreamEntry:
            解析成功的不可变 Entry；格式错误时抛出明确异常。
    """

    if not isinstance(fields, dict):
        raise CorruptLongTaskStreamMessageError(
            "Stream Entry 字段必须是字典"
        )
    payload = fields.get("payload")
    if payload is None:
        payload = fields.get(b"payload")
    if payload is None:
        raise CorruptLongTaskStreamMessageError(
            "Stream Entry 缺少 payload"
        )
    try:
        message = LongTaskQueueMessage.model_validate_json(
            _decode_text(payload)
        )
    except (ValueError, TypeError) as exc:
        raise CorruptLongTaskStreamMessageError(
            "Stream payload 无法恢复成长任务队列消息"
        ) from exc
    return LongTaskStreamEntry(
        stream_id=_decode_text(stream_id),
        message=message,
    )


def _parse_xautoclaim_page(raw_claim: Any) -> tuple[str, list[Any]]:
    """
    规范化 redis-py 返回的 XAUTOCLAIM 单页结果。

    参数含义：
        raw_claim:
            Redis 6.2 常见的两段式结果，或 Redis 7 附带已删除消息编号的
            三段式结果。

    返回值含义：
        tuple[str, list[Any]]:
            下一扫描位置和当前页成功认领的原始 Stream Entry 列表。
    """

    if not isinstance(raw_claim, (list, tuple)) or len(raw_claim) < 2:
        raise CorruptLongTaskStreamMessageError(
            "XAUTOCLAIM 返回格式无效"
        )
    next_start_id = _decode_text(raw_claim[0])
    raw_entries = raw_claim[1]
    if not isinstance(raw_entries, (list, tuple)):
        raise CorruptLongTaskStreamMessageError(
            "XAUTOCLAIM 消息列表格式无效"
        )
    return next_start_id, list(raw_entries)


def _decode_text(value: Any) -> str:
    """
    将 Redis 返回的 bytes 或普通值规范化成字符串。

    参数含义：
        value:
            Redis 客户端返回的原始值。

    返回值含义：
        str:
            UTF-8 解码后的文本或普通字符串表示。
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_default_consumer_name() -> str:
    """
    为单进程单 Worker 模式生成当前消费者的稳定名称。

    返回值含义：
        str:
            由固定前缀、机器名和当前进程号组成的 Consumer Name。
    """

    return f"long-task-worker-{socket.gethostname()}-{os.getpid()}"


def _require_non_empty(value: str, *, field_name: str) -> str:
    """
    规范并校验 Stream Key、消费者组或消息编号等必填文本。

    参数含义：
        value:
            待规范化的文本。
        field_name:
            错误信息中使用的字段名称。

    返回值含义：
        str:
            去除首尾空白后的非空文本。
    """

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized
