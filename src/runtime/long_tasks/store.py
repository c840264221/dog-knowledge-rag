"""长任务快照存储契约与 Redis MVP 实现。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import ValidationError

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskCommitReceipt,
    LongTaskCommitRequest,
    LongTaskEvent,
    utc_now,
)


DEFAULT_LONG_TASK_KEY_PREFIX = "dog-agent:long-task:v1"
DEFAULT_LONG_TASK_STREAM_KEY = "dog-agent:long-task:v1:stream"

_COMPARE_AND_SET_SCRIPT = """
local current_json = redis.call("GET", KEYS[1])
if not current_json then
    return {-1, -1}
end

local decode_ok, current_task = pcall(cjson.decode, current_json)
if not decode_ok or current_task["version"] == nil then
    return {-2, -1}
end

local current_version = tonumber(current_task["version"])
local expected_version = tonumber(ARGV[1])
if current_version ~= expected_version then
    return {0, current_version}
end

redis.call("SET", KEYS[1], ARGV[2])
return {1, tonumber(ARGV[3])}
"""

_RELIABLE_COMMIT_SCRIPT = """
local receipt_json = redis.call("GET", KEYS[4])
if receipt_json then
    local receipt_ok, receipt = pcall(cjson.decode, receipt_json)
    if not receipt_ok
        or receipt["request_fingerprint"] == nil
        or receipt["submission_fingerprint"] == nil then
        return {-6, -1, receipt_json}
    end
    if receipt["request_fingerprint"] ~= ARGV[6] then
        return {-3, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    if receipt["submission_fingerprint"] ~= ARGV[7] then
        return {-5, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    return {2, tonumber(receipt["task_version"] or -1), receipt_json}
end

local task_ok, next_task = pcall(cjson.decode, ARGV[3])
local events_ok, event_blueprints = pcall(cjson.decode, ARGV[8])
local receipt_template_ok, receipt = pcall(cjson.decode, ARGV[9])
if not task_ok or not events_ok or not receipt_template_ok then
    return {-7, -1, ""}
end
if #event_blueprints < 1 then
    return {-7, -1, ""}
end
if tostring(next_task["task_id"] or "") ~= ARGV[5]
    or tonumber(next_task["version"] or -1) ~= tonumber(ARGV[4]) then
    return {-7, -1, ""}
end

local function type_name(key)
    local result = redis.call("TYPE", key)
    if type(result) == "table" then
        return result["ok"]
    end
    return result
end

local task_type = type_name(KEYS[1])
local sequence_type = type_name(KEYS[2])
local events_type = type_name(KEYS[3])
local receipt_type = type_name(KEYS[4])
local stream_type = type_name(KEYS[5])
if (task_type ~= "none" and task_type ~= "string")
    or (sequence_type ~= "none" and sequence_type ~= "string")
    or (events_type ~= "none" and events_type ~= "list")
    or (receipt_type ~= "none" and receipt_type ~= "string")
    or (stream_type ~= "none" and stream_type ~= "stream") then
    return {-8, -1, ""}
end

local current_json = redis.call("GET", KEYS[1])
if ARGV[1] == "create" then
    if current_json then
        return {-4, -1, ""}
    end
else
    if not current_json then
        return {-1, -1, ""}
    end
    local current_ok, current_task = pcall(cjson.decode, current_json)
    if not current_ok or current_task["version"] == nil then
        return {-2, -1, ""}
    end
    local current_version = tonumber(current_task["version"])
    if current_version ~= tonumber(ARGV[2]) then
        return {0, current_version, ""}
    end
end

local current_sequence = tonumber(redis.call("GET", KEYS[2]) or "0")
if current_sequence == nil then
    return {-2, -1, ""}
end
local sequence_start = current_sequence + 1
local event_ids = {}
local event_jsons = {}
for index, event in ipairs(event_blueprints) do
    event["sequence"] = current_sequence + index
    event_ids[index] = event["event_id"]
    event_jsons[index] = cjson.encode(event)
end
local sequence_end = current_sequence + #event_blueprints

local queue_message_id = nil
redis.call("SET", KEYS[1], ARGV[3])
redis.call("SET", KEYS[2], tostring(sequence_end))
for _, event_json in ipairs(event_jsons) do
    redis.call("RPUSH", KEYS[3], event_json)
end
if ARGV[10] ~= "" then
    queue_message_id = redis.call(
        "XADD", KEYS[5], "*", "payload", ARGV[10]
    )
end

receipt["event_ids"] = event_ids
receipt["event_sequence_start"] = sequence_start
receipt["event_sequence_end"] = sequence_end
if queue_message_id then
    receipt["queue_message_id"] = queue_message_id
else
    receipt["queue_message_id"] = cjson.null
end
local committed_receipt_json = cjson.encode(receipt)
redis.call("SET", KEYS[4], committed_receipt_json)
return {1, tonumber(ARGV[4]), committed_receipt_json}
"""


class LongTaskStoreError(RuntimeError):
    """表示长任务快照存储操作失败。"""


class LongTaskNotFoundError(LongTaskStoreError):
    """表示需要更新的长任务不存在。"""


class LongTaskAlreadyExistsError(LongTaskStoreError):
    """表示相同 task_id 的长任务已经创建。"""


class LongTaskVersionConflictError(LongTaskStoreError):
    """表示保存时读取到的任务版本已经落后。"""


class CorruptLongTaskSnapshotError(LongTaskStoreError):
    """表示 Redis 中的长任务 JSON 已损坏或不符合契约。"""


class LongTaskCommitFingerprintConflictError(LongTaskStoreError):
    """表示同一个 commit_id 被用于两次不同的业务动作。"""


class LongTaskCommitContentConflictError(LongTaskStoreError):
    """表示同一业务动作生成了两份不同的最终提交内容。"""


class CorruptLongTaskCommitReceiptError(LongTaskStoreError):
    """表示 Redis 中的业务提交回执已损坏或不符合契约。"""


class LongTaskStore(Protocol):
    """定义长任务最新快照存储必须提供的最小能力。"""

    async def create(self, task: LongTask) -> LongTask:
        """
        仅在 task_id 尚不存在时创建任务。

        参数含义：
            task:
                首次持久化的版本 1 任务快照。

        返回值含义：
            LongTask:
                创建成功后返回原任务快照。
        """

        ...

    async def load(self, task_id: str) -> LongTask | None:
        """
        根据任务编号读取并校验最新快照。

        参数含义：
            task_id:
                需要查询的长任务唯一编号。

        返回值含义：
            LongTask | None:
                找到时返回类型化快照，不存在时返回 None。
        """

        ...

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        使用乐观锁覆盖任务最新快照。

        参数含义：
            task:
                已完成状态迁移、版本应为 expected_version + 1 的新快照。
            expected_version:
                调用方修改任务前读取到的旧版本号。

        返回值含义：
            LongTask:
                原子保存成功后的新任务快照。
        """

        ...


class LongTaskReliableCommitStore(LongTaskStore, Protocol):
    """定义快照、事件、通知和回执的可靠业务提交能力。"""

    async def commit(
        self,
        request: LongTaskCommitRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子接受一次创建或更新业务提交。

        参数含义：
            request:
                已通过领域校验的完整提交请求。

        返回值含义：
            LongTaskCommitReceipt:
                首次提交或相同内容幂等重试对应的持久化回执。
        """

        ...


class RedisLongTaskStore:
    """
    使用 Redis String 保存最新 LongTask JSON 快照。

    功能：
        使用 SET NX 防止重复创建，使用 Lua Compare-And-Set 保证版本检查和
        SET 在 Redis 内原子完成；业务提交使用独立 Lua，把快照、事件、
        可选 Stream 消息和回执放进同一 Redis 原子执行边界。

    参数含义：
        redis_client:
            已由 RedisProvider 启动并完成健康检查的异步 Redis 客户端。
        key_prefix:
            Redis Key 命名空间；默认包含项目名、领域名和结构版本。
        stream_key:
            可靠提交可选通知写入的 Redis Stream Key，必须与 Worker 一致。

    返回值含义：
        RedisLongTaskStore:
            实现 LongTaskStore 契约的 Redis 快照存储。
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        key_prefix: str = DEFAULT_LONG_TASK_KEY_PREFIX,
        stream_key: str = DEFAULT_LONG_TASK_STREAM_KEY,
    ) -> None:
        normalized_prefix = str(key_prefix or "").strip().rstrip(":")
        if not normalized_prefix:
            raise ValueError("长任务 Redis Key 前缀不能为空")
        self._redis = redis_client
        self._key_prefix = normalized_prefix
        normalized_stream_key = str(stream_key or "").strip()
        if not normalized_stream_key:
            raise ValueError("长任务 Redis Stream Key 不能为空")
        self._stream_key = normalized_stream_key

    async def create(self, task: LongTask) -> LongTask:
        """
        使用 SET NX 原子创建版本 1 的长任务快照。

        参数含义：
            task:
                尚未写入 Store 的初始长任务。

        返回值含义：
            LongTask:
                创建成功后返回原任务；Key 已存在时抛出重复创建异常。
        """

        if task.version != 1:
            raise ValueError("首次创建的长任务 version 必须为 1")
        created = await self._redis.set(
            self._task_key(task.task_id),
            task.model_dump_json(),
            nx=True,
        )
        if not created:
            raise LongTaskAlreadyExistsError(
                f"长任务已经存在: {task.task_id}"
            )
        return task

    async def load(self, task_id: str) -> LongTask | None:
        """
        从 Redis 读取 JSON，并重新执行 LongTask 契约校验。

        参数含义：
            task_id:
                需要加载的长任务唯一编号。

        返回值含义：
            LongTask | None:
                找到且校验成功时返回任务，不存在时返回 None。
        """

        normalized_task_id = str(task_id or "").strip()
        if not normalized_task_id:
            raise ValueError("task_id 不能为空")
        raw_task = await self._redis.get(self._task_key(normalized_task_id))
        if raw_task is None:
            return None
        try:
            task = LongTask.model_validate_json(raw_task)
        except (ValidationError, ValueError, TypeError) as exc:
            raise CorruptLongTaskSnapshotError(
                f"长任务快照无法恢复: {normalized_task_id}"
            ) from exc
        if task.task_id != normalized_task_id:
            raise CorruptLongTaskSnapshotError(
                "Redis Key 中的 task_id 与快照内容不一致"
            )
        return task

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        比较 Redis 当前版本，并原子覆盖为调用方的新快照。

        参数含义：
            task:
                状态机产生的新任务快照。
            expected_version:
                调用方开始本次修改时读取到的版本号。

        返回值含义：
            LongTask:
                Redis 成功保存后的新任务快照。
        """

        if expected_version < 1:
            raise ValueError("expected_version 必须大于等于 1")
        if task.version != expected_version + 1:
            raise ValueError(
                "待保存任务 version 必须等于 expected_version + 1"
            )

        result = await self._redis.eval(
            _COMPARE_AND_SET_SCRIPT,
            1,
            self._task_key(task.task_id),
            expected_version,
            task.model_dump_json(),
            task.version,
        )
        result_code, current_version = self._parse_cas_result(result)
        if result_code == -1:
            raise LongTaskNotFoundError(
                f"长任务不存在: {task.task_id}"
            )
        if result_code == -2:
            raise CorruptLongTaskSnapshotError(
                f"长任务快照缺少合法版本: {task.task_id}"
            )
        if result_code == 0:
            raise LongTaskVersionConflictError(
                "长任务版本冲突: "
                f"expected={expected_version}, actual={current_version}"
            )
        if result_code != 1:
            raise LongTaskStoreError(
                f"Redis 返回未知保存结果: {result!r}"
            )
        return task

    async def commit(
        self,
        request: LongTaskCommitRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子保存任务快照、业务事件、可选队列消息和提交回执。

        参数含义：
            request:
                包含稳定 commit_id、提交指纹和全部提交草稿的请求。

        返回值含义：
            LongTaskCommitReceipt:
                首次成功或相同内容幂等重试对应的已提交回执。
        """

        committed_at = utc_now()
        event_blueprints = self._build_event_blueprints(
            request=request,
            committed_at=committed_at,
        )
        receipt_template = {
            "commit_id": request.commit_id,
            "request_fingerprint": request.request_fingerprint,
            "submission_fingerprint": request.submission_fingerprint,
            "task_id": request.task.task_id,
            "task_version": request.task.version,
            "event_ids": [item["event_id"] for item in event_blueprints],
            "event_sequence_start": 1,
            "event_sequence_end": len(event_blueprints),
            "queue_message_id": None,
            "committed_at": committed_at.isoformat(),
        }
        result = await self._redis.eval(
            _RELIABLE_COMMIT_SCRIPT,
            5,
            self._task_key(request.task.task_id),
            self._event_sequence_key(request.task.task_id),
            self._events_key(request.task.task_id),
            self._receipt_key(
                request.task.task_id,
                request.commit_id,
            ),
            self._stream_key,
            request.operation,
            request.expected_version or 0,
            request.task.model_dump_json(),
            request.task.version,
            request.task.task_id,
            request.request_fingerprint,
            request.submission_fingerprint,
            json.dumps(
                event_blueprints,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                receipt_template,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            (
                request.queue_message.model_dump_json()
                if request.queue_message is not None
                else ""
            ),
        )
        result_code, current_version, raw_receipt = (
            self._parse_commit_result(result)
        )
        if result_code == -1:
            raise LongTaskNotFoundError(
                f"长任务不存在: {request.task.task_id}"
            )
        if result_code == -2:
            raise CorruptLongTaskSnapshotError(
                f"长任务提交状态已损坏: {request.task.task_id}"
            )
        if result_code == -3:
            raise LongTaskCommitFingerprintConflictError(
                "commit_id 已用于不同业务动作: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code == -4:
            raise LongTaskAlreadyExistsError(
                f"长任务已经存在: {request.task.task_id}"
            )
        if result_code == -5:
            raise LongTaskCommitContentConflictError(
                "同一业务动作生成了不同提交内容: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code == 0:
            raise LongTaskVersionConflictError(
                "长任务版本冲突: "
                f"expected={request.expected_version}, "
                f"actual={current_version}"
            )
        if result_code == -6:
            raise CorruptLongTaskCommitReceiptError(
                "长任务提交回执无法恢复: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code in {-7, -8}:
            raise LongTaskStoreError(
                "Redis 拒绝了非法可靠提交结构: "
                f"code={result_code}"
            )
        if result_code not in {1, 2}:
            raise LongTaskStoreError(
                f"Redis 返回未知可靠提交结果: {result!r}"
            )
        return self._parse_receipt(raw_receipt)

    async def load_receipt(
        self,
        *,
        task_id: str,
        commit_id: str,
    ) -> LongTaskCommitReceipt | None:
        """
        按任务和逻辑提交编号读取已持久化回执。

        参数含义：
            task_id:
                回执所属长任务编号。
            commit_id:
                调用方重试时保持稳定的提交编号。

        返回值含义：
            LongTaskCommitReceipt | None:
                找到时返回类型化回执，不存在时返回 None。
        """

        raw_receipt = await self._redis.get(
            self._receipt_key(task_id, commit_id)
        )
        if raw_receipt is None:
            return None
        return self._parse_receipt(raw_receipt)

    async def load_events(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[LongTaskEvent]:
        """
        按提交顺序分页读取一份任务的不可变业务事件。

        参数含义：
            task_id:
                准备查询事件历史的任务编号。
            after_sequence:
                只返回 sequence 大于该值的事件，0 表示从头读取。
            limit:
                本次最多返回的事件数量。

        返回值含义：
            list[LongTaskEvent]:
                已按 sequence 排序并通过契约校验的事件列表。
        """

        if after_sequence < 0:
            raise ValueError("after_sequence 不能小于 0")
        if limit < 1 or limit > 1000:
            raise ValueError("limit 必须在 1 到 1000 之间")
        start_index = after_sequence
        raw_events = await self._redis.lrange(
            self._events_key(task_id),
            start_index,
            start_index + limit - 1,
        )
        try:
            return [
                LongTaskEvent.model_validate_json(raw_event)
                for raw_event in raw_events
            ]
        except (ValidationError, ValueError, TypeError) as exc:
            raise LongTaskStoreError(
                f"长任务事件历史无法恢复: {task_id}"
            ) from exc

    def _event_sequence_key(self, task_id: str) -> str:
        """返回一份任务的事件序号 Redis Key。"""

        return f"{self._key_prefix}:event-sequence:{task_id}"

    def _events_key(self, task_id: str) -> str:
        """返回一份任务的不可变事件列表 Redis Key。"""

        return f"{self._key_prefix}:events:{task_id}"

    def _receipt_key(self, task_id: str, commit_id: str) -> str:
        """返回一份业务提交回执的 Redis Key。"""

        normalized_commit_id = str(commit_id or "").strip()
        if not normalized_commit_id:
            raise ValueError("commit_id 不能为空")
        return (
            f"{self._key_prefix}:receipt:{task_id}:"
            f"{normalized_commit_id}"
        )

    @staticmethod
    def _build_event_blueprints(
        *,
        request: LongTaskCommitRequest,
        committed_at: Any,
    ) -> list[dict[str, Any]]:
        """把事件草稿转换成等待 Lua 分配 sequence 的稳定事件骨架。"""

        blueprints: list[dict[str, Any]] = []
        for index, draft in enumerate(request.event_drafts):
            event_identity = (
                f"{request.task.task_id}:{request.commit_id}:{index}"
            )
            event_id = "event_" + hashlib.sha256(
                event_identity.encode("utf-8")
            ).hexdigest()[:32]
            blueprints.append(
                {
                    "event_id": event_id,
                    "task_id": request.task.task_id,
                    "task_version": request.task.version,
                    "commit_id": request.commit_id,
                    "step_id": draft.step_id,
                    "sequence": 1,
                    "event_type": draft.event_type,
                    "actor_type": draft.actor_type,
                    "actor_id": draft.actor_id,
                    "payload": draft.payload,
                    "correlation_id": draft.correlation_id,
                    "schema_version": draft.schema_version,
                    "created_at": committed_at.isoformat(),
                }
            )
        return blueprints

    @staticmethod
    def _parse_commit_result(result: Any) -> tuple[int, int, Any]:
        """解析可靠提交 Lua 返回的结果码、版本和回执 JSON。"""

        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise LongTaskStoreError(
                f"Redis 返回非法可靠提交结果: {result!r}"
            )
        try:
            return int(result[0]), int(result[1]), result[2]
        except (TypeError, ValueError) as exc:
            raise LongTaskStoreError(
                f"Redis 返回非法可靠提交结果: {result!r}"
            ) from exc

    @staticmethod
    def _parse_receipt(raw_receipt: Any) -> LongTaskCommitReceipt:
        """把 Redis JSON 恢复为经过严格校验的提交回执。"""

        if isinstance(raw_receipt, bytes):
            raw_receipt = raw_receipt.decode("utf-8")
        try:
            return LongTaskCommitReceipt.model_validate_json(raw_receipt)
        except (ValidationError, ValueError, TypeError) as exc:
            raise CorruptLongTaskCommitReceiptError(
                "长任务提交回执无法恢复"
            ) from exc

    def _task_key(self, task_id: str) -> str:
        """
        构建一份长任务最新快照的 Redis Key。

        参数含义：
            task_id:
                长任务唯一编号。

        返回值含义：
            str:
                ``{prefix}:task:{task_id}`` 格式的 Redis Key。
        """

        normalized_task_id = str(task_id or "").strip()
        if not normalized_task_id:
            raise ValueError("task_id 不能为空")
        return f"{self._key_prefix}:task:{normalized_task_id}"

    @staticmethod
    def _parse_cas_result(result: Any) -> tuple[int, int]:
        """
        把 Redis Lua 返回值转换为结果码和当前版本号。

        参数含义：
            result:
                Redis ``EVAL`` 返回的二元素列表或元组。

        返回值含义：
            tuple[int, int]:
                第一项为保存结果码，第二项为 Redis 当前版本号。
        """

        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise LongTaskStoreError(
                f"Redis 返回非法保存结果: {result!r}"
            )
        try:
            return int(result[0]), int(result[1])
        except (TypeError, ValueError) as exc:
            raise LongTaskStoreError(
                f"Redis 返回非法保存结果: {result!r}"
            ) from exc
