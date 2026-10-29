"""长任务快照存储契约与 Redis MVP 实现。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from pydantic import ValidationError

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskCommitReceipt,
    LongTaskCommitRequest,
    LongTaskEvent,
    utc_now,
)
from src.runtime.long_tasks.failure_contracts import (
    LongTaskDeadLetterRecord,
    LongTaskFailureCommitRequest,
    LongTaskRetryCancellationRequest,
    LongTaskRetryPlan,
    LongTaskRetryDispatchRequest,
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

_FAILURE_COMMIT_SCRIPT = """
local receipt_json = redis.call("GET", KEYS[4])
if receipt_json then
    local receipt_ok, receipt = pcall(cjson.decode, receipt_json)
    if not receipt_ok
        or receipt["request_fingerprint"] == nil
        or receipt["submission_fingerprint"] == nil then
        return {-6, -1, receipt_json}
    end
    if receipt["request_fingerprint"] ~= ARGV[5] then
        return {-3, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    if receipt["submission_fingerprint"] ~= ARGV[6] then
        return {-5, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    return {2, tonumber(receipt["task_version"] or -1), receipt_json}
end

local task_ok, next_task = pcall(cjson.decode, ARGV[2])
local events_ok, event_blueprints = pcall(cjson.decode, ARGV[7])
local receipt_template_ok, receipt = pcall(cjson.decode, ARGV[8])
if not task_ok or not events_ok or not receipt_template_ok then
    return {-7, -1, ""}
end
if #event_blueprints < 1 then
    return {-7, -1, ""}
end
if tostring(next_task["task_id"] or "") ~= ARGV[4]
    or tonumber(next_task["version"] or -1) ~= tonumber(ARGV[3]) then
    return {-7, -1, ""}
end

local retry_plan = nil
if ARGV[9] ~= "" then
    local retry_ok, decoded_retry = pcall(cjson.decode, ARGV[9])
    if not retry_ok then
        return {-7, -1, ""}
    end
    retry_plan = decoded_retry
end
local dead_letter = nil
if ARGV[10] ~= "" then
    local dead_letter_ok, decoded_dead_letter = pcall(cjson.decode, ARGV[10])
    if not dead_letter_ok then
        return {-7, -1, ""}
    end
    dead_letter = decoded_dead_letter
end
if retry_plan ~= nil and dead_letter ~= nil then
    return {-7, -1, ""}
end
if retry_plan ~= nil and (
    tostring(retry_plan["task_id"] or "") ~= ARGV[4]
    or retry_plan["failure_commit_id"] ~= receipt["commit_id"]
) then
    return {-7, -1, ""}
end
if dead_letter ~= nil and (
    tostring(dead_letter["task_id"] or "") ~= ARGV[4]
    or dead_letter["failure_commit_id"] ~= receipt["commit_id"]
) then
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
if (task_type ~= "none" and task_type ~= "string")
    or (sequence_type ~= "none" and sequence_type ~= "string")
    or (events_type ~= "none" and events_type ~= "list")
    or (receipt_type ~= "none" and receipt_type ~= "string") then
    return {-8, -1, ""}
end
if retry_plan ~= nil and type_name(KEYS[5]) ~= "none" then
    return {-9, -1, ""}
end
if dead_letter ~= nil and type_name(KEYS[6]) ~= "none" then
    return {-9, -1, ""}
end
if retry_plan ~= nil then
    local schedule_type = type_name(KEYS[7])
    if schedule_type ~= "none" and schedule_type ~= "zset" then
        return {-8, -1, ""}
    end
end

local current_json = redis.call("GET", KEYS[1])
if not current_json then
    return {-1, -1, ""}
end
local current_ok, current_task = pcall(cjson.decode, current_json)
if not current_ok or current_task["version"] == nil then
    return {-2, -1, ""}
end
local current_version = tonumber(current_task["version"])
if current_version ~= tonumber(ARGV[1]) then
    return {0, current_version, ""}
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

redis.call("SET", KEYS[1], ARGV[2])
redis.call("SET", KEYS[2], tostring(sequence_end))
for _, event_json in ipairs(event_jsons) do
    redis.call("RPUSH", KEYS[3], event_json)
end
if retry_plan ~= nil then
    redis.call("SET", KEYS[5], ARGV[9])
    redis.call("ZADD", KEYS[7], tonumber(ARGV[11]), retry_plan["retry_plan_id"])
end
if dead_letter ~= nil then
    redis.call("SET", KEYS[6], ARGV[10])
end

receipt["event_ids"] = event_ids
receipt["event_sequence_start"] = sequence_start
receipt["event_sequence_end"] = sequence_end
local committed_receipt_json = cjson.encode(receipt)
redis.call("SET", KEYS[4], committed_receipt_json)
return {1, tonumber(ARGV[3]), committed_receipt_json}
"""

_CLAIM_DUE_RETRY_PLAN_SCRIPT = """
local function type_name(key)
    local result = redis.call("TYPE", key)
    if type(result) == "table" then
        return result["ok"]
    end
    return result
end

local plan_type = type_name(KEYS[1])
local schedule_type = type_name(KEYS[2])
local lock_type = type_name(KEYS[3])
if (plan_type ~= "none" and plan_type ~= "string")
    or (schedule_type ~= "none" and schedule_type ~= "zset")
    or (lock_type ~= "none" and lock_type ~= "string") then
    return {-4, ""}
end

local plan_json = redis.call("GET", KEYS[1])
if not plan_json then
    return {-1, ""}
end
local plan_ok, plan = pcall(cjson.decode, plan_json)
if not plan_ok or tostring(plan["retry_plan_id"] or "") ~= ARGV[1] then
    return {-2, ""}
end
if plan["status"] ~= "scheduled" and plan["status"] ~= "dispatching" then
    return {0, ""}
end

local scheduled_score = redis.call("ZSCORE", KEYS[2], ARGV[1])
if not scheduled_score then
    return {-3, ""}
end
if tonumber(scheduled_score) > tonumber(ARGV[2]) then
    return {0, ""}
end

local current_claim_id = redis.call("GET", KEYS[3])
if current_claim_id then
    if current_claim_id == ARGV[3]
        and plan["dispatch_claim_id"] == ARGV[3] then
        return {2, plan_json}
    end
    return {0, ""}
end

local lock_created = redis.call(
    "SET", KEYS[3], ARGV[3], "NX", "PX", tonumber(ARGV[5])
)
if not lock_created then
    return {0, ""}
end

plan["status"] = "dispatching"
plan["dispatcher_id"] = ARGV[4]
plan["dispatch_claim_id"] = ARGV[3]
plan["dispatch_lease_expires_at"] = ARGV[6]
local claimed_plan_json = cjson.encode(plan)
redis.call("SET", KEYS[1], claimed_plan_json)
return {1, claimed_plan_json}
"""

_CANCEL_CLAIMED_RETRY_PLAN_SCRIPT = """
local function type_name(key)
    local result = redis.call("TYPE", key)
    if type(result) == "table" then
        return result["ok"]
    end
    return result
end

local plan_type = type_name(KEYS[1])
local schedule_type = type_name(KEYS[2])
local lock_type = type_name(KEYS[3])
if (plan_type ~= "none" and plan_type ~= "string")
    or (schedule_type ~= "none" and schedule_type ~= "zset")
    or (lock_type ~= "none" and lock_type ~= "string") then
    return {-3, ""}
end

local plan_json = redis.call("GET", KEYS[1])
if not plan_json then
    return {-1, ""}
end
local plan_ok, plan = pcall(cjson.decode, plan_json)
if not plan_ok
    or tostring(plan["retry_plan_id"] or "") ~= ARGV[1]
    or tostring(plan["task_id"] or "") ~= ARGV[2]
    or tostring(plan["step_id"] or "") ~= ARGV[3] then
    return {-2, ""}
end

if plan["status"] == "cancelled" then
    if plan["cancellation_reason"] == ARGV[6] then
        return {2, plan_json}
    end
    return {0, ""}
end
if plan["status"] ~= "dispatching"
    or plan["dispatcher_id"] ~= ARGV[4]
    or plan["dispatch_claim_id"] ~= ARGV[5] then
    return {0, ""}
end
if redis.call("GET", KEYS[3]) ~= ARGV[5] then
    return {0, ""}
end
if not redis.call("ZSCORE", KEYS[2], ARGV[1]) then
    return {0, ""}
end

plan["status"] = "cancelled"
plan["dispatcher_id"] = cjson.null
plan["dispatch_claim_id"] = cjson.null
plan["dispatch_lease_expires_at"] = cjson.null
plan["cancellation_reason"] = ARGV[6]
plan["cancelled_at"] = ARGV[7]
local cancelled_plan_json = cjson.encode(plan)
redis.call("SET", KEYS[1], cancelled_plan_json)
redis.call("ZREM", KEYS[2], ARGV[1])
redis.call("DEL", KEYS[3])
return {1, cancelled_plan_json}
"""

_COMMIT_RETRY_DISPATCH_SCRIPT = """
local receipt_json = redis.call("GET", KEYS[7])
if receipt_json then
    local receipt_ok, receipt = pcall(cjson.decode, receipt_json)
    if not receipt_ok
        or receipt["request_fingerprint"] == nil
        or receipt["submission_fingerprint"] == nil then
        return {-6, -1, receipt_json}
    end
    if receipt["request_fingerprint"] ~= ARGV[8] then
        return {-3, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    if receipt["submission_fingerprint"] ~= ARGV[9] then
        return {-5, tonumber(receipt["task_version"] or -1), receipt_json}
    end
    return {2, tonumber(receipt["task_version"] or -1), receipt_json}
end

local task_ok, next_task = pcall(cjson.decode, ARGV[2])
local events_ok, event_blueprints = pcall(cjson.decode, ARGV[10])
local receipt_ok, receipt = pcall(cjson.decode, ARGV[11])
local message_ok, queue_message = pcall(cjson.decode, ARGV[12])
if not task_ok or not events_ok or not receipt_ok or not message_ok then
    return {-7, -1, ""}
end
if #event_blueprints < 1 then
    return {-7, -1, ""}
end
if tostring(next_task["task_id"] or "") ~= ARGV[4]
    or tonumber(next_task["version"] or -1) ~= tonumber(ARGV[3])
    or next_task["status"] ~= "queued" then
    return {-7, -1, ""}
end
if tostring(queue_message["task_id"] or "") ~= ARGV[4]
    or tonumber(queue_message["task_version"] or -1) ~= tonumber(ARGV[3])
    or queue_message["reason"] ~= "retry" then
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
local plan_type = type_name(KEYS[2])
local schedule_type = type_name(KEYS[3])
local lock_type = type_name(KEYS[4])
local sequence_type = type_name(KEYS[5])
local events_type = type_name(KEYS[6])
local receipt_type = type_name(KEYS[7])
local stream_type = type_name(KEYS[8])
if task_type ~= "string"
    or plan_type ~= "string"
    or schedule_type ~= "zset"
    or (lock_type ~= "none" and lock_type ~= "string")
    or (sequence_type ~= "none" and sequence_type ~= "string")
    or (events_type ~= "none" and events_type ~= "list")
    or (receipt_type ~= "none" and receipt_type ~= "string")
    or (stream_type ~= "none" and stream_type ~= "stream") then
    return {-8, -1, ""}
end

local current_json = redis.call("GET", KEYS[1])
local plan_json = redis.call("GET", KEYS[2])
local current_ok, current_task = pcall(cjson.decode, current_json)
local plan_ok, plan = pcall(cjson.decode, plan_json)
if not current_ok or current_task["version"] == nil or not plan_ok then
    return {-2, -1, ""}
end
local current_version = tonumber(current_task["version"])
if current_version ~= tonumber(ARGV[1]) then
    return {0, current_version, ""}
end
if tostring(plan["retry_plan_id"] or "") ~= ARGV[5]
    or tostring(plan["task_id"] or "") ~= ARGV[4]
    or tonumber(plan["failure_task_version"] or -1) ~= tonumber(ARGV[1])
    or plan["status"] ~= "dispatching"
    or plan["dispatcher_id"] ~= ARGV[6]
    or plan["dispatch_claim_id"] ~= ARGV[7] then
    return {-9, current_version, ""}
end
local lock_claim_id = redis.call("GET", KEYS[4])
if lock_claim_id ~= ARGV[7] then
    return {-9, current_version, ""}
end
if not redis.call("ZSCORE", KEYS[3], ARGV[5]) then
    return {-9, current_version, ""}
end

local current_sequence = tonumber(redis.call("GET", KEYS[5]) or "0")
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

local queue_message_id = redis.call(
    "XADD", KEYS[8], "*", "payload", ARGV[12]
)
plan["status"] = "dispatched"
plan["dispatcher_id"] = cjson.null
plan["dispatch_claim_id"] = cjson.null
plan["dispatch_lease_expires_at"] = cjson.null
plan["queue_message_id"] = queue_message_id
plan["dispatched_at"] = ARGV[13]

redis.call("SET", KEYS[1], ARGV[2])
redis.call("SET", KEYS[2], cjson.encode(plan))
redis.call("SET", KEYS[5], tostring(sequence_end))
for _, event_json in ipairs(event_jsons) do
    redis.call("RPUSH", KEYS[6], event_json)
end
redis.call("ZREM", KEYS[3], ARGV[5])
redis.call("DEL", KEYS[4])

receipt["event_ids"] = event_ids
receipt["event_sequence_start"] = sequence_start
receipt["event_sequence_end"] = sequence_end
receipt["queue_message_id"] = queue_message_id
local committed_receipt_json = cjson.encode(receipt)
redis.call("SET", KEYS[7], committed_receipt_json)
return {1, tonumber(ARGV[3]), committed_receipt_json}
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


class LongTaskFailureRecordConflictError(LongTaskStoreError):
    """表示重试计划或死信编号已被另一份未关联提交占用。"""


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

    async def commit_failure(
        self,
        request: LongTaskFailureCommitRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子保存失败快照、事件、计划或死信以及提交回执。

        参数含义：
            request:
                已通过失败领域规则校验的完整提交请求。

        返回值含义：
            LongTaskCommitReceipt：首次成功或幂等重放对应的回执。
        """

        ...

    async def load_due_retry_plan_ids(
        self,
        *,
        due_at: datetime,
        limit: int = 100,
    ) -> list[str]:
        """
        按计划时间读取已经到期的重试计划编号。

        参数含义：
            due_at:
                查询截止时间，必须包含有效时区。
            limit:
                本次最多返回的计划数量。

        返回值含义：
            list[str]：按最早执行时间排序的到期计划编号。
        """

        ...

    async def claim_due_retry_plan(
        self,
        *,
        retry_plan_id: str,
        dispatcher_id: str,
        dispatch_claim_id: str,
        now: datetime,
        lease_seconds: float,
    ) -> LongTaskRetryPlan | None:
        """
        原子领取一个已经到期的重试计划发布权。

        参数含义：
            retry_plan_id:
                准备领取的重试计划编号。
            dispatcher_id:
                当前调度器实例身份。
            dispatch_claim_id:
                当前领取请求生成的稳定发布权令牌。
            now:
                本次领取判断使用的当前时间。
            lease_seconds:
                发布权租约持续秒数。

        返回值含义：
            LongTaskRetryPlan | None：
                领取成功或相同令牌重放时返回 dispatching 计划；
                尚未到期或已被其他调度器领取时返回 None。
        """

        ...

    async def commit_retry_dispatch(
        self,
        request: LongTaskRetryDispatchRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子完成已领取重试计划的消息发布。

        参数含义：
            request:
                包含发布令牌、queued 新快照、事件和重试消息的提交请求。

        返回值含义：
            LongTaskCommitReceipt：首次成功或幂等重放对应的提交回执。
        """

        ...

    async def cancel_claimed_retry_plan(
        self,
        request: LongTaskRetryCancellationRequest,
    ) -> LongTaskRetryPlan:
        """
        原子取消当前调度器持有发布权的失效计划。

        参数含义：
            request:
                包含计划编号、发布权令牌和取消原因的请求。

        返回值含义：
            LongTaskRetryPlan：状态已经变为 cancelled 的最新计划。
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

    async def commit_failure(
        self,
        request: LongTaskFailureCommitRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子保存失败任务、事件、重试计划或死信以及提交回执。

        功能：
            在同一个 Redis Lua 脚本中先检查回执和任务版本，再一次性写入
            全部失败数据。任何检查不通过时都不会写入其中任何一项。

        参数含义：
            request:
                包含失败新快照、幂等指纹、事件以及计划或死信的请求。

        返回值含义：
            LongTaskCommitReceipt：
                首次成功提交或相同内容幂等重放对应的持久化回执。
        """

        committed_at = utc_now()
        event_blueprints = self._build_event_blueprints(
            request=request,
            committed_at=committed_at,
        )
        retry_plan = request.retry_plan
        dead_letter = request.dead_letter_record
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
            "retry_plan_id": (
                retry_plan.retry_plan_id if retry_plan is not None else None
            ),
            "dead_letter_id": (
                dead_letter.dead_letter_id
                if dead_letter is not None
                else None
            ),
            "committed_at": committed_at.isoformat(),
        }
        result = await self._redis.eval(
            _FAILURE_COMMIT_SCRIPT,
            7,
            self._task_key(request.task.task_id),
            self._event_sequence_key(request.task.task_id),
            self._events_key(request.task.task_id),
            self._receipt_key(request.task.task_id, request.commit_id),
            self._retry_plan_key(
                retry_plan.retry_plan_id
                if retry_plan is not None
                else f"unused:{request.commit_id}"
            ),
            self._dead_letter_key(
                dead_letter.dead_letter_id
                if dead_letter is not None
                else f"unused:{request.commit_id}"
            ),
            self._retry_schedule_key(),
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
            retry_plan.model_dump_json() if retry_plan is not None else "",
            (
                dead_letter.model_dump_json()
                if dead_letter is not None
                else ""
            ),
            (
                retry_plan.scheduled_at.timestamp()
                if retry_plan is not None
                else 0
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
                f"长任务失败提交状态已损坏: {request.task.task_id}"
            )
        if result_code == -3:
            raise LongTaskCommitFingerprintConflictError(
                "failure commit_id 已用于不同业务动作: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code == -5:
            raise LongTaskCommitContentConflictError(
                "同一失败动作生成了不同提交内容: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code == 0:
            raise LongTaskVersionConflictError(
                "长任务失败提交版本冲突: "
                f"expected={request.expected_version}, "
                f"actual={current_version}"
            )
        if result_code == -6:
            raise CorruptLongTaskCommitReceiptError(
                "长任务失败提交回执无法恢复: "
                f"task_id={request.task.task_id}, "
                f"commit_id={request.commit_id}"
            )
        if result_code in {-7, -8}:
            raise LongTaskStoreError(
                "Redis 拒绝了非法失败提交结构: "
                f"code={result_code}"
            )
        if result_code == -9:
            raise LongTaskFailureRecordConflictError(
                "重试计划或死信编号已被其他提交占用"
            )
        if result_code not in {1, 2}:
            raise LongTaskStoreError(
                f"Redis 返回未知失败提交结果: {result!r}"
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

    async def load_retry_plan(
        self,
        retry_plan_id: str,
    ) -> LongTaskRetryPlan | None:
        """
        根据计划编号读取已经原子保存的重试计划。

        参数含义：
            retry_plan_id:
                需要读取的稳定重试计划编号。

        返回值含义：
            LongTaskRetryPlan | None：找到时返回计划，不存在时返回 None。
        """

        normalized_id = str(retry_plan_id or "").strip()
        raw_plan = await self._redis.get(
            self._retry_plan_key(normalized_id)
        )
        if raw_plan is None:
            return None
        try:
            plan = LongTaskRetryPlan.model_validate_json(raw_plan)
        except (ValidationError, ValueError, TypeError) as exc:
            raise LongTaskStoreError("重试计划无法恢复") from exc
        if plan.retry_plan_id != normalized_id:
            raise LongTaskStoreError(
                "Redis Key 中的 retry_plan_id 与计划内容不一致"
            )
        return plan

    async def load_due_retry_plan_ids(
        self,
        *,
        due_at: datetime,
        limit: int = 100,
    ) -> list[str]:
        """
        按执行时间读取已经到期的重试计划编号。

        功能：
            查询重试时间排序集合中分数小于等于 due_at 的成员，供后续
            调度器逐个领取。这里只读取，不修改计划状态或删除索引。

        参数含义：
            due_at:
                本次认为已经到期的截止时间，必须包含有效时区。
            limit:
                本次最多读取的计划数量，范围为 1 到 1000。

        返回值含义：
            list[str]：按 scheduled_at 从早到晚排列的重试计划编号。
        """

        if due_at.tzinfo is None or due_at.utcoffset() is None:
            raise ValueError("due_at 必须包含时区")
        if limit < 1 or limit > 1000:
            raise ValueError("limit 必须在 1 到 1000 之间")
        raw_ids = await self._redis.zrangebyscore(
            self._retry_schedule_key(),
            "-inf",
            due_at.timestamp(),
            start=0,
            num=limit,
        )
        return [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_ids
        ]

    async def claim_due_retry_plan(
        self,
        *,
        retry_plan_id: str,
        dispatcher_id: str,
        dispatch_claim_id: str,
        now: datetime,
        lease_seconds: float,
    ) -> LongTaskRetryPlan | None:
        """
        原子领取一个到期重试计划的消息发布权。

        功能：
            同时检查计划状态、到期时间和短期锁。领取成功后把计划改为
            dispatching 并写入调度器身份、发布令牌和租约时间。锁过期后，
            其他调度器可以重新领取尚未完成发布的计划。

        参数含义：
            retry_plan_id:
                准备领取的重试计划编号。
            dispatcher_id:
                当前调度器进程或实例的稳定身份。
            dispatch_claim_id:
                本次领取生成的唯一发布权令牌。
            now:
                判断计划是否到期的当前时间，必须包含有效时区。
            lease_seconds:
                当前调度器最多持有发布权的秒数。

        返回值含义：
            LongTaskRetryPlan | None：
                成功时返回状态为 dispatching 的计划；尚未到期、已完成、
                已取消或正被其他调度器持有时返回 None。
        """

        normalized_plan_id = str(retry_plan_id or "").strip()
        normalized_dispatcher_id = str(dispatcher_id or "").strip()
        normalized_claim_id = str(dispatch_claim_id or "").strip()
        if not normalized_plan_id:
            raise ValueError("retry_plan_id 不能为空")
        if not normalized_dispatcher_id:
            raise ValueError("dispatcher_id 不能为空")
        if not normalized_claim_id:
            raise ValueError("dispatch_claim_id 不能为空")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now 必须包含时区")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须大于 0")

        normalized_now = now.astimezone(timezone.utc)
        lease_expires_at = normalized_now + timedelta(
            seconds=lease_seconds
        )
        lease_milliseconds = max(1, int(lease_seconds * 1000))
        result = await self._redis.eval(
            _CLAIM_DUE_RETRY_PLAN_SCRIPT,
            3,
            self._retry_plan_key(normalized_plan_id),
            self._retry_schedule_key(),
            self._retry_dispatch_lock_key(normalized_plan_id),
            normalized_plan_id,
            normalized_now.timestamp(),
            normalized_claim_id,
            normalized_dispatcher_id,
            lease_milliseconds,
            lease_expires_at.isoformat(),
        )
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise LongTaskStoreError(
                f"Redis 返回非法重试计划领取结果: {result!r}"
            )
        try:
            result_code = int(result[0])
        except (TypeError, ValueError) as exc:
            raise LongTaskStoreError(
                f"Redis 返回非法重试计划领取结果: {result!r}"
            ) from exc

        if result_code == 0:
            return None
        if result_code == -1:
            raise LongTaskStoreError(
                f"重试计划不存在: {normalized_plan_id}"
            )
        if result_code == -2:
            raise LongTaskStoreError(
                f"重试计划内容已损坏: {normalized_plan_id}"
            )
        if result_code == -3:
            raise LongTaskStoreError(
                f"重试计划缺少时间索引: {normalized_plan_id}"
            )
        if result_code == -4:
            raise LongTaskStoreError("重试计划领取使用了错误的 Redis 类型")
        if result_code not in {1, 2}:
            raise LongTaskStoreError(
                f"Redis 返回未知重试计划领取结果: {result!r}"
            )

        raw_plan = result[1]
        try:
            claimed_plan = LongTaskRetryPlan.model_validate_json(raw_plan)
        except (ValidationError, ValueError, TypeError) as exc:
            raise LongTaskStoreError("已领取重试计划无法恢复") from exc
        if (
            claimed_plan.retry_plan_id != normalized_plan_id
            or claimed_plan.status != "dispatching"
            or claimed_plan.dispatch_claim_id != normalized_claim_id
        ):
            raise LongTaskStoreError("Redis 返回了不匹配的已领取重试计划")
        return claimed_plan

    async def commit_retry_dispatch(
        self,
        request: LongTaskRetryDispatchRequest,
    ) -> LongTaskCommitReceipt:
        """
        原子完成任务恢复、消息发布和重试计划收尾。

        功能：
            校验任务版本、计划状态、调度器身份和发布令牌后，在同一个
            Redis 脚本中保存 queued/ready 快照、写入 Stream、把计划改为
            dispatched，并移除时间索引和短期发布锁。

        参数含义：
            request:
                已通过领域校验的重试消息发布提交请求。

        返回值含义：
            LongTaskCommitReceipt：
                首次成功或相同内容幂等重放对应的持久化回执。
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
            "retry_plan_id": request.retry_plan_id,
            "dead_letter_id": None,
            "committed_at": committed_at.isoformat(),
        }
        queue_message = request.queue_message
        if queue_message is None:
            raise ValueError("重试发布提交缺少队列消息")
        result = await self._redis.eval(
            _COMMIT_RETRY_DISPATCH_SCRIPT,
            8,
            self._task_key(request.task.task_id),
            self._retry_plan_key(request.retry_plan_id),
            self._retry_schedule_key(),
            self._retry_dispatch_lock_key(request.retry_plan_id),
            self._event_sequence_key(request.task.task_id),
            self._events_key(request.task.task_id),
            self._receipt_key(request.task.task_id, request.commit_id),
            self._stream_key,
            request.expected_version or 0,
            request.task.model_dump_json(),
            request.task.version,
            request.task.task_id,
            request.retry_plan_id,
            request.dispatcher_id,
            request.dispatch_claim_id,
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
            queue_message.model_dump_json(),
            request.dispatched_at.isoformat(),
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
                f"重试发布相关数据已损坏: {request.retry_plan_id}"
            )
        if result_code == -3:
            raise LongTaskCommitFingerprintConflictError(
                "重试发布 commit_id 已用于不同业务动作"
            )
        if result_code == -5:
            raise LongTaskCommitContentConflictError(
                "同一重试发布动作生成了不同提交内容"
            )
        if result_code == 0:
            raise LongTaskVersionConflictError(
                "重试发布任务版本冲突: "
                f"expected={request.expected_version}, "
                f"actual={current_version}"
            )
        if result_code == -6:
            raise CorruptLongTaskCommitReceiptError(
                "重试发布提交回执无法恢复"
            )
        if result_code in {-7, -8}:
            raise LongTaskStoreError(
                "Redis 拒绝了非法重试发布结构: "
                f"code={result_code}"
            )
        if result_code == -9:
            raise LongTaskFailureRecordConflictError(
                "重试计划发布权、状态或时间索引已经发生变化"
            )
        if result_code not in {1, 2}:
            raise LongTaskStoreError(
                f"Redis 返回未知重试发布结果: {result!r}"
            )
        return self._parse_receipt(raw_receipt)

    async def cancel_claimed_retry_plan(
        self,
        request: LongTaskRetryCancellationRequest,
    ) -> LongTaskRetryPlan:
        """
        原子取消失效计划并清理时间索引和发布锁。

        功能：
            只有计划中保存的调度器、发布令牌与 Redis 短期锁全部一致时
            才能取消，防止没有发布权的调度器误删其他实例正在处理的计划。

        参数含义：
            request:
                已通过领域校验的重试计划取消请求。

        返回值含义：
            LongTaskRetryPlan：首次取消或相同取消重放得到的最新计划。
        """

        result = await self._redis.eval(
            _CANCEL_CLAIMED_RETRY_PLAN_SCRIPT,
            3,
            self._retry_plan_key(request.retry_plan_id),
            self._retry_schedule_key(),
            self._retry_dispatch_lock_key(request.retry_plan_id),
            request.retry_plan_id,
            request.task_id,
            request.step_id,
            request.dispatcher_id,
            request.dispatch_claim_id,
            request.cancellation_reason,
            request.cancelled_at.isoformat(),
        )
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise LongTaskStoreError(
                f"Redis 返回非法重试计划取消结果: {result!r}"
            )
        try:
            result_code = int(result[0])
        except (TypeError, ValueError) as exc:
            raise LongTaskStoreError(
                f"Redis 返回非法重试计划取消结果: {result!r}"
            ) from exc
        if result_code == -1:
            raise LongTaskStoreError(
                f"重试计划不存在: {request.retry_plan_id}"
            )
        if result_code == -2:
            raise LongTaskStoreError(
                f"重试计划内容已损坏: {request.retry_plan_id}"
            )
        if result_code == -3:
            raise LongTaskStoreError("重试计划取消使用了错误的 Redis 类型")
        if result_code == 0:
            raise LongTaskFailureRecordConflictError(
                "当前调度器已经不再持有该重试计划的发布权"
            )
        if result_code not in {1, 2}:
            raise LongTaskStoreError(
                f"Redis 返回未知重试计划取消结果: {result!r}"
            )
        try:
            cancelled_plan = LongTaskRetryPlan.model_validate_json(result[1])
        except (ValidationError, ValueError, TypeError) as exc:
            raise LongTaskStoreError("已取消重试计划无法恢复") from exc
        if (
            cancelled_plan.retry_plan_id != request.retry_plan_id
            or cancelled_plan.status != "cancelled"
            or cancelled_plan.cancellation_reason
            != request.cancellation_reason
        ):
            raise LongTaskStoreError("Redis 返回了不匹配的已取消重试计划")
        return cancelled_plan

    async def load_dead_letter(
        self,
        dead_letter_id: str,
    ) -> LongTaskDeadLetterRecord | None:
        """
        根据死信编号读取已经原子保存的死信记录。

        参数含义：
            dead_letter_id:
                需要读取的稳定死信记录编号。

        返回值含义：
            LongTaskDeadLetterRecord | None：
                找到时返回死信记录，不存在时返回 None。
        """

        normalized_id = str(dead_letter_id or "").strip()
        raw_record = await self._redis.get(
            self._dead_letter_key(normalized_id)
        )
        if raw_record is None:
            return None
        try:
            record = LongTaskDeadLetterRecord.model_validate_json(raw_record)
        except (ValidationError, ValueError, TypeError) as exc:
            raise LongTaskStoreError("死信记录无法恢复") from exc
        if record.dead_letter_id != normalized_id:
            raise LongTaskStoreError(
                "Redis Key 中的 dead_letter_id 与死信内容不一致"
            )
        return record

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

    def _retry_plan_key(self, retry_plan_id: str) -> str:
        """返回一份重试计划的 Redis Key。"""

        normalized_id = str(retry_plan_id or "").strip()
        if not normalized_id:
            raise ValueError("retry_plan_id 不能为空")
        return f"{self._key_prefix}:retry-plan:{normalized_id}"

    def _dead_letter_key(self, dead_letter_id: str) -> str:
        """返回一份死信记录的 Redis Key。"""

        normalized_id = str(dead_letter_id or "").strip()
        if not normalized_id:
            raise ValueError("dead_letter_id 不能为空")
        return f"{self._key_prefix}:dead-letter:{normalized_id}"

    def _retry_schedule_key(self) -> str:
        """返回按最早执行时间排序的重试计划索引 Redis Key。"""

        return f"{self._key_prefix}:retry-schedule"

    def _retry_dispatch_lock_key(self, retry_plan_id: str) -> str:
        """返回一个重试计划的短期发布权锁 Redis Key。"""

        normalized_id = str(retry_plan_id or "").strip()
        if not normalized_id:
            raise ValueError("retry_plan_id 不能为空")
        return f"{self._key_prefix}:retry-dispatch-lock:{normalized_id}"

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
