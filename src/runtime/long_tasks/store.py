"""长任务快照存储契约与 Redis MVP 实现。"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from src.runtime.long_tasks.contracts import LongTask


DEFAULT_LONG_TASK_KEY_PREFIX = "dog-agent:long-task:v1"

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


class RedisLongTaskStore:
    """
    使用 Redis String 保存最新 LongTask JSON 快照。

    功能：
        使用 SET NX 防止重复创建，使用 Lua Compare-And-Set 保证版本检查和
        SET 在 Redis 内原子完成。该 MVP 不保存完整 Event 历史。

    参数含义：
        redis_client:
            已由 RedisProvider 启动并完成健康检查的异步 Redis 客户端。
        key_prefix:
            Redis Key 命名空间；默认包含项目名、领域名和结构版本。

    返回值含义：
        RedisLongTaskStore:
            实现 LongTaskStore 契约的 Redis 快照存储。
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        key_prefix: str = DEFAULT_LONG_TASK_KEY_PREFIX,
    ) -> None:
        normalized_prefix = str(key_prefix or "").strip().rstrip(":")
        if not normalized_prefix:
            raise ValueError("长任务 Redis Key 前缀不能为空")
        self._redis = redis_client
        self._key_prefix = normalized_prefix

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
