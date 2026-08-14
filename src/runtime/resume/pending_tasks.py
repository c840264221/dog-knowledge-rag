"""统一等待任务数据模型与活动任务集合。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


PendingTaskType = Literal["tool", "skill", "multi_agent"]
PendingTaskStatus = Literal[
    "awaiting_input",
    "running",
    "completed",
    "cancelled",
    "expired",
]
PendingInputValueType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "object",
]


class PendingTaskError(RuntimeError):
    """表示统一等待任务处理失败。"""


class DuplicatePendingTaskError(PendingTaskError):
    """表示相同任务编号被重复注册。"""


class InvalidPendingTaskRegistrationError(PendingTaskError):
    """表示新注册任务不是 awaiting_input 状态。"""


class PendingTaskNotFoundError(PendingTaskError, LookupError):
    """表示请求的等待任务不存在。"""


class PendingTaskVersionConflictError(PendingTaskError):
    """表示调用方持有的任务版本已经过期。"""


class InvalidPendingTaskTransitionError(PendingTaskError):
    """表示等待任务发生了不允许的状态迁移。"""


class PendingInputContract(BaseModel):
    """
    描述等待任务当前需要用户补充的一个结构化字段。

    参数含义：
        field_id:
            业务稳定字段编号，例如 body_weight 或 database_name。
        value_type:
            字段值类型，例如 string、integer 或 number。
        description:
            面向用户或开发人员的字段说明。
        required:
            是否为任务继续执行的必填字段。
        unit_family:
            可选的单位维度，例如 mass 或 duration。
        accepted_units:
            允许用户提交的单位列表，例如 kg、g。
        enum_values:
            可选的确定性候选值列表。

    返回值含义：
        PendingInputContract:
            可用于候选匹配和参数校验的统一输入契约。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    field_id: str = Field(min_length=1, max_length=128)
    value_type: PendingInputValueType
    description: str = Field(default="", max_length=500)
    required: bool = True
    unit_family: str | None = Field(default=None, max_length=64)
    accepted_units: list[str] = Field(default_factory=list)
    enum_values: list[str] = Field(default_factory=list)

    @field_validator("accepted_units", "enum_values")
    @classmethod
    def validate_unique_non_empty_values(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        校验候选值列表不包含空值或重复值。

        参数含义：
            values:
                Pydantic 已完成基础类型转换的字符串列表。

        返回值含义：
            list[str]:
                去除首尾空白且保持原顺序的合法值列表。
        """

        normalized_values = [value.strip() for value in values]
        if any(not value for value in normalized_values):
            raise ValueError("输入契约候选值不能为空")
        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError("输入契约候选值不能重复")
        return normalized_values


class ToolPendingPayload(BaseModel):
    """保存 Tool 等待任务恢复所需的类型化业务数据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    payload_kind: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    call_id: str | None = Field(default=None, max_length=200)
    resume_state: dict[str, Any] = Field(default_factory=dict)


class SkillPendingPayload(BaseModel):
    """保存 Skill 等待任务恢复所需的类型化业务数据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    payload_kind: Literal["skill"] = "skill"
    skill_id: str = Field(min_length=1, max_length=200)
    inputs: dict[str, Any] = Field(default_factory=dict)
    target_agent: str = Field(min_length=1, max_length=100)
    resume_state: dict[str, Any] = Field(default_factory=dict)


class MultiAgentPendingPayload(BaseModel):
    """保存 Multi-Agent 等待任务恢复所需的类型化业务数据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    payload_kind: Literal["multi_agent"] = "multi_agent"
    collaboration_id: str = Field(min_length=1, max_length=200)
    waiting_step_ids: list[str] = Field(default_factory=list)
    resume_state: dict[str, Any] = Field(default_factory=dict)


PendingTaskPayload = Annotated[
    ToolPendingPayload | SkillPendingPayload | MultiAgentPendingPayload,
    Field(discriminator="payload_kind"),
]


class PendingTaskSnapshot(BaseModel):
    """
    保存一个可跨轮恢复的统一等待任务快照。

    参数含义：
        task_id:
            任务创建时生成且生命周期内不变的全局唯一编号。
        task_kind:
            任务所属业务模块。
        status:
            当前生命周期状态。
        user_id、thread_id:
            任务所有者和会话边界，防止跨用户或跨会话恢复。
        title、pending_prompt:
            用户可读的任务名称和当前澄清问题。
        input_contracts:
            当前等待输入的结构化契约列表。
        payload:
            由任务类型判别并严格校验的模块专属恢复数据。
        version:
            每次状态迁移递增的乐观锁版本号。
        created_at、updated_at:
            ISO 8601 格式的创建和更新时间。

    返回值含义：
        PendingTaskSnapshot:
            可序列化进 DogState 和 Checkpoint 的统一任务快照。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task_id: str = Field(min_length=1, max_length=200)
    task_kind: PendingTaskType
    status: PendingTaskStatus = "awaiting_input"
    user_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    pending_prompt: str = Field(min_length=1, max_length=2_000)
    input_contracts: list[PendingInputContract] = Field(min_length=1)
    payload: PendingTaskPayload
    version: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=lambda: _utc_now_iso())
    updated_at: str = Field(default_factory=lambda: _utc_now_iso())

    @field_validator("task_id", "user_id", "thread_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """
        校验关键标识不包含空白字符。

        参数含义：
            value:
                待校验的任务、用户或会话编号。

        返回值含义：
            str:
                合法的原始编号。
        """

        if any(character.isspace() for character in value):
            raise ValueError("任务关键标识不能包含空白字符")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone_timestamp(cls, value: str) -> str:
        """
        校验任务时间是包含时区的 ISO 8601 字符串。

        参数含义：
            value:
                待校验的创建或更新时间文本。

        返回值含义：
            str:
                格式合法且包含时区的原始时间字符串。
        """

        try:
            parsed_value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("任务时间必须使用 ISO 8601 格式") from exc
        if parsed_value.tzinfo is None:
            raise ValueError("任务时间必须包含时区")
        return value

    @model_validator(mode="after")
    def validate_payload_matches_task_kind(self) -> "PendingTaskSnapshot":
        """
        校验公共任务类型与专属载荷类型一致。

        参数含义：
            无，读取当前模型字段。

        返回值含义：
            PendingTaskSnapshot:
                类型一致时返回当前模型；不一致时抛出校验错误。
        """

        if self.payload.payload_kind != self.task_kind:
            raise ValueError("task_kind 与 payload_kind 必须一致")
        return self


_ALLOWED_TRANSITIONS: dict[PendingTaskStatus, set[PendingTaskStatus]] = {
    "awaiting_input": {"running", "cancelled", "expired"},
    "running": {"awaiting_input", "completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
    "expired": set(),
}
_ACTIVE_STATUSES: set[PendingTaskStatus] = {
    "awaiting_input",
    "running",
}
_TERMINAL_STATUSES: set[PendingTaskStatus] = {
    "completed",
    "cancelled",
    "expired",
}


class PendingTaskCollection:
    """
    统一管理一个会话中的等待任务快照。

    功能：
        使用真正唯一的 task_id 注册、查询、排序和迁移任务状态；集合内部
        保存 Pydantic 模型，写入 DogState 前转换为普通 JSON 兼容字典。

    参数含义：
        tasks:
            可选的首次注册任务集合；其中每个任务都必须处于
            awaiting_input 状态。Checkpoint 恢复应使用 from_state。

    返回值含义：
        PendingTaskCollection:
            当前用户会话的一组活动等待任务。
    """

    def __init__(
        self,
        tasks: Iterable[PendingTaskSnapshot] | None = None,
    ) -> None:
        self._tasks: dict[str, PendingTaskSnapshot] = {}
        for task in tasks or []:
            self.register(task)

    @classmethod
    def from_state(
        cls,
        raw_tasks: Mapping[str, Any] | None,
    ) -> "PendingTaskCollection":
        """
        从 DogState 或 Checkpoint 普通字典恢复活动任务集合。

        参数含义：
            raw_tasks:
                以 task_id 为键的原始任务字典；为空时创建空集合。

        返回值含义：
            PendingTaskCollection:
                完成 Schema 校验并裁剪终态任务后的活动集合。
        """

        collection = cls()
        if raw_tasks is None:
            return collection
        for raw_task_id, raw_task in raw_tasks.items():
            task = PendingTaskSnapshot.model_validate(raw_task)
            if str(raw_task_id) != task.task_id:
                raise ValueError("pending_tasks 字典键必须与 task_id 一致")
            if task.status in _TERMINAL_STATUSES:
                continue
            collection._restore_active_task(task)
        return collection

    def register(self, task: PendingTaskSnapshot) -> None:
        """
        注册一个首次进入等待用户输入状态的统一任务。

        参数含义：
            task:
                已通过 Schema 校验的任务快照。

        返回值含义：
            None:
                注册成功后不返回额外数据；状态不是 awaiting_input 或编号
                重复时抛出明确异常。
        """

        if task.status != "awaiting_input":
            raise InvalidPendingTaskRegistrationError(
                "新等待任务必须以 awaiting_input 状态注册"
            )
        if task.task_id in self._tasks:
            raise DuplicatePendingTaskError(
                f"等待任务已经注册: {task.task_id}"
            )
        self._tasks[task.task_id] = task

    def _restore_active_task(self, task: PendingTaskSnapshot) -> None:
        """
        把 Checkpoint 中已存在的活动任务恢复进当前集合。

        参数含义：
            task:
                已通过 Schema 校验且状态为 awaiting_input 或 running 的任务。

        返回值含义：
            None:
                恢复成功后不返回额外数据。
        """

        if task.status not in _ACTIVE_STATUSES:
            raise InvalidPendingTaskRegistrationError(
                f"不能把终态任务恢复进活动集合: {task.status}"
            )
        if task.task_id in self._tasks:
            raise DuplicatePendingTaskError(
                f"等待任务已经存在: {task.task_id}"
            )
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> PendingTaskSnapshot | None:
        """
        根据任务编号安全查询任务。

        参数含义：
            task_id:
                需要查询的统一任务编号。

        返回值含义：
            PendingTaskSnapshot | None:
                找到时返回任务快照，找不到时返回 None。
        """

        return self._tasks.get(str(task_id or "").strip())

    def refresh_waiting_task(
        self,
        task: PendingTaskSnapshot,
        *,
        expected_version: int,
    ) -> PendingTaskSnapshot:
        """
        刷新等待任务的恢复 Payload，并递增乐观锁版本。

        功能：
            当用户只补充了部分参数、任务仍处于 awaiting_input 时，用最新
            业务快照替换旧 Payload。该方法不允许改变 task_id、task_kind
            或任务状态，避免绕过 transition 状态机。

        参数含义：
            task:
                包含最新等待提示、输入契约和恢复 Payload 的任务快照。
            expected_version:
                调用方读取旧任务时看到的版本号。

        返回值含义：
            PendingTaskSnapshot:
                内容已刷新、版本递增且保留原创建时间的新任务快照。
        """

        current_task = self.require(task.task_id)
        if current_task.version != expected_version:
            raise PendingTaskVersionConflictError(
                "任务版本冲突: "
                f"expected={expected_version}, actual={current_task.version}"
            )
        if current_task.status != "awaiting_input":
            raise InvalidPendingTaskTransitionError(
                "只有 awaiting_input 任务可以刷新恢复快照"
            )
        if task.status != "awaiting_input":
            raise InvalidPendingTaskRegistrationError(
                "刷新后的任务必须保持 awaiting_input 状态"
            )
        if task.task_kind != current_task.task_kind:
            raise ValueError("刷新任务时不能改变 task_kind")
        if (
            task.user_id != current_task.user_id
            or task.thread_id != current_task.thread_id
        ):
            raise ValueError("刷新任务时不能改变用户或会话边界")

        refreshed_task = task.model_copy(
            update={
                "version": current_task.version + 1,
                "created_at": current_task.created_at,
                "updated_at": _utc_now_iso(),
            }
        )
        self._tasks[task.task_id] = refreshed_task
        return refreshed_task

    def require(self, task_id: str) -> PendingTaskSnapshot:
        """
        读取必须存在的任务。

        参数含义：
            task_id:
                必须存在的统一任务编号。

        返回值含义：
            PendingTaskSnapshot:
                对应任务；不存在时抛出 PendingTaskNotFoundError。
        """

        task = self.get(task_id)
        if task is None:
            raise PendingTaskNotFoundError(f"等待任务不存在: {task_id}")
        return task

    def list_tasks(
        self,
        *,
        status: PendingTaskStatus | None = None,
    ) -> list[PendingTaskSnapshot]:
        """
        按创建时间和任务编号稳定列出任务。

        参数含义：
            status:
                可选的状态过滤条件；为空时返回全部任务。

        返回值含义：
            list[PendingTaskSnapshot]:
                排序稳定的任务快照列表。
        """

        return sorted(
            (
                task
                for task in self._tasks.values()
                if status is None or task.status == status
            ),
            key=lambda task: (task.created_at, task.task_id),
        )

    def transition(
        self,
        *,
        task_id: str,
        target_status: PendingTaskStatus,
        expected_version: int | None = None,
    ) -> PendingTaskSnapshot:
        """
        按状态机和可选版本条件迁移任务。

        参数含义：
            task_id:
                需要迁移的统一任务编号。
            target_status:
                目标任务状态。
            expected_version:
                调用方读取任务时看到的版本；不一致时拒绝覆盖新状态。

        返回值含义：
            PendingTaskSnapshot:
                版本加一并更新时间后的新任务快照。迁移到终态时，该快照
                会返回给调用方用于审计，同时任务会从活动集合自动移除。
        """

        task = self.require(task_id)
        if expected_version is not None and task.version != expected_version:
            raise PendingTaskVersionConflictError(
                f"任务版本冲突: expected={expected_version}, actual={task.version}"
            )
        if target_status not in _ALLOWED_TRANSITIONS[task.status]:
            raise InvalidPendingTaskTransitionError(
                f"不允许的任务状态迁移: {task.status} -> {target_status}"
            )
        updated_task = task.model_copy(
            update={
                "status": target_status,
                "version": task.version + 1,
                "updated_at": _utc_now_iso(),
            }
        )
        if target_status in _TERMINAL_STATUSES:
            self._tasks.pop(task.task_id, None)
        else:
            self._tasks[task.task_id] = updated_task
        return updated_task

    def to_state(self) -> dict[str, dict[str, Any]]:
        """
        把活动任务集合转换成 Checkpoint 友好的普通字典。

        参数含义：
            无。

        返回值含义：
            dict[str, dict[str, Any]]:
                以 task_id 为键且不包含 Pydantic 对象的 JSON 兼容字典。
        """

        return {
            task.task_id: task.model_dump(mode="json")
            for task in self.list_tasks()
        }


def build_pending_task_id(
    task_kind: PendingTaskType,
    *,
    token_factory: Callable[[], str] | None = None,
) -> str:
    """
    为新等待任务生成生命周期内稳定的全局唯一编号。

    参数含义：
        task_kind:
            任务类型，用于形成可读前缀。
        token_factory:
            可选的随机令牌生成函数，主要用于测试注入固定值。

    返回值含义：
        str:
            形如 pending_tool_<uuid> 的统一任务编号。
    """

    if task_kind not in {"tool", "skill", "multi_agent"}:
        raise ValueError(f"不支持的等待任务类型: {task_kind}")
    raw_token = (
        token_factory()
        if token_factory is not None
        else uuid4().hex
    )
    normalized_token = str(raw_token or "").strip().replace("-", "")
    if not normalized_token or any(
        character.isspace() for character in normalized_token
    ):
        raise ValueError("任务编号随机令牌不能为空或包含空白字符")
    if not normalized_token.isalnum():
        raise ValueError("任务编号随机令牌只能包含字母和数字")
    return f"pending_{task_kind}_{normalized_token}"


def _utc_now_iso() -> str:
    """
    生成 UTC 时区的 ISO 8601 时间字符串。

    参数含义：
        无。

    返回值含义：
        str:
            可直接写入 JSON 和 Checkpoint 的 UTC 时间。
    """

    return datetime.now(timezone.utc).isoformat()
