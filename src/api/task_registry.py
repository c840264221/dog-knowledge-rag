from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from src.api.schemas import TaskStatusResponse


ApiTaskStatus = Literal[
    "running",
    "completed",
    "interrupted",
    "cancel_requested",
    "failed",
]
AgentBusinessStatus = Literal[
    "completed",
    "partial",
    "failed",
    "cancelled",
    "awaiting_input",
]


@dataclass(frozen=True)
class ApiTaskSnapshot:
    """
    保存一条 API 请求在当前进程中的不可变状态快照。

    功能：
        作为 ApiTaskRegistry 的内部数据对象，记录任务身份、生命周期状态
        和时间；每次更新都生成新对象，避免读取方观察到修改一半的数据。

    参数含义：
        multi_agent_task_id:
            当前任务编号。
        trace_id:
            当前链路追踪编号。
        session_id:
            当前会话编号。
        status:
            当前 API 请求生命周期状态。
        created_at:
            首次登记时间。
        updated_at:
            最近更新时间。
        error_message:
            可选错误摘要。
        business_status:
            主图返回结果后的 Agent 业务状态。

    返回值含义：
        ApiTaskSnapshot:
            只在 API 进程内部传递的不可变任务快照。
    """

    multi_agent_task_id: str
    trace_id: str
    session_id: str
    status: ApiTaskStatus
    created_at: str
    updated_at: str
    error_message: str | None = None
    business_status: AgentBusinessStatus | None = None

    def to_response(self) -> TaskStatusResponse:
        """
        将内部任务快照转换成 HTTP 响应模型。

        参数含义：
            无。

        返回值含义：
            TaskStatusResponse:
                可以安全返回给 API 调用方的任务状态。
        """

        return TaskStatusResponse(
            multi_agent_task_id=self.multi_agent_task_id,
            trace_id=self.trace_id,
            session_id=self.session_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            error_message=self.error_message,
            business_status=self.business_status,
        )


class ApiTaskRegistry:
    """
    保存当前 API 进程中的请求状态。

    功能：
        请求开始时登记 running，执行完成、中断、失败或收到取消请求时更新
        状态，并为状态查询接口提供线程安全快照。

    参数含义：
        无。

    返回值含义：
        ApiTaskRegistry:
            单进程任务状态登记表；未来多进程部署时应替换为 Redis 等共享存储。
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ApiTaskSnapshot] = {}
        self._lock = Lock()

    def start(
        self,
        *,
        multi_agent_task_id: str,
        trace_id: str,
        session_id: str,
    ) -> ApiTaskSnapshot:
        """
        登记或重新启动一条 API 请求。

        参数含义：
            multi_agent_task_id:
                当前任务编号。
            trace_id:
                当前链路追踪编号。
            session_id:
                当前会话编号。

        返回值含义：
            ApiTaskSnapshot:
                status=running 的最新任务快照。
        """

        now = _utc_now()
        with self._lock:
            previous = self._tasks.get(multi_agent_task_id)
            snapshot = ApiTaskSnapshot(
                multi_agent_task_id=multi_agent_task_id,
                trace_id=trace_id,
                session_id=session_id,
                status="running",
                created_at=(
                    previous.created_at
                    if previous is not None
                    else now
                ),
                updated_at=now,
            )
            self._tasks[multi_agent_task_id] = snapshot
            return snapshot

    def update(
        self,
        multi_agent_task_id: str,
        *,
        status: ApiTaskStatus,
        error_message: str | None = None,
        business_status: AgentBusinessStatus | None = None,
    ) -> ApiTaskSnapshot | None:
        """
        更新一条已登记任务的生命周期状态。

        参数含义：
            multi_agent_task_id:
                需要更新的任务编号。
            status:
                新的 API 请求生命周期状态。
            error_message:
                执行失败时可选的错误摘要。
            business_status:
                主图返回结果后的 Agent 业务状态；不传时保留原值。

        返回值含义：
            ApiTaskSnapshot | None:
                找到任务时返回更新后的快照，否则返回 None。
        """

        with self._lock:
            current = self._tasks.get(multi_agent_task_id)
            if current is None:
                return None
            updated = replace(
                current,
                status=status,
                updated_at=_utc_now(),
                error_message=error_message,
                business_status=(
                    current.business_status
                    if business_status is None
                    else business_status
                ),
            )
            self._tasks[multi_agent_task_id] = updated
            return updated

    def get(self, multi_agent_task_id: str) -> ApiTaskSnapshot | None:
        """
        获取指定任务的当前状态快照。

        参数含义：
            multi_agent_task_id:
                需要查询的任务编号。

        返回值含义：
            ApiTaskSnapshot | None:
                找到时返回不可变快照，否则返回 None。
        """

        with self._lock:
            return self._tasks.get(multi_agent_task_id)


def _utc_now() -> str:
    """
    获取 UTC ISO 8601 时间字符串。

    参数含义：
        无。

    返回值含义：
        str:
            带 UTC 时区偏移的当前时间。
    """

    return datetime.now(timezone.utc).isoformat()
