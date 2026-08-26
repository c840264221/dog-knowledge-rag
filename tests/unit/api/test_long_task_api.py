"""持久化长任务状态查询 API 单元测试。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.exception_handlers import register_exception_handlers
from src.api.long_task_services import LongTaskApiQueryService
from src.api.routes.long_tasks import (
    _iterate_long_task_status_events,
    router as long_tasks_router,
)
from src.api.schemas import LongTaskStatusResponse
from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskGoal,
    LongTaskPendingInteraction,
    LongTaskStep,
)
from src.settings.api import ApiSettings


class FakeLongTaskStore:
    """
    为查询服务返回一份确定性的 LongTask 快照。

    参数含义：
        task:
            根据 task_id 返回的可选任务。

    返回值含义：
        FakeLongTaskStore:
            只实现本组测试需要的 load 方法。
    """

    def __init__(self, task: LongTask | None) -> None:
        self._task = task

    async def load(self, task_id: str) -> LongTask | None:
        """任务编号匹配时返回测试快照，否则返回 None。"""

        if self._task is None or self._task.task_id != task_id:
            return None
        return self._task


class FakeLongTaskQueryService:
    """记录路由查询参数并返回预设公开响应。"""

    def __init__(
        self,
        response: LongTaskStatusResponse | None,
    ) -> None:
        self._response = response
        self.calls: list[dict[str, str]] = []

    async def get_status(
        self,
        *,
        task_id: str,
        user_id: str,
    ) -> LongTaskStatusResponse | None:
        """记录任务和用户编号，并返回预设查询结果。"""

        self.calls.append(
            {"task_id": task_id, "user_id": user_id}
        )
        return self._response


class SequenceLongTaskQueryService:
    """按调用顺序返回多份状态，用于模拟 SSE 轮询期间的版本变化。"""

    def __init__(
        self,
        responses: list[LongTaskStatusResponse | None],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def get_status(
        self,
        *,
        task_id: str,
        user_id: str,
    ) -> LongTaskStatusResponse | None:
        """记录查询并返回下一份预设状态。"""

        self.calls.append(
            {"task_id": task_id, "user_id": user_id}
        )
        if not self._responses:
            raise AssertionError("SSE 测试状态已经耗尽")
        return self._responses.pop(0)


class FakeLongTaskApplicationService:
    """记录长任务交互命令并返回预设的最新任务快照。"""

    def __init__(self, task: LongTask) -> None:
        self._task = task
        self.calls: list[dict[str, Any]] = []

    async def respond_to_missing_input(self, **kwargs: Any) -> LongTask:
        """记录路由传入的命令参数并返回预设任务。"""

        self.calls.append(dict(kwargs))
        return self._task


def _build_waiting_task() -> LongTask:
    """
    构造包含完成步骤和等待输入步骤的测试长任务。

    返回值含义：
        LongTask:
            可用于验证进度、公开字段过滤和待交互响应的任务快照。
    """

    task_id = "long_task_001"
    return LongTask(
        task_id=task_id,
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="请制定完整养犬计划，并包含私人备注。",
            objective="制定完整养犬计划",
            expected_output="计划文档",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id=task_id,
                title="整理基础资料",
                assigned_agent="dog_knowledge_agent",
                status="completed",
                output_summary="基础资料已整理。",
                attempt_count=1,
                input_data={"private_note": "不能公开"},
                metadata={"internal": True},
            ),
            LongTaskStep(
                step_id="step_2",
                task_id=task_id,
                title="确认每日运动时间",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="awaiting_input",
                waiting_reason="missing_input",
                attempt_count=1,
            ),
        ],
        status="awaiting_input",
        execution_mode="durable",
        pending_interaction=LongTaskPendingInteraction(
            interaction_id="interaction_001",
            interaction_type="missing_input",
            source_step_ids=["step_2"],
            target_step_ids=["step_2"],
            prompt="每天可以运动多长时间？",
            allowed_actions=["submit_input", "cancel"],
            input_contract={
                "items": [
                    {
                        "step_id": "step_2",
                        "prompt": "每天可以运动多长时间？",
                    }
                ]
            },
            metadata={"batch_id": "internal_batch"},
        ),
        active_step_ids=["step_2"],
        version=4,
        metadata={"internal_trace": "secret"},
    )


def _build_route_app(
    service: Any,
    *,
    application_service: Any | None = None,
) -> FastAPI:
    """
    创建只包含长任务路由的轻量 FastAPI 测试应用。

    参数含义：
        service:
            注入应用状态的查询服务或测试替身。
        application_service:
            可选长任务交互命令服务替身。

    返回值含义：
        FastAPI:
            已关闭 API Key 认证并注册统一异常处理器的测试应用。
    """

    app = FastAPI()
    app.state.api_settings = ApiSettings(
        auth_enabled=False,
        cors_enabled=False,
        rate_limit_enabled=False,
    )
    app.state.long_task_query_service = service
    app.state.long_task_application_service = application_service
    register_exception_handlers(app)
    app.include_router(long_tasks_router)
    return app


def _build_public_status(
    *,
    status: str,
    version: int,
) -> LongTaskStatusResponse:
    """
    构造指定 Task 状态和版本的公开 SSE 测试快照。

    参数含义：
        status:
            当前测试需要模拟的长任务状态。
        version:
            当前测试需要模拟的权威快照版本。

    返回值含义：
        LongTaskStatusResponse:
            字段完整、可以直接进入 SSE 事件的数据对象。
    """

    return LongTaskStatusResponse(
        task_id="long_task_001",
        thread_id="thread_001",
        objective="制定完整养犬计划",
        status=status,
        execution_mode="durable",
        progression_mode="automatic",
        version=version,
        total_step_count=2,
        completed_step_count=(2 if status == "completed" else 1),
        steps=[],
        created_at="2026-08-22T00:00:00+00:00",
        updated_at=f"2026-08-22T00:00:0{version}+00:00",
    )


async def test_query_service_should_filter_internal_task_fields() -> None:
    """验证查询服务校验用户归属并只返回前端安全字段。"""

    service = LongTaskApiQueryService(
        FakeLongTaskStore(_build_waiting_task())
    )

    response = await service.get_status(
        task_id="long_task_001",
        user_id="user_001",
    )

    assert response is not None
    assert response.status == "awaiting_input"
    assert response.completed_step_count == 1
    assert response.total_step_count == 2
    assert response.pending_interaction is not None
    assert response.pending_interaction.interaction_id == (
        "interaction_001"
    )
    public_data = response.model_dump(mode="json")
    assert "original_request" not in public_data
    assert "active_step_ids" not in public_data
    assert "private_note" not in str(public_data)
    assert "internal_batch" not in str(public_data)
    assert "claim_id" not in str(public_data)


async def test_query_service_should_hide_other_users_task() -> None:
    """验证用户编号不匹配时与任务不存在一样返回 None。"""

    service = LongTaskApiQueryService(
        FakeLongTaskStore(_build_waiting_task())
    )

    response = await service.get_status(
        task_id="long_task_001",
        user_id="user_002",
    )

    assert response is None


def test_long_task_route_should_return_status_and_forward_owner() -> None:
    """验证 HTTP 路由转发 task_id、user_id 并返回公开状态。"""

    task = _build_waiting_task()
    public_response = LongTaskStatusResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        objective=task.goal.objective,
        status=task.status,
        execution_mode=task.execution_mode,
        progression_mode=task.progression_mode,
        version=task.version,
        total_step_count=2,
        completed_step_count=1,
        steps=[],
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )
    service = FakeLongTaskQueryService(public_response)

    with TestClient(_build_route_app(service)) as client:
        response = client.get(
            "/v1/long-tasks/long_task_001",
            params={"user_id": "user_001"},
        )

    assert response.status_code == 200
    assert response.json()["task_id"] == "long_task_001"
    assert service.calls == [
        {"task_id": "long_task_001", "user_id": "user_001"}
    ]


def test_long_task_route_should_return_404_for_hidden_task() -> None:
    """验证任务不存在或归属不匹配时路由统一返回 HTTP 404。"""

    service = FakeLongTaskQueryService(None)

    with TestClient(_build_route_app(service)) as client:
        response = client.get(
            "/v1/long-tasks/long_task_001",
            params={"user_id": "user_002"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_long_task_router_should_not_expose_direct_submission() -> None:
    """验证普通用户不能绕过统一 Agent 入口直接创建 LongTask。"""

    app = _build_route_app(FakeLongTaskQueryService(None))

    direct_submission_routes = [
        route
        for route in app.routes
        if (
            getattr(route, "path", "") == "/v1/long-tasks"
            and "POST" in (getattr(route, "methods", set()) or set())
        )
    ]

    assert direct_submission_routes == []


def test_long_task_sse_should_emit_snapshot_and_close_for_terminal() -> None:
    """验证终态任务建立 SSE 后发送初始快照并立即正常关闭。"""

    service = FakeLongTaskQueryService(
        _build_public_status(status="completed", version=6)
    )

    with TestClient(_build_route_app(service)) as client:
        response = client.get(
            "/v1/long-tasks/long_task_001/events",
            params={"user_id": "user_001"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert "event: snapshot" in response.text
    assert '"status":"completed"' in response.text
    assert "event: updated" not in response.text


async def test_sse_iterator_should_emit_only_version_changes() -> None:
    """验证 SSE 依次推送初始、更新和终态，不重复推送相同版本。"""

    initial = _build_public_status(status="awaiting_input", version=4)
    service = SequenceLongTaskQueryService(
        [
            _build_public_status(status="awaiting_input", version=4),
            _build_public_status(status="running", version=5),
            _build_public_status(status="completed", version=6),
        ]
    )

    async def is_disconnected() -> bool:
        """模拟测试客户端始终保持连接。"""

        return False

    async def no_wait(_: float) -> None:
        """跳过真实轮询等待，让状态序列立即推进。"""

        return None

    events = [
        event
        async for event in _iterate_long_task_status_events(
            service=service,
            task_id="long_task_001",
            user_id="user_001",
            initial_status=initial,
            is_disconnected=is_disconnected,
            poll_interval_seconds=1,
            heartbeat_interval_seconds=15,
            sleep=no_wait,
        )
    ]

    assert [event["event"] for event in events] == [
        "snapshot",
        "updated",
        "terminal",
    ]
    assert [event["data"]["version"] for event in events] == [
        4,
        5,
        6,
    ]


def test_interaction_response_route_should_forward_structured_answers() -> None:
    """验证交互响应路由转发逐 Step 回答并返回最新任务状态。"""

    task = _build_waiting_task()
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "status": "running",
            "pending_interaction": None,
            "active_step_ids": [],
            "version": task.version + 1,
        }
    )
    step_data = task.steps[1].model_dump(mode="python")
    step_data.update(
        {
            "status": "ready",
            "waiting_reason": None,
            "version": task.steps[1].version + 1,
        }
    )
    task_data["steps"] = [
        task.steps[0],
        LongTaskStep.model_validate(step_data),
    ]
    application_service = FakeLongTaskApplicationService(
        LongTask.model_validate(task_data)
    )
    app = _build_route_app(
        FakeLongTaskQueryService(None),
        application_service=application_service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/long-tasks/long_task_001/interactions/"
            "interaction_001/responses",
            json={
                "user_id": "user_001",
                "action": "submit_input",
                "answers": {"step_2": {"daily_minutes": 60}},
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["steps"][1]["status"] == "ready"
    assert application_service.calls == [
        {
            "task_id": "long_task_001",
            "user_id": "user_001",
            "interaction_id": "interaction_001",
            "action": "submit_input",
            "answers": {"step_2": {"daily_minutes": 60}},
        }
    ]
