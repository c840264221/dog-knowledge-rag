from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.schemas import (
    CancellationResponse,
    GraphRunResponse,
    TaskStatusResponse,
)


class FakeRuntimeContainer:
    """
    记录 FastAPI 是否正确调用容器生命周期。

    功能：
        替代真实 RuntimeContainer，避免单元测试启动模型、向量库和检查点。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContainer:
            可记录 startup 和 shutdown 调用次数的测试替身。
    """

    def __init__(self) -> None:
        self.startup_calls = 0
        self.shutdown_calls = 0

    async def startup(self) -> None:
        """记录一次测试容器启动。"""

        self.startup_calls += 1

    async def shutdown(self) -> None:
        """记录一次测试容器关闭。"""

        self.shutdown_calls += 1


class FakeAgentApiService:
    """
    为 API 路由提供确定性响应。

    功能：
        记录路由传入的参数，并返回固定完成、恢复和取消结果。

    参数含义：
        无。

    返回值含义：
        FakeAgentApiService:
            不依赖真实 Agent 主图的测试服务。
    """

    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []

    async def chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str | None,
    ) -> GraphRunResponse:
        """记录新对话参数并返回固定完成结果。"""

        self.chat_calls.append(
            {
                "question": question,
                "session_id": session_id,
                "trace_id": trace_id,
            }
        )
        return GraphRunResponse(
            status="completed",
            business_status="completed",
            answer="固定测试回答",
            session_id=session_id,
            thread_id=session_id,
            trace_id=trace_id or "generated_trace",
            multi_agent_task_id="multi_agent_task_test",
            checkpoint_ns="main_graph",
        )

    async def resume(
        self,
        *,
        resume_value: str,
        session_id: str,
        trace_id: str,
    ) -> GraphRunResponse:
        """记录恢复参数并返回固定中断结果。"""

        self.resume_calls.append(
            {
                "resume_value": resume_value,
                "session_id": session_id,
                "trace_id": trace_id,
            }
        )
        return GraphRunResponse(
            status="interrupted",
            business_status="awaiting_input",
            prompt="请继续补充体重",
            session_id=session_id,
            thread_id=session_id,
            trace_id=trace_id,
            multi_agent_task_id="multi_agent_task_test",
            checkpoint_ns="main_graph",
            interrupt_type="user_clarification",
        )

    def cancel(self, multi_agent_task_id: str) -> CancellationResponse:
        """记录取消参数并返回固定成功结果。"""

        self.cancel_calls.append(multi_agent_task_id)
        return CancellationResponse(
            multi_agent_task_id=multi_agent_task_id,
            cancellation_requested=True,
            message="已发送取消请求，正在停止未完成步骤。",
        )

    async def stream_chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """按顺序产出固定 accepted 和 completed 事件。"""

        yield {
            "event": "accepted",
            "data": {
                "status": "running",
                "session_id": session_id,
                "trace_id": trace_id or "generated_trace",
                "multi_agent_task_id": "multi_agent_task_test",
            },
        }
        yield {
            "event": "completed",
            "data": {
                "status": "completed",
                "business_status": "completed",
                "answer": f"{question}：固定流式回答",
            },
        }

    def get_task_status(
        self,
        multi_agent_task_id: str,
    ) -> TaskStatusResponse | None:
        """为已知任务返回固定完成状态。"""

        if multi_agent_task_id != "multi_agent_task_test":
            return None
        return TaskStatusResponse(
            multi_agent_task_id=multi_agent_task_id,
            trace_id="trace_test",
            session_id="session_test",
            status="completed",
            business_status="completed",
            created_at="2026-07-27T00:00:00+00:00",
            updated_at="2026-07-27T00:00:01+00:00",
        )


def build_test_client() -> tuple[
    TestClient,
    FakeRuntimeContainer,
    FakeAgentApiService,
]:
    """
    创建使用确定性替身的 FastAPI 测试客户端。

    参数含义：
        无。

    返回值含义：
        tuple:
            TestClient、假容器和假 API 服务。
    """

    container = FakeRuntimeContainer()
    service = FakeAgentApiService()
    app = create_app(
        runtime_container=container,
        agent_api_service=service,
    )
    return TestClient(app), container, service


def test_app_lifespan_and_health_endpoints() -> None:
    """测试 API 生命周期、存活检查和就绪检查。"""

    client, container, _ = build_test_client()
    assert container.startup_calls == 0

    with client:
        assert container.startup_calls == 1
        health_response = client.get("/health")
        ready_response = client.get("/ready")

        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"
        assert ready_response.status_code == 200
        assert ready_response.json()["status"] == "ready"

    assert container.shutdown_calls == 1


def test_chat_route_should_forward_validated_request() -> None:
    """测试新对话路由会把校验后的参数交给 API 服务。"""

    client, _, service = build_test_client()
    with client:
        response = client.post(
            "/v1/chat",
            json={
                "question": "金毛适合新手吗？",
                "session_id": "session_001",
                "trace_id": "trace_001",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["answer"] == "固定测试回答"
    assert service.chat_calls == [
        {
            "question": "金毛适合新手吗？",
            "session_id": "session_001",
            "trace_id": "trace_001",
        }
    ]


def test_chat_route_should_reject_blank_question() -> None:
    """测试空白问题会在进入 Agent 主图前被 Pydantic 拒绝。"""

    client, _, service = build_test_client()
    with client:
        response = client.post(
            "/v1/chat",
            json={
                "question": "   ",
                "session_id": "session_001",
            },
        )

    assert response.status_code == 422
    assert service.chat_calls == []


def test_resume_route_should_keep_original_identity() -> None:
    """测试恢复路由会保留原 session_id 和 trace_id。"""

    client, _, service = build_test_client()
    with client:
        response = client.post(
            "/v1/chat/resume",
            json={
                "resume_value": "6岁",
                "session_id": "session_resume",
                "trace_id": "trace_resume",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "interrupted"
    assert response.json()["prompt"] == "请继续补充体重"
    assert service.resume_calls[0]["session_id"] == "session_resume"
    assert service.resume_calls[0]["trace_id"] == "trace_resume"


def test_cancel_route_should_forward_multi_agent_task_id() -> None:
    """测试取消路由会把任务编号转交给取消服务。"""

    client, _, service = build_test_client()
    task_id = "multi_agent_task_trace_001"
    with client:
        response = client.post(
            f"/v1/multi-agent/tasks/{task_id}/cancel"
        )

    assert response.status_code == 200
    assert response.json()["cancellation_requested"] is True
    assert service.cancel_calls == [task_id]


def test_stream_route_should_return_sse_events() -> None:
    """测试流式路由会返回符合 SSE 格式的生命周期事件。"""

    client, _, _ = build_test_client()
    with client:
        response = client.post(
            "/v1/chat/stream",
            json={
                "question": "介绍一下金毛",
                "session_id": "session_stream",
                "trace_id": "trace_stream",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert "event: accepted" in response.text
    assert "event: completed" in response.text
    assert "multi_agent_task_test" in response.text


def test_task_status_route_should_return_status_or_404() -> None:
    """测试任务状态路由会返回快照，并为未知任务返回 404。"""

    client, _, _ = build_test_client()
    with client:
        found = client.get("/v1/tasks/multi_agent_task_test")
        missing = client.get("/v1/tasks/missing")

    assert found.status_code == 200
    assert found.json()["status"] == "completed"
    assert missing.status_code == 404
