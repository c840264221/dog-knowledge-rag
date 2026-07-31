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
from src.settings.api import ApiSettings


class SmokeRuntimeContainer:
    """
    为 API 契约冒烟测试提供最小生命周期容器。

    功能：
        记录 FastAPI lifespan 是否正确调用 startup 和 shutdown，不启动真实
        LLM、向量库、Checkpoint 或主图运行时。

    参数含义：
        无。

    返回值含义：
        SmokeRuntimeContainer:
            可注入 create_app 的确定性测试容器。
    """

    def __init__(self) -> None:
        self.startup_count = 0
        self.shutdown_count = 0

    async def startup(self) -> None:
        """记录一次 API 启动阶段调用。"""

        self.startup_count += 1

    async def shutdown(self) -> None:
        """记录一次 API 关闭阶段调用。"""

        self.shutdown_count += 1


class SmokeAgentApiService:
    """
    为完整 HTTP 路由提供确定性的 Agent 服务响应。

    功能：
        模拟完成、等待输入、SSE、取消和状态查询结果，让冒烟测试聚焦
        FastAPI 对外契约，不受模型和外部基础设施波动影响。

    参数含义：
        无。

    返回值含义：
        SmokeAgentApiService:
            可注入 create_app 的 API 服务替身。
    """

    async def chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> GraphRunResponse:
        """返回一条确定性的业务完成响应。"""

        return GraphRunResponse(
            status="completed",
            business_status="completed",
            answer=f"{question}：冒烟测试完成。",
            session_id=session_id,
            thread_id=session_id,
            trace_id=trace_id,
            multi_agent_task_id=f"multi_agent_task_{trace_id}",
            checkpoint_ns="default",
        )

    async def resume(
        self,
        *,
        resume_value: str,
        session_id: str,
        trace_id: str,
    ) -> GraphRunResponse:
        """返回一条确定性的再次等待用户输入响应。"""

        return GraphRunResponse(
            status="interrupted",
            business_status="awaiting_input",
            prompt=f"已收到{resume_value}，请继续补充体重。",
            session_id=session_id,
            thread_id=session_id,
            trace_id=trace_id,
            multi_agent_task_id=f"multi_agent_task_{trace_id}",
            checkpoint_ns="default",
            interrupt_type="user_clarification",
        )

    async def stream_chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """依次产出 SSE 接受事件和完成事件。"""

        task_id = f"multi_agent_task_{trace_id}"
        yield {
            "event": "accepted",
            "data": {
                "session_id": session_id,
                "trace_id": trace_id,
                "multi_agent_task_id": task_id,
            },
        }
        yield {
            "event": "completed",
            "data": {
                "status": "completed",
                "business_status": "completed",
                "answer": f"{question}：流式冒烟测试完成。",
                "session_id": session_id,
                "thread_id": session_id,
                "trace_id": trace_id,
                "multi_agent_task_id": task_id,
                "checkpoint_ns": "default",
                "metadata": {},
            },
        }

    def cancel(self, multi_agent_task_id: str) -> CancellationResponse:
        """返回已经发送取消信号的确定性结果。"""

        return CancellationResponse(
            multi_agent_task_id=multi_agent_task_id,
            cancellation_requested=True,
            message="已发送取消请求。",
        )

    def get_task_status(
        self,
        multi_agent_task_id: str,
    ) -> TaskStatusResponse | None:
        """为已知任务返回完成状态，其他任务返回 None。"""

        if multi_agent_task_id != "multi_agent_task_smoke":
            return None
        return TaskStatusResponse(
            multi_agent_task_id=multi_agent_task_id,
            trace_id="smoke",
            session_id="session_smoke",
            status="completed",
            business_status="completed",
            created_at="2026-07-27T00:00:00+00:00",
            updated_at="2026-07-27T00:00:01+00:00",
        )


def test_api_openapi_and_route_contract_smoke() -> None:
    """
    验证完整 FastAPI 应用的主要对外契约。

    功能：
        启动真实应用生命周期，检查健康接口、OpenAPI Schema、同步对话、
        恢复、SSE、取消和状态查询端点可以通过统一 HTTP 契约协同工作。

    参数含义：
        无。

    返回值含义：
        None。
    """

    container = SmokeRuntimeContainer()
    app = create_app(
        runtime_container=container,
        agent_api_service=SmokeAgentApiService(),
        api_settings=ApiSettings(
            auth_enabled=False,
            cors_enabled=False,
            rate_limit_enabled=False,
        ),
    )

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").json()["status"] == "ready"

        openapi = client.get("/openapi.json").json()
        expected_paths = {
            "/health",
            "/ready",
            "/v1/chat",
            "/v1/chat/stream",
            "/v1/chat/resume",
            "/v1/multi-agent/tasks/{multi_agent_task_id}/cancel",
            "/v1/tasks/{multi_agent_task_id}",
        }
        assert expected_paths <= set(openapi["paths"])
        graph_response_schema = openapi["components"]["schemas"][
            "GraphRunResponse"
        ]
        assert "business_status" in graph_response_schema["required"]
        assert "business_error" in graph_response_schema["properties"]
        assert "AgentBusinessError" in openapi["components"]["schemas"]
        assert "APIKeyHeader" in openapi["components"]["securitySchemes"]
        assert openapi["paths"]["/v1/chat"]["post"]["security"] == [
            {
                "APIKeyHeader": [],
            }
        ]

        chat_response = client.post(
            "/v1/chat",
            json={
                "question": "金毛性格怎么样？",
                "session_id": "session_smoke",
                "trace_id": "smoke",
            },
        )
        assert chat_response.status_code == 200
        assert chat_response.headers["X-Trace-ID"] == "smoke"
        assert chat_response.json()["business_status"] == "completed"

        resume_response = client.post(
            "/v1/chat/resume",
            json={
                "resume_value": "6岁",
                "session_id": "session_smoke",
                "trace_id": "smoke_resume",
            },
        )
        assert resume_response.status_code == 200
        assert resume_response.json()["status"] == "interrupted"
        assert (
            resume_response.json()["business_status"]
            == "awaiting_input"
        )

        stream_response = client.post(
            "/v1/chat/stream",
            json={
                "question": "生成综合方案",
                "session_id": "session_stream",
                "trace_id": "smoke_stream",
            },
        )
        assert stream_response.status_code == 200
        assert stream_response.headers["content-type"].startswith(
            "text/event-stream"
        )
        assert "event: accepted" in stream_response.text
        assert "event: completed" in stream_response.text

        cancel_response = client.post(
            "/v1/multi-agent/tasks/multi_agent_task_smoke/cancel"
        )
        assert cancel_response.status_code == 200
        assert cancel_response.json()["cancellation_requested"] is True

        task_response = client.get(
            "/v1/tasks/multi_agent_task_smoke"
        )
        assert task_response.status_code == 200
        assert task_response.json()["business_status"] == "completed"

    assert container.startup_count == 1
    assert container.shutdown_count == 1
