from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.errors.base import DogAgentError
from tests.unit.api.test_app import (
    FakeAgentApiService,
    FakeRuntimeContainer,
)


def build_error_test_app() -> FastAPI:
    """
    创建包含测试异常路由的 API 应用。

    参数含义：
        无。

    返回值含义：
        FastAPI:
            使用假容器和假 Agent 服务、可触发不同异常的测试应用。
    """

    app = create_app(
        runtime_container=FakeRuntimeContainer(),
        agent_api_service=FakeAgentApiService(),
    )

    @app.get("/test/recoverable-error")
    async def raise_recoverable_error() -> None:
        """抛出可恢复 DogAgentError。"""

        raise DogAgentError("用户输入无法处理", recoverable=True)

    @app.get("/test/internal-error")
    async def raise_internal_error() -> None:
        """抛出不应向调用方暴露原文的未知异常。"""

        raise RuntimeError("数据库密码=should-not-leak")

    return app


def test_validation_error_should_use_unified_contract() -> None:
    """测试 Pydantic 参数错误使用统一响应且不回显原始输入。"""

    app = build_error_test_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            headers={"X-Trace-ID": "trace-validation"},
            json={
                "question": "   ",
                "session_id": "session_001",
            },
        )

    body = response.json()
    assert response.status_code == 422
    assert response.headers["X-Trace-ID"] == "trace-validation"
    assert body["status"] == "error"
    assert body["trace_id"] == "trace-validation"
    assert body["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert body["error"]["details"]
    assert "input" not in body["error"]["details"][0]


def test_http_404_should_use_unified_contract() -> None:
    """测试任务不存在错误使用统一 404 契约。"""

    app = build_error_test_app()
    with TestClient(app) as client:
        response = client.get(
            "/v1/tasks/missing",
            headers={"X-Trace-ID": "trace-missing"},
        )

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert body["trace_id"] == "trace-missing"


def test_recoverable_agent_error_should_keep_public_message() -> None:
    """测试可恢复 Agent 异常可以向调用方返回公开说明。"""

    app = build_error_test_app()
    with TestClient(app) as client:
        response = client.get(
            "/test/recoverable-error",
            headers={"X-Trace-ID": "trace-agent-error"},
        )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "DOG_AGENT_REQUEST_FAILED"
    assert body["error"]["message"] == "用户输入无法处理"


def test_unexpected_error_should_hide_internal_message() -> None:
    """测试未知异常返回 500，但不会泄露内部异常原文。"""

    app = build_error_test_app()
    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get(
            "/test/internal-error",
            headers={"X-Trace-ID": "trace-internal"},
        )

    body = response.json()
    assert response.status_code == 500
    assert response.headers["X-Trace-ID"] == "trace-internal"
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert body["error"]["message"] == "服务器内部错误"
    assert "should-not-leak" not in response.text


def test_body_trace_id_should_be_used_by_response_header() -> None:
    """测试请求体 trace_id 会同步到 API 响应头。"""

    app = build_error_test_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "question": "介绍金毛",
                "session_id": "session_body_trace",
                "trace_id": "trace-from-body",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Trace-ID"] == "trace-from-body"
    assert response.json()["trace_id"] == "trace-from-body"
