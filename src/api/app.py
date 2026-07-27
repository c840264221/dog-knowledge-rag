from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware import ApiRequestLoggingMiddleware
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.services import AgentApiService
from src.api.task_registry import ApiTaskRegistry
from src.runtime.container.init import container as default_container
from src.settings import settings


def create_app(
    *,
    runtime_container: Any = default_container,
    agent_api_service: AgentApiService | None = None,
) -> FastAPI:
    """
    创建并装配 Dog Agent FastAPI 应用。

    功能：
        绑定 RuntimeContainer 生命周期、创建 AgentApiService、注册健康检查
        与 Agent 路由。测试可以注入替身，避免启动真实模型和数据库。

    参数含义：
        runtime_container:
            管理 LLM、RAG、Checkpoint 和 GraphRuntimeService 的运行时容器。
        agent_api_service:
            可选 API 服务替身；未提供时使用容器中的 graph_runtime 创建真实服务。

    返回值含义：
        FastAPI:
            已注册生命周期和路由的 ASGI 应用对象。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """
        管理 FastAPI 与 RuntimeContainer 的共同生命周期。

        参数含义：
            app:
                当前 FastAPI 应用。

        返回值含义：
            AsyncIterator[None]:
                startup 完成后让出应用运行阶段，退出时执行 shutdown。
        """

        app.state.ready = False
        await runtime_container.startup()
        app.state.agent_api_service = (
            agent_api_service
            or AgentApiService(
                graph_runtime=runtime_container.get("graph_runtime"),
                task_registry=ApiTaskRegistry(),
            )
        )
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await runtime_container.shutdown()

    application = FastAPI(
        title=f"{settings.app.app_name} API",
        version="1.18.0",
        description="Dog Agent Framework 的 HTTP API 服务入口。",
        lifespan=lifespan,
    )
    application.state.ready = False
    application.add_middleware(ApiRequestLoggingMiddleware)
    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(chat_router)
    return application


app = create_app()
