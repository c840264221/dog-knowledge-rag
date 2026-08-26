from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware import (
    ApiRateLimitMiddleware,
    ApiRequestBodyLimitMiddleware,
    ApiRequestLoggingMiddleware,
)
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.long_tasks import router as long_tasks_router
from src.api.services import AgentApiService
from src.api.long_task_services import LongTaskApiQueryService
from src.api.task_registry import ApiTaskRegistry
from src.runtime.container.init import container as default_container
from src.runtime.long_tasks.store import RedisLongTaskStore
from src.runtime.long_tasks.application_service import (
    LongTaskApplicationService,
)
from src.runtime.long_tasks.stream import RedisLongTaskStream
from src.settings import settings
from src.settings.api import ApiSettings


def create_app(
    *,
    runtime_container: Any = default_container,
    agent_api_service: AgentApiService | None = None,
    long_task_query_service: LongTaskApiQueryService | None = None,
    long_task_application_service: LongTaskApplicationService | None = None,
    api_settings: ApiSettings | None = None,
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
        long_task_query_service:
            可选长任务查询服务替身；未提供时尝试复用容器中的 RedisProvider
            构建真实只读服务，Redis 未启用时保持不可用。
        long_task_application_service:
            可选长任务命令服务替身；未提供时复用 RedisProvider 构建带 Store
            和 Stream Publisher 的真实应用服务。
        api_settings:
            可选 API 配置；未提供时使用全局 settings.api。测试可以注入开启
            或关闭认证的确定性配置。

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
        app.state.long_task_query_service = (
            long_task_query_service
            or _build_long_task_query_service(runtime_container)
        )
        app.state.long_task_application_service = (
            long_task_application_service
            or _build_long_task_application_service(runtime_container)
        )
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await runtime_container.shutdown()

    resolved_api_settings = api_settings or settings.api
    application = FastAPI(
        title=f"{settings.app.app_name} API",
        version="1.20.0",
        description="Dog Agent Framework 的 HTTP API 服务入口。",
        lifespan=lifespan,
    )
    application.state.ready = False
    application.state.api_settings = resolved_api_settings
    application.add_middleware(
        ApiRequestBodyLimitMiddleware,
        max_body_bytes=(
            resolved_api_settings.max_request_body_bytes
        ),
    )
    if resolved_api_settings.rate_limit_enabled:
        application.add_middleware(
            ApiRateLimitMiddleware,
            request_limit=(
                resolved_api_settings.rate_limit_requests
            ),
            window_seconds=(
                resolved_api_settings.rate_limit_window_seconds
            ),
            trusted_proxy_cidrs=(
                resolved_api_settings.trusted_proxy_cidrs
            ),
        )
    if resolved_api_settings.cors_enabled:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=(
                resolved_api_settings.cors_allowed_origins
            ),
            allow_credentials=(
                resolved_api_settings.cors_allow_credentials
            ),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=[
                "Accept",
                "Content-Type",
                "X-API-Key",
                "X-Trace-ID",
            ],
            expose_headers=[
                "Retry-After",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-Trace-ID",
            ],
            max_age=600,
        )
    application.add_middleware(ApiRequestLoggingMiddleware)
    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(long_tasks_router)
    return application


def _build_long_task_query_service(
    runtime_container: Any,
) -> LongTaskApiQueryService | None:
    """
    使用容器中已经启动的 RedisProvider 装配长任务查询服务。

    参数含义：
        runtime_container:
            当前 FastAPI 应用共享的运行时容器或测试替身。

    返回值含义：
        LongTaskApiQueryService | None:
            Redis 已注册且启用时返回真实查询服务；测试替身没有 Redis 或
            Redis 配置关闭时返回 None，由依赖层对查询请求返回 HTTP 503。
    """

    try:
        redis_provider = runtime_container.get("redis")
    except (AttributeError, ValueError):
        return None
    if not redis_provider.enabled:
        return None
    return LongTaskApiQueryService(
        RedisLongTaskStore(redis_provider.client)
    )


def _build_long_task_application_service(
    runtime_container: Any,
) -> LongTaskApplicationService | None:
    """
    使用容器中已经启动的 RedisProvider 装配长任务命令服务。

    参数含义：
        runtime_container:
            当前 FastAPI 应用共享的运行时容器或测试替身。

    返回值含义：
        LongTaskApplicationService | None:
            Redis 已启用时返回同时持有 Store 和 Stream Publisher 的应用
            服务；依赖不可用时返回 None，由 API 依赖层返回 HTTP 503。
    """

    try:
        redis_provider = runtime_container.get("redis")
    except (AttributeError, ValueError):
        return None
    if not redis_provider.enabled:
        return None
    redis_client = redis_provider.client
    return LongTaskApplicationService(
        RedisLongTaskStore(redis_client),
        queue_publisher=RedisLongTaskStream(redis_client),
    )


app = create_app()
