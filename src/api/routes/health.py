from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="检查 API 进程是否存活",
)
async def health() -> HealthResponse:
    """
    返回 API 进程存活状态。

    功能：
        只要 FastAPI 仍能处理请求就返回成功，不检查 LLM、向量库等依赖。

    参数含义：
        无。

    返回值含义：
        HealthResponse:
            status=ok 的存活检查结果。
    """

    return HealthResponse(
        status="ok",
        service="dog-agent-api",
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="检查 Agent 运行依赖是否就绪",
)
async def ready(request: Request) -> HealthResponse:
    """
    返回 RuntimeContainer 是否已经完成启动。

    功能：
        区分“HTTP 进程活着”和“Agent 依赖可以接收业务流量”。容器尚未完成
        startup 时返回 HTTP 503。

    参数含义：
        request:
            FastAPI 当前请求，用于读取 app.state.ready。

    返回值含义：
        HealthResponse:
            status=ready 的就绪检查结果。
    """

    if not bool(getattr(request.app.state, "ready", False)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent Runtime 尚未就绪",
        )
    return HealthResponse(
        status="ready",
        service="dog-agent-api",
    )
