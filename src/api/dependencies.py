import secrets
from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.api.services import AgentApiService
from src.settings.api import ApiSettings


API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="Dog Agent API 业务接口访问密钥。",
)


def get_agent_api_service(request: Request) -> AgentApiService:
    """
    从 FastAPI 应用状态中获取 Agent API 服务。

    功能：
        让路由函数通过 Dependency Injection（依赖注入）取得业务服务，
        避免路由内部直接导入全局 Container。

    参数含义：
        request:
            FastAPI 当前 HTTP 请求对象。

    返回值含义：
        AgentApiService:
            应用启动时已经装配好的 Agent API 业务服务。
    """

    return request.app.state.agent_api_service


def require_api_key(
    request: Request,
    provided_api_key: Annotated[
        str | None,
        Security(api_key_header),
    ] = None,
) -> None:
    """
    校验受保护业务接口携带的 API Key。

    功能：
        认证未启用时保持现有调用方式；启用后从 X-API-Key 请求头读取密钥，
        使用恒定时间比较验证调用方身份，失败时阻止请求进入 Agent 主图。

    参数含义：
        request:
            FastAPI 当前请求，用于读取应用启动时注入的 ApiSettings。
        provided_api_key:
            FastAPI 从 X-API-Key 请求头提取的可选调用方密钥。

    返回值含义：
        None:
            认证关闭或密钥正确时不返回业务数据；认证失败时抛出 HTTP 401。
    """

    api_settings: ApiSettings = request.app.state.api_settings
    if not api_settings.auth_enabled:
        return

    expected_api_key = api_settings.auth_key.get_secret_value()
    if not secrets.compare_digest(
        provided_api_key or "",
        expected_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 缺失或无效",
            headers={
                "WWW-Authenticate": (
                    f'ApiKey header="{API_KEY_HEADER_NAME}"'
                ),
            },
        )
