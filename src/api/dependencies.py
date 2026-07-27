from fastapi import Request

from src.api.services import AgentApiService


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
