import uvicorn

from src.settings import settings


def main() -> None:
    """
    启动 Dog Agent Framework FastAPI 服务。

    功能：
        使用 Uvicorn（ASGI 服务启动器）加载 src.api.app:app，并在开发模式
        下启动 HTTP 服务。

    参数含义：
        无。

    返回值含义：
        None:
            当前进程会持续运行，直到用户停止服务。
    """

    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
        access_log=settings.api.access_log,
    )


if __name__ == "__main__":
    main()
