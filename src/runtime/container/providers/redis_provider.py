"""Redis 异步客户端 Provider。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis

from src.logger import logger
from src.settings.redis import RedisSettings


RedisClientFactory = Callable[..., Any]


class RedisProvider:
    """
    创建并管理应用共享的 Redis 异步客户端。

    功能：
        在 Container 启动时创建连接池并执行 PING 健康检查，在 Container
        关闭时释放客户端。Provider 不承载队列、缓存或租约业务逻辑。

    参数含义：
        settings:
            Redis 开关、连接地址、超时和连接池配置。
        client_factory:
            可选客户端工厂，单元测试可注入 Fake Client；生产默认使用
            ``Redis.from_url``。

    返回值含义：
        RedisProvider:
            可由 RuntimeContainer 管理生命周期的 Redis 服务提供者。
    """

    def __init__(
        self,
        settings: RedisSettings,
        *,
        client_factory: RedisClientFactory | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or Redis.from_url
        self._client: Any | None = None

    @property
    def enabled(self) -> bool:
        """
        返回当前配置是否启用 Redis。

        返回值含义：
            bool:
                ``REDIS_ENABLED`` 为真时返回 True，否则返回 False。
        """

        return self._settings.enabled

    @property
    def client(self) -> Any:
        """
        返回已经完成启动和健康检查的 Redis 客户端。

        返回值含义：
            Any:
                可执行异步 Redis 命令的共享客户端。

        异常：
            RuntimeError:
                Redis 未启用或 Provider 尚未成功启动时抛出。
        """

        if not self.enabled:
            raise RuntimeError("Redis 当前未启用")
        if self._client is None:
            raise RuntimeError("RedisProvider 尚未启动")
        return self._client

    async def startup(self) -> None:
        """
        创建共享 Redis 客户端并执行 PING 健康检查。

        功能：
            Redis 未启用时安全跳过；启用时按照配置创建连接池，连接失败
            则立即释放客户端并抛出明确异常，避免应用带病启动。

        返回值含义：
            None。
        """

        if not self.enabled:
            logger.info("Redis 未启用，跳过客户端启动")
            return
        if self._client is not None:
            logger.warning("RedisProvider 已启动")
            return

        client = self._client_factory(
            self._settings.url,
            decode_responses=True,
            socket_connect_timeout=(
                self._settings.connect_timeout_seconds
            ),
            socket_timeout=self._settings.socket_timeout_seconds,
            health_check_interval=(
                self._settings.health_check_interval_seconds
            ),
            max_connections=self._settings.max_connections,
        )
        try:
            ping_result = await client.ping()
            if ping_result is not True:
                raise RuntimeError("Redis PING 未返回 True")
        except Exception as exc:
            await client.aclose()
            raise RuntimeError("Redis 启动健康检查失败") from exc

        self._client = client
        logger.info("RedisProvider 启动完成")

    async def shutdown(self) -> None:
        """
        关闭 Redis 客户端及其连接池。

        返回值含义：
            None。Provider 未启动时保持幂等，不执行额外操作。
        """

        client = self._client
        self._client = None
        if client is None:
            return
        await client.aclose()
        logger.info("RedisProvider 已关闭")
