"""RedisProvider 生命周期单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from src.runtime.container.providers.redis_provider import RedisProvider
from src.settings.redis import RedisSettings


class FakeRedisClient:
    """记录 PING 与关闭调用的测试 Redis 客户端。"""

    def __init__(
        self,
        *,
        ping_result: bool = True,
        ping_error: Exception | None = None,
    ) -> None:
        self.ping_result = ping_result
        self.ping_error = ping_error
        self.ping_count = 0
        self.close_count = 0

    async def ping(self) -> bool:
        """
        返回预设 PING 结果或抛出预设异常。

        返回值含义：
            bool:
                测试配置的健康检查结果。
        """

        self.ping_count += 1
        if self.ping_error is not None:
            raise self.ping_error
        return self.ping_result

    async def aclose(self) -> None:
        """
        记录异步关闭调用。

        返回值含义：
            None。
        """

        self.close_count += 1


class RecordingRedisFactory:
    """记录 Provider 传入连接参数的测试客户端工厂。"""

    def __init__(self, client: FakeRedisClient) -> None:
        self.client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeRedisClient:
        """
        记录连接地址和选项并返回预设客户端。

        参数含义：
            url:
                Provider 请求连接的 Redis URL。
            kwargs:
                Provider 传入的连接池、超时和解码选项。

        返回值含义：
            FakeRedisClient:
                测试共享客户端。
        """

        self.calls.append((url, kwargs))
        return self.client


@pytest.mark.asyncio
async def test_disabled_redis_provider_should_skip_client_creation() -> None:
    """验证 Redis 关闭时 Provider 不创建网络客户端。"""

    fake_client = FakeRedisClient()
    factory = RecordingRedisFactory(fake_client)
    provider = RedisProvider(
        RedisSettings(enabled=False, _env_file=None),
        client_factory=factory,
    )

    await provider.startup()
    await provider.shutdown()

    assert provider.enabled is False
    assert factory.calls == []
    assert fake_client.ping_count == 0
    assert fake_client.close_count == 0
    with pytest.raises(RuntimeError, match="未启用"):
        _ = provider.client


@pytest.mark.asyncio
async def test_enabled_redis_provider_should_ping_and_close_client() -> None:
    """验证启用后创建共享客户端、执行 PING 并在关闭时释放连接池。"""

    fake_client = FakeRedisClient()
    factory = RecordingRedisFactory(fake_client)
    settings = RedisSettings(
        enabled=True,
        url="redis://redis:6379/2",
        connect_timeout_seconds=2,
        socket_timeout_seconds=3,
        health_check_interval_seconds=15,
        max_connections=12,
        _env_file=None,
    )
    provider = RedisProvider(settings, client_factory=factory)

    await provider.startup()

    assert provider.client is fake_client
    assert fake_client.ping_count == 1
    assert factory.calls == [
        (
            "redis://redis:6379/2",
            {
                "decode_responses": True,
                "socket_connect_timeout": 2.0,
                "socket_timeout": 3.0,
                "health_check_interval": 15,
                "max_connections": 12,
            },
        )
    ]

    await provider.shutdown()

    assert fake_client.close_count == 1
    with pytest.raises(RuntimeError, match="尚未启动"):
        _ = provider.client


@pytest.mark.asyncio
async def test_redis_provider_should_close_failed_health_check_client() -> None:
    """验证 Redis PING 失败时释放半初始化客户端并拒绝启动。"""

    fake_client = FakeRedisClient(
        ping_error=ConnectionError("redis unavailable")
    )
    provider = RedisProvider(
        RedisSettings(enabled=True, _env_file=None),
        client_factory=RecordingRedisFactory(fake_client),
    )

    with pytest.raises(RuntimeError, match="健康检查失败"):
        await provider.startup()

    assert fake_client.ping_count == 1
    assert fake_client.close_count == 1
    with pytest.raises(RuntimeError, match="尚未启动"):
        _ = provider.client


@pytest.mark.asyncio
async def test_redis_provider_lifecycle_should_be_idempotent() -> None:
    """验证重复启动和重复关闭不会重复创建或释放客户端。"""

    fake_client = FakeRedisClient()
    factory = RecordingRedisFactory(fake_client)
    provider = RedisProvider(
        RedisSettings(enabled=True, _env_file=None),
        client_factory=factory,
    )

    await provider.startup()
    await provider.startup()
    await provider.shutdown()
    await provider.shutdown()

    assert len(factory.calls) == 1
    assert fake_client.ping_count == 1
    assert fake_client.close_count == 1
