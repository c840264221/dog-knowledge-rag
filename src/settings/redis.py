"""Redis 运行时配置。"""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from src.settings.base import BaseAppSettings


class RedisSettings(BaseAppSettings):
    """
    管理 Redis 客户端连接与连接池配置。

    功能：
        从 ``REDIS_`` 前缀环境变量读取是否启用、连接地址、超时和连接池
        上限，并在应用启动前拒绝不受支持的连接协议。

    参数含义：
        无。字段值通过默认值、初始化参数或环境变量提供。

    返回值含义：
        RedisSettings:
            已完成类型转换和连接地址校验的 Redis 配置对象。
    """

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    socket_timeout_seconds: float = Field(default=5.0, gt=0)
    health_check_interval_seconds: int = Field(default=30, ge=0)
    max_connections: int = Field(default=20, ge=1)

    @field_validator("url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        """
        校验并规范 Redis 连接地址。

        功能：
            只允许 redis 或 rediss 协议，并要求地址包含主机名，避免配置
            错误延迟到 Provider 启动时才暴露。

        参数含义：
            cls:
                当前 RedisSettings 类型，由 Pydantic 自动传入。
            value:
                待校验的 Redis URL。

        返回值含义：
            str:
                去除首尾空白后的合法 Redis URL。
        """

        normalized_value = str(value or "").strip()
        parsed_url = urlsplit(normalized_value)
        if parsed_url.scheme not in {"redis", "rediss"}:
            raise ValueError("Redis URL 只允许 redis 或 rediss 协议")
        if not parsed_url.hostname:
            raise ValueError("Redis URL 必须包含主机名")
        return normalized_value
