from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import SettingsConfigDict

from src.settings.base import BaseAppSettings


ApiEnvironment = Literal[
    "development",
    "test",
    "production",
]
ApiLogLevel = Literal[
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]


class ApiSettings(BaseAppSettings):
    """
    管理 FastAPI 和 Uvicorn 的启动配置。

    功能：
        从 API_ 前缀的环境变量读取监听地址、端口、运行环境和日志选项，
        管理可选 API Key 认证，并在服务启动前拒绝当前架构尚不支持的
        多进程配置或不完整安全配置。

    参数含义：
        无。字段值通过默认值、初始化参数或环境变量提供。

    返回值含义：
        ApiSettings:
            已完成类型转换和部署约束校验的 API 配置对象。
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    environment: ApiEnvironment = "development"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = False
    log_level: ApiLogLevel = "info"
    access_log: bool = True
    auth_enabled: bool = False
    auth_key: SecretStr = SecretStr("")
    cors_enabled: bool = False
    cors_allowed_origins: list[str] = Field(default_factory=list)
    cors_allow_credentials: bool = False
    max_request_body_bytes: int = Field(
        default=65_536,
        ge=1_024,
        le=10 * 1_024 * 1_024,
    )
    rate_limit_enabled: bool = False
    rate_limit_requests: int = Field(default=60, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3_600,
    )

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_allowed_origins(
        cls,
        origins: list[str],
    ) -> list[str]:
        """
        规范并校验浏览器跨域来源白名单。

        功能：
            去除空白和末尾斜杠、保持声明顺序去重，并拒绝通配符、路径、
            查询参数以及非 HTTP(S) 来源，避免把 URL 和 Origin 混为一谈。

        参数含义：
            cls:
                当前 ApiSettings 类型，由 Pydantic 自动传入。
            origins:
                配置中声明的浏览器来源列表。

        返回值含义：
            list[str]:
                已规范化且可交给 CORSMiddleware 的精确来源白名单。
        """

        normalized_origins: list[str] = []
        for raw_origin in origins:
            origin = str(raw_origin).strip().rstrip("/")
            if not origin:
                continue
            if origin == "*":
                raise ValueError(
                    "API_CORS_ALLOWED_ORIGINS 不允许使用通配符 *"
                )
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme not in {"http", "https"}
                or not parsed_origin.netloc
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
            ):
                raise ValueError(
                    "CORS 来源必须是仅包含协议、主机和可选端口的 "
                    f"HTTP(S) Origin: {origin}"
                )
            if origin not in normalized_origins:
                normalized_origins.append(origin)
        return normalized_origins

    @model_validator(mode="after")
    def validate_deployment_mode(self) -> ApiSettings:
        """
        校验当前 API 架构支持的部署模式。

        功能：
            阻止多 Worker 导致进程内任务注册表和取消令牌彼此隔离，并阻止
            生产环境启用仅用于开发热更新的 reload。

        参数含义：
            self:
                当前已经完成字段解析的 API 配置对象。

        返回值含义：
            ApiSettings:
                部署模式合法时返回当前配置；不合法时由 Pydantic 抛出
                ValidationError。
        """

        if self.workers != 1:
            raise ValueError(
                "当前 ApiTaskRegistry 和取消令牌只支持单进程，"
                "API_WORKERS 必须为 1"
            )
        if self.environment == "production" and self.reload:
            raise ValueError("生产环境不能启用 API_RELOAD")
        if (
            self.auth_enabled
            and not self.auth_key.get_secret_value().strip()
        ):
            raise ValueError(
                "启用 API Key 认证时必须配置 API_AUTH_KEY"
            )
        if self.cors_enabled and not self.cors_allowed_origins:
            raise ValueError(
                "启用 CORS 时必须配置 API_CORS_ALLOWED_ORIGINS"
            )
        return self
