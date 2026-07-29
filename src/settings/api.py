from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
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
        并在服务启动前拒绝当前架构尚不支持的多进程配置。

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
        return self
