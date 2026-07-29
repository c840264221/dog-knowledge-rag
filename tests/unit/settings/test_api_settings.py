"""
API Settings 测试。

功能：
    验证 Uvicorn 启动参数默认值、环境变量覆盖和单进程部署约束。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.api import ApiSettings


API_ENVIRONMENT_VARIABLES = [
    "API_ENVIRONMENT",
    "API_HOST",
    "API_PORT",
    "API_WORKERS",
    "API_RELOAD",
    "API_LOG_LEVEL",
    "API_ACCESS_LOG",
]


def _clear_api_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    清除可能影响 API Settings 测试的环境变量。

    参数含义：
        monkeypatch:
            pytest 提供的临时环境变量修改工具。

    返回值含义：
        None:
            只清理当前测试进程环境，不返回业务数据。
    """

    for variable_name in API_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def test_api_settings_should_keep_local_development_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证默认配置继续兼容原有本机启动方式。

    参数含义：
        monkeypatch:
            pytest 提供的临时环境变量修改工具。

    返回值含义：
        None。
    """

    _clear_api_environment(monkeypatch)

    api_settings = ApiSettings()

    assert api_settings.environment == "development"
    assert api_settings.host == "127.0.0.1"
    assert api_settings.port == 8000
    assert api_settings.workers == 1
    assert api_settings.reload is False
    assert api_settings.log_level == "info"
    assert api_settings.access_log is True


def test_api_settings_should_read_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 API_ 环境变量可以覆盖启动参数并完成类型转换。

    参数含义：
        monkeypatch:
            pytest 提供的临时环境变量修改工具。

    返回值含义：
        None。
    """

    _clear_api_environment(monkeypatch)
    monkeypatch.setenv("API_ENVIRONMENT", "test")
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_ACCESS_LOG", "false")

    api_settings = ApiSettings()

    assert api_settings.environment == "test"
    assert api_settings.host == "0.0.0.0"
    assert api_settings.port == 9000
    assert api_settings.access_log is False


def test_api_settings_should_reject_multiple_workers() -> None:
    """
    验证进程内任务注册表尚未共享时不能启动多个 Worker。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="API_WORKERS 必须为 1",
    ):
        ApiSettings(workers=2)


def test_api_settings_should_reject_reload_in_production() -> None:
    """
    验证生产环境不能启用开发热更新。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="生产环境不能启用 API_RELOAD",
    ):
        ApiSettings(
            environment="production",
            reload=True,
        )
