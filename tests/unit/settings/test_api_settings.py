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
    "API_AUTH_ENABLED",
    "API_AUTH_KEY",
    "API_CORS_ENABLED",
    "API_CORS_ALLOWED_ORIGINS",
    "API_CORS_ALLOW_CREDENTIALS",
    "API_MAX_REQUEST_BODY_BYTES",
    "API_RATE_LIMIT_ENABLED",
    "API_RATE_LIMIT_REQUESTS",
    "API_RATE_LIMIT_WINDOW_SECONDS",
    "API_TRUSTED_PROXY_CIDRS",
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
    assert api_settings.auth_enabled is False
    assert api_settings.auth_key.get_secret_value() == ""
    assert api_settings.cors_enabled is False
    assert api_settings.cors_allowed_origins == []
    assert api_settings.cors_allow_credentials is False
    assert api_settings.max_request_body_bytes == 65_536
    assert api_settings.rate_limit_enabled is False
    assert api_settings.rate_limit_requests == 60
    assert api_settings.rate_limit_window_seconds == 60
    assert api_settings.trusted_proxy_cidrs == []


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
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_AUTH_KEY", "test-secret")
    monkeypatch.setenv("API_CORS_ENABLED", "true")
    monkeypatch.setenv(
        "API_CORS_ALLOWED_ORIGINS",
        '["http://localhost:3000","https://app.example.com/"]',
    )
    monkeypatch.setenv("API_CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("API_MAX_REQUEST_BODY_BYTES", "131072")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_REQUESTS", "120")
    monkeypatch.setenv("API_RATE_LIMIT_WINDOW_SECONDS", "30")
    monkeypatch.setenv(
        "API_TRUSTED_PROXY_CIDRS",
        '["127.0.0.1","10.0.0.23/24","127.0.0.1/32"]',
    )

    api_settings = ApiSettings()

    assert api_settings.environment == "test"
    assert api_settings.host == "0.0.0.0"
    assert api_settings.port == 9000
    assert api_settings.access_log is False
    assert api_settings.auth_enabled is True
    assert (
        api_settings.auth_key.get_secret_value()
        == "test-secret"
    )
    assert api_settings.cors_enabled is True
    assert api_settings.cors_allowed_origins == [
        "http://localhost:3000",
        "https://app.example.com",
    ]
    assert api_settings.cors_allow_credentials is True
    assert api_settings.max_request_body_bytes == 131_072
    assert api_settings.rate_limit_enabled is True
    assert api_settings.rate_limit_requests == 120
    assert api_settings.rate_limit_window_seconds == 30
    assert api_settings.trusted_proxy_cidrs == [
        "127.0.0.1/32",
        "10.0.0.0/24",
    ]


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


def test_api_settings_should_require_key_when_auth_is_enabled() -> None:
    """
    验证开启 API Key 认证时不能遗漏实际密钥。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="必须配置 API_AUTH_KEY",
    ):
        ApiSettings(
            auth_enabled=True,
            auth_key="   ",
        )


def test_api_settings_should_hide_secret_key_in_representation() -> None:
    """
    验证 ApiSettings 的文本表示不会泄露真实 API Key。

    参数含义：
        无。

    返回值含义：
        None。
    """

    api_settings = ApiSettings(
        auth_enabled=True,
        auth_key="do-not-print-this-key",
    )

    assert "do-not-print-this-key" not in repr(api_settings)


def test_api_settings_should_require_origins_when_cors_enabled() -> None:
    """
    验证开启 CORS 时必须声明至少一个可信浏览器来源。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="必须配置 API_CORS_ALLOWED_ORIGINS",
    ):
        ApiSettings(cors_enabled=True)


@pytest.mark.parametrize(
    "invalid_origin",
    [
        "*",
        "localhost:3000",
        "ftp://example.com",
        "https://example.com/app",
        "https://example.com?source=test",
    ],
)
def test_api_settings_should_reject_invalid_cors_origin(
    invalid_origin: str,
) -> None:
    """
    验证 CORS 白名单拒绝通配符和不是标准 Origin 的地址。

    参数含义：
        invalid_origin:
            当前测试准备交给 ApiSettings 的非法来源。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        ApiSettings(
            cors_enabled=True,
            cors_allowed_origins=[invalid_origin],
        )


@pytest.mark.parametrize(
    "invalid_max_body_bytes",
    [
        1_023,
        10 * 1_024 * 1_024 + 1,
    ],
)
def test_api_settings_should_reject_unsafe_body_limit(
    invalid_max_body_bytes: int,
) -> None:
    """
    验证请求体上限不能过小或大到失去基础保护意义。

    参数含义：
        invalid_max_body_bytes:
            当前准备校验的非法请求体字节上限。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        ApiSettings(
            max_request_body_bytes=invalid_max_body_bytes,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("rate_limit_requests", 0),
        ("rate_limit_requests", 10_001),
        ("rate_limit_window_seconds", 0),
        ("rate_limit_window_seconds", 3_601),
    ],
)
def test_api_settings_should_reject_invalid_rate_limit(
    field_name: str,
    invalid_value: int,
) -> None:
    """
    验证限流请求数和窗口秒数必须位于安全配置范围内。

    参数含义：
        field_name:
            当前准备覆盖的 ApiSettings 字段名称。
        invalid_value:
            当前字段使用的非法整数值。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        ApiSettings(
            **{
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    "invalid_cidr",
    [
        "not-an-ip",
        "10.0.0.1/99",
        "example.com",
    ],
)
def test_api_settings_should_reject_invalid_trusted_proxy_cidr(
    invalid_cidr: str,
) -> None:
    """
    验证可信代理列表只能包含合法 IP 或 CIDR。

    参数含义：
        invalid_cidr:
            当前准备交给 ApiSettings 的非法代理网络字符串。

    返回值含义：
        None。
    """

    with pytest.raises(
        ValidationError,
        match="API_TRUSTED_PROXY_CIDRS",
    ):
        ApiSettings(trusted_proxy_cidrs=[invalid_cidr])
