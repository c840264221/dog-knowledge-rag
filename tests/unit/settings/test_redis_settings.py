"""Redis 配置单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.redis import RedisSettings


def test_redis_settings_should_use_safe_local_defaults() -> None:
    """
    验证未配置 Redis 时默认关闭，并使用本机开发连接参数。

    返回值含义：
        None。
    """

    settings = RedisSettings(_env_file=None)

    assert settings.enabled is False
    assert settings.url == "redis://localhost:6379/0"
    assert settings.connect_timeout_seconds == 5.0
    assert settings.socket_timeout_seconds == 5.0
    assert settings.health_check_interval_seconds == 30
    assert settings.max_connections == 20


def test_redis_settings_should_read_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 ``REDIS_`` 前缀环境变量可以覆盖默认配置。

    参数含义：
        monkeypatch:
            pytest 提供的临时环境变量修改工具。

    返回值含义：
        None。
    """

    monkeypatch.setenv("REDIS_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/2")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "40")

    settings = RedisSettings(_env_file=None)

    assert settings.enabled is True
    assert settings.url == "redis://redis:6379/2"
    assert settings.max_connections == 40


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http://localhost:6379/0",
        "redis:///0",
        "",
    ],
)
def test_redis_settings_should_reject_invalid_url(
    invalid_url: str,
) -> None:
    """
    验证非法协议、缺失主机或空 Redis URL 会被配置层拒绝。

    参数含义：
        invalid_url:
            当前需要验证的非法连接地址。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="Redis URL"):
        RedisSettings(url=invalid_url, _env_file=None)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("connect_timeout_seconds", 0),
        ("socket_timeout_seconds", 0),
        ("health_check_interval_seconds", -1),
        ("max_connections", 0),
    ],
)
def test_redis_settings_should_reject_invalid_limits(
    field_name: str,
    invalid_value: int,
) -> None:
    """
    验证 Redis 超时、健康检查间隔和连接池上限不能越界。

    参数含义：
        field_name:
            当前需要覆盖的配置字段名称。
        invalid_value:
            不符合字段约束的测试值。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        RedisSettings(
            **{field_name: invalid_value},
            _env_file=None,
        )
