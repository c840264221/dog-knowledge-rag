"""V1.22 生产部署环境配置预检测试。"""

from __future__ import annotations

from pathlib import Path

from scripts.deployment.validate_v122_release_env import (
    load_env_file,
    validate_release_environment,
)


def _valid_release_values() -> dict[str, str]:
    """
    构造一组满足生产发布要求的测试配置。

    参数含义：
        无。

    返回值含义：
        dict[str, str]:
            不包含真实凭据、仅供单元测试使用的合格配置。
    """

    return {
        "DOG_AGENT_API_IMAGE": (
            "ghcr.io/example-owner/dog-agent-api:1.22.0"
        ),
        "API_AUTH_ENABLED": "true",
        "API_AUTH_KEY": "a" * 32,
        "API_RATE_LIMIT_ENABLED": "true",
        "DEBUG": "false",
    }


def test_release_environment_should_accept_fixed_secure_configuration() -> None:
    """验证固定版本镜像和最低安全配置能够通过预检。"""

    assert validate_release_environment(_valid_release_values()) == []


def test_release_environment_should_collect_all_unsafe_reasons() -> None:
    """验证预检会一次返回全部问题且不会回显密钥原文。"""

    secret = "too-short"
    errors = validate_release_environment(
        {
            "DOG_AGENT_API_IMAGE": "dog-agent-api:latest",
            "API_AUTH_ENABLED": "false",
            "API_AUTH_KEY": secret,
            "API_RATE_LIMIT_ENABLED": "false",
            "DEBUG": "true",
        }
    )

    assert len(errors) == 5
    assert any("固定版本号" in error for error in errors)
    assert any("API_AUTH_ENABLED" in error for error in errors)
    assert any("API_AUTH_KEY" in error for error in errors)
    assert any("API_RATE_LIMIT_ENABLED" in error for error in errors)
    assert any("DEBUG" in error for error in errors)
    assert all(secret not in error for error in errors)


def test_release_environment_should_allow_versioned_rollback_override() -> None:
    """验证回滚镜像可以覆盖环境文件中的当前发布镜像。"""

    values = _valid_release_values()
    values["DOG_AGENT_API_IMAGE"] = "invalid"

    errors = validate_release_environment(
        values,
        image_override="ghcr.io/example-owner/dog-agent-api:1.21.0",
    )

    assert errors == []


def test_load_env_file_should_support_quotes_export_and_comments(
    tmp_path: Path,
) -> None:
    """验证环境文件读取器兼容常见 dotenv 写法。"""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# production\n"
        "export API_AUTH_ENABLED=true\n"
        'API_AUTH_KEY="secret # remains inside quotes"\n'
        "DEBUG=false # disable debug\n",
        encoding="utf-8",
    )

    values = load_env_file(env_file)

    assert values == {
        "API_AUTH_ENABLED": "true",
        "API_AUTH_KEY": "secret # remains inside quotes",
        "DEBUG": "false",
    }
