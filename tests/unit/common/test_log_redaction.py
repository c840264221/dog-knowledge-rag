from __future__ import annotations

from collections import namedtuple

import pytest

from src.common.log_redaction import (
    REDACTED_VALUE,
    redact_log_record,
    redact_sensitive_text,
    redact_sensitive_value,
)
from src.logger import logger


@pytest.mark.parametrize(
    ("unsafe_text", "secret_value"),
    [
        ("X-API-Key=super-secret-key", "super-secret-key"),
        (
            "Authorization: Bearer header.payload.signature",
            "header.payload.signature",
        ),
        ('{"password": "database-password"}', "database-password"),
        ("{'access_token': 'access-token-value'}", "access-token-value"),
        ("client_secret='client-secret-value'", "client-secret-value"),
        ("请求密钥=chinese-secret-value", "chinese-secret-value"),
    ],
)
def test_redact_sensitive_text_should_hide_common_secret_formats(
    unsafe_text: str,
    secret_value: str,
) -> None:
    """
    验证常见请求头、JSON、映射和赋值文本中的敏感值会被遮盖。

    参数含义：
        unsafe_text:
            当前包含敏感信息的原始日志文本。
        secret_value:
            不允许出现在清洗结果中的原始值。

    返回值含义：
        None。
    """

    redacted_text = redact_sensitive_text(unsafe_text)

    assert secret_value not in redacted_text
    assert REDACTED_VALUE in redacted_text


def test_redact_sensitive_value_should_clean_nested_mapping() -> None:
    """
    验证嵌套结构按字段名脱敏并保留普通可观测指标。

    参数含义：
        无。

    返回值含义：
        None。
    """

    redacted_value = redact_sensitive_value(
        {
            "headers": {
                "X-API-Key": "api-key-value",
                "Authorization": "Bearer token-value",
            },
            "database_password": "database-password",
            "token_count": 123,
            "trace_id": "trace-safe",
        }
    )

    assert redacted_value["headers"]["X-API-Key"] == REDACTED_VALUE
    assert redacted_value["headers"]["Authorization"] == REDACTED_VALUE
    assert redacted_value["database_password"] == REDACTED_VALUE
    assert redacted_value["token_count"] == 123
    assert redacted_value["trace_id"] == "trace-safe"


def test_redact_sensitive_text_should_keep_normal_message() -> None:
    """
    验证不包含凭据的普通日志不会被无意义改写。

    参数含义：
        无。

    返回值含义：
        None。
    """

    message = (
        "API 请求完成: status_code=200 "
        "latency_ms=15.20 token_count=42"
    )

    assert redact_sensitive_text(message) == message


def test_redact_log_record_should_clean_message_extra_and_exception() -> None:
    """
    验证统一日志记录清洗覆盖消息、结构化扩展字段和异常原文。

    参数含义：
        无。

    返回值含义：
        None。
    """

    record_exception_type = namedtuple(
        "RecordException",
        ["type", "value", "traceback"],
    )
    record = {
        "message": "X-API-Key=message-secret",
        "extra": {
            "api_key": "extra-secret",
            "trace_id": "trace-safe",
        },
        "exception": record_exception_type(
            ValueError,
            ValueError("password=exception-secret"),
            None,
        ),
    }

    redact_log_record(record)

    rendered_record = str(record)
    assert "message-secret" not in rendered_record
    assert "extra-secret" not in rendered_record
    assert "exception-secret" not in rendered_record
    assert record["extra"]["trace_id"] == "trace-safe"


def test_project_logger_should_redact_before_writing_sink() -> None:
    """
    验证项目实际 Logger 会在 Sink 接收前清洗消息、extra 和异常。

    参数含义：
        无。

    返回值含义：
        None。
    """

    captured_logs: list[str] = []
    sink_id = logger.add(
        lambda message: captured_logs.append(str(message)),
        format="{message}\n{extra}\n{exception}",
        enqueue=False,
    )
    try:
        try:
            raise ValueError("password=exception-secret")
        except ValueError:
            logger.bind(
                api_key="extra-secret",
            ).exception(
                "Authorization: Bearer message-secret"
            )
    finally:
        logger.remove(sink_id)

    rendered_log = "".join(captured_logs)
    assert REDACTED_VALUE in rendered_log
    assert "message-secret" not in rendered_log
    assert "extra-secret" not in rendered_log
    assert "exception-secret" not in rendered_log
