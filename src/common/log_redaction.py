from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REDACTED_VALUE = "[REDACTED]"

_SENSITIVE_KEY_NAMES = {
    "authorization",
    "proxy_authorization",
    "x_api_key",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "cookie",
    "set_cookie",
    "credential",
    "credentials",
    "密码",
    "密钥",
    "令牌",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_refresh_token",
    "_password",
    "_passwd",
    "_secret",
)
_TEXT_KEY_PATTERN = (
    r"authorization|proxy[-_ ]authorization|x[-_ ]api[-_ ]key|"
    r"api[-_ ]?key|access[-_ ]token|refresh[-_ ]token|id[-_ ]token|"
    r"password|passwd|client[-_ ]secret|secret|set[-_ ]cookie|cookie|"
    r"credentials?|密码|密钥|令牌"
)
_DOUBLE_QUOTED_JSON_PATTERN = re.compile(
    rf'(?i)("(?:{_TEXT_KEY_PATTERN})"\s*:\s*")[^"]*(")'
)
_SINGLE_QUOTED_MAPPING_PATTERN = re.compile(
    rf"(?i)('(?:{_TEXT_KEY_PATTERN})'\s*:\s*')[^']*(')"
)
_QUOTED_ASSIGNMENT_PATTERN = re.compile(
    rf"(?i)((?<![\w])(?:{_TEXT_KEY_PATTERN})(?![\w])"
    rf"\s*[:=]\s*)([\"'])[^\"']*([\"'])"
)
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)((?:authorization|proxy[-_ ]authorization)"
    r"\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_PATTERN = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"
)
_CHINESE_ASSIGNMENT_PATTERN = re.compile(
    r"((?:[\u4e00-\u9fff]{0,8})?(?:密码|密钥|令牌)"
    r"\s*[:=]\s*)(?!\[REDACTED\])[^\s,;&}\]]+"
)
_UNQUOTED_ASSIGNMENT_PATTERN = re.compile(
    rf"(?i)((?<![\w])(?:{_TEXT_KEY_PATTERN})(?![\w])"
    rf"\s*[:=]\s*)(?!\[REDACTED\])[^\s,;&}}\]]+"
)


def redact_log_record(record: dict[str, Any]) -> None:
    """
    在 Loguru 写入任意 Sink 前统一清洗日志记录。

    功能：
        脱敏 message、extra 和异常原文中的 API Key、Authorization、
        Cookie、密码及 Token，同时保留日志结构和 trace_id。异常原文包含
        敏感值时移除该条 traceback，避免源码行或局部变量再次泄漏。

    参数含义：
        record:
            Loguru patch 回调提供的可变日志记录字典。

    返回值含义：
        None:
            直接原地修改日志记录，不创建新的日志事件。
    """

    record["message"] = redact_sensitive_text(
        str(record.get("message") or "")
    )
    record["extra"] = redact_sensitive_value(
        record.get("extra", {})
    )

    exception = record.get("exception")
    if exception is None:
        return
    exception_value = getattr(exception, "value", None)
    sanitized_message = redact_sensitive_text(
        str(exception_value or "")
    )
    if sanitized_message == str(exception_value or ""):
        return

    exception_type = getattr(exception, "type", type(exception_value))
    exception_type_name = getattr(
        exception_type,
        "__name__",
        "Exception",
    )
    sanitized_exception = RuntimeError(
        f"{exception_type_name}: {sanitized_message}"
    )
    record["exception"] = type(exception)(
        RuntimeError,
        sanitized_exception,
        None,
    )


def redact_sensitive_value(
    value: Any,
    *,
    key: str | None = None,
) -> Any:
    """
    递归清洗结构化日志值中的敏感字段。

    参数含义：
        value:
            字符串、字典、列表、元组或其他准备写入日志的值。
        key:
            当前值在父级映射中的字段名称。

    返回值含义：
        Any:
            保持原有主要容器结构、敏感内容替换为固定掩码后的值。
    """

    if key is not None and _is_sensitive_key(key):
        return REDACTED_VALUE
    if isinstance(value, Mapping):
        return {
            item_key: redact_sensitive_value(
                item_value,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            redact_sensitive_value(item)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            redact_sensitive_value(item)
            for item in value
        )
    if isinstance(value, set):
        return {
            redact_sensitive_value(item)
            for item in value
        }
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def redact_sensitive_text(text: str) -> str:
    """
    清洗已格式化日志文本中的常见密钥表达形式。

    参数含义：
        text:
            可能包含 JSON、请求头或 key=value 片段的日志文本。

    返回值含义：
        str:
            敏感值已替换为 [REDACTED] 的安全日志文本。
    """

    redacted_text = _DOUBLE_QUOTED_JSON_PATTERN.sub(
        rf"\1{REDACTED_VALUE}\2",
        text,
    )
    redacted_text = _SINGLE_QUOTED_MAPPING_PATTERN.sub(
        rf"\1{REDACTED_VALUE}\2",
        redacted_text,
    )
    redacted_text = _QUOTED_ASSIGNMENT_PATTERN.sub(
        rf"\1\2{REDACTED_VALUE}\3",
        redacted_text,
    )
    redacted_text = _AUTHORIZATION_PATTERN.sub(
        rf"\1{REDACTED_VALUE}",
        redacted_text,
    )
    redacted_text = _BEARER_PATTERN.sub(
        f"Bearer {REDACTED_VALUE}",
        redacted_text,
    )
    redacted_text = _CHINESE_ASSIGNMENT_PATTERN.sub(
        rf"\1{REDACTED_VALUE}",
        redacted_text,
    )
    return _UNQUOTED_ASSIGNMENT_PATTERN.sub(
        rf"\1{REDACTED_VALUE}",
        redacted_text,
    )


def _is_sensitive_key(key: str) -> bool:
    """
    判断结构化日志字段名是否表示敏感凭据。

    参数含义：
        key:
            当前结构化日志字段名称。

    返回值含义：
        bool:
            字段应整体脱敏时返回 True。
    """

    normalized_key = re.sub(
        r"[^0-9a-zA-Z\u4e00-\u9fff]+",
        "_",
        key,
    ).strip("_").lower()
    return (
        normalized_key in _SENSITIVE_KEY_NAMES
        or normalized_key.endswith(_SENSITIVE_KEY_SUFFIXES)
    )
