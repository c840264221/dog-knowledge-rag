"""
SQLite 只读 SQL 安全守卫。

功能：
    为 SQLite MCP Readonly Client 提供 SQL 安全校验和 limit 归一化。

设计说明：
    当前 V1.9.4 是 MVP，不引入 SQL parser。
    这里使用保守字符串规则：
    1. 只允许 SELECT 或 WITH 开头。
    2. 禁止明显写操作关键词。
    3. 禁止多语句分号。
    4. 强制由 client 控制 LIMIT。

专业名词：
    Guard：
        守卫，用于在执行前拦截危险输入。
    Readonly SQL：
        只读 SQL，只查询数据，不修改数据库。
"""

from __future__ import annotations

import re


FORBIDDEN_SQL_KEYWORDS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "replace",
    "truncate",
    "update",
    "vacuum",
}

SQL_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


def normalize_sql_limit(
    limit: int | None,
    default_limit: int,
    max_limit: int,
) -> int:
    """
    归一化 SQL 查询行数限制。

    功能：
        将用户传入的 limit 限制在 1 到 max_limit 之间。
        如果 limit 为空或非法，则使用 default_limit。

    参数：
        limit:
            用户传入的行数限制。

        default_limit:
            默认行数限制。

        max_limit:
            最大允许行数。

    返回值：
        int:
            归一化后的行数限制。
    """

    try:
        parsed_limit = int(
            limit
            if limit is not None
            else default_limit
        )
    except (
        TypeError,
        ValueError,
    ):
        parsed_limit = default_limit

    if parsed_limit <= 0:
        parsed_limit = default_limit

    return min(
        parsed_limit,
        max_limit,
    )


def ensure_readonly_sql(
    sql: str,
) -> str:
    """
    校验 SQL 是否为只读查询。

    功能：
        校验 SQL 必须以 SELECT 或 WITH 开头，
        且不能包含写操作关键词和多语句分号。
        校验通过后返回去除前后空白的 SQL。

    参数：
        sql:
            待执行的 SQL。

    返回值：
        str:
            归一化后的只读 SQL。

    异常：
        ValueError:
            SQL 为空、不是只读查询或包含危险关键词时抛出。
    """

    if not isinstance(
        sql,
        str,
    ):
        raise ValueError(
            "SQL 必须是字符串。"
        )

    normalized_sql = sql.strip()

    if not normalized_sql:
        raise ValueError(
            "SQL 不能为空。"
        )

    lowered_sql = normalized_sql.lower()

    if ";" in normalized_sql.rstrip(";"):
        raise ValueError(
            "不允许执行多语句 SQL。"
        )

    lowered_sql = lowered_sql.rstrip(";").strip()

    if not (
        lowered_sql.startswith("select ")
        or lowered_sql.startswith("with ")
    ):
        raise ValueError(
            "SQLite MCP 只允许 SELECT 或 WITH 查询。"
        )

    tokens = set(
        re.findall(
            r"[A-Za-z_]+",
            lowered_sql,
        )
    )
    forbidden_tokens = tokens & FORBIDDEN_SQL_KEYWORDS

    if forbidden_tokens:
        raise ValueError(
            f"SQL 包含禁止关键词: {sorted(forbidden_tokens)}"
        )

    return normalized_sql.rstrip(";").strip()


def ensure_safe_identifier(
    identifier: str,
    field_name: str,
) -> str:
    """
    校验 SQLite 标识符是否安全。

    功能：
        限制 table_name 等标识符只能包含字母、数字和下划线，
        且必须以字母或下划线开头，避免通过表名参数注入 SQL。

    参数：
        identifier:
            待校验的标识符。

        field_name:
            字段名称，用于错误信息。

    返回值：
        str:
            校验后的标识符。

    异常：
        ValueError:
            标识符为空或格式非法时抛出。
    """

    if not isinstance(
        identifier,
        str,
    ):
        raise ValueError(
            f"{field_name} 必须是字符串。"
        )

    normalized_identifier = identifier.strip()

    if not SQL_IDENTIFIER_PATTERN.match(
        normalized_identifier
    ):
        raise ValueError(
            f"{field_name} 只能包含字母、数字和下划线，且不能以数字开头。"
        )

    return normalized_identifier
