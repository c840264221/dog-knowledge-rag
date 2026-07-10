"""
SQLite MCP 只读守卫测试。

功能：
    验证 readonly_guard 能限制危险 SQL，并归一化查询 limit。
"""

from __future__ import annotations

import pytest

from src.mcp.sqlite.readonly_guard import (
    ensure_readonly_sql,
    ensure_safe_identifier,
    normalize_sql_limit,
)


def test_normalize_sql_limit_should_use_default_when_invalid() -> None:
    """
    测试非法 limit 会回退默认值。

    功能：
        limit 为空、负数或无法转换为整数时，使用 default_limit。

    参数：
        无。

    返回值：
        None。
    """

    assert normalize_sql_limit(
        limit=None,
        default_limit=20,
        max_limit=100,
    ) == 20
    assert normalize_sql_limit(
        limit=-1,
        default_limit=20,
        max_limit=100,
    ) == 20
    assert normalize_sql_limit(
        limit="bad",
        default_limit=20,
        max_limit=100,
    ) == 20


def test_normalize_sql_limit_should_cap_to_max_limit() -> None:
    """
    测试 limit 超过最大值时会被压到 max_limit。

    功能：
        防止一次查询返回过多数据。

    参数：
        无。

    返回值：
        None。
    """

    assert normalize_sql_limit(
        limit=999,
        default_limit=20,
        max_limit=100,
    ) == 100


def test_ensure_readonly_sql_should_allow_select_and_with() -> None:
    """
    测试允许 SELECT 和 WITH 查询。

    功能：
        只读查询会返回去除前后空白和结尾分号后的 SQL。

    参数：
        无。

    返回值：
        None。
    """

    assert ensure_readonly_sql(
        " SELECT * FROM dogs; "
    ) == "SELECT * FROM dogs"
    assert ensure_readonly_sql(
        "WITH recent AS (SELECT * FROM dogs) SELECT * FROM recent"
    ).startswith(
        "WITH recent"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE dogs SET name = 'x'",
        "DELETE FROM dogs",
        "DROP TABLE dogs",
        "INSERT INTO dogs(name) VALUES ('x')",
        "PRAGMA writable_schema = 1",
        "SELECT * FROM dogs; DROP TABLE dogs",
    ],
)
def test_ensure_readonly_sql_should_reject_dangerous_sql(
    sql: str,
) -> None:
    """
    测试危险 SQL 会被拒绝。

    功能：
        写操作、多语句和 PRAGMA 等高风险 SQL 不允许执行。

    参数：
        sql:
            待测试的 SQL。

    返回值：
        None。
    """

    with pytest.raises(
        ValueError,
    ):
        ensure_readonly_sql(
            sql
        )


def test_ensure_safe_identifier_should_allow_simple_table_name() -> None:
    """
    测试安全表名可以通过校验。

    功能：
        字母、数字、下划线组合的表名允许使用。

    参数：
        无。

    返回值：
        None。
    """

    assert ensure_safe_identifier(
        identifier="memory_items_1",
        field_name="table_name",
    ) == "memory_items_1"


def test_ensure_safe_identifier_should_reject_injection_like_name() -> None:
    """
    测试疑似注入的表名会被拒绝。

    功能：
        表名中包含分号、空格或 SQL 片段时不允许通过。

    参数：
        无。

    返回值：
        None。
    """

    with pytest.raises(
        ValueError,
    ):
        ensure_safe_identifier(
            identifier="dogs; DROP TABLE dogs",
            field_name="table_name",
        )
