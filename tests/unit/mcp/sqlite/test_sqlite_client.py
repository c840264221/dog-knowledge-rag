"""
SQLite MCP 只读客户端测试。

功能：
    使用临时 SQLite 数据库验证 list_tables、describe_table、select_rows 和 readonly_query。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.mcp.sqlite.client import SQLiteMcpReadonlyClient


def create_test_database(
    database_path: Path,
) -> None:
    """
    创建测试 SQLite 数据库。

    功能：
        在临时路径中创建 dogs 表，并插入两行测试数据。

    参数：
        database_path:
            SQLite 数据库文件路径。

    返回值：
        None。
    """

    connection = sqlite3.connect(
        database_path
    )

    try:
        connection.execute(
            """
            CREATE TABLE dogs (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                size TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO dogs(name, size) VALUES (?, ?)",
            [
                (
                    "Golden Retriever",
                    "large",
                ),
                (
                    "Poodle",
                    "medium",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def build_test_client(
    tmp_path: Path,
) -> SQLiteMcpReadonlyClient:
    """
    构建测试 SQLite MCP 只读客户端。

    功能：
        创建临时数据库，并将其注册为 allowlist 中的 dogs_db。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        SQLiteMcpReadonlyClient:
            测试客户端。
    """

    database_path = tmp_path / "dogs.sqlite3"
    create_test_database(
        database_path=database_path,
    )

    return SQLiteMcpReadonlyClient(
        allowed_databases={
            "dogs_db": database_path,
        },
        default_limit=1,
        max_limit=2,
    )


def test_list_tables_should_return_user_tables(
    tmp_path: Path,
) -> None:
    """
    测试列出 SQLite 用户表。

    功能：
        确认 list_tables 能返回 dogs 表。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    result = client.list_tables(
        database_name="dogs_db",
    )

    assert result.database_name == "dogs_db"
    assert result.tables == [
        "dogs",
    ]


def test_describe_table_should_return_columns(
    tmp_path: Path,
) -> None:
    """
    测试查看 SQLite 表结构。

    功能：
        确认 describe_table 能返回 dogs 表字段信息。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    result = client.describe_table(
        database_name="dogs_db",
        table_name="dogs",
    )

    assert result.table_name == "dogs"
    assert [
        column.name
        for column in result.columns
    ] == [
        "id",
        "name",
        "size",
    ]
    assert result.columns[0].primary_key is True
    assert result.columns[1].not_null is True


def test_select_rows_should_return_limited_rows(
    tmp_path: Path,
) -> None:
    """
    测试查询表前 N 行。

    功能：
        确认 select_rows 使用传入 limit，并返回普通 dict 行数据。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    result = client.select_rows(
        database_name="dogs_db",
        table_name="dogs",
        limit=1,
    )

    assert result.limit == 1
    assert result.rows == [
        {
            "id": 1,
            "name": "Golden Retriever",
            "size": "large",
        }
    ]


def test_select_rows_should_cap_limit(
    tmp_path: Path,
) -> None:
    """
    测试 select_rows 会限制最大返回行数。

    功能：
        请求 limit 超过 max_limit 时，实际只返回 max_limit 行。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    result = client.select_rows(
        database_name="dogs_db",
        table_name="dogs",
        limit=99,
    )

    assert result.limit == 2
    assert len(
        result.rows
    ) == 2


def test_run_readonly_query_should_return_rows(
    tmp_path: Path,
) -> None:
    """
    测试执行只读 SQL 查询。

    功能：
        SELECT 查询会被包装并追加 LIMIT，返回受控行数。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    result = client.run_readonly_query(
        database_name="dogs_db",
        sql="SELECT name FROM dogs ORDER BY id",
        limit=2,
    )

    assert result.row_count == 2
    assert result.rows == [
        {
            "name": "Golden Retriever",
        },
        {
            "name": "Poodle",
        },
    ]


def test_run_readonly_query_should_reject_write_sql(
    tmp_path: Path,
) -> None:
    """
    测试写 SQL 会被拒绝。

    功能：
        run_readonly_query 不允许 UPDATE 等写操作。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    with pytest.raises(
        ValueError,
    ):
        client.run_readonly_query(
            database_name="dogs_db",
            sql="UPDATE dogs SET name = 'x'",
        )


def test_client_should_reject_database_outside_allowlist(
    tmp_path: Path,
) -> None:
    """
    测试拒绝访问白名单外数据库。

    功能：
        用户只能传数据库别名，不能绕过 allowlist 访问任意路径。

    参数：
        tmp_path:
            pytest 临时目录。

    返回值：
        None。
    """

    client = build_test_client(
        tmp_path=tmp_path,
    )

    with pytest.raises(
        ValueError,
    ):
        client.list_tables(
            database_name="unknown_db",
        )
