"""
SQLite MCP 只读客户端。

功能：
    提供只读方式查看 SQLite 数据库的能力：
    1. 列出表。
    2. 查看表结构。
    3. 查看表前 N 行。
    4. 执行只读 SQL 查询。

设计原则：
    1. 只允许访问 allowlist（白名单）中的数据库别名。
    2. 不允许用户直接传入任意数据库路径。
    3. 只允许 SELECT / WITH 查询。
    4. 限制返回行数，避免一次性读取过多数据。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.mcp.sqlite.readonly_guard import (
    ensure_readonly_sql,
    ensure_safe_identifier,
    normalize_sql_limit,
)
from src.mcp.sqlite.schemas import (
    SQLiteColumnInfo,
    SQLiteDescribeTableResult,
    SQLiteListTablesResult,
    SQLiteQueryResult,
    SQLiteSelectRowsResult,
)


class SQLiteMcpReadonlyClient:
    """
    SQLite MCP 只读客户端。

    功能：
        通过数据库白名单访问 SQLite 文件，并提供只读查询能力。

    参数：
        allowed_databases:
            数据库别名到 SQLite 文件路径的映射。

        default_limit:
            默认返回行数。

        max_limit:
            最大返回行数。

    返回值：
        SQLiteMcpReadonlyClient:
            可执行只读 SQLite 查询的客户端对象。
    """

    def __init__(
        self,
        allowed_databases: dict[str, str | Path],
        default_limit: int = 20,
        max_limit: int = 100,
    ) -> None:
        """
        初始化 SQLite MCP 只读客户端。

        功能：
            保存数据库白名单，并归一化默认 limit 和最大 limit。

        参数：
            allowed_databases:
                数据库别名到 SQLite 文件路径的映射。

            default_limit:
                默认返回行数。

            max_limit:
                最大返回行数。

        返回值：
            None。
        """

        self.allowed_databases = {
            name: Path(
                path
            )
            for name, path in allowed_databases.items()
        }
        self.max_limit = max(
            1,
            int(
                max_limit
            ),
        )
        self.default_limit = normalize_sql_limit(
            limit=default_limit,
            default_limit=20,
            max_limit=self.max_limit,
        )

    def resolve_database_path(
        self,
        database_name: str,
    ) -> Path:
        """
        根据数据库别名解析 SQLite 文件路径。

        功能：
            只允许访问初始化时传入白名单的数据库别名。

        参数：
            database_name:
                数据库别名。

        返回值：
            Path:
                SQLite 文件路径。

        异常：
            ValueError:
                数据库别名不存在或数据库文件不存在时抛出。
        """

        if database_name not in self.allowed_databases:
            raise ValueError(
                f"数据库不在允许访问列表中: {database_name}"
            )

        database_path = self.allowed_databases[database_name]

        if not database_path.exists():
            raise ValueError(
                f"SQLite 数据库文件不存在: {database_path}"
            )

        return database_path

    def connect_readonly(
        self,
        database_name: str,
    ) -> sqlite3.Connection:
        """
        以只读模式连接 SQLite 数据库。

        功能：
            使用 SQLite URI mode=ro 打开数据库，降低误写风险。

        参数：
            database_name:
                数据库别名。

        返回值：
            sqlite3.Connection:
                SQLite 只读连接。
        """

        database_path = self.resolve_database_path(
            database_name=database_name,
        )
        connection = sqlite3.connect(
            f"file:{database_path}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row

        return connection

    def list_tables(
        self,
        database_name: str,
    ) -> SQLiteListTablesResult:
        """
        列出 SQLite 数据库中的用户表。

        功能：
            查询 sqlite_master，返回 type='table' 且非 sqlite_ 内部表的表名。

        参数：
            database_name:
                数据库别名。

        返回值：
            SQLiteListTablesResult:
                表名列表结果。
        """

        connection = self.connect_readonly(
            database_name=database_name,
        )

        try:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        finally:
            connection.close()

        return SQLiteListTablesResult(
            database_name=database_name,
            tables=[
                str(
                    row["name"]
                )
                for row in rows
            ],
        )

    def describe_table(
        self,
        database_name: str,
        table_name: str,
    ) -> SQLiteDescribeTableResult:
        """
        查看 SQLite 表结构。

        功能：
            使用 PRAGMA table_info 读取字段信息。
            table_name 会先经过安全标识符校验。

        参数：
            database_name:
                数据库别名。

            table_name:
                表名。

        返回值：
            SQLiteDescribeTableResult:
                表字段结构结果。
        """

        safe_table_name = ensure_safe_identifier(
            identifier=table_name,
            field_name="table_name",
        )

        connection = self.connect_readonly(
            database_name=database_name,
        )

        try:
            rows = connection.execute(
                f"PRAGMA table_info({safe_table_name})"
            ).fetchall()
        finally:
            connection.close()

        return SQLiteDescribeTableResult(
            database_name=database_name,
            table_name=safe_table_name,
            columns=[
                SQLiteColumnInfo(
                    name=str(
                        row["name"]
                    ),
                    type=str(
                        row["type"]
                        or ""
                    ),
                    not_null=bool(
                        row["notnull"]
                    ),
                    default_value=row["dflt_value"],
                    primary_key=bool(
                        row["pk"]
                    ),
                )
                for row in rows
            ],
        )

    def select_rows(
        self,
        database_name: str,
        table_name: str,
        limit: int | None = None,
    ) -> SQLiteSelectRowsResult:
        """
        查询 SQLite 表前 N 行。

        功能：
            校验 table_name 后执行 SELECT * FROM table LIMIT ?。

        参数：
            database_name:
                数据库别名。

            table_name:
                表名。

            limit:
                用户请求返回的最大行数。

        返回值：
            SQLiteSelectRowsResult:
                表行数据结果。
        """

        safe_table_name = ensure_safe_identifier(
            identifier=table_name,
            field_name="table_name",
        )
        resolved_limit = normalize_sql_limit(
            limit=limit,
            default_limit=self.default_limit,
            max_limit=self.max_limit,
        )

        connection = self.connect_readonly(
            database_name=database_name,
        )

        try:
            rows = connection.execute(
                f"SELECT * FROM {safe_table_name} LIMIT ?",
                (
                    resolved_limit,
                ),
            ).fetchall()
        finally:
            connection.close()

        return SQLiteSelectRowsResult(
            database_name=database_name,
            table_name=safe_table_name,
            rows=rows_to_dicts(
                rows=rows,
            ),
            limit=resolved_limit,
        )

    def run_readonly_query(
        self,
        database_name: str,
        sql: str,
        limit: int | None = None,
    ) -> SQLiteQueryResult:
        """
        执行只读 SQL 查询。

        功能：
            先通过 readonly_guard 校验 SQL，
            再用子查询包裹原 SQL 并追加 LIMIT，确保返回行数受控。

        参数：
            database_name:
                数据库别名。

            sql:
                用户输入的只读 SQL。

            limit:
                用户请求返回的最大行数。

        返回值：
            SQLiteQueryResult:
                只读查询结果。
        """

        readonly_sql = ensure_readonly_sql(
            sql=sql,
        )
        resolved_limit = normalize_sql_limit(
            limit=limit,
            default_limit=self.default_limit,
            max_limit=self.max_limit,
        )
        executable_sql = (
            "SELECT * FROM ("
            f"{readonly_sql}"
            ") AS readonly_query_result LIMIT ?"
        )

        connection = self.connect_readonly(
            database_name=database_name,
        )

        try:
            rows = connection.execute(
                executable_sql,
                (
                    resolved_limit,
                ),
            ).fetchall()
        finally:
            connection.close()

        result_rows = rows_to_dicts(
            rows=rows,
        )

        return SQLiteQueryResult(
            database_name=database_name,
            sql=readonly_sql,
            rows=result_rows,
            row_count=len(
                result_rows
            ),
            limit=resolved_limit,
        )


def rows_to_dicts(
    rows: list[sqlite3.Row],
) -> list[dict[str, Any]]:
    """
    将 sqlite3.Row 列表转换成普通 dict 列表。

    功能：
        避免上层直接依赖 sqlite3.Row 对象，方便后续写入 ToolResult.content。

    参数：
        rows:
            sqlite3.Row 列表。

    返回值：
        list[dict[str, Any]]:
            普通字典格式的行数据列表。
    """

    return [
        dict(
            row
        )
        for row in rows
    ]
