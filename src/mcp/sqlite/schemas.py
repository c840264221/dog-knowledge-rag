"""
SQLite MCP 只读结果 Schema。

功能：
    定义 SQLite MCP Readonly Client 返回给上层的标准数据结构。

专业名词：
    Schema：
        数据结构 / 数据模型，用于约束字段和类型。
    Readonly：
        只读，只允许查询，不允许修改数据库。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SQLiteColumnInfo(BaseModel):
    """
    SQLite 表字段信息。

    功能：
        描述 SQLite 表中的一个字段。

    参数：
        name:
            字段名称。

        type:
            字段类型。

        not_null:
            是否不允许为空。

        default_value:
            默认值。

        primary_key:
            是否为主键字段。

    返回值：
        SQLiteColumnInfo:
            Pydantic 数据对象，可通过 model_dump 转成普通 dict。
    """

    name: str = Field(
        description="字段名称",
    )
    type: str = Field(
        default="",
        description="字段类型",
    )
    not_null: bool = Field(
        default=False,
        description="是否不允许为空",
    )
    default_value: Any = Field(
        default=None,
        description="字段默认值",
    )
    primary_key: bool = Field(
        default=False,
        description="是否主键",
    )


class SQLiteListTablesResult(BaseModel):
    """
    SQLite 表列表结果。

    功能：
        描述某个允许访问数据库中的全部表名。

    参数：
        database_name:
            数据库别名。

        tables:
            表名列表。

    返回值：
        SQLiteListTablesResult:
            Pydantic 数据对象。
    """

    database_name: str = Field(
        description="数据库别名",
    )
    tables: list[str] = Field(
        default_factory=list,
        description="表名列表",
    )


class SQLiteDescribeTableResult(BaseModel):
    """
    SQLite 表结构结果。

    功能：
        描述某张表的字段结构。

    参数：
        database_name:
            数据库别名。

        table_name:
            表名。

        columns:
            字段信息列表。

    返回值：
        SQLiteDescribeTableResult:
            Pydantic 数据对象。
    """

    database_name: str = Field(
        description="数据库别名",
    )
    table_name: str = Field(
        description="表名",
    )
    columns: list[SQLiteColumnInfo] = Field(
        default_factory=list,
        description="字段信息列表",
    )


class SQLiteSelectRowsResult(BaseModel):
    """
    SQLite 表行数据结果。

    功能：
        描述从某张表中读取到的前 N 行数据。

    参数：
        database_name:
            数据库别名。

        table_name:
            表名。

        rows:
            行数据列表。

        limit:
            实际使用的行数限制。

    返回值：
        SQLiteSelectRowsResult:
            Pydantic 数据对象。
    """

    database_name: str = Field(
        description="数据库别名",
    )
    table_name: str = Field(
        description="表名",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="行数据列表",
    )
    limit: int = Field(
        description="实际查询行数限制",
    )


class SQLiteQueryResult(BaseModel):
    """
    SQLite 只读 SQL 查询结果。

    功能：
        描述一次只读 SQL 查询的返回数据。

    参数：
        database_name:
            数据库别名。

        sql:
            实际执行的 SQL。

        rows:
            行数据列表。

        row_count:
            返回行数。

        limit:
            实际使用的行数限制。

    返回值：
        SQLiteQueryResult:
            Pydantic 数据对象。
    """

    database_name: str = Field(
        description="数据库别名",
    )
    sql: str = Field(
        description="实际执行的 SQL",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="行数据列表",
    )
    row_count: int = Field(
        default=0,
        description="返回行数",
    )
    limit: int = Field(
        description="实际查询行数限制",
    )
