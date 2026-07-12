import sqlite3

import pytest
from pydantic import ValidationError

from src.memory.memory_document_mapper import (
    memory_dict_to_document,
)
from src.memory.memory_schema import (
    MemoryOutput,
)
from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore,
)


def test_memory_output_should_accept_standard_contract() -> None:
    """
    测试 MemoryOutput 接受完整的标准记忆抽取结果。

    功能：
        验证统一记忆类型和 importance 重要程度字段可以正常通过校验。

    参数：
        无。

    返回值：
        None，pytest 根据断言判断测试是否通过。
    """

    result = MemoryOutput(
        should_save=True,
        memory_type="profile",
        content="用户正在学习企业级 Agent 开发",
        confidence=0.95,
        importance=0.9,
        reason="用户明确表达了稳定背景信息",
    )

    assert result.memory_type == "profile"
    assert result.importance == 0.9


@pytest.mark.parametrize(
    "field_name,field_value",
    [
        ("memory_type", "temporary_question"),
        ("confidence", 1.1),
        ("importance", -0.1),
    ],
)
def test_memory_output_should_reject_invalid_contract_value(
        field_name: str,
        field_value: object,
) -> None:
    """
    测试 MemoryOutput 拒绝非法记忆契约字段。

    功能：
        验证非法类型以及超出 0 到 1 范围的分数不会进入后续链路。

    参数：
        field_name: 需要替换的字段名称。
        field_value: 用于触发校验失败的非法字段值。

    返回值：
        None，pytest 根据异常断言判断测试是否通过。
    """

    payload = {
        "should_save": True,
        "memory_type": "preference",
        "content": "用户希望使用中文回答",
        "confidence": 0.9,
        "importance": 0.8,
        "reason": "用户明确表达了长期偏好",
    }
    payload[field_name] = field_value

    with pytest.raises(ValidationError):
        MemoryOutput.model_validate(payload)


def test_sqlite_memory_store_should_migrate_legacy_table(
        tmp_path,
) -> None:
    """
    测试 SQLiteMemoryStore 自动迁移旧版记忆表。

    功能：
        先创建不包含新版字段的旧表，再验证 Store 初始化时只补充缺失列。

    参数：
        tmp_path: pytest 提供的临时目录，避免修改真实记忆数据库。

    返回值：
        None，pytest 根据表结构断言判断测试是否通过。
    """

    db_path = tmp_path / "legacy_memory.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            strength REAL DEFAULT 1,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, memory_type, content)
        )
        """
    )
    connection.commit()
    connection.close()

    store = SQLiteMemoryStore(
        db_path=db_path
    )

    try:
        columns = {
            str(row[1])
            for row in store.conn.execute(
                "PRAGMA table_info(user_memory)"
            ).fetchall()
        }

        assert {
            "source",
            "importance",
            "updated_at",
            "expires_at",
        }.issubset(columns)
    finally:
        store.close()


def test_memory_fields_should_flow_from_sqlite_to_chroma_metadata(
        tmp_path,
) -> None:
    """
    测试新版记忆字段从 SQLite 传递到 Chroma metadata。

    功能：
        写入一条完整记忆，回查 SQLite 后转换为向量文档并验证元数据。

    参数：
        tmp_path: pytest 提供的临时目录，避免修改真实记忆数据库。

    返回值：
        None，pytest 根据 SQLite 记录和 Document metadata 判断是否通过。
    """

    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory.sqlite3"
    )

    try:
        memory_id = store.add_memory(
            user_id="user_001",
            memory_type="preference",
            content="用户希望技术名词附带中文解释",
            confidence=0.95,
            source="conversation",
            importance=0.9,
            expires_at="2027-01-01T00:00:00",
        )

        assert memory_id is not None

        memory = store.get_memory_by_id(
            memory_id
        )
        assert memory is not None

        document = memory_dict_to_document(
            memory
        )

        assert document.page_content == "用户希望技术名词附带中文解释"
        assert document.metadata["memory_id"] == str(memory_id)
        assert document.metadata["source"] == "conversation"
        assert document.metadata["importance"] == 0.9
        assert document.metadata["updated_at"]
        assert document.metadata["expires_at"] == "2027-01-01T00:00:00"
    finally:
        store.close()
