import sqlite3

import pytest
from pydantic import ValidationError

from src.memory.memory_document_mapper import (
    memory_dict_to_document,
)
from src.memory.memory_manager import (
    MemoryManager,
)
from src.memory.memory_schema import (
    MemoryOutput,
)
from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore,
)
from src.graph.nodes.memory_extract_node import (
    is_memory_save_success,
)


class FakeMemoryVectorStore:
    """记录 MemoryManager 对 Chroma 执行的删除和写入操作。"""

    def __init__(self) -> None:
        self.deleted_ids: list[list[str]] = []
        self.added_items: list[dict] = []

    def delete(self, ids: list[str]) -> None:
        """记录删除的 Chroma 文档编号。"""

        self.deleted_ids.append(
            ids
        )

    def add_documents(
            self,
            documents,
            ids: list[str],
    ) -> None:
        """记录重新写入的文档和 Chroma 文档编号。"""

        self.added_items.append(
            {
                "documents": documents,
                "ids": ids,
            }
        )


class FakeVectorStoreProvider:
    """向 MemoryManager 提供测试用 memory_db。"""

    def __init__(self) -> None:
        self.memory_db = FakeMemoryVectorStore()


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


def test_memory_manager_should_create_and_reinforce_memory(
        tmp_path,
) -> None:
    """
    测试记忆首次创建和重复强化使用同一条记录。

    功能：
        验证首次保存返回 created，重复保存返回 reinforced 且 strength 增加。

    参数：
        tmp_path: pytest 临时目录，用于创建隔离的 SQLite 数据库。

    返回值：
        None，pytest 根据断言判断测试是否通过。
    """

    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory_manager.sqlite3"
    )
    manager = MemoryManager(
        store=store
    )

    try:
        created = manager.save_memory(
            user_id="user_001",
            memory_type="favorite_dog",
            content="金毛",
            confidence=0.9,
            importance=0.8,
        )
        reinforced = manager.save_memory(
            user_id="user_001",
            memory_type="favorite_dog",
            content="用户喜欢金毛",
            confidence=0.8,
            importance=0.9,
        )

        assert created["action"] == "created"
        assert reinforced["action"] == "reinforced"
        assert reinforced["memory_id"] == created["memory_id"]
        assert reinforced["strength"] > created["strength"]
        assert reinforced["importance"] == 0.9
    finally:
        store.close()


def test_memory_manager_should_reactivate_inactive_memory(
        tmp_path,
) -> None:
    """
    测试发生偏好反转后可以重新激活历史失效记忆。

    功能：
        依次保存喜欢、厌恶、再次喜欢金毛，验证旧 favorite_dog 记录被复用。

    参数：
        tmp_path: pytest 临时目录，用于创建隔离的 SQLite 数据库。

    返回值：
        None，pytest 根据状态和编号断言判断测试是否通过。
    """

    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory_reactivation.sqlite3"
    )
    vectorstore_provider = FakeVectorStoreProvider()
    manager = MemoryManager(
        store=store,
        vectorstore_provider=vectorstore_provider,
    )

    try:
        favorite = manager.save_memory(
            user_id="user_001",
            memory_type="favorite_dog",
            content="金毛",
            confidence=0.9,
            importance=0.8,
            expires_at="2027-01-01T00:00:00",
        )
        dislike = manager.save_memory(
            user_id="user_001",
            memory_type="dislike",
            content="金毛",
            confidence=0.95,
            importance=0.9,
        )
        reactivated = manager.save_memory(
            user_id="user_001",
            memory_type="favorite_dog",
            content="金毛",
            confidence=0.98,
            importance=0.95,
        )

        favorite_record = store.get_memory_by_id(
            favorite["memory_id"]
        )
        dislike_record = store.get_memory_by_id(
            dislike["memory_id"]
        )

        assert reactivated["action"] == "reactivated"
        assert reactivated["memory_id"] == favorite["memory_id"]
        assert favorite_record["status"] == "active"
        assert favorite_record["expires_at"] is None
        assert dislike_record["status"] == "inactive"

        last_vector_write = (
            vectorstore_provider
            .memory_db
            .added_items[-1]
        )
        assert last_vector_write["ids"] == [
            f"memory_{favorite['memory_id']}"
        ]
        assert (
            last_vector_write["documents"][0].metadata["status"]
            == "active"
        )
        assert (
            last_vector_write["documents"][0].metadata["expires_at"]
            == ""
        )
    finally:
        store.close()


def test_memory_manager_should_skip_empty_content(
        tmp_path,
) -> None:
    """
    测试空记忆内容不会写入数据库。

    功能：
        验证归一化后为空的内容返回 skipped 且没有 memory_id。

    参数：
        tmp_path: pytest 临时目录，用于创建隔离的 SQLite 数据库。

    返回值：
        None，pytest 根据保存结果判断测试是否通过。
    """

    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory_skipped.sqlite3"
    )
    manager = MemoryManager(
        store=store
    )

    try:
        result = manager.save_memory(
            user_id="user_001",
            memory_type="preference",
            content="   ",
            confidence=0.9,
        )

        assert result["action"] == "skipped"
        assert result["memory_id"] is None
    finally:
        store.close()


@pytest.mark.parametrize(
    "save_result,expected",
    [
        ({"action": "created", "memory_id": 1}, True),
        ({"action": "reinforced", "memory_id": 1}, True),
        ({"action": "reactivated", "memory_id": 1}, True),
        ({"action": "skipped", "memory_id": None}, False),
        ({"action": "failed", "memory_id": None}, False),
        (None, False),
    ],
)
def test_is_memory_save_success_should_match_real_write_action(
        save_result,
        expected: bool,
) -> None:
    """
    测试节点只把真实创建或更新动作标记为保存成功。

    功能：
        验证 created、reinforced、reactivated 与 skipped、failed 的区别。

    参数：
        save_result: MemoryManager 返回的保存结果。
        expected: 期望的成功判断。

    返回值：
        None，pytest 根据断言判断测试是否通过。
    """

    assert is_memory_save_success(
        save_result
    ) is expected
